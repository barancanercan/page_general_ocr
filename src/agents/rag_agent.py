import logging
import time
import ollama
from typing import List, Tuple, Optional, Dict, Any, Union
from collections import defaultdict

from src.config import settings
from src.services.embedding_service import EmbeddingService
from src.services.vector_db_service import VectorDBService
from src.utils.military_extraction import extract_units
from src.utils.normalization import normalize_unit_name
from src.agents.memory import GlobalMemory, get_long_term_memory, QueryClassifier, DecisionEngine

logger = logging.getLogger(__name__)

class RAGAgent:

    SYSTEM_PROMPT = """Türk İstiklal Harbi uzmanı askeri tarih analistisin. Görevi: ANALİZ ET, KARAR VER, GEREKÇELENDIR.

CEVAP FORMATI:
## Durum Analizi
- Koşullar ve kritik faktörler

## Değerlendirme
- Askeri yaklaşım ve strateji
- Alternatifler (varsa)

## Gerekçe
- Askeri mantık ve doktrin bağlantısı

## Kaynaklar
- (Kitap, Sayfa) formatında

SORU TİPLERİ:
- Faktüel (kim/ne/nerede): Doğrudan bilgi + kaynak
- Analitik (nasıl/strateji): Durum → Karar → Gerekçe
- Nedensel (neden): Koşullar → Sebep-Sonuç
- Karşılaştırmalı: Tablo + analiz

DOKTRİNLER: Mevzi/hareketli savunma, kuşatma, yarma, sıklet merkezi, ikmal hattı.

ÜSLUP: Askeri, kesin, analitik. Her bilgiye kaynak ekle."""


    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.vector_db = VectorDBService()
        self._unit_map = {}
        self.memory = GlobalMemory.get_memory("default")
        self.long_term_memory = get_long_term_memory()

    def get_ingested_books(self) -> List[str]:
        try:
            return self.vector_db.get_ingested_books()
        except Exception as e:
            logger.error(f"Failed to get ingested books: {e}", exc_info=True)
            return []

    def get_all_units(self) -> List[str]:
        """
        Veritabanındaki tüm birlikleri çeker, normalize eder ve gruplar.
        Arayüze sadece temiz isimleri döndürür.
        """
        try:
            raw_units = self.vector_db.get_all_units()
            self._unit_map = defaultdict(list)
            seen_normalized = set()  # Tekrar edenleri önlemek için
            
            for raw in raw_units:
                clean = normalize_unit_name(raw)
                if clean:
                    # Aynı normalize edilmiş isim farklı orijinal versiyonlarla gelebilir
                    # Sadece ilkini kullan
                    if clean not in seen_normalized:
                        self._unit_map[clean].append(raw)
                        seen_normalized.add(clean)
                    else:
                        # Varyasyon olarak ekle
                        self._unit_map[clean].append(raw)
            
            # Temiz isimleri alfabetik döndür
            return sorted(list(self._unit_map.keys()))
        except Exception as e:
            logger.error(f"Failed to get units: {e}", exc_info=True)
            return []

    def _get_unit_variations(self, unit_name: str) -> List[str]:
        if not unit_name or unit_name == "Tüm Birlikler":
            return []
        
        if not self._unit_map:
            self.get_all_units()
            
        return self._unit_map.get(unit_name, [unit_name])

    def _format_source_ref(self, hit: Dict[str, Any]) -> str:
        book = hit.get("book_title", "Bilinmeyen Kitap")
        page = hit.get("page_num", "?")
        return f"({book}, Sayfa {page})"

    def _build_context(self, hits: List[Dict[str, Any]]) -> str:
        if not hits:
            return "" 
        
        parts = []
        for hit in hits:
            book = hit.get("book_title", "Bilinmeyen Kitap")
            page = hit.get("page_num", "?")
            text = hit.get('text', '')
            parts.append(f"[Kaynak: {book}, Sayfa: {page}]\n{text}")
            
        return "\n\n---\n\n".join(parts)

    def _diversify_results(self, candidates: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        top_k = min(top_k, len(candidates))
        guaranteed_top_n = 2
        if top_k < guaranteed_top_n:
            return candidates[:top_k]

        final_selection = []
        seen_books = set()

        for cand in candidates[:guaranteed_top_n]:
            final_selection.append(cand)
            seen_books.add(cand.get("book_title"))
        
        remaining_candidates = candidates[guaranteed_top_n:]
        for cand in remaining_candidates:
            if len(final_selection) >= top_k:
                break
            book_title = cand.get("book_title")
            if book_title not in seen_books:
                final_selection.append(cand)
                seen_books.add(book_title)
        
        if len(final_selection) < top_k:
            selected_docs_content = {doc.get('text') for doc in final_selection}
            for cand in remaining_candidates:
                if len(final_selection) >= top_k:
                    break
                if cand.get('text') not in selected_docs_content:
                    final_selection.append(cand)

        final_selection.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        
        return final_selection

    def chat(self, question: str, history: Optional[List[Tuple[str, str]]] = None, book_filter: Optional[str] = None, unit_filter: Optional[str] = None) -> Tuple[str, List[str], Dict[str, float]]:
        timing = {"embed": 0.0, "search": 0.0, "rerank": 0.0, "llm": 0.0, "total": 0.0}
        t_start = time.time()

        try:
            t0 = time.time()
            question_entities = extract_units(question)
            query_vector = self.embedding_service.embed_query(question)
            timing["embed"] = time.time() - t0
            
            target_units = self._get_unit_variations(unit_filter)
            
            logger.info(f"Question: '{question}' | Unit Filter: {unit_filter} (Variations: {target_units}) | Book Filter: {book_filter}")

            t1 = time.time()
            candidates = self.vector_db.hybrid_search(
                query_vector, 
                entities=question_entities,
                top_k=settings.RAG_FETCH_K, 
                book_filter=book_filter,
                unit_filter=target_units
            )
            timing["search"] = time.time() - t1
            
            if not candidates:
                msg = "Aradığınız kriterlere uygun kaynak bulunamadı."
                if unit_filter:
                    msg += f" (Seçilen Birlik: {unit_filter})"
                return msg, [], timing

            t2 = time.time()
            candidate_texts = [c.get("text", "") for c in candidates]
            rerank_scores = self.embedding_service.rerank(question, candidate_texts)
            
            for i, score in enumerate(rerank_scores):
                candidates[i]["rerank_score"] = score
            
            candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
            timing["rerank"] = time.time() - t2

            final_hits = self._diversify_results(candidates, settings.RAG_TOP_K)
            
            context = self._build_context(final_hits)
            
            if not context:
                 return "Verilen kaynaklarda bu konu hakkında spesifik bir bilgi bulunmamaktadır.", [], timing

            user_prompt = f"""Kullanıcı Sorusu: "{question}"

--- KAYNAK METİNLER ---
{context}
"""
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]

            t3 = time.time()
            response = ollama.chat(
                model=settings.CHAT_MODEL,
                messages=messages,
                options={"temperature": 0.0}
            )
            answer = response["message"]["content"].strip()
            timing["llm"] = time.time() - t3
            
            timing["total"] = time.time() - t_start
            
            sources = sorted(list(set(self._format_source_ref(hit) for hit in final_hits)))

            return answer, sources, timing

        except Exception as e:
            logger.error(f"RAG chat error: {e}", exc_info=True)
            return "Sistemde beklenmeyen bir hata oluştu. Lütfen tekrar deneyin.", [], timing

    def chat_with_context(self, question: str, history: Optional[List[Tuple[str, str]]] = None, book_filter: Optional[str] = None, unit_filter: Optional[Union[str, List[str]]] = None, context_data: Optional[List[Dict[str, Any]]] = None, session_id: str = "default") -> Tuple[str, List[str], Dict[str, float]]:
        timing = {"llm": 0.0, "total": 0.0}
        t_start = time.time()

        try:
            if not context_data:
                return "Önce 'Verileri Getir' butonuna tıklayarak birlik verilerini yükleyin.", [], timing

            self.memory = GlobalMemory.get_memory(session_id)
            
            query_classification = QueryClassifier.classify(question)
            logger.info(f"Query classified as: {query_classification['type']} (confidence: {query_classification['confidence']:.2f})")
            
            decision_analysis = DecisionEngine.analyze(
                question, 
                query_classification["type"], 
                context_data
            )
            
            prior_context = self._get_prior_context(question, context_data)
            
            context = self._build_context_from_data(context_data, question)
            
            conversation_history = self._format_conversation_history(history)
            
            long_term_context = self.long_term_memory.get_context_for_query(question)
            
            decision_section = ""
            if decision_analysis["requires_decision"]:
                decision_section = f"""
=== ASKERİ DURUM ANALİZİ ===
Soru Tipi: {query_classification['type'].upper()}
Odak Noktası: {decision_analysis['analysis'].get('focus', 'N/A')}

"""
                if decision_analysis.get("decisions"):
                    decision_section += "Karar Seçenekleri:\n"
                    for i, decision in enumerate(decision_analysis["decisions"], 1):
                        decision_section += f"  {i}. {decision.get('type', 'N/A')}\n"
                        for opt in decision.get("sub_options", []):
                            decision_section += f"     - {opt}\n"
                
                if decision_analysis.get("reasoning"):
                    decision_section += "\nAskeri Değerlendirme:\n"
                    for reason in decision_analysis["reasoning"]:
                        decision_section += f"  • {reason}\n"

            user_prompt = f"""Kullanıcı Soruları ve Cevapları:
{conversation_history}

Kullanıcının Önceki Sorularından Elde Edilen Bilgiler:
{prior_context}

=== UZUN VADELİ HAFIZA (Tarihi Corpus + Ontology) ===
{long_term_context}

{decision_section}

---
YENİ KULLANICI SORUSU: "{question}"

--- KAYNAK METİNLER ---
{context}
"""

            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]

            t3 = time.time()
            response = ollama.chat(
                model=settings.CHAT_MODEL,
                messages=messages,
                options={"temperature": 0.0}
            )
            answer = response["message"]["content"].strip()
            timing["llm"] = time.time() - t3

            timing["total"] = time.time() - t_start

            # Extract sources from filtered data (not full context_data)
            filtered_items = self._get_filtered_items(context_data, question)
            sources = sorted(list(set(
                f"({item.get('Kitap', 'Bilinmiyor')}, Sayfa {item.get('Sayfa', '?')})"
                for item in filtered_items
            )))
            
            self.memory.add_message("user", question)
            self.memory.add_message("assistant", answer)

            # Use actual message count instead of manual tracking
            if len(self.memory.messages) >= 10:
                self._create_summary()

            return answer, sources, timing

        except Exception as e:
            logger.error(f"RAG chat_with_context error: {e}", exc_info=True)
            return f"Sistemde bir hata oluştu: {e}", [], timing

    def _get_prior_context(self, current_question: str, current_data: List[Dict[str, Any]]) -> str:
        memory_context = self.memory.get_recent_context()
        
        if not memory_context:
            return "Bu konuşmadaki ilk soru."
        
        current_keywords = set(current_question.lower().split())
        relevant_info = []
        
        for msg in self.memory.messages:
            if msg["role"] == "assistant":
                if any(kw in msg["content"].lower() for kw in current_keywords):
                    relevant_info.append(msg["content"][:300])
        
        if relevant_info:
            return "\n\n".join(relevant_info[:2])
        
        return memory_context

    def _format_conversation_history(self, history: Optional[List[Tuple[str, str]]]) -> str:
        if not history:
            return "Önceki konuşma yok."
        
        lines = []
        for i, (user_msg, assistant_msg) in enumerate(history[-5:], 1):
            lines.append(f"{i}- Soru: {user_msg[:150]}")
            lines.append(f"  Cevap: {assistant_msg[:200]}...")
        
        return "\n".join(lines)

    def _create_summary(self):
        summary_prompt = """Aşağıdaki konuşmanın kısa bir özetini çıkar. 
Önemli sorulan konuları ve verilen cevapların özetini yaz.

Konuşma:
"""
        for msg in self.memory.messages:
            role = "Kullanıcı" if msg["role"] == "user" else "Asistan"
            summary_prompt += f"{role}: {msg['content'][:300]}\n"

        try:
            response = ollama.chat(
                model=settings.CHAT_MODEL,
                messages=[{"role": "user", "content": summary_prompt}],
                options={"temperature": 0.0}
            )
            summary = response["message"]["content"].strip()
            self.memory.update_summary(summary)
        except Exception as e:
            logger.warning(f"Summary creation failed: {e}")

    def _get_filtered_items(self, data: List[Dict[str, Any]], question: str = "") -> List[Dict[str, Any]]:
        """Returns filtered and scored items for source extraction."""
        if not data or not question:
            return data[:30] if data else []

        question_lower = question.lower()
        keywords = self._extract_keywords(question_lower)

        scored_data = []
        for item in data:
            text = item.get("Metin", "").lower()
            units = item.get("Birlikler", "").lower()

            score = 0
            for kw in keywords:
                if kw in text:
                    score += 3
                if kw in units:
                    score += 5

            book_title = item.get("Kitap", "").lower()
            if any(kw in book_title for kw in keywords):
                score += 2

            scored_data.append((score, item))

        scored_data.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored_data[:30]]

    def _build_context_from_data(self, data: List[Dict[str, Any]], question: str = "") -> str:
        if not data:
            return ""

        # Use shared filtering method
        filtered_data = self._get_filtered_items(data, question) if question else data[:30]

        parts = []
        seen_texts = set()
        for item in filtered_data:
            text = item.get("Metin", "")
            if text in seen_texts:
                continue
            seen_texts.add(text)

            book = item.get("Kitap", "Bilinmeyen Kitap")
            page = item.get("Sayfa", "?")
            parts.append(f"[Kaynak: {book}, Sayfa: {page}]\n{text}")

        return "\n\n---\n\n".join(parts)

    def _extract_keywords(self, text: str) -> List[str]:
        stop_words = {
            'bir', 've', 'ile', 'için', 'hakkında', 'ne', 'nasıl', 'neden', 'nereye',
            'kim', 'hangi', 'bu', 'şu', 'mi', 'mı', 'mu', 'mü', 'da', 'de',
            'ya', 'veya', 'ya da', 'ama', 'fakat', 'lakin', 'ancak', 'çünkü',
            'zira', 'oysa', 'öyleyse', 'demek', 'ki', 'işte', 'hani', 'ayette', 'ey'
        }
        
        words = text.replace('?', ' ').replace('.', ' ').replace(',', ' ').replace(':', ' ').split()
        keywords = [w for w in words if len(w) > 2 and w not in stop_words]
        
        unique_keywords = []
        seen = set()
        for kw in keywords:
            if kw not in seen:
                unique_keywords.append(kw)
                seen.add(kw)
        
        return unique_keywords[:15]
