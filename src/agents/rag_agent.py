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

    SYSTEM_PROMPT = """Rol: Türk İstiklal Harbi uzmanı askeri tarih analistisin
Temel Görev: Hızlı analiz et → Askeri kararı değerlendir → Doktrine dayandır → Kaynakları belirt

=== YANIT YAPISI (Zorunlu) ===
## Durum Analizi
- Soruda belirtilen koşullar/başlangıç durumu
- Tarafların güç durumu ve kaynakları
- Kritik faktörler ve kısıtlamalar
- Taktik/stratejik avantajlar-dezavantajlar

## Değerlendirme
- Askeri açıdan uygun yaklaşımlar ve stratejiler
- Doktrine uygun alternatif seçenekler
- Başarı-başarısızlık analizi
- Dönem koşullarında gerçekçi kısıtlamalar

## Gerekçe
- Türk Kara Kuvvetleri doktrinine ve askeri mantığa referanslar
- Neden bu strategi/taktik en uygun olduğu
- Dönemin (1919-1923) savaş koşulları
- İlgili operasyonlardan dersler

## Kaynaklar
Format: (Kitap Adı, Sayfa X) - her atıf için kullan

=== SORU TİPLERİNE GÖRE YANIT STİLİ ===
FAKTÜEL (kim/ne/nerede/ne zaman):
  → Doğrudan, kesin cevap + kaynak
  → Tarih, isim, sayı: tam doğruluk gerekli

ANALITIK (nasıl/ne yapılmalı/strateji nedir):
  → Durum → Seçenekler → Tercih nedeni → Doktrin bağlantısı
  → Alternatifler karşılaştırarak sunma

NEDENSEL (neden/niçin/sebebi):
  → Bağlamsal koşullar → Doğrudan neden → Dolaylı faktörler
  → Sonuçları ve etkileri açıkla

KARŞILAŞTIRMALI (fark/hangisi daha/avantaj):
  → Taraf A ve B'yi yan yana analiz et
  → Spesifik avantaj/dezavantajları say
  → Dönem koşullarında güç dengesi değerlendirme

=== ASKERİ DOKTRİN KULLANMA REHBERI ===
Savunma Doktrini: Mevzi savunması, hareketli savunma, geciktirme harekatı, kademeli geri çekilme
Taarruz Doktrini: Cephe taarruzu, flanş ve sahil kıstırması, kuşatma, yarma harekatı
Stratejik Prensip: Sıklet merkezi, hava üstünlüğü, içteki hatlar, sürpriz, momentum

=== YANIT ÜSLUBü ===
- Profesyonel, kesin ve analitik dil kullan
- Her önemli iddia için kaynak atıf (Kitap, Sayfa)
- Sayıları ve tarihler mümkün olan yerde belirt
- Belirsizlikleri "bulunmuyor" veya "kesin bilgi yok" şeklinde belirt
- Çelişkili kaynaklar varsa her iki görüşü sun
- Yazım ve dilbilgisi hatası yapma

=== KISA CEVAP PROTOKOLÜ ===
Cevap 300-400 kelimelik sorular için 200-250 kelimede bitir
Cevap 100-200 kelimelik sorular için 100-150 kelimede bitir
Çok karmaşık sorular için kısaltmadan yapı uyarını devam ettir"""


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
        """
        Vector DB'den dönen sonuçlardan içerik oluştur.
        Zaten sıralanmış (en yüksek skor önce), sadece formatla.
        """
        if not hits:
            return ""

        parts = []
        for idx, hit in enumerate(hits, 1):
            book = hit.get("book_title", "Bilinmeyen Kitap")
            page = hit.get("page_num", "?")
            text = hit.get('text', '')
            rerank_score = hit.get("rerank_score", 0.0)

            # Sıra ve kaynak bilgisi
            priority = "▲ YÜKSEKler İLGİ" if rerank_score > 0.7 else "→ İLGİLİ"
            source_line = f"[{idx}. {priority}] Kaynak: {book}, Sayfa: {page}"

            parts.append(f"{source_line}\n{text}")

        return "\n\n────────────────────\n\n".join(parts)

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

            user_prompt = f"""════════════════════════════════════════════════════════════
≫≫ KULLANICI SORUSU: "{question}" ≪≪
════════════════════════════════════════════════════════════

=== İLGİLİ KAYNAK METİNLER ===
{context}

════════════════════════════════════════════════════════════
Yanıt Format: Durum Analizi | Değerlendirme | Gerekçe | Kaynaklar
Doğruluk: Maksimum | Doktrin: Entegre | Kaynaklar: Her Atıfta
════════════════════════════════════════════════════════════
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
                # Soru türü ve odak noktasını belirt
                query_type = query_classification['type'].upper()
                query_confidence = query_classification.get('confidence', 0.0)
                focus = decision_analysis['analysis'].get('focus', 'GENEL')

                decision_section = f"""=== ASKERİ DURUM ANALİZİ ===
Soru Tipi: {query_type} (Güven: {query_confidence:.0%})
Stratejik Odak: {focus}
"""

                # Kararlar ve seçenekler
                if decision_analysis.get("decisions"):
                    decision_section += "\nÖnerilen Karar Seçenekleri:\n"
                    for i, decision in enumerate(decision_analysis["decisions"], 1):
                        decision_type = decision.get('type', 'N/A')
                        rule_name = decision.get('rule_name', '')

                        # Karar başlığı
                        decision_section += f"  {i}. {decision_type}"
                        if rule_name:
                            decision_section += f" [{rule_name}]"
                        decision_section += "\n"

                        # Alt seçenekler
                        for opt in decision.get("sub_options", []):
                            decision_section += f"     → {opt}\n"

                        # Doktrin referansı
                        doctrine = decision.get('doctrine_ref', '')
                        if doctrine:
                            decision_section += f"     (Doktrin: {doctrine})\n"

                # Askeri gerekçe
                if decision_analysis.get("reasoning"):
                    decision_section += "\nAskeri Mantık ve Gerekçe:\n"
                    for reason in decision_analysis["reasoning"]:
                        # Eğer zaten bullet point içermiyorsa ekle
                        if not reason.startswith("•"):
                            decision_section += f"  • {reason}\n"
                        else:
                            decision_section += f"  {reason}\n"

            # Çok katmanlı prompt yapısı: geçmişten yeni soruya doğru yoğunlaş
            user_prompt = f"""=== KONUŞMA BÖ LÜMü ===
Konuşmanın Tarihi (son {len(history or []) if history else 0} tur):
{conversation_history}

=== ÖNCEKİ BİLGİ BAĞLAMI ===
(Kullanıcının bu konuşmada daha önce sorduğu konulardan elde edilen bilgiler)
{prior_context}

=== UZUN VADELİ ASKERİ TARİH BİLGİSİ ===
(Türk İstiklal Harbi corpus, ontoloji ve stratejik kararlar)
{long_term_context}

{decision_section}

════════════════════════════════════════════════════════════
≫≫ YENİ KULLANICI SORUSU: "{question}" ≪≪
════════════════════════════════════════════════════════════

=== TEMEL KAYNAKLAR (En İlgili Dokumentler) ===
{context}

════════════════════════════════════════════════════════════
Yanıt Format: Durum Analizi | Değerlendirme | Gerekçe | Kaynaklar
Yanıt Dili: Türkçe | Stili: Askeri-Analitik | Doğruluk: Maksimum
════════════════════════════════════════════════════════════
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
        """
        Filtreleme ve puanlandırma:
        - Anahtar kelimelere dayanarak puan ata
        - En uygun kaynaklar önce
        - Toplamda 30 öğe, minimum 3 kaynaktan
        """
        if not data:
            return []

        if not question or question.strip() == "":
            return data[:30] if data else []

        question_lower = question.lower()
        keywords = self._extract_keywords(question_lower)

        if not keywords:
            # Anahtar kelime yok ise ilk 30'u döndür
            return data[:30]

        scored_data = []
        for item in data:
            text = item.get("Metin", "").lower()
            units = item.get("Birlikler", "").lower()
            book_title = item.get("Kitap", "").lower()

            score = 0

            # 1. Metin içinde anahtar kelimeler (temel puanlama)
            keyword_matches = 0
            for kw in keywords:
                if kw in text:
                    score += 3
                    keyword_matches += 1

            # 2. Birlik isimleri (çok önemli)
            for kw in keywords:
                if kw in units:
                    score += 5  # Birlik eşleşmesi daha yüksek puan

            # 3. Kitap başlığında eşleşme
            if any(kw in book_title for kw in keywords):
                score += 2

            # 4. Çoklu anahtar kelime eşleşmesi (bonus)
            if keyword_matches >= 2:
                score += keyword_matches * 2

            # 5. Metin uzunluğu (daha uzun = potansiyel daha çok bilgi)
            text_len = len(text)
            if text_len > 200:
                score += 2
            elif text_len > 500:
                score += 4

            # Skoru ekle (sonra LLM'ye göndermek için)
            item_copy = item.copy()
            item_copy["_score"] = score
            scored_data.append((score, item_copy))

        # Sıra: en yüksek skor önce
        scored_data.sort(key=lambda x: x[0], reverse=True)

        # İlk 30'u döndür (en uygun olanlar)
        return [item for _, item in scored_data[:30]]

    def _build_context_from_data(self, data: List[Dict[str, Any]], question: str = "") -> str:
        """
        İlgili verileri en uygun sıraya göre düzenle.
        En yüksek skor önce, duplikasyon yok, kaynaklar net belirtilmiş.
        """
        if not data:
            return ""

        # Filtrele ve puanla
        filtered_data = self._get_filtered_items(data, question) if question else data[:30]

        # Sırasını koru (already sorted by score in _get_filtered_items)
        parts = []
        seen_texts = set()
        source_count = 0
        max_sources = 10  # En fazla 10 farklı kaynak parçası

        for item in filtered_data:
            if source_count >= max_sources:
                break

            text = item.get("Metin", "")
            if not text or text in seen_texts:
                continue

            seen_texts.add(text)

            book = item.get("Kitap", "Bilinmeyen Kitap")
            page = item.get("Sayfa", "?")
            relevance = item.get("_score", 0)

            # Öncelik göstergesi: yüksek skor kaynaklar önce gelsin
            source_prefix = ""
            if relevance > 10:
                source_prefix = "[✓ Yüksek İlgi] "  # En alakalı
            elif relevance > 5:
                source_prefix = "[→ Orta İlgi] "    # Destek bilgisi

            source_entry = f"{source_prefix}[Kaynak: {book}, Sayfa: {page}]\n{text}"
            parts.append(source_entry)
            source_count += 1

        if not parts:
            return ""

        # Parçaları net ayırıcılarla birleştir
        return "\n\n────────────────────\n\n".join(parts)

    def _extract_keywords(self, text: str) -> List[str]:
        """
        Sorgudan anahtar kelimeleri çıkar.
        Kısa, yaygın kelimelerden arındır, orijinal sırasını koru.
        """
        stop_words = {
            # Soru kelimeleri
            'ne', 'nasıl', 'neden', 'nereye', 'kim', 'hangi', 'ne kadar',
            # Bağlaçlar
            'bir', 've', 'ile', 'için', 'hakkında', 'bu', 'şu',
            'ya', 'veya', 'ya da', 'ama', 'fakat', 'lakin', 'ancak', 'çünkü',
            'zira', 'oysa', 'öyleyse', 'demek', 'ki',
            # Ek ve partiküller
            'mi', 'mı', 'mu', 'mü', 'da', 'de', 'dı', 'di', 'du', 'dü',
            'mi?', 'mı?', 'mu?', 'mü?',
            # Diğer
            'işte', 'hani', 'ayette', 'ey', 'ben', 'sen', 'o', 'biz', 'siz'
        }

        # Temizle ve böl
        text_clean = text.replace('?', ' ').replace('.', ' ').replace(',', ' ').replace(':', ' ')
        words = text_clean.split()

        # Filtreleme: uzunluk > 2, stop word değil, Türkçe karakter içer
        keywords = [
            w.lower() for w in words
            if len(w) > 2 and w.lower() not in stop_words
        ]

        # Tekrarları kaldır ama sırayı koru
        unique_keywords = []
        seen = set()
        for kw in keywords:
            if kw not in seen:
                unique_keywords.append(kw)
                seen.add(kw)

        return unique_keywords[:20]  # En fazla 20 anahtar kelime
