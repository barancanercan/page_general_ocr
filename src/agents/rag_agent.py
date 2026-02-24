import logging
import time
import re
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


def extract_dates(text: str) -> List[str]:
    """
    Türkçe tarih formatlarından tarih çıkarır.
    Örnek: "2 Ağustos 1913", "12 Mart 1922", "1919-1922"
    """
    turkish_months = [
        'ocak', 'şubat', 'mart', 'nisan', 'mayıs', 'haziran',
        'temmuz', 'ağustos', 'eylül', 'ekim', 'kasım', 'aralık'
    ]
    
    dates_found = []
    
    # Format: GG AA YYYY veya GG AY YYYY
    for month in turkish_months:
        pattern = rf'(\d{{1,2}})\s+{month}\s+(\d{{4}})'
        matches = re.findall(pattern, text.lower())
        for day, year in matches:
            dates_found.append(f"{day} {month} {year}")
            dates_found.append(year)  # Yılı da ekle
    
    # Sadece yıl (4 haneli)
    year_pattern = r'\b(19\d{2}|18[789]\d)\b'
    years = re.findall(year_pattern, text)
    dates_found.extend(years)
    
    return list(set(dates_found))

class RAGAgent:

    SYSTEM_PROMPT = """Sen Türk askeri tarih uzmanısın. TÜRKÇE CEVAP VER.

ZORUNLU KURALLAR:
1. HER ZAMAN TÜRKÇE YAZ - İngilizce cevap verme!
2. Soruyu DİKKATLİCE OKU ve ne sorulduğunu anla
3. Kaynaklardaki bilgilerle DOĞRUDAN CEVAP VER
4. "Tamamdır", "Hazırım", "Lütfen sorun" gibi boş laflar YAZMA
5. Kişi ismi soruluyorsa, o kişinin bilgilerini kaynaklardan bul ve yaz
6. Cevabına hemen bilgiyle başla"""


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

    def _get_unit_variations(self, unit_name: Optional[str]) -> List[str]:
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
        """
        if not hits:
            return ""

        parts = []
        for idx, hit in enumerate(hits, 1):
            book = hit.get("book_title", "Bilinmeyen Kitap")
            page = hit.get("page_num", "?")
            text = hit.get('text', '')

            parts.append(f"[{idx}] {book}, Sayfa {page}:\n{text}")

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

            user_prompt = f"""Kaynaklara dayanarak şu soruyu cevapla:

"{question}"

=== KAYNAKLAR ===
{context}
=================

Bu soruya kaynakları kullanarak cevap ver. Eğer tam bilgi yoksa, ilgili bulduğun her şeyi yaz."""
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]

            t3 = time.time()
            try:
                response = ollama.chat(
                    model=settings.CHAT_MODEL,
                    messages=messages,
                    options={
                        "temperature": 0.1,
                        "num_ctx": 4096,
                        "num_predict": 512
                    }
                )
                answer = response["message"]["content"].strip()
            except Exception as e:
                logger.error(f"Ollama error: {e}")
                answer = f"Model hatası: {str(e)[:100]}"
            timing["llm"] = time.time() - t3

            timing["total"] = time.time() - t_start

            # Sadece ilk 8 kaynağı göster
            sources = sorted(list(set(self._format_source_ref(hit) for hit in final_hits[:8])))

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

            # unit_filter'ı context oluşturmada kullan - seçilen birliğe öncelik ver
            context = self._build_context_from_data(context_data, question, unit_filter)
            
            conversation_history = self._format_conversation_history(history)
            
            long_term_context = self.long_term_memory.get_context_for_query(question)
            
            decision_section = ""
            if decision_analysis["requires_decision"]:
                query_type = query_classification['type'].upper()
                query_confidence = query_classification.get('confidence', 0.0)
                focus = decision_analysis['analysis'].get('focus', 'GENEL')

                decision_section = f"""=== ASKERİ DURUM ANALİZİ ===
Soru Tipi: {query_type} (Güven: {query_confidence:.0%})
Stratejik Odak: {focus}
"""

                if decision_analysis.get("decisions"):
                    decision_section += "\nÖnerilen Karar Seçenekleri:\n"
                    for i, decision in enumerate(decision_analysis["decisions"], 1):
                        decision_type = decision.get('type', 'N/A')
                        rule_name = decision.get('rule_name', '')

                        decision_section += f"  {i}. {decision_type}"
                        if rule_name:
                            decision_section += f" [{rule_name}]"
                        decision_section += "\n"

                        for opt in decision.get("sub_options", []):
                            decision_section += f"     → {opt}\n"

                        doctrine = decision.get('doctrine_ref', '')
                        if doctrine:
                            decision_section += f"     (Doktrin: {doctrine})\n"

                if decision_analysis.get("reasoning"):
                    decision_section += "\nAskeri Mantık ve Gerekçe:\n"
                    for reason in decision_analysis["reasoning"]:
                        if not reason.startswith("•"):
                            decision_section += f"  • {reason}\n"
                        else:
                            decision_section += f"  {reason}\n"

            # Seçilen birlik bilgisini hazırla
            unit_context = ""
            if unit_filter and unit_filter != "Tüm Birlikler":
                if isinstance(unit_filter, list):
                    unit_name = unit_filter[0] if unit_filter else ""
                else:
                    unit_name = unit_filter
                unit_context = f"[SEÇİLEN BİRLİK: {unit_name}]\n\n"

            # Detaylı açıklama promptu
            user_prompt = f"""{unit_context}Kaynaklara dayanarak şu soruyu cevapla:

"{question}"

=== KAYNAKLAR ===
{context}
=================

Bu soruya kaynakları kullanarak cevap ver. Eğer tam bilgi yoksa, ilgili bulduğun her şeyi yaz."""

            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]

            # DEBUG: Context boyutunu logla
            context_len = len(user_prompt)
            logger.info(f"Context boyutu: {context_len} karakter")
            print(f"[DEBUG] Context boyutu: {context_len} karakter")  # Terminal'de görünsün

            t3 = time.time()
            try:
                print(f"[DEBUG] Ollama çağrılıyor: {settings.CHAT_MODEL}")
                response = ollama.chat(
                    model=settings.CHAT_MODEL,
                    messages=messages,
                    options={
                        "temperature": 0.1,
                        "num_ctx": 4096,
                        "num_predict": 512
                    }
                )
                answer = response["message"]["content"].strip()
                print(f"[DEBUG] Cevap alındı: {len(answer)} karakter")
            except Exception as e:
                logger.error(f"Ollama error: {e}")
                print(f"[DEBUG] HATA: {e}")
                answer = f"Model hatası: {str(e)[:200]}"
            timing["llm"] = time.time() - t3

            timing["total"] = time.time() - t_start

            # Filtrelenmiş verilerden kaynak çıkar
            filtered_items = self._get_filtered_items(context_data, question, unit_filter)
            
            # Sadece context'teki ilk 8 kaynağı göster
            context_sources = []
            seen_books = set()
            for item in filtered_items[:8]:
                book = item.get('Kitap', 'Bilinmiyor')
                if book not in seen_books:
                    context_sources.append(f"({book}, Sayfa {item.get('Sayfa', '?')})")
                    seen_books.add(book)
            sources = context_sources
            
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

    def _extract_proper_names(self, text: str) -> List[str]:
        """
        Metinden özel isimleri (kişi adları) çıkar.
        Büyük harfle başlayan ardışık kelimeler.
        """
        # Türkçe büyük harfler
        upper_chars = "ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ"

        words = text.split()
        names = []
        current_name = []

        for word in words:
            # Soru işareti, nokta vs. temizle
            clean_word = word.strip("?,.:;!()\"'")
            if clean_word and clean_word[0] in upper_chars and len(clean_word) > 1:
                current_name.append(clean_word)
            else:
                if len(current_name) >= 2:  # En az 2 kelimelik isim
                    names.append(" ".join(current_name).lower())
                current_name = []

        # Son ismi de ekle
        if len(current_name) >= 2:
            names.append(" ".join(current_name).lower())

        return names

    def _get_filtered_items(self, data: List[Dict[str, Any]], question: str = "", unit_filter: Optional[Union[str, List[str]]] = None) -> List[Dict[str, Any]]:
        """
        Filtreleme ve puanlandırma:
        - İSİM EŞLEŞMESİ - EN YÜKSEK ÖNCELİK (kişi adları)
        - Tarih eşleşmesi varsa yüksek öncelik ver
        - Seçilen birliğe (unit_filter) öncelik ver
        - Anahtar kelimelere dayanarak ek puan ata
        """
        if not data:
            return []

        if not question or question.strip() == "":
            return data[:30] if data else []

        question_lower = question.lower()
        keywords = self._extract_keywords(question_lower)

        # Tarih çıkarma
        query_dates = extract_dates(question)

        # İsim çıkarma (kişi adları)
        proper_names = self._extract_proper_names(question)

        # Unit filter'ı normalize et
        unit_variations = []
        if unit_filter:
            if isinstance(unit_filter, list):
                unit_variations = [u.lower() for u in unit_filter]
            else:
                unit_variations = [unit_filter.lower()]

        scored_data = []
        for item in data:
            text = item.get("Metin", "").lower()
            units = item.get("Birlikler", "").lower()
            book_title = item.get("Kitap", "").lower()

            score = 0

            # 0. İSİM EŞLEŞMESİ - EN YÜKSEK ÖNCELİK (+1000 puan)
            # Kişi adları (örn: "İbrahim ÇOLAK") metinde geçiyorsa
            for name in proper_names:
                if name in text:
                    score += 1000
                    break
                # İsmin parçaları da kontrol et (sadece soyad vs.)
                name_parts = name.split()
                matching_parts = sum(1 for part in name_parts if part in text and len(part) > 2)
                if matching_parts >= 2:
                    score += 800
                    break
                elif matching_parts == 1 and len(name_parts) <= 2:
                    score += 400

            # 1. TARİH EŞLEŞMESİ - ÇOK ÖNEMLİ
            if query_dates:
                for q_date in query_dates:
                    # Tam tarih eşleşmesi (örn: "2 ağustos 1913") +500
                    if q_date in text:
                        score += 500
                        break

                    # Tarih parçalarıyla eşleşme +300
                    if len(q_date) > 4:  # Tam tarih
                        parts = q_date.split()
                        if len(parts) >= 2:
                            year = parts[-1] if parts[-1].isdigit() else ""
                            if year and year in text:
                                score += 300
                                break

                    # Sadece yıl eşleşmesi +100
                    if q_date.isdigit() and len(q_date) == 4:
                        if q_date in text:
                            score += 100
                            break

            # 2. SEÇİLEN BİRLİK EŞLEŞMESİ (+50 puan)
            if unit_variations:
                for uv in unit_variations:
                    if uv in units:
                        score += 50
                        break

            # 3. Metin içinde anahtar kelimeler (temel puanlama)
            keyword_matches = 0
            for kw in keywords:
                # Genel askeri terimler, birlik filtresi varsa atla
                if kw in ["ordu", "kolordu", "tümen", "alay", "tugay", "tabur"]:
                    if unit_variations:
                        continue
                if kw in text:
                    score += 3
                    keyword_matches += 1

            # 4. Kitap başlığında eşleşme
            if any(kw in book_title for kw in keywords):
                score += 2

            # 5. Çoklu anahtar kelime eşleşmesi (bonus)
            if keyword_matches >= 2:
                score += keyword_matches * 2

            # 6. Metin uzunluğu (daha uzun = potansiyel daha çok bilgi)
            text_len = len(text)
            if text_len > 500:
                score += 4
            elif text_len > 200:
                score += 2

            item_copy = item.copy()
            item_copy["_score"] = score
            scored_data.append((score, item_copy))

        # Sıra: en yüksek skor önce
        scored_data.sort(key=lambda x: x[0], reverse=True)

        # İlk 40'ı döndür (en uygun olanlar)
        return [item for _, item in scored_data[:40]]

    def _build_context_from_data(self, data: List[Dict[str, Any]], question: str = "", unit_filter: Optional[Union[str, List[str]]] = None) -> str:
        """
        İlgili verileri en uygun sıraya göre düzenle.
        En yüksek skor önce, duplikasyon yok, kaynaklar net belirtilmiş.
        Seçilen birlik (unit_filter) varsa ona öncelik ver.
        """
        if not data:
            return ""

        # Filtrele ve puanla - unit_filter'ı da geçir
        filtered_data = self._get_filtered_items(data, question, unit_filter) if question else data[:30]

        # DEBUG: Hangi kayıtlar seçildi?
        logger.info(f"=== DEBUG: Soru: '{question}' ===")
        for i, item in enumerate(filtered_data[:5]):
            score = item.get("_score", 0)
            book = item.get("Kitap", "?")
            page = item.get("Sayfa", "?")
            text_preview = item.get("Metin", "")[:100]
            logger.info(f"  [{i+1}] Skor:{score} | {book}, s.{page} | {text_preview}...")

        # Sırasını koru (already sorted by score in _get_filtered_items)
        parts = []
        seen_texts = set()
        source_count = 0
        max_sources = 8  # 8 kaynak - detay için yeterli

        for item in filtered_data:
            if source_count >= max_sources:
                break

            text = item.get("Metin", "")
            if not text or text in seen_texts:
                continue

            seen_texts.add(text)

            book = item.get("Kitap", "Bilinmeyen Kitap")
            page = item.get("Sayfa", "?")

            source_entry = f"[{source_count + 1}] {book}, Sayfa {page}:\n{text}"
            parts.append(source_entry)
            source_count += 1

        if not parts:
            return ""

        # Parçaları net ayırıcılarla birleştir
        context = "\n\n---\n\n".join(parts)

        # Context'i 6000 karakterle sınırla (model için güvenli)
        if len(context) > 6000:
            context = context[:6000] + "\n[... devamı kısaltıldı ...]"

        return context

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
