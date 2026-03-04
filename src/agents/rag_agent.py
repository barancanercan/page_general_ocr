import logging
import time
import re
from openai import OpenAI, APIError, RateLimitError, APITimeoutError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from typing import List, Tuple, Optional, Dict, Any, Union
from collections import defaultdict

from src.config import settings
from src.services.embedding_service import EmbeddingService
from src.services.vector_db_service import VectorDBService
from src.utils.military_extraction import extract_units
from src.utils.normalization import normalize_unit_name
from src.agents.memory import GlobalMemory, get_long_term_memory, QueryClassifier, DecisionEngine

logger = logging.getLogger(__name__)

# Turkce kucuk harf donusumu
TR_LOWER_MAP = str.maketrans("ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ", "abcçdefgğhıijklmnoöprsştuüvyz")

def tr_lower(text: str) -> str:
    """Turkce karakterleri dogru sekilde kucuk harfe cevirir"""
    return text.translate(TR_LOWER_MAP).lower()

# OpenAI client
_openai_client: Optional[OpenAI] = None

def get_openai_client() -> OpenAI:
    """OpenAI client singleton"""
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
    return _openai_client


def extract_dates(text: str) -> List[str]:
    """
    Turkce tarih formatlarindan tarih cikarir.
    Ornek: "2 Agustos 1913", "12 Mart 1922", "1919-1922"
    """
    turkish_months = [
        'ocak', 'subat', 'mart', 'nisan', 'mayis', 'haziran',
        'temmuz', 'agustos', 'eylul', 'ekim', 'kasim', 'aralik'
    ]

    dates_found = []

    # Format: GG AA YYYY veya GG AY YYYY
    for month in turkish_months:
        pattern = rf'(\d{{1,2}})\s+{month}\s+(\d{{4}})'
        matches = re.findall(pattern, text.lower())
        for day, year in matches:
            dates_found.append(f"{day} {month} {year}")
            dates_found.append(year)

    # Sadece yil (4 haneli)
    year_pattern = r'\b(19\d{2}|18[789]\d)\b'
    years = re.findall(year_pattern, text)
    dates_found.extend(years)

    return list(set(dates_found))


def is_complex_query(question: str, context_len: int = 0) -> bool:
    """
    Sorunun karmasik olup olmadigini belirler.
    Karmasik sorular GPT-4o, basit sorular GPT-4o-mini kullanir.
    """
    question_lower = question.lower()

    # Karmasik soru isaretleri
    complex_keywords = [
        'analiz', 'karsilastir', 'acikla', 'detayli', 'neden', 'nasil',
        'strateji', 'taktik', 'planlama', 'iliskisi', 'etki', 'sonuc',
        'degerlendirme', 'yorum', 'baglan', 'fark', 'benzerlik'
    ]

    # Basit soru isaretleri
    simple_keywords = [
        'kim', 'ne zaman', 'nerede', 'kac', 'hangi', 'listele', 'say'
    ]

    # Soru uzunlugu kontrolu (uzun sorular genelde karmasik)
    if len(question) > 150:
        return True

    # Context cok buyukse karmasik model kullan
    if context_len > 4000:
        return True

    # Keyword kontrolu
    for kw in complex_keywords:
        if kw in question_lower:
            return True

    # Birden fazla soru isareti (coklu soru)
    if question.count('?') > 1:
        return True

    return False


class RAGAgent:

    SYSTEM_PROMPT = """Sen Turk askeri tarih uzmanisir. TURKCE CEVAP VER.

ZORUNLU KURALLAR:
1. HER ZAMAN TURKCE YAZ - Ingilizce cevap verme!
2. Soruyu DIKKATICE OKU ve ne soruldugunu anla
3. Kaynaklardaki bilgilerle DOGRUDAN CEVAP VER
4. "Tamam", "Hazirim", "Lutfen sorun" gibi bos laflar YAZMA
5. Kisi ismi soruluyorsa, o kisinin bilgilerini kaynaklardan bul ve yaz
6. Cevabina hemen bilgiyle basla"""


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
        Veritabanindaki tum birlikleri ceker, normalize eder ve gruplar.
        Arayuze sadece temiz isimleri dondurur.
        """
        try:
            raw_units = self.vector_db.get_all_units()
            self._unit_map = defaultdict(list)
            seen_normalized = set()

            for raw in raw_units:
                clean = normalize_unit_name(raw)
                if clean:
                    if clean not in seen_normalized:
                        self._unit_map[clean].append(raw)
                        seen_normalized.add(clean)
                    else:
                        self._unit_map[clean].append(raw)

            return sorted(list(self._unit_map.keys()))
        except Exception as e:
            logger.error(f"Failed to get units: {e}", exc_info=True)
            return []

    def _get_unit_variations(self, unit_name: Optional[str]) -> List[str]:
        if not unit_name or unit_name == "Tum Birlikler":
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
        Vector DB'den donen sonuclardan icerik olustur.
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

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((RateLimitError, APITimeoutError)),
        before_sleep=lambda retry_state: logger.warning(f"OpenAI retry {retry_state.attempt_number}/3...")
    )
    def _call_openai(self, messages: List[Dict[str, str]], use_hard_model: bool = False) -> str:
        """
        OpenAI API cagirisi yapar. Rate limiting ve retry destegi var.
        use_hard_model=True: gpt-4o (karmasik gorevler)
        use_hard_model=False: gpt-4o-mini (kolay gorevler)
        """
        client = get_openai_client()
        model = settings.CHAT_MODEL_HARD if use_hard_model else settings.CHAT_MODEL_EASY

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.1,
                max_tokens=1024,
                timeout=60.0
            )
            return response.choices[0].message.content.strip()
        except RateLimitError:
            logger.warning("Rate limit - retry edilecek")
            raise
        except APITimeoutError:
            logger.warning("Timeout - retry edilecek")
            raise
        except APIError as e:
            logger.error(f"OpenAI API hatasi: {e.status_code}")
            return "API hatasi olustu. Lutfen tekrar deneyin."
        except Exception as e:
            logger.error(f"Beklenmeyen hata: {e}")
            return "Bir hata olustu. Lutfen tekrar deneyin."

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
                msg = "Aradiginiz kriterlere uygun kaynak bulunamadi."
                if unit_filter:
                    msg += f" (Secilen Birlik: {unit_filter})"
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
                return "Verilen kaynaklarda bu konu hakkinda spesifik bir bilgi bulunmamaktadir.", [], timing

            user_prompt = f"""Kaynaklara dayanarak su soruyu cevapla:

"{question}"

=== KAYNAKLAR ===
{context}
=================

Bu soruya kaynaklari kullanarak cevap ver. Eger tam bilgi yoksa, ilgili buldugon her seyi yaz."""

            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]

            # Karmasiklik kontrolu
            use_hard = is_complex_query(question, len(context))
            model_used = settings.CHAT_MODEL_HARD if use_hard else settings.CHAT_MODEL_EASY

            t3 = time.time()
            try:
                answer = self._call_openai(messages, use_hard_model=use_hard)
                logger.info(f"Model used: {model_used}")
            except Exception as e:
                logger.error(f"OpenAI error: {e}")
                answer = f"Model hatasi: {str(e)[:100]}"
            timing["llm"] = time.time() - t3

            timing["total"] = time.time() - t_start

            sources = sorted(list(set(self._format_source_ref(hit) for hit in final_hits[:8])))

            return answer, sources, timing

        except Exception as e:
            logger.error(f"RAG chat error: {e}", exc_info=True)
            return "Sistemde beklenmeyen bir hata olustu. Lutfen tekrar deneyin.", [], timing

    def chat_with_context(self, question: str, history: Optional[List[Tuple[str, str]]] = None, book_filter: Optional[str] = None, unit_filter: Optional[Union[str, List[str]]] = None, context_data: Optional[List[Dict[str, Any]]] = None, session_id: str = "default") -> Tuple[str, List[str], Dict[str, float]]:
        timing = {"llm": 0.0, "total": 0.0}
        t_start = time.time()

        try:
            if not context_data:
                return "Once 'Verileri Getir' butonuna tiklayarak birlik verilerini yukleyin.", [], timing

            self.memory = GlobalMemory.get_memory(session_id)

            query_classification = QueryClassifier.classify(question)
            logger.info(f"Query classified as: {query_classification['type']} (confidence: {query_classification['confidence']:.2f})")

            decision_analysis = DecisionEngine.analyze(
                question,
                query_classification["type"],
                context_data
            )

            prior_context = self._get_prior_context(question, context_data)

            context = self._build_context_from_data(context_data, question, unit_filter)

            conversation_history = self._format_conversation_history(history)

            long_term_context = self.long_term_memory.get_context_for_query(question)

            decision_section = ""
            if decision_analysis["requires_decision"]:
                query_type = query_classification['type'].upper()
                query_confidence = query_classification.get('confidence', 0.0)
                focus = decision_analysis['analysis'].get('focus', 'GENEL')

                decision_section = f"""=== ASKERI DURUM ANALIZI ===
Soru Tipi: {query_type} (Guven: {query_confidence:.0%})
Stratejik Odak: {focus}
"""

                if decision_analysis.get("decisions"):
                    decision_section += "\nOnerilen Karar Secenekleri:\n"
                    for i, decision in enumerate(decision_analysis["decisions"], 1):
                        decision_type = decision.get('type', 'N/A')
                        rule_name = decision.get('rule_name', '')

                        decision_section += f"  {i}. {decision_type}"
                        if rule_name:
                            decision_section += f" [{rule_name}]"
                        decision_section += "\n"

                        for opt in decision.get("sub_options", []):
                            decision_section += f"     -> {opt}\n"

                        doctrine = decision.get('doctrine_ref', '')
                        if doctrine:
                            decision_section += f"     (Doktrin: {doctrine})\n"

                if decision_analysis.get("reasoning"):
                    decision_section += "\nAskeri Mantik ve Gerekce:\n"
                    for reason in decision_analysis["reasoning"]:
                        if not reason.startswith("*"):
                            decision_section += f"  * {reason}\n"
                        else:
                            decision_section += f"  {reason}\n"

            unit_context = ""
            if unit_filter and unit_filter != "Tum Birlikler":
                if isinstance(unit_filter, list):
                    unit_name = unit_filter[0] if unit_filter else ""
                else:
                    unit_name = unit_filter
                unit_context = f"[SECILEN BIRLIK: {unit_name}]\n\n"

            user_prompt = f"""{unit_context}Kaynaklara dayanarak su soruyu cevapla:

"{question}"

=== KAYNAKLAR ===
{context}
=================

Bu soruya kaynaklari kullanarak cevap ver. Eger tam bilgi yoksa, ilgili buldugon her seyi yaz."""

            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]

            context_len = len(user_prompt)
            logger.info(f"Context boyutu: {context_len} karakter")

            # Karmasiklik kontrolu
            use_hard = is_complex_query(question, context_len)
            model_used = settings.CHAT_MODEL_HARD if use_hard else settings.CHAT_MODEL_EASY

            t3 = time.time()
            try:
                logger.info(f"OpenAI model: {model_used}")
                answer = self._call_openai(messages, use_hard_model=use_hard)
                logger.info(f"Cevap: {len(answer)} karakter")
            except Exception as e:
                logger.error(f"OpenAI error: {e}")
                answer = "Cevap olusturulamadi. Lutfen tekrar deneyin."
            timing["llm"] = time.time() - t3

            timing["total"] = time.time() - t_start

            filtered_items = self._get_filtered_items(context_data, question, unit_filter)

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

            if len(self.memory.messages) >= 10:
                self._create_summary()

            return answer, sources, timing

        except Exception as e:
            logger.error(f"RAG chat_with_context error: {e}", exc_info=True)
            return f"Sistemde bir hata olustu: {e}", [], timing

    def _get_prior_context(self, current_question: str, current_data: List[Dict[str, Any]]) -> str:
        memory_context = self.memory.get_recent_context()

        if not memory_context:
            return "Bu konusmadaki ilk soru."

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
            return "Onceki konusma yok."

        lines = []
        for i, (user_msg, assistant_msg) in enumerate(history[-5:], 1):
            lines.append(f"{i}- Soru: {user_msg[:150]}")
            lines.append(f"  Cevap: {assistant_msg[:200]}...")

        return "\n".join(lines)

    def _create_summary(self):
        summary_prompt = """Asagidaki konusmanin kisa bir ozetini cikar.
Onemli sorulan konulari ve verilen cevaplarin ozetini yaz.

Konusma:
"""
        for msg in self.memory.messages:
            role = "Kullanici" if msg["role"] == "user" else "Asistan"
            summary_prompt += f"{role}: {msg['content'][:300]}\n"

        try:
            messages = [{"role": "user", "content": summary_prompt}]
            # Ozet basit bir gorev - gpt-4o-mini kullan
            summary = self._call_openai(messages, use_hard_model=False)
            self.memory.update_summary(summary)
        except Exception as e:
            logger.warning(f"Summary creation failed: {e}")

    def _extract_proper_names(self, text: str) -> List[str]:
        """
        Metinden ozel isimleri (kisi adlari) cikarir.
        Buyuk harfle baslayan ardisik kelimeler.
        """
        upper_chars = "ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZabcçdefgğhıijklmnoöprsştuüvyz"

        words = text.split()
        names = []
        current_name = []

        for word in words:
            clean_word = word.strip("?,.:;!()\"'")
            if clean_word and len(clean_word) > 1:
                first_char = clean_word[0]
                # Buyuk harfle baslayan kelimeler
                if first_char.isupper() or first_char in "İÇĞÖŞÜ":
                    current_name.append(clean_word)
                else:
                    if len(current_name) >= 2:
                        names.append(tr_lower(" ".join(current_name)))
                    elif len(current_name) == 1 and len(current_name[0]) > 3:
                        # Tek kelimelik isimler de ekle (soyad vs)
                        names.append(tr_lower(current_name[0]))
                    current_name = []
            else:
                if len(current_name) >= 2:
                    names.append(tr_lower(" ".join(current_name)))
                current_name = []

        if len(current_name) >= 2:
            names.append(tr_lower(" ".join(current_name)))
        elif len(current_name) == 1 and len(current_name[0]) > 3:
            names.append(tr_lower(current_name[0]))

        return names

    def _get_filtered_items(self, data: List[Dict[str, Any]], question: str = "", unit_filter: Optional[Union[str, List[str]]] = None) -> List[Dict[str, Any]]:
        """
        Filtreleme ve puanlandirma:
        - ISIM ESLESMESI - EN YUKSEK ONCELIK (kisi adlari)
        - Tarih eslesmesi varsa yuksek oncelik ver
        - Secilen birlige (unit_filter) oncelik ver
        - Anahtar kelimelere dayanarak ek puan ata
        """
        if not data:
            return []

        if not question or question.strip() == "":
            return data[:30] if data else []

        question_lower = tr_lower(question)
        keywords = self._extract_keywords(question_lower)

        query_dates = extract_dates(question)
        proper_names = self._extract_proper_names(question)

        unit_variations = []
        if unit_filter:
            if isinstance(unit_filter, list):
                unit_variations = [tr_lower(u) for u in unit_filter]
            else:
                unit_variations = [tr_lower(unit_filter)]

        scored_data = []
        for item in data:
            text = tr_lower(item.get("Metin", ""))
            units = tr_lower(item.get("Birlikler", ""))
            book_title = tr_lower(item.get("Kitap", ""))

            score = 0

            # 0. ISIM ESLESMESI - EN YUKSEK ONCELIK (+1000 puan)
            # Ayrica soruda gecen tum onemli kelimeleri dogrudan ara
            question_words = [tr_lower(w) for w in question.split() if len(w) > 3]
            for qw in question_words:
                if qw in text:
                    score += 100  # Her eslesen kelime icin +100

            for name in proper_names:
                name_lower = tr_lower(name)
                if name_lower in text:
                    score += 1000
                    break
                name_parts = name_lower.split()
                matching_parts = sum(1 for part in name_parts if part in text and len(part) > 2)
                if matching_parts >= 2:
                    score += 800
                    break
                elif matching_parts == 1:
                    score += 400

            # 1. TARIH ESLESMESI - COK ONEMLI
            if query_dates:
                for q_date in query_dates:
                    if q_date in text:
                        score += 500
                        break
                    if len(q_date) > 4:
                        parts = q_date.split()
                        if len(parts) >= 2:
                            year = parts[-1] if parts[-1].isdigit() else ""
                            if year and year in text:
                                score += 300
                                break
                    if q_date.isdigit() and len(q_date) == 4:
                        if q_date in text:
                            score += 100
                            break

            # 2. SECILEN BIRLIK ESLESMESI (+50 puan)
            if unit_variations:
                for uv in unit_variations:
                    if uv in units:
                        score += 50
                        break

            # 3. Metin icinde anahtar kelimeler
            keyword_matches = 0
            for kw in keywords:
                if kw in ["ordu", "kolordu", "tumen", "alay", "tugay", "tabur"]:
                    if unit_variations:
                        continue
                if kw in text:
                    score += 3
                    keyword_matches += 1

            # 4. Kitap basliginda eslesme
            if any(kw in book_title for kw in keywords):
                score += 2

            # 5. Coklu anahtar kelime eslesmesi (bonus)
            if keyword_matches >= 2:
                score += keyword_matches * 2

            # 6. Metin uzunlugu
            text_len = len(text)
            if text_len > 500:
                score += 4
            elif text_len > 200:
                score += 2

            item_copy = item.copy()
            item_copy["_score"] = score
            scored_data.append((score, item_copy))

        scored_data.sort(key=lambda x: x[0], reverse=True)

        return [item for _, item in scored_data[:40]]

    def _build_context_from_data(self, data: List[Dict[str, Any]], question: str = "", unit_filter: Optional[Union[str, List[str]]] = None) -> str:
        """
        Ilgili verileri en uygun siraya gore duzenle.
        """
        if not data:
            return ""

        filtered_data = self._get_filtered_items(data, question, unit_filter) if question else data[:30]

        logger.info(f"=== DEBUG: Soru: '{question}' ===")
        for i, item in enumerate(filtered_data[:5]):
            score = item.get("_score", 0)
            book = item.get("Kitap", "?")
            page = item.get("Sayfa", "?")
            text_preview = item.get("Metin", "")[:100]
            logger.info(f"  [{i+1}] Skor:{score} | {book}, s.{page} | {text_preview}...")

        parts = []
        seen_texts = set()
        source_count = 0
        max_sources = 8

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

        context = "\n\n---\n\n".join(parts)

        if len(context) > 6000:
            context = context[:6000] + "\n[... devami kisaltildi ...]"

        return context

    def _extract_keywords(self, text: str) -> List[str]:
        """
        Sorgudan anahtar kelimeleri cikarir.
        """
        stop_words = {
            'ne', 'nasil', 'neden', 'nereye', 'kim', 'hangi', 'ne kadar',
            'bir', 've', 'ile', 'icin', 'hakkinda', 'bu', 'su',
            'ya', 'veya', 'ya da', 'ama', 'fakat', 'lakin', 'ancak', 'cunku',
            'zira', 'oysa', 'oyleyse', 'demek', 'ki',
            'mi', 'mi', 'mu', 'mu', 'da', 'de', 'di', 'di', 'du', 'du',
            'mi?', 'mi?', 'mu?', 'mu?',
            'iste', 'hani', 'ayette', 'ey', 'ben', 'sen', 'o', 'biz', 'siz'
        }

        text_clean = text.replace('?', ' ').replace('.', ' ').replace(',', ' ').replace(':', ' ')
        words = text_clean.split()

        keywords = [
            tr_lower(w) for w in words
            if len(w) > 2 and tr_lower(w) not in stop_words
        ]

        unique_keywords = []
        seen = set()
        for kw in keywords:
            if kw not in seen:
                unique_keywords.append(kw)
                seen.add(kw)

        return unique_keywords[:20]
