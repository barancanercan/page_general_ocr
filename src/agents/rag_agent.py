import logging
import time
import ollama
from typing import List, Tuple, Optional, Dict, Any
from collections import defaultdict

from src.config import settings
from src.services.embedding_service import EmbeddingService
from src.services.vector_db_service import VectorDBService
from src.utils.military_extraction import extract_units

logger = logging.getLogger(__name__)

class RAGAgent:
    """
    Final RAG Agent: Fast, single-stage pipeline with Re-ranking and Diversification.
    """

    SYSTEM_PROMPT = """Sen, kaynakları titizlikle analiz eden bir Türk askeri tarihi araştırmacısısın. Görevin, sana aşağıda `--- KAYNAK METİNLER ---` başlığı altında sunulan metinleri kullanarak kullanıcının sorusuna kapsamlı bir cevap oluşturmaktır.

**KESİN KURALLAR:**
1.  **SENTEZLE VE BİRLEŞTİR:** Farklı kaynaklardaki bilgileri bir araya getirerek tutarlı bir anlatı oluştur.
2.  **ASLA PES ETME:** Eğer kaynaklarda konuyla ilgili en ufak bir bilgi bile varsa, "bilgi bulunmamaktadır" demek **KESİNLİKLE YASAKTIR**. Bu cevabı yalnızca ve yalnızca sana *hiçbir* kaynak metin verilmediğinde kullanabilirsin.
3.  **KAYNAK GÖSTER:** Cevabındaki her bir bilgi parçası için, cümlenin veya paragrafın sonuna, metnin başındaki `[Kaynak: ...]` etiketini kullanarak `(Kitap Adı, Sayfa X)` formatında referans ver.
4.  **YAPISAL VE NET OL:** Cevabını, kullanıcının kolayca anlayabileceği şekilde maddeler veya net paragraflar halinde düzenle.
5.  **SADECE VERİLEN KAYNAKLARI KULLAN:** Dışarıdan asla bilgi ekleme. Tüm cevabın, sana sunulan metinlere dayanmalıdır.
"""

    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.vector_db = VectorDBService()

    def get_ingested_books(self) -> List[str]:
        """
        Retrieves a list of all unique book titles from the vector database.
        """
        try:
            return self.vector_db.get_ingested_books()
        except Exception as e:
            logger.error(f"Failed to get ingested books: {e}", exc_info=True)
            return []

    def _format_source_ref(self, hit: Dict[str, Any]) -> str:
        book = hit.get("book_title", "Bilinmeyen Kitap")
        page = hit.get("page_num", "?")
        return f"({book}, Sayfa {page})"

    def _build_context(self, hits: List[Dict[str, Any]]) -> str:
        if not hits:
            return "" # Return empty string if no hits
        
        # A daha temiz ve LLM'in kafasını karıştırmayacak format
        parts = []
        for hit in hits:
            book = hit.get("book_title", "Bilinmeyen Kitap")
            page = hit.get("page_num", "?")
            text = hit.get('text', '')
            parts.append(f"[Kaynak: {book}, Sayfa: {page}]\n{text}")
            
        return "\n\n---\n\n".join(parts)

    def _diversify_results(self, candidates: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """
        Selects a diverse set of results using a hybrid strategy.
        It guarantees the top N most relevant results are included, then fills the rest
        with diverse sources from other books.
        """
        if not candidates:
            return []

        top_k = min(top_k, len(candidates))
        guaranteed_top_n = 2 # Performans için 3'ten 2'ye düşürüldü
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

    def chat(self, question: str, history: Optional[List[Tuple[str, str]]] = None, book_filter: Optional[str] = None) -> Tuple[str, List[str], Dict[str, float]]:
        timing = {"embed": 0.0, "search": 0.0, "rerank": 0.0, "llm": 0.0, "total": 0.0}
        t_start = time.time()

        try:
            t0 = time.time()
            question_entities = extract_units(question)
            query_vector = self.embedding_service.embed_query(question)
            timing["embed"] = time.time() - t0
            logger.info(f"Question: '{question}' | Extracted Entities: {question_entities}")

            t1 = time.time()
            candidates = self.vector_db.hybrid_search(
                query_vector, 
                entities=question_entities,
                top_k=settings.RAG_FETCH_K, 
                book_filter=book_filter
            )
            timing["search"] = time.time() - t1
            
            if not candidates:
                return "Aradığınız kriterlere uygun kaynak bulunamadı. Lütfen farklı bir soru sorun veya veritabanına yeni belgeler ekleyin.", [], timing

            t2 = time.time()
            candidate_texts = [c.get("text", "") for c in candidates]
            rerank_scores = self.embedding_service.rerank(question, candidate_texts)
            
            for i, score in enumerate(rerank_scores):
                candidates[i]["rerank_score"] = score
            
            candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
            timing["rerank"] = time.time() - t2

            final_hits = self._diversify_results(candidates, settings.RAG_TOP_K)
            logger.info(f"Selected {len(final_hits)} diversified sources for LLM from {len(set(c.get('book_title') for c in final_hits))} unique books.")

            context = self._build_context(final_hits)
            
            if not context:
                 return "Verilen kaynaklarda bu konu hakkında spesifik bir bilgi bulunmamaktadır.", [], timing

            # Kullanıcı sorusunu ve kaynakları tek bir mesaja birleştirme
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
                options={"temperature": 0.1}
            )
            answer = response["message"]["content"].strip()
            timing["llm"] = time.time() - t3
            
            timing["total"] = time.time() - t_start
            
            sources = sorted(list(set(self._format_source_ref(hit) for hit in final_hits)))

            return answer, sources, timing

        except Exception as e:
            logger.error(f"RAG chat error: {e}", exc_info=True)
            return f"Sistemde bir hata oluştu: {e}", [], timing
