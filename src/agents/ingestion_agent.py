import logging
import time
import os
from pathlib import Path
from typing import List, Callable, Optional

from src.config import settings
from src.core.models import Paragraph
from src.services.ocr_service import OCRService
from src.services.embedding_service import EmbeddingService
from src.services.vector_db_service import VectorDBService
from src.utils.text_processing import split_paragraphs
from src.utils.military_extraction import extract_units

logger = logging.getLogger(__name__)

class IngestionAgent:
    """Agent responsible for ingesting PDFs into the system."""

    def __init__(self):
        self.ocr_service = OCRService()
        self.vector_db = VectorDBService()
        self.embedding_service = EmbeddingService()

    def _validate_pdf_path(self, pdf_path: str) -> Path:
        path = Path(pdf_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"PDF dosyası bulunamadı: {pdf_path}")
        if path.suffix.lower() != '.pdf':
            raise ValueError("Sadece PDF dosyaları kabul edilir")
        return path

    def _make_book_title(self, pdf_path: str) -> str:
        stem = Path(pdf_path).stem
        import re
        title = re.sub(r'[_\-]+', ' ', stem)
        return title.strip()

    def _make_paragraph_id(self, book_title: str, page_num: int, para_idx: int) -> str:
        import re
        safe = re.sub(r'\W+', '_', book_title.lower()).strip('_')
        return f"{safe}_p{page_num:03d}_{para_idx:03d}"

    def ingest_pdf(self, pdf_path: str, max_pages: int = 0, progress_callback: Optional[Callable[[str], None]] = None) -> dict:
        def _progress(msg: str):
            logger.info(msg)
            if progress_callback:
                progress_callback(msg)

        validated_path = self._validate_pdf_path(pdf_path)
        pdf_path_str = str(validated_path)

        book_title = self._make_book_title(pdf_path_str)

        if self.vector_db.is_book_ingested(book_title):
            msg = f"'{book_title}' already ingested, skipping."
            _progress(msg)
            return {"status": "skipped", "message": msg, "paragraphs": 0}

        try:
            # 1. OCR (Sequential to avoid pypdfium2 access violations and Ollama overload)
            _progress(f"[1/4] Starting OCR for: {book_title}")
            t0 = time.time()
            
            # Get total pages first
            try:
                import pypdfium2 as pdfium
                with open(pdf_path_str, 'rb') as pdf_file:
                    pdf = pdfium.PdfDocument(pdf_file)
                    total_pages = len(pdf)
            except Exception as e:
                return {"status": "error", "message": f"Could not read PDF: {e}", "paragraphs": 0}

            pages_to_process = total_pages
            if max_pages > 0:
                pages_to_process = min(total_pages, max_pages)

            page_results = []
            
            # Process pages sequentially
            for i in range(pages_to_process):
                try:
                    # Update progress for UI
                    _progress(f"Processing page {i+1}/{pages_to_process}...")
                    result = self.ocr_service.process_page(pdf_path, i)
                    page_results.append(result)
                except Exception as exc:
                    logger.error(f"Page {i+1} processing failed: {exc}")

            _progress(f"[1/4] OCR completed: {len(page_results)} pages ({time.time()-t0:.1f}s)")

            # 2. Paragraph Splitting & Entity Extraction
            _progress("[2/4] Splitting paragraphs and extracting entities...")
            t1 = time.time()
            all_paragraphs = []
            
            for page_result in page_results:
                if not page_result.success or not page_result.text.strip():
                    continue
                
                texts = split_paragraphs(page_result.text, min_length=settings.MIN_PARAGRAPH_LENGTH)
                page_para_count = len(texts)
                
                for idx, text in enumerate(texts):
                    units = extract_units(text)
                    para = Paragraph(
                        paragraph_id=self._make_paragraph_id(book_title, page_result.page_num, idx),
                        text=text,
                        book_title=book_title,
                        page_num=page_result.page_num,
                        paragraph_index=idx,
                        page_paragraph_count=page_para_count,
                        military_units=units,
                        confidence=page_result.confidence,
                        model_used=page_result.model_used
                    )
                    all_paragraphs.append(para)

            if not all_paragraphs:
                msg = "No paragraphs extracted."
                _progress(msg)
                return {"status": "error", "message": msg, "paragraphs": 0}

            _progress(f"[2/4] {len(all_paragraphs)} paragraphs extracted ({time.time()-t1:.1f}s)")

            # 3. Embedding
            _progress("[3/4] Generating embeddings...")
            t2 = time.time()
            texts = [p.text for p in all_paragraphs]
            vectors = self.embedding_service.embed_texts(texts)
            _progress(f"[3/4] {len(vectors)} embeddings generated ({time.time()-t2:.1f}s)")

            # 4. Vector DB Upsert
            _progress("[4/4] Saving to Vector DB...")
            t3 = time.time()
            self.vector_db.upsert_paragraphs(all_paragraphs, vectors)
            _progress(f"[4/4] Saved to Vector DB ({time.time()-t3:.1f}s)")

            total_time = time.time() - t0
            msg = f"Completed: {len(all_paragraphs)} paragraphs saved. (Total: {total_time:.1f}s)"
            _progress(msg)
            return {"status": "ok", "message": msg, "paragraphs": len(all_paragraphs)}

        except Exception as e:
            msg = f"Error: {e}"
            logger.error(msg, exc_info=True)
            _progress(msg)
            return {"status": "error", "message": msg, "paragraphs": 0}
