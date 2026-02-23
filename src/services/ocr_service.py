import os
import time
import logging
import threading
import ollama
import pypdfium2 as pdfium
from PIL import Image
from typing import List, Optional

from src.config import settings
from src.core.models import PageResult, ProcessingStats
from src.utils.text_processing import post_process_text, calculate_confidence

logger = logging.getLogger(__name__)

class OCRService:
    _instance = None
    _lock = threading.Lock()
    _client: Optional[ollama.Client] = None
    
    OCR_PROMPT = (
        "You are an OCR engine. "
        "Extract all visible text from this image ONCE. "
        "Do NOT repeat any text. "
        "Do NOT explain. Do NOT summarize. Do NOT add anything. "
        "Return only the extracted text without any repetition."
    )

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance
    
    @property
    def client(self) -> ollama.Client:
        if self._client is None:
            self._client = ollama.Client(timeout=settings.OCR_TIMEOUT)
        assert self._client is not None
        return self._client
    
    def __init__(self):
        pass

    def _pdf_page_to_image(self, pdf_path: str, page_idx: int) -> str:
        """Converts a PDF page to an image file."""
        try:
            settings.TEMP_DIR.mkdir(exist_ok=True)
            with open(pdf_path, 'rb') as pdf_file:
                pdf = pdfium.PdfDocument(pdf_file)
                page = pdf[page_idx]
                bitmap = page.render(scale=int(settings.OCR_DPI / 72))
                img = bitmap.to_pil()

                if img.width > settings.OCR_MAX_IMAGE_WIDTH:
                    ratio = settings.OCR_MAX_IMAGE_WIDTH / img.width
                    img = img.resize(
                        (settings.OCR_MAX_IMAGE_WIDTH, int(img.height * ratio)),
                        Image.Resampling.LANCZOS,
                    )

                if img.mode != "RGB":
                    img = img.convert("RGB")

                path = settings.TEMP_DIR / f"page_{page_idx}_{time.time()}.png"
                img.save(path, "PNG")
                return str(path)
        except Exception as e:
            logger.error(f"Failed to convert page {page_idx} to image: {e}")
            raise

    def _run_ocr_model(self, model_name: str, image_path: str) -> str:
        """Runs the OCR model on an image."""
        try:
            response = self.client.generate(
                model=model_name,
                prompt=self.OCR_PROMPT,
                images=[image_path],
                options={
                    "temperature": settings.OCR_TEMPERATURE,
                    "num_predict": settings.OCR_NUM_PREDICT,
                    "repeat_penalty": settings.OCR_REPETITION_PENALTY,
                    "top_k": settings.OCR_TOP_K,
                    "top_p": settings.OCR_TOP_P,
                },
            )
            return response["response"].strip()
        except ollama.ResponseError as e:
            logger.error(f"Ollama error: {e}")
            raise
        except Exception as e:
            logger.error(f"OCR error: {e}")
            raise

    def process_page(self, pdf_path: str, page_idx: int) -> PageResult:
        """Processes a single page with fallback logic."""
        page_num = page_idx + 1
        start_time = time.time()
        image_path = None

        try:
            image_path = self._pdf_page_to_image(pdf_path, page_idx)
            
            # Try small model first
            model_used = settings.OCR_MODEL_SMALL
            raw_text = self._run_ocr_model(model_used, image_path)
            text = post_process_text(raw_text)
            confidence = calculate_confidence(text)

            # Fallback logic
            if confidence < settings.CONFIDENCE_THRESHOLD or len(text) < settings.MIN_TEXT_LENGTH:
                logger.info(f"Page {page_num}: Low confidence ({confidence:.2f}) or short text. Retrying with large model...")
                model_used = settings.OCR_MODEL_LARGE
                raw_text = self._run_ocr_model(model_used, image_path)
                text = post_process_text(raw_text)
                confidence = calculate_confidence(text)

            elapsed = time.time() - start_time
            logger.info(f"Page {page_num} processed with {model_used} - Conf: {confidence:.2f}, Time: {elapsed:.1f}s")

            return PageResult(
                page_num=page_num,
                text=text,
                confidence=confidence,
                processing_time=elapsed,
                model_used=model_used
            )

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Error processing page {page_num}: {e}")
            return PageResult(
                page_num=page_num,
                text="",
                confidence=0.0,
                processing_time=elapsed,
                model_used="failed",
                success=False,
                error_message=str(e)
            )
        finally:
            if image_path and os.path.exists(image_path):
                try:
                    os.remove(image_path)
                except OSError:
                    pass

    def process_pdf(self, pdf_path: str, max_pages: int = 0) -> List[PageResult]:
        """Processes an entire PDF."""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        try:
            with open(pdf_path, 'rb') as pdf_file:
                pdf = pdfium.PdfDocument(pdf_file)
                total_pages = len(pdf)
        except Exception as e:
            raise Exception(f"Could not read PDF: {e}")

        pages_to_process = total_pages
        if max_pages > 0:
            pages_to_process = min(total_pages, max_pages)

        logger.info(f"Processing {pages_to_process}/{total_pages} pages from {os.path.basename(pdf_path)}")

        results = []
        # Sequential processing for now, can be parallelized by the caller or here
        for i in range(pages_to_process):
            results.append(self.process_page(pdf_path, i))

        return results
