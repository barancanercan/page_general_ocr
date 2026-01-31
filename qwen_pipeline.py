#!/usr/bin/env python3
"""
Qwen2.5-VL OCR Pipeline
========================
Qwen2.5-VL vision modeli ile PDF'lerden metin cikarma.
Ollama uzerinden yerel GPU ile calisir.
"""

import os
import sys
import time
import json
import re
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Optional, Dict, Any

import pypdfium2 as pdfium
from PIL import Image
import ollama


# ============================================================================
# Configuration
# ============================================================================

class Config:
    MODEL_NAME: str = "qwen2.5vl:7b"
    MODEL_TIMEOUT: int = 600
    MAX_IMAGE_WIDTH: int = 1800
    DPI: int = 240
    IMAGE_FORMAT: str = "PNG"
    TEMPERATURE: float = 0.0
    NUM_PREDICT: int = 6000
    MAX_PAGES: int = 0  # 0 = tum sayfalar
    OUTPUT_DIR: str = "output"
    TEMP_DIR: str = ".temp_ocr"
    LOG_LEVEL: int = logging.INFO
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class PageResult:
    page_num: int  # PDF sayfa numarasi (1-indexed)
    text: str
    confidence: float
    processing_time: float
    success: bool = True
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProcessingStats:
    total_pages: int = 0
    processed_pages: int = 0
    successful_pages: int = 0
    failed_pages: int = 0
    total_time: float = 0.0
    total_characters: int = 0
    average_confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# Exceptions
# ============================================================================

class QwenOCRError(Exception):
    pass

class ModelNotFoundError(QwenOCRError):
    pass

class PDFProcessingError(QwenOCRError):
    pass

class ImageProcessingError(QwenOCRError):
    pass

class OCRError(QwenOCRError):
    pass


# ============================================================================
# Logging
# ============================================================================

def setup_logging(name: str = "QwenOCR") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(Config.LOG_LEVEL)
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(Config.LOG_LEVEL)
    handler.setFormatter(logging.Formatter(Config.LOG_FORMAT))
    logger.addHandler(handler)
    return logger


# ============================================================================
# Pipeline
# ============================================================================

class QwenOCRPipeline:
    OCR_PROMPT = (
        "You are an OCR engine. "
        "Extract all visible text from this image. "
        "Preserve original formatting as much as possible. "
        "Do not explain. Do not summarize. Do not add anything. "
        "Return only the text."
    )

    TURKISH_CHARS = r'a-zA-ZçğıöşüÇĞİÖŞÜ'

    def __init__(self, max_pages: int = Config.MAX_PAGES,
                 logger: Optional[logging.Logger] = None):
        self.max_pages = max_pages
        self.logger = logger or setup_logging()
        self.client = ollama.Client(timeout=Config.MODEL_TIMEOUT)
        self.stats = ProcessingStats()
        self.results: List[PageResult] = []
        self._verify_model()

    def _verify_model(self):
        try:
            models = self.client.list()
            model_names = [m.model for m in models.models]
            if Config.MODEL_NAME not in model_names:
                raise ModelNotFoundError(
                    f"Model '{Config.MODEL_NAME}' bulunamadi. "
                    f"Calistir: ollama pull {Config.MODEL_NAME}"
                )
            self.logger.info(f"Model '{Config.MODEL_NAME}' hazir")
        except ModelNotFoundError:
            raise
        except Exception as e:
            raise ModelNotFoundError(f"Model dogrulanamadi: {e}")

    # -- Image --

    def _pdf_page_to_image(self, pdf_path: str, page_idx: int) -> str:
        try:
            os.makedirs(Config.TEMP_DIR, exist_ok=True)
            pdf = pdfium.PdfDocument(pdf_path)
            page = pdf[page_idx]
            bitmap = page.render(scale=Config.DPI / 72)
            img = bitmap.to_pil()
            pdf.close()

            if img.width > Config.MAX_IMAGE_WIDTH:
                ratio = Config.MAX_IMAGE_WIDTH / img.width
                img = img.resize(
                    (Config.MAX_IMAGE_WIDTH, int(img.height * ratio)),
                    Image.Resampling.LANCZOS,
                )

            if img.mode != "RGB":
                img = img.convert("RGB")

            path = os.path.join(Config.TEMP_DIR, f"page_{page_idx}.png")
            img.save(path, Config.IMAGE_FORMAT)
            return path
        except Exception as e:
            raise ImageProcessingError(f"Sayfa goruntuye donusturulemedi: {e}")

    # -- Text --

    def _fix_text(self, text: str) -> str:
        tc = self.TURKISH_CHARS
        text = re.sub(rf'([{tc}])-\s*\n\s*([{tc}])', r'\1\2', text)
        text = re.sub(rf'([{tc}])[–—]\s*\n\s*([{tc}])', r'\1\2', text)
        text = text.replace('\u00AD', '')
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def _confidence(self, text: str) -> float:
        if not text:
            return 0.0
        score = min(1.0, len(text) / 2000)
        if re.search(r'[çğıöşüÇĞİÖŞÜ]', text):
            score = min(1.0, score + 0.1)
        return round(score, 3)

    # -- OCR --

    def _ocr(self, image_path: str) -> str:
        try:
            response = self.client.generate(
                model=Config.MODEL_NAME,
                prompt=self.OCR_PROMPT,
                images=[image_path],
                options={
                    "temperature": Config.TEMPERATURE,
                    "num_predict": Config.NUM_PREDICT,
                },
            )
            return response["response"].strip()
        except Exception as e:
            raise OCRError(f"OCR basarisiz: {e}")

    # -- Process --

    def process_page(self, pdf_path: str, page_idx: int) -> PageResult:
        page_num = page_idx + 1
        start = time.time()
        try:
            self.logger.info(f"Sayfa {page_num} isleniyor...")
            img_path = self._pdf_page_to_image(pdf_path, page_idx)
            raw = self._ocr(img_path)
            text = self._fix_text(raw)
            conf = self._confidence(text)
            elapsed = time.time() - start

            self.logger.info(
                f"Sayfa {page_num} tamamlandi - "
                f"{elapsed:.1f}s, {len(text)} karakter, guven: {conf:.2f}"
            )
            return PageResult(page_num, text, conf, elapsed)

        except Exception as e:
            elapsed = time.time() - start
            self.logger.error(f"Sayfa {page_num} hata: {e}")
            return PageResult(page_num, "", 0.0, elapsed, False, str(e))

    def process_pdf(self, pdf_path: str) -> List[PageResult]:
        start = time.time()

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF bulunamadi: {pdf_path}")

        try:
            pdf = pdfium.PdfDocument(pdf_path)
            total = len(pdf)
            pdf.close()
        except Exception as e:
            raise PDFProcessingError(f"PDF okunamadi: {e}")

        count = total if self.max_pages <= 0 else min(total, self.max_pages)

        self.logger.info("=" * 60)
        self.logger.info(f"PDF: {os.path.basename(pdf_path)}")
        self.logger.info(f"Toplam sayfa: {total}, islenecek: {count}")
        self.logger.info(f"DPI: {Config.DPI}, Max genislik: {Config.MAX_IMAGE_WIDTH}px")
        self.logger.info("=" * 60)

        self.results = []
        for i in range(count):
            result = self.process_page(pdf_path, i)
            self.results.append(result)
            self.stats.processed_pages += 1
            if result.success:
                self.stats.successful_pages += 1
                self.stats.total_characters += len(result.text)
            else:
                self.stats.failed_pages += 1

        self.stats.total_pages = total
        self.stats.total_time = time.time() - start

        if self.stats.successful_pages > 0:
            avg = sum(r.confidence for r in self.results if r.success)
            self.stats.average_confidence = round(
                avg / self.stats.successful_pages, 3
            )

        self._print_summary()
        return self.results

    def _print_summary(self):
        s = self.stats
        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("OZET")
        self.logger.info("=" * 60)
        self.logger.info(f"Toplam: {s.total_pages} | Islenen: {s.processed_pages}")
        self.logger.info(f"Basarili: {s.successful_pages} | Basarisiz: {s.failed_pages}")
        self.logger.info(f"Toplam karakter: {s.total_characters}")
        self.logger.info(f"Ortalama guven: {s.average_confidence:.3f}")
        self.logger.info(f"Toplam sure: {s.total_time:.1f}s")
        if s.processed_pages > 0:
            self.logger.info(f"Sayfa basina: {s.total_time / s.processed_pages:.1f}s")
        self.logger.info("=" * 60)

    # -- Save --

    def save_results(self, pdf_path: str,
                     results: Optional[List[PageResult]] = None,
                     fmt: str = "txt") -> str:
        if results is None:
            results = self.results
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = Path(pdf_path).stem

        if fmt == "json":
            path = os.path.join(Config.OUTPUT_DIR, f"{name}_{ts}.json")
            self._save_json(path, results)
        else:
            path = os.path.join(Config.OUTPUT_DIR, f"{name}_{ts}.txt")
            self._save_txt(path, results)

        self.logger.info(f"Sonuclar kaydedildi: {path}")
        return path

    def _save_txt(self, path: str, results: List[PageResult]):
        with open(path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(f"\n{'=' * 60}\n")
                f.write(f"Sayfa {r.page_num}\n")
                f.write(f"{'=' * 60}\n")
                f.write(f"Guven: {r.confidence:.3f} | Sure: {r.processing_time:.1f}s\n")
                if r.error_message:
                    f.write(f"Hata: {r.error_message}\n")
                f.write(f"\n{r.text}\n")

    def _save_json(self, path: str, results: List[PageResult]):
        data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "model": Config.MODEL_NAME,
                "stats": self.stats.to_dict(),
            },
            "pages": [r.to_dict() for r in results],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def cleanup(self):
        d = Config.TEMP_DIR
        if os.path.exists(d):
            for f in os.listdir(d):
                p = os.path.join(d, f)
                if os.path.isfile(p):
                    os.remove(p)


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Qwen2.5-VL OCR Pipeline")
    parser.add_argument("pdf_path", help="PDF dosya yolu")
    parser.add_argument("--max-pages", type=int, default=Config.MAX_PAGES,
                        help="Islenecek max sayfa (0=tumu)")
    parser.add_argument("--format", choices=["txt", "json"], default="txt",
                        help="Cikti formati")
    parser.add_argument("--dpi", type=int, default=Config.DPI,
                        help="Render DPI")
    args = parser.parse_args()

    Config.DPI = args.dpi
    logger = setup_logging()
    pipeline = None

    try:
        pipeline = QwenOCRPipeline(max_pages=args.max_pages, logger=logger)
        results = pipeline.process_pdf(args.pdf_path)
        pipeline.save_results(args.pdf_path, results, args.format)
        logger.info("Pipeline tamamlandi")
    except (ModelNotFoundError, FileNotFoundError, PDFProcessingError) as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.error(f"Beklenmeyen hata: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if pipeline:
            pipeline.cleanup()


if __name__ == "__main__":
    main()
