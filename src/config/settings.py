import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
TEMP_DIR = BASE_DIR / ".temp_ocr"
QDRANT_PATH = BASE_DIR / "qdrant_data"

def init_directories():
    """Gerekli dizinleri oluşturur."""
    TEMP_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

init_directories()

OCR_MODEL_SMALL: str = os.getenv("OCR_MODEL_SMALL", "qwen2.5vl:3b")
OCR_MODEL_LARGE: str = os.getenv("OCR_MODEL_LARGE", "qwen2.5vl:7b")
CHAT_MODEL: str = os.getenv("CHAT_MODEL", "gemma3:latest")

OCR_TIMEOUT: int = int(os.getenv("OCR_TIMEOUT", "1200"))
OCR_NUM_PREDICT: int = int(os.getenv("OCR_NUM_PREDICT", "3000"))
OCR_TEMPERATURE: float = float(os.getenv("OCR_TEMPERATURE", "0.1"))
OCR_REPETITION_PENALTY: float = float(os.getenv("OCR_REPETITION_PENALTY", "1.3"))
OCR_TOP_K: int = int(os.getenv("OCR_TOP_K", "40"))
OCR_TOP_P: float = float(os.getenv("OCR_TOP_P", "0.9"))
OCR_DPI: int = int(os.getenv("OCR_DPI", "150"))
OCR_MAX_IMAGE_WIDTH: int = int(os.getenv("OCR_MAX_IMAGE_WIDTH", "1200"))
CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.6"))
MIN_TEXT_LENGTH: int = int(os.getenv("MIN_TEXT_LENGTH", "50"))

EMBED_MODEL: str = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANK_MODEL: str = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
EMBED_DIM: int = int(os.getenv("EMBED_DIM", "384"))

COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "paragraphs")

RAG_FETCH_K: int = int(os.getenv("RAG_FETCH_K", "20"))
RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "5"))
RAG_HISTORY_TURNS: int = int(os.getenv("RAG_HISTORY_TURNS", "4"))

MIN_PARAGRAPH_LENGTH: int = int(os.getenv("MIN_PARAGRAPH_LENGTH", "200"))

GRADIO_PORT: int = int(os.getenv("GRADIO_PORT", "7860"))
GRADIO_TITLE: str = os.getenv("GRADIO_TITLE", "PageGeneralOCR Pro")
