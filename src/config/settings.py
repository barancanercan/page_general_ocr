import os
from pathlib import Path
from dotenv import load_dotenv

# .env dosyasini yukle
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
TEMP_DIR = BASE_DIR / ".temp_ocr"
QDRANT_PATH = BASE_DIR / "qdrant_data"

def init_directories():
    """Gerekli dizinleri olusturur."""
    TEMP_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

init_directories()

# OpenAI API (Chat/RAG icin)
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

# Chat Model Settings (OpenAI)
# Zor gorevler: karmasik analiz, detayli sorular
CHAT_MODEL_HARD: str = os.getenv("CHAT_MODEL_HARD", "gpt-4o")
# Kolay gorevler: basit sorular, ozet
CHAT_MODEL_EASY: str = os.getenv("CHAT_MODEL_EASY", "gpt-4o-mini")

# OCR Settings (Ollama - eski haliyle)
OCR_MODEL_SMALL: str = os.getenv("OCR_MODEL_SMALL", "qwen2.5vl:3b")
OCR_MODEL_LARGE: str = os.getenv("OCR_MODEL_LARGE", "qwen2.5vl:7b")
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

# Embedding Settings (local - degismeyecek)
EMBED_MODEL: str = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANK_MODEL: str = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
EMBED_DIM: int = int(os.getenv("EMBED_DIM", "384"))

COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "paragraphs")

# RAG Settings
RAG_FETCH_K: int = int(os.getenv("RAG_FETCH_K", "30"))
RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "8"))
RAG_HISTORY_TURNS: int = int(os.getenv("RAG_HISTORY_TURNS", "4"))

MIN_PARAGRAPH_LENGTH: int = int(os.getenv("MIN_PARAGRAPH_LENGTH", "200"))
