import os
from pathlib import Path

# Base Directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Directories
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
TEMP_DIR = BASE_DIR / ".temp_ocr"
QDRANT_PATH = BASE_DIR / "qdrant_data"

# Create directories if they don't exist
TEMP_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Ollama Models
OCR_MODEL_SMALL = "qwen2.5vl:3b"
OCR_MODEL_LARGE = "qwen2.5vl:7b"
CHAT_MODEL = "gemma3:latest" # Kullanıcının kesin isteği üzerine ayarlandı.

# OCR Settings
OCR_TIMEOUT = 1200
OCR_NUM_PREDICT = 3000
OCR_TEMPERATURE = 0.0
OCR_DPI = 150
OCR_MAX_IMAGE_WIDTH = 1200
CONFIDENCE_THRESHOLD = 0.6
MIN_TEXT_LENGTH = 50

# Embedding & Re-ranking
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Vector DB
COLLECTION_NAME = "paragraphs"

# RAG Settings (Optimized for better performance)
RAG_FETCH_K = 20  # Aday sayısı (Önceki: 25)
RAG_TOP_K = 5     # LLM'e verilecek en iyi belge sayısı (Önceki: 7)
RAG_HISTORY_TURNS = 4

# Paragraph Parsing
MIN_PARAGRAPH_LENGTH = 200

# Gradio
GRADIO_PORT = 7860
GRADIO_TITLE = "PageGeneralOCR Pro"
