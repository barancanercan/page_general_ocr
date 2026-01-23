import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Ollama
OLLAMA_CHAT = "http://localhost:11434/api/chat"
OLLAMA_EMBED = "http://localhost:11434/api/embeddings"
LLM_MODEL = "qwen2.5:3b"
EMBED_MODEL = "nomic-embed-text"

# OCR
OCR_LANG = "tur+eng"
OCR_DPI = 200
MAX_PAGES = 10

# Paths
DATA_DIR = "data"
CACHE_DIR = ".cache"
