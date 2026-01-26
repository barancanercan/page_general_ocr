"""Pipeline Configuration"""

import torch

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Models
LLM_MODEL = "ytu-ce-cosmos/turkish-gpt2-large"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# OCR
OCR_LANG = "tur+eng"
OCR_DPI = 200

# Processing
BATCH_SIZE = 8
WORKERS = 4

# Paths
DATA_DIR = "data"
CACHE_DIR = ".cache"