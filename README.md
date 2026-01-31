# Qwen2.5-VL OCR Pipeline

Qwen2.5-VL vision modeli ile PDF belgelerinden metin cikarma pipeline'i. Ollama uzerinden yerel GPU ile calisir.

## Gereksinimler

- Python 3.10+
- NVIDIA GPU (CUDA)
- [Ollama](https://ollama.com/)

## Kurulum

```bash
# Ollama'yi kur ve modeli indir
ollama pull qwen2.5vl:7b

# Python bagimliliklar
pip install -r requirements.txt
```

## Kullanim

```bash
# Tum sayfalari isle
python qwen_pipeline.py belge.pdf

# Ilk 20 sayfa
python qwen_pipeline.py belge.pdf --max-pages 20

# JSON cikti
python qwen_pipeline.py belge.pdf --format json

# Farkli DPI
python qwen_pipeline.py belge.pdf --dpi 300
```

## Cikti

Sonuclar `output/` klasorune kaydedilir.

**TXT formati:**
```
============================================================
Sayfa 1
============================================================
Guven: 0.954 | Sure: 33.5s

[sayfa metni]
```

**JSON formati:**
```json
{
  "metadata": {
    "timestamp": "2026-01-31T...",
    "model": "qwen2.5vl:7b",
    "stats": { ... }
  },
  "pages": [
    {
      "page_num": 1,
      "text": "...",
      "confidence": 0.954,
      "processing_time": 33.5,
      "success": true
    }
  ]
}
```

## Konfigürasyon

`qwen_pipeline.py` icindeki `Config` sinifi:

| Parametre | Varsayilan | Aciklama |
|-----------|-----------|----------|
| `MODEL_NAME` | `qwen2.5vl:7b` | Ollama model adi |
| `MODEL_TIMEOUT` | `600` | Model timeout (saniye) |
| `DPI` | `240` | PDF render DPI |
| `MAX_IMAGE_WIDTH` | `1800` | Max goruntu genisligi (px) |
| `TEMPERATURE` | `0.0` | Model sicakligi (0=deterministik) |
| `NUM_PREDICT` | `6000` | Max token sayisi |
| `MAX_PAGES` | `0` | Max sayfa (0=tumu) |

## Lisans

MIT
