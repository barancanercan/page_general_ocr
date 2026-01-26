# PDF OCR Pipeline

GPU destekli yerel LLM kullanarak PDF belgelerinden metin çıkarma, temizleme ve embedding oluşturma pipeline'ı.

## Özellikler

- PDF sayfalarını görüntüye dönüştürme
- Tesseract OCR ile metin çıkarma (Türkçe + İngilizce)
- Yerel LLM ile OCR hatalarını düzeltme
- Askeri birlik isimlerini otomatik tespit
- Sentence Transformers ile embedding oluşturma
- CUDA GPU desteği

## Gereksinimler

- Python 3.10+
- NVIDIA GPU (CUDA 12.1+)
- Tesseract OCR

### Sistem Bağımlılıkları

```bash
# Ubuntu/Debian
sudo apt install tesseract-ocr tesseract-ocr-tur

# Fedora
sudo dnf install tesseract tesseract-langpack-tur
```

## Kurulum

```bash
# Repository'yi klonla
git clone <repo-url>
cd page_general_ocr

# Virtual environment oluştur
python -m venv .venv
source .venv/bin/activate

# PyTorch (CUDA)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Diğer bağımlılıklar
pip install -r requirements.txt
```

## Kullanım

```bash
python main.py <pdf_dosyası> [pdf_dosyası2 ...]
```

### Örnek

```bash
python main.py data/belge.pdf
```

Çıktı: `data/belge.jsonl`

## Çıktı Formatı

Her satır bir paragraf içerir (JSON Lines):

```json
{
  "id": "belge_p42_001",
  "embedding": [0.123, -0.456, ...],
  "document": "Temizlenmiş paragraf metni...",
  "metadata": {
    "book_id": "belge",
    "index": 1,
    "page": 42,
    "units": ["3. Kolordu", "15. Tümen"],
    "confidence": 0.892
  }
}
```

## Konfigürasyon

`config/settings.py` dosyasını düzenleyerek ayarları değiştirebilirsiniz:

```python
# Device
DEVICE = "cuda"  # veya "cpu"

# Models
LLM_MODEL = "ytu-ce-cosmos/turkish-gpt2-large"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# OCR
OCR_LANG = "tur+eng"
OCR_DPI = 200

# Processing
BATCH_SIZE = 8
WORKERS = 4
```

### Model Seçenekleri

**LLM Modelleri:**
| Model | VRAM | Açıklama |
|-------|------|----------|
| `ytu-ce-cosmos/turkish-gpt2-large` | ~2GB | Türkçe GPT-2 |
| `dbmdz/bert-base-turkish-cased` | ~1GB | Türkçe BERT |

**Embedding Modelleri:**
| Model | Boyut | Açıklama |
|-------|-------|----------|
| `paraphrase-multilingual-MiniLM-L12-v2` | 384 | Çok dilli, hızlı |
| `paraphrase-multilingual-mpnet-base-v2` | 768 | Çok dilli, yüksek kalite |

## Proje Yapısı

```
page_general_ocr/
├── config/
│   └── settings.py      # Konfigürasyon
├── core/
│   └── models.py        # Veri modelleri
├── entity/
│   └── military.py      # Askeri birlik pattern'leri
├── llm/
│   └── engine.py        # LLM motoru
├── ocr/
│   └── extractor.py     # PDF ve OCR işlemleri
├── paragraph/
│   └── classifier.py    # Paragraf çıkarma
├── pipelines/
│   └── process.py       # Ana pipeline
├── main.py              # Giriş noktası
└── requirements.txt     # Bağımlılıklar
```

## Pipeline Adımları

1. **PDF Yükleme**: PDF sayfaları görüntüye dönüştürülür
2. **OCR**: Tesseract ile metin çıkarılır
3. **Paragraf Çıkarma**: Metin bloklara ayrılır
4. **LLM Temizleme**: OCR hataları düzeltilir
5. **Birlik Tespiti**: Askeri birimler çıkarılır
6. **Embedding**: Vektör temsilleri oluşturulur

## Lisans

MIT
