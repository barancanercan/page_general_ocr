<p align="center">
  <img src="logo.png" alt="PageGeneral Logo" width="120">
</p>

# PageGeneral

![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

**PageGeneral**, taranmis tarihi askeri belgeleri (PDF) isleyerek dijitallestiren, anlamlandiran ve bu belgeler uzerinde yapay zeka destekli sorgulama yapilmasina olanak taniyan bir RAG (Retrieval-Augmented Generation) platformudur.

Ozellikle **Turk Istiklal Harbi** gibi karmasik, eski Turkce terimler ve askeri terminoloji iceren belgeler uzerinde yuksek dogrulukla calismak uzere optimize edilmistir.

---

## Temel Yetenekler

### Veri Mufettisi - Yapay Zeka Asistan
- **Birlik Odakli Sorgulama:** Sectiginiz birlik (tumen, alay, kolordu) verileri getirilir ve yapay zeka bu veriler uzerinden sorularinizi yanitlar.
- **Sorgu-Anlam Eslestirme:** Sorulan soruya en uygun kaynaklari onceliklendiren akilli filtreleme sistemi.
- **Baglamsal Hafiza:** Konusma gecmisini ve kullanicinin onceki sorularini hatirlayarak tutarli yanitlar uretir.

### Cok Katmanli Hafiza Sistemi
- **Kisa Vadeli Hafiza:** Mevcut oturumdaki son 5 mesajdan itibaren konusma baglamini korur.
- **Uzun Vadeli Hafiza:** Tarihi corpus ve askeri ontology ile zenginlestirilmis bilgi tabani.
- **Oturum Bazli Hafiza:** Her birlik icin ayri hafiza alani.

### Akilli Bilgi Tabani
- **Tarihi Corpus:** Muharebeler, savaslar, stratejik kararlar ve diplomatik surecler.
- **Askeri Ontology:** Rutbe hiyerarsisi, harekat turleri, taktik kavramlar, lojistik terimler.
- **Semantik Arama:** Hem corpus hem ontology'de sorgu bazli akilli arama.

---

## Sistem Mimarisi

```
+-----------------------------------------------------------------------------------+
|                         KULLANICI ARAYUZU (Streamlit)                             |
|  +-----------------------------------------------------------------------------+  |
|  |                    VERI MUFETTISI SEKMESI                                   |  |
|  |  +-------------+    +-------------+    +--------------------------+         |  |
|  |  | Birlik Sec  | -> | Verileri    | -> |  Yapay Zeka Sohbeti     |         |  |
|  |  | Filtreleme  |    | Getir       |    |  (RAG + Hafiza)         |         |  |
|  |  +-------------+    +-------------+    +--------------------------+         |  |
|  +-----------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------+
                                        |
                                        v
+-----------------------------------------------------------------------------------+
|                           RAG AGENT PIPELINE                                      |
|                                                                                   |
|  +---------------+    +---------------+    +---------------+    +-------------+   |
|  |   Sorgu       |    |  Sorgu        |    |  Kaynak       |    |    LLM      |   |
|  |   Analizi     | -> |  Bazli        | -> |  Birlestirme  | -> |  (OpenAI)   |   |
|  |  (Keywords)   |    |  Filtreleme   |    |  + Re-rank    |    |             |   |
|  +---------------+    +---------------+    +---------------+    +-------------+   |
|                                                                                   |
|  +-----------------------------------------------------------------------------+  |
|  |                    COK KATMANLI HAFIZA SISTEMI                              |  |
|  |  +----------------+    +----------------+    +--------------------+          |  |
|  |  |  Kisa Vadeli   |    | Uzun Vadeli    |    |   Uzun Vadeli      |          |  |
|  |  |  Hafiza        |    | Hafiza         |    |   (Ontology)       |          |  |
|  |  | (Conversation) |    | (Corpus)       |    |                    |          |  |
|  |  +----------------+    +----------------+    +--------------------+          |  |
|  +-----------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------+
                                        |
                                        v
+-----------------------------------------------------------------------------------+
|                           VERI DEPO (Qdrant + FileSystem)                         |
|                                                                                   |
|  +-----------------------------+    +------------------------------------------+  |
|  |   Qdrant Vector DB          |    |   data/memory/                          |  |
|  |   (Islenmis Paragraflar)    |    |   +-- military_corpus.json              |  |
|  |   - book_title              |    |   |   (Tarihi muharebeler, savaslar)    |  |
|  |   - page_num                |    |   +-- military_ontology.json            |  |
|  |   - military_units          |    |   (Askeri terimler, rutbeler)           |  |
|  |   - text                    |    |                                          |  |
|  +-----------------------------+    +------------------------------------------+  |
+-----------------------------------------------------------------------------------+
```

---

## Dizin Yapisi

```
page_general_ocr/
├── streamlit_app.py                # Streamlit arayuzu
├── requirements.txt                # Python bagimliliklari
├── .env.example                    # Ornek environment dosyasi
│
├── src/                            # Ana kaynak kodu
│   ├── agents/                     # RAG ve veri isleme agentlari
│   │   ├── rag_agent.py            # Ana RAG pipeline (OpenAI entegrasyonu)
│   │   ├── memory.py               # Hafiza sistemi (Kisa + Uzun vadeli)
│   │   └── ingestion_agent.py      # PDF isleme ve indeksleme
│   │
│   ├── services/                   # Temel servisler
│   │   ├── ocr_service.py          # PDF OCR isleme (Ollama qwen2.5vl)
│   │   ├── embedding_service.py    # Metin embedding (sentence-transformers)
│   │   └── vector_db_service.py    # Qdrant vektor veritabani
│   │
│   ├── utils/                      # Yardimci fonksiyonlar
│   │   ├── text_processing.py      # Metin temizleme ve paragraf ayirma
│   │   ├── normalization.py        # Askeri birim ismi normalizasyonu
│   │   └── military_extraction.py  # Askeri birim cikarma
│   │
│   ├── config/                     # Konfigurasyon
│   │   ├── settings.py             # Ana ayarlar (model, path, timeout)
│   │   └── constants.py            # Sabit degerler
│   │
│   └── core/                       # Veri modelleri
│       └── models.py               # Pydantic veri modelleri
│
├── data/
│   ├── memory/                     # Uzun vadeli hafiza
│   │   ├── military_corpus.json    # Tarihi bilgiler
│   │   └── military_ontology.json  # Askeri terimler
│   └── raw/
│       └── books/
│           └── *.pdf               # Kaynak kitaplar (Git'e dahil degil)
│
└── qdrant_data/                    # Qdrant veritabani
```

---

## Kurulum ve Baslangic

### Gereksinimler

| Bilesen | Gereksinim |
|---------|------------|
| Python | 3.10+ |
| Ollama | OCR islemleri icin |
| OpenAI API | Chat/RAG islemleri icin |
| GPU | Onerilen: 8GB+ VRAM (OCR icin) |

### 1. Kurulum

```bash
git clone https://github.com/your-username/page_general_ocr.git
cd page_general_ocr
pip install -r requirements.txt
```

### 2. Environment Yapilandirmasi

`.env` dosyasi olusturun:

```bash
cp .env.example .env
```

**Zorunlu degiskenler:**

```env
# OpenAI API (Chat/RAG icin - ZORUNLU)
OPENAI_API_KEY=sk-your-openai-api-key-here

# OCR Modelleri (Ollama)
OCR_MODEL_SMALL=qwen2.5vl:3b
OCR_MODEL_LARGE=qwen2.5vl:7b
```

### 3. Model Kurulumu

**OCR icin Ollama modelleri:**
```bash
ollama pull qwen2.5vl:3b
ollama pull qwen2.5vl:7b
```

**Not:** Chat/RAG islemleri OpenAI API uzerinden yapilir, yerel model gerektirmez.

### 4. Calistirma

```bash
streamlit run streamlit_app.py
```

Tarayicinizda `http://localhost:8501` adresine giderek arayuze erisebilirsiniz.

---

## Model Mimarisi

### OCR Islemleri (Ollama - Yerel)

| Model | Kullanim | Aciklama |
|-------|----------|----------|
| `qwen2.5vl:3b` | Hizli tarama | Dusuk VRAM, hizli sonuc |
| `qwen2.5vl:7b` | Detayli tarama | Yuksek dogruluk, daha fazla VRAM |

### Chat/RAG Islemleri (OpenAI API)

| Model | Kullanim | Aciklama |
|-------|----------|----------|
| `gpt-4o` | Zor gorevler | Karmasik analiz, derin akil yurutme |
| `gpt-4o-mini` | Kolay gorevler | Hizli yanitlar, maliyet optimizasyonu |

Sistem, sorgunun zorluguna gore otomatik model secimi yapar:
- **Basit sorular** (factual, kisa yanitlar) -> `gpt-4o-mini`
- **Karmasik sorular** (analiz, karsilastirma, stratejik degerlendirme) -> `gpt-4o`

---

## Kullanim Kilavuzu

### Adim 1: Veri Yukleme
1. "Belge Yukle" sekmesine gidin
2. PDF dosyalarinizi surukleyip birakin
3. Sistem otomatik olarak OCR islemini baslatir

### Adim 2: Veri Mufettisi
1. "Veri Mufettisi" sekmesine gelin
2. Ilgilendiginiz birligi secin (Orn: "57. Tumen")
3. Istege bagli olarak belirli kitaplari filtreleyin
4. "Verileri Getir" butonuna basin

### Adim 3: Yapay Zeka ile Sohbet
Veriler getirildikten sonra sohbet kutusundan sorularinizi sorun:
- "Bu tumen hangi cephelerde savasti?"
- "Karsilastigi zorluklar nelerdi?"
- "Komutanlari kimlerdi?"

---

## Yapilandirma

### Environment Degiskenleri

| Degisken | Aciklama | Varsayilan |
|----------|----------|------------|
| `OPENAI_API_KEY` | OpenAI API anahtari | (Zorunlu) |
| `OCR_MODEL_SMALL` | Hizli OCR modeli | `qwen2.5vl:3b` |
| `OCR_MODEL_LARGE` | Detayli OCR modeli | `qwen2.5vl:7b` |
| `OCR_TIMEOUT` | OCR islemi timeout (saniye) | `1200` |
| `OCR_DPI` | Tarama cozunurlugu | `150` |
| `RAG_FETCH_K` | Ilk aramada getirilecek belge sayisi | `30` |
| `RAG_TOP_K` | LLM'e gonderilecek en iyi belge sayisi | `8` |

---

## Hata Ayiklama

### Yaygin Hatalar

**1. OpenAI API Hatasi**
```
Error: Invalid API Key
```
Cozum: `.env` dosyasinda `OPENAI_API_KEY` degerini kontrol edin.

**2. Ollama Baglanti Hatasi (OCR)**
```
Error: connection refused
```
Cozum: Ollama'nin calistigini dogrulayin:
```bash
ollama list
```

**3. OCR Model Bulunamadi**
```
Error: model 'qwen2.5vl:3b' not found
```
Cozum: Modeli indirin:
```bash
ollama pull qwen2.5vl:3b
```

**4. VRAM Yetersiz**
Cozum: Daha kucuk OCR modeli kullanin (`qwen2.5vl:3b`).

---

## Streamlit Cloud Deployment

### Hizli Baslangic

1. GitHub'a push edin
2. [share.streamlit.io](https://share.streamlit.io) adresine gidin
3. Repoyu secin ve `streamlit_app.py` dosyasini belirtin
4. **Settings > Secrets** bolumune asagidaki degiskenleri ekleyin:

```toml
OPENAI_API_KEY = "sk-your-api-key"
```

### Notlar
- Streamlit Cloud ucretsiz tier'da 1GB RAM siniri vardir
- Qdrant local mode buyuk veri setlerinde yavas olabilir
- Production icin Qdrant Cloud onerilir

---

## Lisans

MIT License.

---

<p align="center">
  <sub>Baran Can Ercan tarafindan gelistirilmistir.</sub>
</p>
