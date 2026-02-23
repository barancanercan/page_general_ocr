# 🛡️ PageGeneralOCR: Askeri Tarih İstihbarat Platformu

![Project Status](https://img.shields.io/badge/Status-Production%20Ready-green)
![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

**PageGeneralOCR**, taranmış tarihi askeri belgeleri (PDF) işleyerek dijitalleştiren, anlamlandıran ve bu belgeler üzerinde yapay zeka destekli istihbarat sorgulamaları yapılmasına olanak tanıyan, kurumsal seviyede bir RAG (Retrieval-Augmented Generation) platformudur.

Özellikle **Türk İstiklal Harbi** gibi karmaşık, eski Türkçe terimler ve askeri terminoloji içeren belgeler üzerinde yüksek doğrulukla çalışmak üzere optimize edilmiştir.

---

## 🌟 Temel Yetenekler

### 1. 🔍 Veri Müfettişi - Yapay Zeka Asistan
*   **Birlik Odaklı Sorgulama:** Seçtiğiniz birlik (tümen, alay, kolordu) verileri getirilir ve yapay zeka bu veriler üzerinden sorularınızı yanıtlar.
*   **Sorgu-Anlam Eşleştirme:** Sorulan soruya en uygun kaynakları önceliklendiren akıllı filtreleme sistemi.
*   **Bağlamsal Hafıza:** Konuşma geçmişini ve kullanıcının önceki sorularını hatırlayarak tutarlı yanıtlar üretir.

### 2. 🧠 Çok Katmanlı Hafıza Sistemi
*   **Kısa Vadeli Hafıza:** Mevcut oturumdaki son 5 mesajdan itibaren konuşma bağlamını korur.
*   **Uzun Vadeli Hafıza:** Tarihi corpus ve askeri ontology ile zenginleştirilmiş bilgi tabanı.
*   **Oturum Bazlı Hafıza:** Her birlik için ayrı hafıza alanı - farklı birliklerle yapılan konuşmalar karışmaz.

### 3. 📚 Akıllı Bilgi Tabanı
*   **Tarihi Corpus:** Muharebeler, savaşlar, stratejik kararlar ve diplomatik süreçler.
*   **Askeri Ontology:** Rütbe hiyerarşisi, harekat türleri, taktik kavramlar, lojistik terimler.
*   **Semantik Arama:** Hem corpus hem ontology'de sorgu bazlı akıllı arama.

---

## 🏗️ Sistem Mimarisi

### Genel Mimari Şeması

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    KULLANICI ARAYÜZÜ (Gradio / Streamlit)                      │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                    VERİ MÜFETTİŞİ SEKMESİ                              │    │
│  │  ┌─────────────┐    ┌─────────────┐    ┌──────────────────────────┐   │    │
│  │  │ Birlik Seç │ ─► │ Verileri   │ ─► │  Yapay Zeka Sohbeti    │   │    │
│  │  │ Filtreleme │    │ Getir       │    │  (RAG + Hafıza)        │   │    │
│  │  └─────────────┘    └─────────────┘    └──────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           RAG AGENT PİPEİNE                                    │
│                                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌────────────┐  │
│  │   Sorgu     │    │  Sorgu      │    │  Kaynak     │    │    LLM    │  │
│  │   Analizi   │ ─► │  Bazlı      │ ─► │  Birleştirme│ ─► │  (Ollama) │  │
│  │  (Keywords) │    │  Filtreleme │    │  + Re-rank  │    │            │  │
│  └──────────────┘    └──────────────┘    └──────────────┘    └────────────┘  │
│                                                                                 │
│  ┌────────────────────────────────────────────────────────────────────────┐    │
│  │                    ÇOK KATMANLI HAFIZA SİSTEMİ                         │    │
│  │  ┌───────────────┐    ┌───────────────┐    ┌────────────────────┐   │    │
│  │  │  Kısa Vadeli  │    │ Uzun Vadeli   │    │   Uzun Vadeli     │   │    │
│  │  │  Hafıza       │    │ Hafıza        │    │   (Ontology)       │   │    │
│  │  │ (Conversation)│    │ (Corpus)      │    │                    │   │    │
│  │  └───────────────┘    └───────────────┘    └────────────────────┘   │    │
│  └────────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           VERİ DEPO (Qdrant + FileSystem)                       │
│                                                                                 │
│  ┌─────────────────────────────┐    ┌────────────────────────────────────────┐ │
│  │   Qdrant Vector DB          │    │   data/memory/                        │ │
│  │   (İşlenmiş Paragraflar)   │    │   ├── military_corpus.json           │ │
│  │   - book_title              │    │   │   (Tarihi muharebeler, savaşlar)  │ │
│  │   - page_num                │    │   └── military_ontology.json          │ │
│  │   - military_units          │    │   (Askeri terimler, rütbeler)        │ │
│  │   - text                    │    │                                        │ │
│  └─────────────────────────────┘    └────────────────────────────────────────┘ │
│                                                                                 │
│  ┌────────────────────────────────────────────────────────────────────────┐    │
│  │   data/raw/books/*.pdf (Kaynak kitaplar)                              │    │
│  └────────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### RAG Pipeline Detayı

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        RAG AGENT CHAT PIPELINE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. KULLANICI SORUSU                                                       │
│     "57. Tümen'in karşılaştığı zorlukları detaylı ver"                     │
│                           │                                                 │
│                           ▼                                                 │
│  2. ÖNCEKİ KONUŞMA & HAFIZA                                                │
│     ┌──────────────────────────────────────────────────────────────────┐   │
│     │ • Kısa Vadeli: Son 5 mesaj, özet bilgi                        │   │
│     │ • Uzun Vadeli: Corpus + Ontology araması                      │   │
│     │ • Session: Birlik bazlı izole hafıza                           │   │
│     └──────────────────────────────────────────────────────────────────┘   │
│                           │                                                 │
│                           ▼                                                 │
│  3. SORGU BAZLI FİLTRELEME                                                │
│     ┌──────────────────────────────────────────────────────────────────┐   │
│     │ • Anahtar kelime çıkarma (stopwords temizleme)                │   │
│     │ • Kaynakları sorgu ile eşleşmeye göre skorla                  │   │
│     │ • Aynı içerikleri tekrarlama (deduplication)                   │   │
│     │ • Max 30 kaynak                                                 │   │
│     └──────────────────────────────────────────────────────────────────┘   │
│                           │                                                 │
│                           ▼                                                 │
│  4. VEKTÖR ARAMA (Qdrant)                                                  │
│     ┌──────────────────────────────────────────────────────────────────┐   │
│     │ • Semantic similarity + Entity matching                        │   │
│     │ • Birlik filtresi (seçili birlik varyasyonları)              │   │
│     │ • Kitap filtresi                                               │   │
│     │ • top_k = 20 aday                                              │   │
│     └──────────────────────────────────────────────────────────────────┘   │
│                           │                                                 │
│                           ▼                                                 │
│  5. RE-RANKING (Cross-Encoder)                                             │
│     ┌──────────────────────────────────────────────────────────────────┐   │
│     │ • Query + Candidate text → Relevance score                    │   │
│     │ • En alakalı 5 kaynak seçilir                                │   │
│     │ • Kitap çeşitliliği sağlanır                                 │   │
│──┘   │
│                           │     └────────────────────────────────────────────────────────────────                                                 │
│                           ▼                                                 │
│  6. LLM PROMPT OLUŞTURMA                                                  │
│     ┌──────────────────────────────────────────────────────────────────┐   │
│     │ SYSTEM_PROMPT +                                                 │   │
│     │ • Konuşma geçmişi                                               │   │
│     │ • Önceki cevaplardan ilgili bilgiler                          │   │
│     │ • Uzun vadeli hafıza (corpus + ontology)                       │   │
│     │ • Kullanıcının yeni sorusu                                    │   │
│     │ • Kaynak metinler                                              │   │
│     └──────────────────────────────────────────────────────────────────┘   │
│                           │                                                 │
│                           ▼                                                 │
│  7. LLM (Ollama - gemma3/qwen3)                                           │
│     ┌──────────────────────────────────────────────────────────────────┐   │
│     │ • Temperature: 0.0 ( tutarlılık için)                         │   │
│     │ • Her bilgiye kaynak referansı eklenir                        │   │
│     │ • Detaylı, tekrarsız, Türkçe cevaplar                         │   │
│     └──────────────────────────────────────────────────────────────────┘   │
│                           │                                                 │
│                           ▼                                                 │
│  8. CEVAP + KAYNAKLAR + PERFORMANS                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Hafıza Sistemi Mimarisi

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      ÇOK KATMANLI HAFIZA MİMARİSİ                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    KISA VADELİ HAFIZA                               │    │
│  │                    (GlobalMemory)                                  │    │
│  │  ┌────────────────────────────────────────────────────────────┐    │    │
│  │  │  Mesaj Buffer (deque)                                     │    │    │
│  │  │  [Soru1, Cevap1, Soru2, Cevap2, Soru3, Cevap3, ...]     │    │    │
│  │  │                                                             │    │    │
│  │  │  • Max 10 mesaj tutar (5 soru-cevap çifti)              │    │    │
│  │  │  • Her 10 mesajda bir özet oluşturulur                  │    │    │
│  │  └────────────────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    UZUN VADELİ HAFIZA                              │    │
│  │                    (LongTermMemory)                                │    │
│  │                                                                     │    │
│  │  ┌─────────────────────┐  ┌─────────────────────────────────────┐  │    │
│  │  │ military_corpus.json│  │    military_ontology.json          │  │    │
│  │  ├─────────────────────┤  ├─────────────────────────────────────┤  │    │
│  │  │ topics:             │  │ turkish_military_ontology:          │  │    │
│  │  │  - inonu_1          │  │   rank_and_command_authority:      │  │    │
│  │  │  - inonu_2          │  │   - subay_komuta_hiyerarsisi      │  │    │
│  │  │  - sakarya          │  │   - astsubay_idari_yapi           │  │    │
│  │  │  - buyuk_taarruz   │  │   operational_terminology:        │  │    │
│  │  │  - ...             │  │   - harekat_turleri               │  │    │
│  │  │                    │  │   - taktik_kavramlar              │  │    │
│  │  │ entities:           │  │   logistics_and_sustainment:       │  │    │
│  │  │  - birlikler       │  │   - supply_classes                │  │    │
│  │  │  - komutanlar      │  │   - infrastructure                │  │    │
│  │  │  - cephane         │  │   intelligence_and_reconnaissance  │  │    │
│  │  │  - savaslar        │  │   historical_semantic_mapping     │  │    │
│  │  │                    │  │   staff_sections                   │  │    │
│  │  └─────────────────────┘  └─────────────────────────────────────┘  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    GLOBAL MEMORY                                   │    │
│  │                    (Session Yönetimi)                             │    │
│  │  ┌────────────────────────────────────────────────────────────┐    │    │
│  │  │  Session ID → Memory Instance                               │    │    │
│  │  │                                                             │    │    │
│  │  │  "insp_57._tumen" → ConversationMemory (57. Tümen oturumu)│    │    │
│  │  │  "insp_3._kolordu" → ConversationMemory (3. Kolordu)       │    │    │
│  │  │  "insp_default" → ConversationMemory (Genel)              │    │    │
│  │  └────────────────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📂 Dizin Yapısı

```
page_general_ocr/
├── streamlit_app.py                # Streamlit arayüzü (Ana UI)
├── src/
│   ├── main.py                     # Gradio arayüzü (Alternatif UI)
│   ├── config/
│   │   └── settings.py             # Yapılandırma parametreleri
│   ├── agents/
│   │   ├── rag_agent.py            # Ana RAG pipeline
│   │   ├── memory.py               # Hafıza sistemi (Kısa + Uzun vadeli)
│   │   └── ingestion_agent.py      # PDF işleme ve indeksleme
│   ├── services/
│   │   ├── vector_db_service.py    # Qdrant vektör veritabanı
│   │   ├── embedding_service.py    # Embedding ve re-ranking
│   │   └── ocr_service.py          # PDF OCR işleme
│   └── utils/
│       ├── military_extraction.py   # Birlik varlık çıkarma
│       ├── normalization.py        # Birlik adı normalize
│       └── text_processing.py      # Metin işleme
├── data/
│   ├── memory/                     # Uzun vadeli hafıza
│   │   ├── military_corpus.json   # Tarihi bilgiler
│   │   └── military_ontology.json # Askeri terimler
│   └── raw/
│       └── books/
│           └── *.pdf               # Kaynak kitaplar
└── qdrant_data/                    # Qdrant veritabanı dosyaları
```

---

## 🚀 Kurulum ve Başlangıç

### Gereksinimler
*   **Donanım:** NVIDIA GPU (Önerilen: 8GB+ VRAM)
*   **Yazılım:** Python 3.10+, [Ollama](https://ollama.com/)

### 1. Kurulum
```bash
git clone https://github.com/your-username/page_general_ocr.git
cd page_general_ocr
pip install -r requirements.txt
```

### 2. Environment Yapılandırması
```bash
# Örnek yapılandırma dosyasını kopyalayın
cp .env.example .env

# .env dosyasını düzenleyin ve gerekli değerleri ayarlayın
```

### 3. Modellerin Hazırlanması
```bash
# OCR Modelleri
ollama pull qwen2.5vl:3b
ollama pull qwen2.5vl:7b

# Sohbet Modeli
ollama pull gemma3:latest
# veya
ollama pull qwen3:8b
```

### 4. Çalıştırma

**Streamlit (Önerilen):**
```bash
streamlit run streamlit_app.py
```
Tarayıcınızda **`http://localhost:8501`** adresine giderek arayüze erişebilirsiniz.

**Gradio (Alternatif):**
```bash
python src/main.py
```
Gradio arayüzü için **`http://localhost:7860`** adresini kullanın.

---

## 🖥️ Kullanım Kılavuzu

### 📂 Adım 1: Veri Yükleme
*   **"Belge Yükle"** sekmesine gidin.
*   PDF dosyalarınızı sürükleyip bırakın.
*   Sistem otomatik olarak OCR işlemini başlatacak ve verileri indeksleyecektir.

### 🔍 Adım 2: Veri Müfettişi
*   **"Veri Müfettişi"** sekmesine gelin.
*   **Birlik Seçimi:** İlgilendiğiniz birliği seçin (Örn: "57. Tümen").
*   **Kitap Filtreleme:** İsteğe bağlı olarak belirli kitapları seçin.
*   **"Verileri Getir"** butonuna basın.

### 💬 Adım 3: Yapay Zeka ile Sohbet
*   Veriler getirildikten sonra aynı sekmenin alt kısmındaki sohbet kutusundan sorularınızı sorun.
*   *"Bu tümen hangi cephelerde savaştı?"*
*   *"Karşılaştığı zorluklar nelerdi?"*
*   *"Komutanları kimlerdi?"*

**Önemli:** Yapay zeka, sadece "Verileri Getir" ile çekilen veriler üzerinden yanıt verir. Bu sayede:
*   Halüsinasyon riski minimuma iner
*   Sorgu daha tutarlı ve odaklı olur
*   Kaynaklar şeffaf ve doğrulanabilir

---

## 🤖 LLM System Prompt Stratejisi

```python
SYSTEM_PROMPT = """
Sen, Türk İstiklal Harbi konusunda uzmanlaşmış bir askeri tarih araştırmacısısın.

KURALLAR:
1. DİL: Tüm cevaplar MUTLAKA TÜRKÇE olmalıdır.
2. TEKRARLAMA YASAK: Aynı bilgiyi TEKRAR YAZMA.
3. AŞIRI DETAY VER: Her bilgi parçasını en ince detayına kadar açıkla.
4. SORUNUN KAPSAMINA GÖRE YAPILA:
   - Birlik sorusu: Tarihler, cepheler, komutanlar, muharebeler, kayıp/zafer
   - Lojistik sorusu: Cephane miktarı, nakil güzergahları
   - Stratejik sorusu: Karar nedenleri, alternatifler, sonuçlar
5. KAYNAK GÖSTERİMİ: Her bilgiye (Kitap, Sayfa X) referansı ekle.
6. SADECE KAYNAKLARDAKİ BİLGİLERİ KULLAN.
7. Cevap yapısı: Başlık + Detaylı Paragraflar + Tablolar + Sonuç
8. UZUN VADELİ HAFIZA: Corpus ve Ontology'deki bilgileri destekleyici olarak kullan.
"""
```

---

## ⚙️ Yapılandırma

`src/config/settings.py` dosyası veya `.env` dosyası üzerinden sistemin tüm parametrelerini özelleştirebilirsiniz:

### Environment Variables

| Değişken | Açıklama | Varsayılan |
|---------|----------|------------|
| `OCR_MODEL_SMALL` | Hızlı tarama modeli | `qwen2.5vl:3b` |
| `OCR_MODEL_LARGE` | Detaylı tarama modeli | `qwen2.5vl:7b` |
| `CHAT_MODEL` | Sohbet LLM modeli | `gemma3:latest` |
| `OCR_TIMEOUT` | OCR işlemi timeout (saniye) | `1200` |
| `OCR_DPI` | Tarama çözünürlüğü | `150` |
| `RAG_FETCH_K` | İlk aramada getirilecek belge sayısı | `20` |
| `RAG_TOP_K` | LLM'e gönderilecek en iyi belge sayısı | `5` |
| `GRADIO_PORT` | Gradio sunucu portu | `7860` |

### settings.py Parametreleri

| Parametre | Açıklama | Varsayılan |
|-----------|----------|------------|
| `CONFIDENCE_THRESHOLD` | Model değişim eşiği | `0.6` |

---

## 🐛 Hata Ayıklama

### Yaygın Hatalar ve Çözümler

#### 1. Ollama Bağlantı Hatası
```
Error: connection refused
```
**Çözüm:** Ollama'nın çalıştığını doğrulayın:
```bash
ollama list
```

#### 2. Model Bulunamadı
```
Error: model 'xxx' not found
```
**Çözüm:** Modeli indirin:
```bash
ollama pull qwen2.5vl:3b
ollama pull gemma3:latest
```

#### 3. VRAM Yetersiz
**Çözüm:** Daha küçük model kullanın veya batch size'ı azaltın.

#### 4. Qdrant Veritabanı Hatası
```bash
# Veritabanını sıfırla
rm -rf qdrant_data/
```

### Loglama

Detaylı loglar için:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performans İzleme

Gradio arayüzünde her yanıtın altında performans metrikleri görüntülenir:
- Arama süresi
- Yeniden sıralama süresi
- LLM yanıt süresi
- Toplam süre

---

## 📚 Veri Dosyaları

### military_corpus.json
Tarihi bilgiler - muharebeler, savaşlar, stratejik analizler:

```json
{
  "topics": {
    "inonu_1": {
      "title": "1. İnönü Muharebesi",
      "date_range": "1921-01-06 / 1921-01-11",
      "phases": [...],
      "commanders": [...],
      "forces": {...}
    }
  },
  "entities": {
    "command_structure": {...},
    "units": {...},
    "key_commanders": {...},
    "logistics": {...},
    "weaponry": {...}
  }
}
```

### military_ontology.json
Askeri terminoloji ve doktrinler:

```json
{
  "turkish_military_ontology": {
    "rank_and_command_authority": {
      "subay_komuta_hiyerarsisi": {...},
      "astsubay_idari_yapi": {...}
    },
    "operational_terminology": {
      "harekat_turleri": {...},
      "taktik_kavramlar": {...}
    },
    "logistics_and_sustainment": {...},
    "intelligence_and_reconnaissance": {...},
    "historical_semantic_mapping": {...}
  }
}
```

---

## 🤝 Katkıda Bulunma

Bu proje açık kaynaklıdır. Hata bildirimleri, özellik istekleri ve Pull Request'ler memnuniyetle karşılanır.

## 📄 Lisans

MIT License.

---

## 🧠 Askeri Zeka Sistemi (Military Intelligence System)

### Query Classifier - Soru Tipi Analizi

Sistem, kullanıcı sorgularını otomatik olarak sınıflandırır:

| Tip | Anahtar Kelimeler | Örnek |
|-----|------------------|-------|
| **factual** | nerede, ne zaman, kim, kaç | "57. Tümen nerede savaştı?" |
| **analytical** | ne yapılmalı, nasıl, strateji | "Bu durumda ne yapılmalı?" |
| **causal** | neden, niçin, sebep | "Neden geri çekildiler?" |
| **counterfactual** | olsaydı, söyleydi | "Şöyle olsaydı ne olurdu?" |
| **comparative** | fark, avantaj, dezavantaj | "İnönü vs Sakarya farkı?" |

### Decision Engine - Karar Motoru

Askeri durumları analiz eder ve karar önerileri üretir:

```python
# Örnek karar analizi
{
    "query_type": "analytical",
    "requires_decision": True,
    "decisions": [
        {
            "type": "防守 (Savunma)",
            "sub_options": ["Mevzi savunması", "Hareketli savunma"],
            "doctrine_ref": "Alan savunması doktrini"
        }
    ],
    "reasoning": [
        "Düşman kuvvetleri üstün",
        "Mühimmat durumu kritik",
        "Bu koşullarda en uygun seçenek: Savunma"
    ]
}
```

### Zorunlu Output Formatı

Tüm LLM cevapları bu formatta olmak ZORUNDADIR:

```
## 1. DURUM ANALİZİ
[Bu soruda hangi askeri durum inceleniyor?]

## 2. KARAR/DEĞERLENDİRME
[Askeri açıdan en uygun yaklaşım]

## 3. GEREKÇE
[Neden bu karar? Hangi doktrine dayanıyor?]

## 4. KAYNAK ANALİZİ
[İlgili kaynaklardan kanıtlar]
```

### Veri Dosyaları

```
data/memory/
├── military_corpus.json      # Tarihi muharebeler ve savaşlar
├── military_ontology.json    # Askeri terimler ve doktrinler
├── military_decisions.json   # Stratejik karar örnekleri
└── micro_decisions.json     # Taktik seviye mikro kararlar
```

### Micro Decisions Örnekleri

Sistem, birim seviyesi karar örneklerini kullanır:

| Durum | Karar | Gerekçe |
|-------|-------|---------|
| MG ateşi altında | Yan manevra | Doğrudan ilerleme yüksek kayıp |
| Mühimmat kritik | Geciktirme | Cephaneden tasarruf şart |
| Kanat açık | İhtiyat kaydırma | Kuşatmayı önleme |
| İletişim kesildi | Subay inisiyatifi | Merkezi komuta olmadan riskli |
| Kuşatılma riski | Breakout | Zayıf noktadan çıkış |

---

## 📊 Sistem Skoru (Güncel)

| Alan | Önceki | Güncel |
|------|--------|--------|
| Bilgi Çekme | 7/10 | 7/10 |
| Veri Yapısı | 3/10 | 8/10 |
| Karar Üretme | 1/10 | 6/10 |
| Askeri Düşünce | 2/10 | 7/10 |
| Reasoning | 2/10 | 7/10 |

**Hedef:** "Tarih Anlatan Model" → "Karar Veren Komutan"
