# 🛡️ PageGeneralOCR: Askeri Tarih Belge Analiz ve İstihbarat Platformu

![Project Status](https://img.shields.io/badge/Status-Production%20Ready-green)
![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

**PageGeneralOCR**, taranmış tarihi askeri belgeleri (PDF) işleyerek dijitalleştiren, anlamlandıran ve bu belgeler üzerinde yapay zeka destekli istihbarat sorgulamaları yapılmasına olanak tanıyan, kurumsal seviyede bir RAG (Retrieval-Augmented Generation) platformudur.

Özellikle **Türk İstiklal Harbi** gibi karmaşık, eski Türkçe terimler ve askeri terminoloji içeren belgeler üzerinde yüksek doğrulukla çalışmak üzere optimize edilmiştir.

---

## 🌟 Temel Yetenekler

### 1. 🧠 Akıllı OCR ve Dijitalleştirme
*   **Hibrit Model Mimarisi:** Hız için `Qwen2.5-VL-3B`, zorlu ve silik metinler için otomatik devreye giren `Qwen2.5-VL-7B` modellerini kullanır.
*   **Dinamik İyileştirme:** Düşük güven skorlu (confidence score) sayfaları otomatik tespit eder ve daha güçlü modellerle yeniden işler.
*   **Varlık Tespiti (NER):** Metin içerisindeki askeri birlikleri (Tümen, Kolordu, Alay vb.) otomatik olarak tanır ve metaveri olarak etiketler.

### 2. 🔍 Gelişmiş Semantik Arama (RAG)
*   **Vektör Tabanlı Hafıza:** İşlenen her paragraf, `Qdrant` vektör veritabanında anlamsal olarak indekslenir.
*   **Birlik Odaklı Sorgulama:** *"3. Tümen hangi cephelerde savaştı?"* gibi sorularda, sistem sadece ilgili birliğin verilerine odaklanarak halüsinasyonu önler ve nokta atışı cevaplar verir.
*   **Çapraz Doğrulama (Re-ranking):** Bulunan sonuçlar, `Cross-Encoder` modelleri ile sorgu alaka düzeyine göre yeniden sıralanır.

### 3. 📊 Veri Müfettişi ve Analiz
*   **Şeffaf Veri Erişimi:** Veritabanındaki ham verilere doğrudan erişim sağlar.
*   **Filtreleme ve İhracat:** Kitap, sayfa veya birlik bazında filtreleme yapabilir ve sonuçları CSV formatında indirebilirsiniz.

---

## 🏗️ Sistem Mimarisi

Proje, modüler ve ölçeklenebilir bir mikro-servis mimarisi üzerine kurulmuştur:

```mermaid
graph TD
    A[PDF Belgeleri] -->|Ingestion Agent| B(Akıllı OCR Pipeline)
    B -->|Metin & Varlıklar| C{Vektörleştirme}
    C -->|Embeddings| D[(Qdrant Veritabanı)]
    
    E[Kullanıcı] -->|Soru| F[RAG Agent]
    F -->|Semantik Arama| D
    D -->|Alakalı Paragraflar| G[Re-Ranking]
    G -->|Optimize Bağlam| H[LLM (Gemma/Qwen)]
    H -->|Cevap| E
```

---

## 🚀 Kurulum ve Başlangıç

### Gereksinimler
*   **Donanım:** NVIDIA GPU (Önerilen: 8GB+ VRAM)
*   **Yazılım:** Python 3.10+, [Ollama](https://ollama.com/)

### 1. Kurulum
Projeyi klonlayın ve bağımlılıkları yükleyin:

```bash
git clone https://github.com/your-username/page-general-ocr.git
cd page-general-ocr
pip install -r requirements.txt
```

### 2. Modellerin Hazırlanması
Gerekli yapay zeka modellerini Ollama üzerinden indirin:

```bash
# OCR Modelleri
ollama pull qwen2.5vl:3b
ollama pull qwen2.5vl:7b

# Sohbet Modeli
ollama pull gemma3:latest
```

### 3. Çalıştırma
Uygulamayı başlatın:

```bash
python app.py
```
Tarayıcınızda **`http://localhost:7860`** adresine giderek arayüze erişebilirsiniz.

---

## 🖥️ Kullanım Kılavuzu

### 📂 Adım 1: Veri Yükleme
*   **"Belge Yükle"** sekmesine gidin.
*   PDF dosyalarınızı sürükleyip bırakın.
*   Sistem otomatik olarak OCR işlemini başlatacak ve verileri indeksleyecektir.
*   **Toplu Yükleme:** `data/` klasöründeki tüm PDF'leri işlemek için:
    ```bash
    python ingest.py
    ```

### 💬 Adım 2: İstihbarat ve Sorgulama
*   **"Asistan"** sekmesine gelin.
*   **Birlik Seçimi:** (Opsiyonel) Sorgunuzu belirli bir birlikle sınırlandırmak için arama kutusuna birliğin adını yazın (Örn: "57. Tümen").
*   Sorunuzu sorun ve yapay zekanın belgelerden derlediği kanıta dayalı cevabı inceleyin.

### 🧐 Adım 3: Veri Denetimi
*   **"Veri Müfettişi"** sekmesinde, sistemin çıkardığı ham metinleri kontrol edebilir, hatalı okumaları tespit edebilir ve veri setinizi CSV olarak dışa aktarabilirsiniz.

---

## ⚙️ Yapılandırma

`src/config/settings.py` dosyası üzerinden sistemin tüm parametrelerini özelleştirebilirsiniz:

| Parametre | Açıklama | Varsayılan |
|-----------|----------|------------|
| `OCR_MODEL_SMALL` | Hızlı tarama modeli | `qwen2.5vl:3b` |
| `OCR_MODEL_LARGE` | Detaylı tarama modeli | `qwen2.5vl:7b` |
| `CONFIDENCE_THRESHOLD` | Model değişim eşiği | `0.6` |
| `RAG_FETCH_K` | İlk aramada getirilecek belge sayısı | `20` |
| `RAG_TOP_K` | LLM'e gönderilecek en iyi belge sayısı | `5` |

---

## 🤝 Katkıda Bulunma

Bu proje açık kaynaklıdır. Hata bildirimleri, özellik istekleri ve Pull Request'ler memnuniyetle karşılanır.

## 📄 Lisans

MIT License. Detaylar için `LICENSE` dosyasına bakınız.
