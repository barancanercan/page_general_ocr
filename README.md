# PageGeneralOCR: Akıllı Döküman Analiz ve Sorgulama Platformu

**PageGeneralOCR**, taranmış PDF belgeleriniz için gelişmiş bir **Optik Karakter Tanıma (OCR)** ve **Akıllı Sorgulama (RAG)** platformudur. Bu sistem, PDF'lerdeki metinleri yüksek doğrulukla çıkarmakla kalmaz, aynı zamanda bu belgelerden oluşan bilgi havuzunda doğal dilde sorular sormanıza ve sentezlenmiş, kanıta dayalı yanıtlar almanıza olanak tanır.

![Sistem Arayüzü](https://i.imgur.com/your-screenshot-url.png) <!-- TODO: Arayüz ekran görüntüsü ekleyin -->

## 🚀 Temel Özellikler

- **Yüksek Başarımlı OCR:** Güçlü `Qwen-VL` ailesi modellerini kullanarak taranmış ve zorlu PDF'lerden bile metin, tablo ve diğer verileri yüksek doğrulukla çıkarır.
- **Akıllı Soru-Cevap (RAG):** Belgelerinizle sohbet edin. Sistem, sorunuza en uygun cevapları birden çok kaynaktan toplar, sentezler ve kanıtlarıyla birlikte sunar.
- **Gelişmiş Çeşitlendirme:** Sorgu sonuçları, en alakalı bilgileri garanti altına alırken aynı zamanda tüm belge koleksiyonundan çeşitli bakış açıları sunacak şekilde akıllıca sıralanır ve çeşitlendirilir.
- **Yerel ve Güvenli:** Tüm işlemler (OCR ve LLM) `Ollama` aracılığıyla yerel makinenizde çalışır. Verileriniz asla dışarı çıkmaz.
- **Kullanıcı Dostu Arayüz:** `Gradio` tabanlı basit ve etkileşimli arayüzü ile PDF yüklemek ve anında sorgulamaya başlamak çok kolaydır.

## 🏛️ Mimari

Sistem, iki ana ajan etrafında şekillenen modern bir RAG mimarisi kullanır:

1.  **Ingestion Agent (Veri Alım Ajanı):**
    - PDF belgelerini sayfa sayfa işler.
    - Her sayfayı `Qwen-VL` modelini kullanarak OCR işleminden geçirir.
    - Çıkarılan metin paragraflarını, gömme (embedding) modelleri aracılığıyla vektörlere dönüştürür.
    - Bu vektörleri ve ilgili meta verileri (kitap adı, sayfa numarası vb.) `Qdrant` vektör veritabanına kaydeder.

2.  **RAG Agent (Sorgulama Ajanı):**
    - Kullanıcı sorgusunu alır ve bir vektöre dönüştürür.
    - Vektör veritabanında geniş bir anlamsal arama (`k=25`) yaparak potansiyel olarak ilgili tüm belgeleri bulur.
    - `Cross-Encoder` tabanlı bir modelle bu aday belgeleri yeniden sıralayarak (`re-ranking`) en alakalı olanları belirler.
    - Hem en yüksek alaka düzeyini hem de kaynak çeşitliliğini sağlamak için hibrit bir **çeşitlendirme** stratejisi uygular ve nihai bağlam için en iyi belgeleri (`k=7`) seçer.
    - Bu zenginleştirilmiş ve çeşitlendirilmiş bağlamı, güçlü bir sistem talimatıyla birlikte `Ollama` üzerindeki bir LLM'e göndererek sentezlenmiş ve kanıta dayalı bir cevap oluşturur.

## 🛠️ Kurulum

### Gereksinimler
- Python 3.10+
- [Poetry](https://python-poetry.org/docs/#installation) (önerilir) veya `pip`
- [Ollama](https://ollama.com/)
- CUDA destekli bir NVIDIA GPU (önerilir)

### Adım 1: Gerekli Modelleri İndirin
Uygulamanın ihtiyaç duyduğu tüm modelleri Ollama aracılığıyla yerel olarak indirin.

```bash
# OCR için (Yüksek performanslı model)
ollama pull qwen2.5vl:7b

# Sohbet ve sentez için (Örnek model, başka bir model de kullanabilirsiniz)
ollama pull gemma3:latest
```

### Adım 2: Projeyi Kurun

**Poetry ile (Önerilen):**
```bash
# Projeyi klonlayın
git clone https://github.com/your-username/pagegeneralocr.git
cd pagegeneralocr

# Bağımlılıkları kurun
poetry install
```

**pip ile:**
```bash
# Projeyi klonlayın
git clone https://github.com/your-username/pagegeneralocr.git
cd pagegeneralocr

# Bağımlılıkları kurun
pip install -r requirements.txt
```

## 🏃‍♂️ Çalıştırma

Uygulamayı başlatmak için aşağıdaki komutu çalıştırın:

```bash
# Poetry kullanıyorsanız
poetry run python src/main.py

# pip kullanıyorsanız
python src/main.py
```

Uygulama varsayılan olarak `http://127.0.0.1:7860` adresinde çalışmaya başlayacaktır.

## 📋 Kullanım

1.  **PDF Yükleme:** Arayüzdeki "PDF Yükle" alanını kullanarak bir veya daha fazla PDF belgesini sisteme yükleyin. "Başla" butonuna tıkladığınızda, OCR ve indeksleme süreci başlayacaktır.
2.  **Sorgulama:** Yükleme tamamlandıktan sonra, "Sohbet" sekmesine geçin.
    - İsterseniz, aramayı belirli bir kitapla sınırlamak için "Kitap Filtresi"ni kullanın.
    - Sorunuzu metin kutusuna yazın ve "Gönder"e tıklayın.
3.  **Sonuçları İnceleme:** Model, sorunuza en uygun cevabı, kullandığı kaynakları ve işlem performansını gösterecektir.

## ⚙️ Yapılandırma

Tüm önemli ayarlar `src/config/settings.py` dosyasında bulunur. Bu dosyayı düzenleyerek aşağıdaki gibi parametreleri değiştirebilirsiniz:

- **Modeller:** `CHAT_MODEL`, `OCR_MODEL_LARGE`, `EMBED_MODEL`, `RERANK_MODEL`
- **RAG Parametreleri:** `RAG_FETCH_K` (ilk arama boyutu), `RAG_TOP_K` (nihai bağlam boyutu)
- **OCR Ayarları:** `OCR_DPI`, `CONFIDENCE_THRESHOLD` vb.

## 📄 Lisans

Bu proje MIT Lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakın.
