# Askeri Belge Analiz Sistemi (Pro)

Bu proje, askeri belgelerin (PDF) OCR ile taranması, metinlerin çıkarılması, vektörel veritabanına kaydedilmesi ve RAG (Retrieval-Augmented Generation) mimarisi ile sorgulanmasını sağlayan profesyonel bir sistemdir.

## Özellikler

*   **Modüler Mimari**: `src/` klasörü altında ayrıştırılmış servisler, ajanlar ve veri modelleri.
*   **Akıllı OCR**: 
    *   Öncelikle hızlı ve optimize edilmiş küçük model (`qwen2.5vl:3b`) kullanılır.
    *   Güven skoru düşükse veya metin yetersizse otomatik olarak daha büyük ve yetenekli modele (`qwen2.5vl:7b`) geçer.
    *   Paralel işleme ile yüksek performans.
*   **Gelişmiş Metin İşleme**: Tekrarlayan metinleri temizleme, Türkçe karakter düzeltmeleri ve askeri birlik tespiti.
*   **Vektör Veritabanı**: Qdrant ile ölçeklenebilir semantik arama.
*   **Ajan Tabanlı Yapı**:
    *   `IngestionAgent`: PDF işleme sürecini yönetir.
    *   `RAGAgent`: Kullanıcı sorularını cevaplar.
    *   `SupervisorAgent`: Sistem durumunu raporlar.
*   **Kullanıcı Arayüzü**: Gradio tabanlı modern web arayüzü.

## Kurulum

1.  Gerekli kütüphaneleri yükleyin:
    ```bash
    pip install -r requirements.txt
    ```

2.  Ollama modellerini indirin:
    ```bash
    ollama pull qwen2.5vl:3b
    ollama pull qwen2.5vl:7b
    ollama pull qwen2.5:7b
    ```

## Çalıştırma

Projeyi başlatmak için ana dizindeki `run_new.py` dosyasını çalıştırın:

```bash
python run_new.py
```

Tarayıcınızda `http://localhost:7860` adresine gidin.

## Proje Yapısı

```
page_general_ocr/
├── run_new.py              # Başlatma dosyası
├── src/
│   ├── agents/             # İş mantığını yöneten ajanlar
│   │   ├── ingestion_agent.py
│   │   ├── rag_agent.py
│   │   └── supervisor_agent.py
│   ├── config/             # Ayarlar
│   │   └── settings.py
│   ├── core/               # Veri modelleri
│   │   └── models.py
│   ├── services/           # Temel servisler (OCR, DB, Embedding)
│   │   ├── ocr_service.py
│   │   ├── embedding_service.py
│   │   └── vector_db_service.py
│   ├── utils/              # Yardımcı fonksiyonlar
│   │   ├── text_processing.py
│   │   └── military_extraction.py
│   └── main.py             # UI ve uygulama mantığı
└── ...
```

## Temizlenecek Dosyalar

Aşağıdaki eski dosyalar artık kullanılmamaktadır ve silinebilir:

*   `app.py`
*   `rag.py`
*   `run.py`
*   `run_2.py`
*   `config.py`
*   `ingest.py`
*   `models.py`
*   `clear_db.py`
*   `military.py`
*   `test_ocr.py`
*   `vectordb.py`
*   `embedding.py`
*   `paragraph.py`
*   `qwen_pipeline.py`
