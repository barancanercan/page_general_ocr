import logging
from pathlib import Path
import time

from src.config import settings
from src.agents.ingestion_agent import IngestionAgent

# Konsolda detaylı bilgi görebilmek için loglama ayarları
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("Ingest")

def run_ingest():
    """
    'data' klasörünü tarar, içindeki PDF dosyalarını bulur ve daha önce
    veritabanına eklenmemiş olanları işleyerek kaydeder.
    """
    logger.info("--- Toplu Belge İşleme Süreci Başlatıldı ---")
    
    # 'data' klasörünün varlığını kontrol et
    if not settings.DATA_DIR.exists():
        logger.error(f"Veri klasörü bulunamadı: {settings.DATA_DIR}")
        logger.error("Lütfen proje ana dizininde 'data' adında bir klasör oluşturun ve PDF'lerinizi içine kopyalayın.")
        return

    pdf_files = list(settings.DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        logger.warning(f"'{settings.DATA_DIR}' klasöründe hiç PDF dosyası bulunamadı. İşlem yapacak bir şey yok.")
        return

    logger.info(f"Toplam {len(pdf_files)} adet PDF dosyası bulundu. İşleme başlanıyor...")
    
    ingestion_agent = IngestionAgent()
    total_start_time = time.time()
    processed_count = 0
    skipped_count = 0
    error_count = 0

    for i, pdf_path in enumerate(pdf_files):
        file_start_time = time.time()
        logger.info(f"--- Dosya {i+1}/{len(pdf_files)} işleniyor: {pdf_path.name} ---")

        progress_messages = []
        def on_progress(msg: str):
            logger.info(f"  > {msg}")
            progress_messages.append(msg)

        try:
            result = ingestion_agent.ingest_pdf(
                pdf_path=str(pdf_path), 
                max_pages=0, # Tüm sayfaları işle
                progress_callback=on_progress
            )
            
            if result.get("status") == "skipped":
                skipped_count += 1
                logger.warning(f"'{pdf_path.name}' dosyası atlandı. Sebep: {result.get('message')}")
            elif result.get("status") == "ok":
                processed_count += 1
                logger.info(f"'{pdf_path.name}' başarıyla işlendi ve veritabanına eklendi.")
            else:
                error_count += 1
                logger.error(f"'{pdf_path.name}' işlenirken bir hata oluştu. Sebep: {result.get('message')}")

        except Exception as e:
            error_count += 1
            logger.critical(f"'{pdf_path.name}' dosyasında kritik bir hata meydana geldi: {e}", exc_info=True)
        
        file_duration = time.time() - file_start_time
        logger.info(f"'{pdf_path.name}' dosyasının işlemi {file_duration:.2f} saniyede tamamlandı.")

    total_duration = time.time() - total_start_time
    logger.info("--- Toplu Belge İşleme Süreci Tamamlandı ---")
    logger.info("Özet:")
    logger.info(f"  - Toplam Dosya Sayısı: {len(pdf_files)}")
    logger.info(f"  - Başarıyla İşlenen/Eklenen: {processed_count}")
    logger.info(f"  - Atlanan (Daha Önce Eklenmiş): {skipped_count}")
    logger.info(f"  - Hatalı: {error_count}")
    logger.info(f"  - Toplam Süre: {total_duration / 60:.2f} dakika.")

if __name__ == "__main__":
    run_ingest()
