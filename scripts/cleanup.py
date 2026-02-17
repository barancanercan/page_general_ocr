import shutil
import os
import logging
from pathlib import Path

# Loglama ayarları
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("Cleanup")

def remove_directory(path):
    """Bir klasörü ve içeriğini güvenli bir şekilde siler."""
    if os.path.exists(path):
        try:
            shutil.rmtree(path)
            logger.info(f"✅ SİLİNDİ: {path}")
        except Exception as e:
            logger.error(f"❌ HATA ({path}): {e}")
    else:
        logger.info(f"ℹ️ BULUNAMADI (Zaten yok): {path}")

def cleanup_system():
    logger.info("--- 🧹 Sistem Temizliği Başlatılıyor ---")
    
    # Proje kök dizini (scripts klasörünün bir üstü)
    root_dir = Path(__file__).resolve().parent.parent
    
    # Silinecek geçici klasörler (Kök dizine göre)
    temp_dirs = [
        root_dir / ".temp_ocr",
        root_dir / ".temp_ocr_run",
        root_dir / "src" / "__pycache__",
        root_dir / "src" / "agents" / "__pycache__",
        root_dir / "src" / "config" / "__pycache__",
        root_dir / "src" / "core" / "__pycache__",
        root_dir / "src" / "services" / "__pycache__",
        root_dir / "src" / "utils" / "__pycache__",
        root_dir / "__pycache__"
    ]

    for dir_path in temp_dirs:
        remove_directory(str(dir_path))

    # Eski dosyaları da temizleyelim
    old_files = [
        root_dir / "run_new.py",
        root_dir / "bulk_ingest.py",
        root_dir / "cleanup_final.py",
        root_dir / "cleanup_project.py"
    ]
    
    for file_path in old_files:
        if file_path.exists():
            try:
                os.remove(file_path)
                logger.info(f"✅ SİLİNDİ: {file_path.name}")
            except Exception as e:
                logger.error(f"❌ HATA ({file_path.name}): {e}")

    logger.info("--- ✨ Temizlik Tamamlandı ---")

if __name__ == "__main__":
    cleanup_system()
