import shutil
import os
import logging
from pathlib import Path

# Loglama ayarları
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("Cleanup")

PROTECTED_DIRS = {'.', '..', 'C:\\', 'D:\\', 'qdrant_data', 'data', 'src', 'scripts'}

def _is_safe_path(path: Path) -> bool:
    resolved = path.resolve()
    if '..' in str(path):
        return False
    for protected in PROTECTED_DIRS:
        if protected in str(resolved):
            return False
    return True

def remove_directory(path, confirm: bool = False):
    """Bir klasörü ve içeriğini güvenli bir şekilde siler."""
    if not confirm:
        logger.info("Silme işlemi iptal edildi. Devam etmek için confirm=True kullanın.")
        return
    
    target_path = Path(path)
    if not _is_safe_path(target_path):
        logger.error(f"Hata: Bu klasör silinemez! Güvenli değil: {path}")
        return
    
    if os.path.exists(path):
        try:
            shutil.rmtree(path)
            logger.info(f"✅ SİLİNDİ: {path}")
        except Exception as e:
            logger.error(f"❌ HATA ({path}): {e}")
    else:
        logger.info(f"ℹ️ BULUNAMADI (Zaten yok): {path}")

def cleanup_system(confirm: bool = False):
    logger.info("--- 🧹 Sistem Temizliği Başlatılıyor ---")
    
    if not confirm:
        print("Silme işlemi iptal edildi. Devam etmek için confirm=True kullanın.")
        return
    
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
        remove_directory(str(dir_path), confirm=confirm)

    # Eski dosyaları da temizleyelim
    old_files = [
        root_dir / "run_new.py",
        root_dir / "bulk_ingest.py",
        root_dir / "cleanup_final.py",
        root_dir / "cleanup_project.py"
    ]
    
    for file_path in old_files:
        if file_path.exists():
            if not _is_safe_path(file_path):
                logger.error(f"Hata: Bu dosya silinemez! Güvenli değil: {file_path}")
                continue
            try:
                os.remove(file_path)
                logger.info(f"✅ SİLİNDİ: {file_path.name}")
            except Exception as e:
                logger.error(f"❌ HATA ({file_path.name}): {e}")

    logger.info("--- ✨ Temizlik Tamamlandı ---")

if __name__ == "__main__":
    import sys
    confirm = '--confirm' in sys.argv or '-y' in sys.argv
    cleanup_system(confirm=confirm)
