"""
Tekrar eden metinleri tespit edip sadece sorunlu sayfaları yeniden işler.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from typing import List, Dict, Tuple
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from src.config import settings
from src.services.ocr_service import OCRService
from src.services.embedding_service import EmbeddingService
from src.utils.text_processing import detect_phrase_repetition, detect_tail_repetition_detailed, split_paragraphs
from src.utils.military_extraction import extract_units

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def detect_repetition_in_text(text: str) -> dict:
    """
    Metinde OCR kaynaklı tekrar olup olmadığını kontrol eder.
    Sadece metnin sonunda 5+ ardışık tekrar varsa True döner.
    """
    return detect_tail_repetition_detailed(text, min_phrase_len=2, min_repeats=5)

def scan_database_for_repetitions() -> List[Dict]:
    """Veritabanını tarar ve tekrar içeren kayıtları bulur."""
    client = QdrantClient(path=str(settings.QDRANT_PATH))

    problematic_pages = []
    offset = None

    logger.info("Veritabanı taranıyor...")

    while True:
        results, offset = client.scroll(
            collection_name=settings.COLLECTION_NAME,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False
        )

        if not results:
            break

        for point in results:
            payload = point.payload
            text = payload.get("text", "")

            analysis = detect_repetition_in_text(text)

            if analysis["has_repetition"]:
                problematic_pages.append({
                    "id": point.id,
                    "book_title": payload.get("book_title", ""),
                    "page_num": payload.get("page_num", 0),
                    "original_text": text,
                    "payload": payload,
                    "repetition_count": analysis["repeat_count"],
                    "repeated_phrase": analysis["repeated_phrase"],
                    "tail_preview": analysis["tail_preview"]
                })

        if offset is None:
            break

    logger.info(f"Toplam {len(problematic_pages)} sorunlu kayıt bulundu")
    return problematic_pages

PDF_FOLDER = r"C:\Users\user\Desktop\page_general_ocr\data\raw\books"


def find_pdf_for_book(book_title: str) -> str:
    """book_title'dan PDF dosyasını bulur."""
    if not book_title:
        return ""

    # Klasördeki tüm PDF'leri listele
    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith('.pdf')]

    # Direkt eşleşme dene
    direct_match = book_title + ".pdf"
    for pdf in pdf_files:
        if pdf.lower() == direct_match.lower():
            return os.path.join(PDF_FOLDER, pdf)

    # Kısmi eşleşme dene (book_title PDF isminde geçiyor mu)
    book_lower = book_title.lower().replace("_", " ").replace("-", " ")
    for pdf in pdf_files:
        pdf_lower = pdf.lower().replace("_", " ").replace("-", " ").replace(".pdf", "")
        if book_lower in pdf_lower or pdf_lower in book_lower:
            return os.path.join(PDF_FOLDER, pdf)

    return ""


def reprocess_page_ocr(point_id: str, book_title: str, page_num: int, payload: dict) -> bool:
    """Tek bir sayfayı yeniden OCR'lar ve veritabanını günceller."""
    pdf_path = find_pdf_for_book(book_title)

    if not pdf_path or not os.path.exists(pdf_path):
        logger.error(f"PDF bulunamadı: {book_title}")
        return False

    try:
        ocr_service = OCRService()
        embed_service = EmbeddingService()
        client = QdrantClient(path=str(settings.QDRANT_PATH))

        # Sayfayı yeniden OCR'la (0-indexed)
        page_result = ocr_service.process_page(pdf_path, page_num - 1)

        if not page_result.success or not page_result.text:
            logger.warning(f"OCR başarısız: {book_title} sayfa {page_num}")
            return False

        # Tekrar kontrolü
        analysis = detect_repetition_in_text(page_result.text)
        if analysis["has_repetition"]:
            logger.warning(f"Yeniden OCR sonrası hala tekrar var: {book_title} s.{page_num}")
            # Yine de temizlenmiş halini kaydet
            from src.utils.text_processing import detect_tail_repetition
            cleaned_text, _, _, _ = detect_tail_repetition(page_result.text, 2, 5)
        else:
            cleaned_text = page_result.text

        # Yeni embedding
        new_vector = embed_service.embed_query(cleaned_text)

        # Payload güncelle
        updated_payload = payload.copy()
        updated_payload["text"] = cleaned_text
        updated_payload["pdf_path"] = pdf_path
        updated_payload["reprocessed"] = True

        # Upsert
        client.upsert(
            collection_name=settings.COLLECTION_NAME,
            points=[PointStruct(
                id=point_id,
                vector=new_vector,
                payload=updated_payload
            )]
        )

        logger.info(f"✓ Yeniden OCR: {book_title} s.{page_num}")
        return True

    except Exception as e:
        logger.error(f"Yeniden OCR hatası: {e}")
        return False

def main():
    print("=" * 60)
    print("TEKRAR TESPİT VE DÜZELTME ARACI")
    print("=" * 60)

    # 1. Sorunlu sayfaları bul
    problematic = scan_database_for_repetitions()

    if not problematic:
        print("\n✓ Hiçbir tekrar sorunu tespit edilmedi!")
        return

    # 2. Detaylı rapor göster (ilk 30 örnek)
    print(f"\n{len(problematic)} sorunlu sayfa bulundu.")
    print("\n" + "=" * 80)
    print("İLK 30 ÖRNEK (Detaylı):")
    print("=" * 80)

    for i, p in enumerate(problematic[:30]):
        print(f"\n[{i+1}] Kitap: {p['book_title']}, Sayfa: {p['page_num']}")
        print(f"    Tekrar eden phrase: \"{p['repeated_phrase']}\" x{p['repetition_count']+1}")
        print(f"    Metnin sonu:")
        print(f"    ...{p['tail_preview'][-120:]}")
        print("-" * 80)

    # 3. Kullanıcıya sor
    print("\n" + "-" * 60)
    choice = input("Sorunlu sayfaları yeniden işlemek ister misiniz? (e/h/sayı): ").strip().lower()

    if choice == 'h':
        print("İptal edildi.")
        return
    elif choice.isdigit():
        limit = int(choice)
        problematic = problematic[:limit]
        print(f"\nSadece ilk {limit} sayfa işlenecek.")

    # 4. Sayfaları yeniden OCR'la
    print(f"\nPDF klasörü: {PDF_FOLDER}")
    print("Sayfalar yeniden OCR'lanıyor...\n")

    success_count = 0
    fail_count = 0

    for i, p in enumerate(problematic):
        print(f"[{i+1}/{len(problematic)}] {p['book_title']} s.{p['page_num']}...", end=" ")

        result = reprocess_page_ocr(
            point_id=p["id"],
            book_title=p["book_title"],
            page_num=p["page_num"],
            payload=p["payload"]
        )
        if result:
            success_count += 1
            print("✓")
        else:
            fail_count += 1
            print("✗")

    print("\n" + "=" * 60)
    print(f"SONUÇ: {success_count} başarılı, {fail_count} başarısız")
    print("=" * 60)

if __name__ == "__main__":
    main()
