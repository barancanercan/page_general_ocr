#!/usr/bin/env python3
"""
Veritabanındaki kalan OCR tekrarlarını tespit edip temizler.
Streamlit kapalıyken çalıştırın.
"""
import sys
sys.path.insert(0, '.')

from qdrant_client import models
from src.services.vector_db_service import VectorDBService
from src.utils.text_processing import detect_tail_repetition

def main():
    print("Veritabanı taranıyor...")

    client = VectorDBService.get_client()
    offset = None
    problematic = []
    total_scanned = 0

    # Tüm kayıtları tara
    while True:
        result, next_offset = client.scroll(
            collection_name='paragraphs',
            limit=500,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not result:
            break

        total_scanned += len(result)

        for point in result:
            text = point.payload.get('text', '')
            cleaned, has_rep, count, phrase = detect_tail_repetition(text)
            if has_rep:
                problematic.append({
                    'id': point.id,
                    'payload': point.payload,
                    'cleaned_text': cleaned,
                    'phrase': phrase[:50] if phrase else '',
                    'count': count
                })

        if next_offset is None:
            break
        offset = next_offset
        print(f"  Taranan: {total_scanned}, Sorunlu: {len(problematic)}", end='\r')

    print(f"\nToplam taranan: {total_scanned}")
    print(f"Sorunlu kayıt: {len(problematic)}")

    if not problematic:
        print("Temizlenecek kayıt yok!")
        return

    # İlk 5 örneği göster
    print("\nÖrnek sorunlu kayıtlar:")
    for p in problematic[:5]:
        book = p['payload'].get('book_title', '')
        page = p['payload'].get('page_num', '')
        print(f"  - {book}, Sayfa {page}: '{p['phrase']}...' ({p['count']} tekrar)")

    # Kullanıcı onayı
    answer = input(f"\n{len(problematic)} kayıt temizlenecek. Devam? (e/h): ")
    if answer.lower() != 'e':
        print("İptal edildi.")
        return

    # Kayıtları güncelle
    print("\nKayıtlar güncelleniyor...")
    updated = 0
    failed = 0

    for p in problematic:
        try:
            # Payload'ı güncelle
            new_payload = p['payload'].copy()
            new_payload['text'] = p['cleaned_text']

            # Qdrant'ta güncelle
            client.set_payload(
                collection_name='paragraphs',
                payload=new_payload,
                points=[p['id']]
            )
            updated += 1

            if updated % 50 == 0:
                print(f"  Güncellenen: {updated}/{len(problematic)}", end='\r')

        except Exception as e:
            failed += 1
            print(f"\nHata: {p['id']} - {e}")

    print(f"\n\nTamamlandı!")
    print(f"  Güncellenen: {updated}")
    print(f"  Başarısız: {failed}")

if __name__ == "__main__":
    main()
