import os
import json
from typing import List

from core.models import Paragraph
from core.cache import get_cache, set_cache
from ocr.extractor import pdf_to_images, ocr_page, detect_page_number
from paragraph.classifier import split_paragraphs
from entity.military import extract_divisions
from embedding.ollama import clean_text, embed
from scorer.confidence import (
    calc_ocr_confidence,
    calc_para_quality,
    calc_entity_certainty,
    calc_final_confidence
)


def process_pdf(pdf_path: str) -> List[Paragraph]:
    book_id = os.path.basename(pdf_path).rsplit('.', 1)[0]
    images = pdf_to_images(pdf_path)
    paragraphs = []
    global_idx = 0

    for page_idx, img in enumerate(images):
        print(f"[{page_idx+1}/{len(images)}] İşleniyor...")

        # OCR with cache
        cache_key = f"{book_id}_page_{page_idx}"
        cached = get_cache("ocr", cache_key)
        if cached:
            raw_text, raw_ocr_conf = cached["text"], cached["confidence"]
        else:
            raw_text, raw_ocr_conf = ocr_page(img)
            set_cache("ocr", cache_key, {"text": raw_text, "confidence": raw_ocr_conf})

        source_page = detect_page_number(raw_text)
        blocks = split_paragraphs(raw_text)

        for block in blocks:
            clean = clean_text(block)

            # Final filtre: temizlenmiş metin 200+ karakter olmalı
            if len(clean) < 200:
                continue

            global_idx += 1
            divisions = extract_divisions(clean)

            # Embedding with cache
            embed_cached = get_cache("embedding", clean[:100])
            if embed_cached:
                vec = embed_cached
            else:
                vec = embed(clean)
                set_cache("embedding", clean[:100], vec)

            # Confidence hesaplama
            ocr_conf = calc_ocr_confidence(raw_ocr_conf)
            para_qual = calc_para_quality(clean)
            entity_cert = calc_entity_certainty(divisions, clean)
            final_conf = calc_final_confidence(ocr_conf, para_qual, entity_cert)

            para = Paragraph(
                book_id=book_id,
                paragraph_index=global_idx,
                document=clean,
                embedding=vec,
                source_page=source_page,
                division=divisions,
                ocr_confidence=ocr_conf,
                para_quality=para_qual,
                entity_certainty=entity_cert,
                confidence=final_conf
            )
            paragraphs.append(para)

    return paragraphs


def save_jsonl(paragraphs: List[Paragraph], output_path: str):
    with open(output_path, 'w', encoding='utf-8') as f:
        for p in paragraphs:
            f.write(json.dumps(p.to_dict(), ensure_ascii=False) + '\n')
