import os
import json
from typing import List

from core.models import Paragraph
from core.cache import get_cache, set_cache
from ocr.extractor import pdf_to_images, ocr_page, detect_page_number
from paragraph.classifier import split_paragraphs
from entity.military import extract_divisions
from embedding.ollama import clean_text, embed


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
            raw_text, confidence = cached["text"], cached["confidence"]
        else:
            raw_text, confidence = ocr_page(img)
            set_cache("ocr", cache_key, {"text": raw_text, "confidence": confidence})

        source_page = detect_page_number(raw_text)
        blocks = split_paragraphs(raw_text)

        for block in blocks:
            global_idx += 1
            clean = clean_text(block)
            divisions = extract_divisions(clean)

            # Embedding with cache
            embed_cached = get_cache("embedding", clean[:100])
            if embed_cached:
                vec = embed_cached
            else:
                vec = embed(clean)
                set_cache("embedding", clean[:100], vec)

            para = Paragraph(
                book_id=book_id,
                paragraph_index=global_idx,
                document=clean,
                embedding=vec,
                source_page=source_page,
                division=divisions,
                confidence=confidence
            )
            paragraphs.append(para)

    return paragraphs


def save_jsonl(paragraphs: List[Paragraph], output_path: str):
    with open(output_path, 'w', encoding='utf-8') as f:
        for p in paragraphs:
            f.write(json.dumps(p.to_dict(), ensure_ascii=False) + '\n')
