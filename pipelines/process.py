"""Main processing pipeline"""

import os
import json
import time
from typing import List

from core.models import Paragraph
from ocr.extractor import pdf_to_images, ocr_parallel, detect_page_number
from paragraph.classifier import extract as extract_paragraphs
from entity.military import UNIT_PATTERNS
from llm import get_engine
from config.settings import BATCH_SIZE


def process_pdf(pdf_path: str) -> List[Paragraph]:
    """Process PDF and extract paragraphs with embeddings."""
    start = time.time()
    book_id = os.path.basename(pdf_path).rsplit('.', 1)[0]

    engine = get_engine()

    print(f"\n{'='*50}")
    print(f"Processing: {book_id}")
    print(f"{'='*50}")

    # 1. PDF to images
    print("\n[1/6] Loading PDF...")
    images = pdf_to_images(pdf_path)
    print(f"      {len(images)} pages")

    # 2. OCR
    print("\n[2/6] Running OCR...")
    ocr_results = ocr_parallel(images)

    # 3. Extract paragraphs
    print("\n[3/6] Extracting paragraphs...")
    blocks = []
    for i, (text, conf) in enumerate(ocr_results):
        page = detect_page_number(text)
        for para in extract_paragraphs(text):
            blocks.append({"text": para, "conf": conf, "page": page})
    print(f"      {len(blocks)} paragraphs")

    if not blocks:
        return []

    # 4. Clean with LLM
    print("\n[4/6] Cleaning text...")
    for i, block in enumerate(blocks):
        block["clean"] = engine.clean_ocr(block["text"])
        if (i + 1) % 10 == 0:
            print(f"\r      {i+1}/{len(blocks)}", end="", flush=True)
    print(f"\r      {len(blocks)}/{len(blocks)}")

    # Filter short texts
    valid = [b for b in blocks if len(b["clean"]) >= 200]
    print(f"      {len(valid)} valid paragraphs")

    if not valid:
        return []

    # 5. Extract units
    print("\n[5/6] Extracting units...")
    for i, block in enumerate(valid):
        block["units"] = engine.extract_units(block["clean"], UNIT_PATTERNS)
        if (i + 1) % 50 == 0:
            print(f"\r      {i+1}/{len(valid)}", end="", flush=True)
    print(f"\r      {len(valid)}/{len(valid)}")

    # 6. Generate embeddings
    print("\n[6/6] Generating embeddings...")
    texts = [b["clean"] for b in valid]
    embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        embeddings.extend(engine.embed_batch(batch))
        print(f"\r      {min(i+BATCH_SIZE, len(texts))}/{len(texts)}", end="", flush=True)
    print()

    # Build paragraphs
    paragraphs = []
    for i, block in enumerate(valid):
        para = Paragraph(
            book_id=book_id,
            index=i + 1,
            text=block["clean"],
            embedding=embeddings[i],
            page=block["page"],
            units=block["units"],
            confidence=block["conf"]
        )
        paragraphs.append(para)

    elapsed = time.time() - start
    print(f"\nDone: {len(paragraphs)} paragraphs in {elapsed:.0f}s")

    return paragraphs


def save_jsonl(paragraphs: List[Paragraph], path: str):
    """Save paragraphs to JSONL file."""
    with open(path, 'w', encoding='utf-8') as f:
        for p in paragraphs:
            f.write(json.dumps(p.to_dict(), ensure_ascii=False) + '\n')
    print(f"Saved: {path}")