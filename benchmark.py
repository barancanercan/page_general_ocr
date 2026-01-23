import os
import json
import requests
import pytesseract
import pypdfium2 as pdfium
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = ""   # GPU tamamen kapalı

PDF_NAME = "istiklal_harbi.pdf"
OLLAMA_CHAT = "http://localhost:11434/api/chat"
OLLAMA_EMBED = "http://localhost:11434/api/embeddings"
LLM_MODEL = "qwen2.5:3b"
EMBED_MODEL = "nomic-embed-text"

# ===============================
# PDF → Image
# ===============================

def pdf_to_images(path, dpi=200, max_pages=10):
    pdf = pdfium.PdfDocument(path)
    images = []

    for i in range(min(len(pdf), max_pages)):
        page = pdf.get_page(i)
        bitmap = page.render(scale=dpi / 72)
        images.append(bitmap.to_pil())

    return images


# ===============================
# OCR (Tesseract)
# ===============================

def ocr_page(img):
    """OCR ile metin çıkar ve ortalama confidence döndür."""
    text = pytesseract.image_to_string(img, lang="tur+eng")

    # Confidence için data al
    data = pytesseract.image_to_data(img, lang="tur+eng", output_type=pytesseract.Output.DICT)
    confidences = [int(c) for c in data['conf'] if int(c) > 0]
    avg_conf = sum(confidences) / len(confidences) / 100.0 if confidences else 0.0

    return text, round(avg_conf, 3)


# ===============================
# Paragraf bölücü
# ===============================

def split_paragraphs(text):
    blocks = []
    for p in text.split("\n\n"):
        p = p.strip()
        if len(p) > 30:
            blocks.append(p)
    return blocks


# ===============================
# Ollama → Metin Temizleyici
# ===============================

def ollama_clean(text):
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are a document normalization engine. You NEVER output instructions, explanations or bullet points. You only output the cleaned document text in Turkish."
            },
            {
                "role": "user",
                "content": text
            }
        ],
        "stream": False
    }

    r = requests.post(OLLAMA_CHAT, json=payload, timeout=300)
    r.raise_for_status()
    return r.json()["message"]["content"]


# ===============================
# Ollama → Embedding
# ===============================

def embed(text):
    r = requests.post(
        OLLAMA_EMBED,
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=120
    )
    r.raise_for_status()
    return r.json()["embedding"]


# ===============================
# PDF → JSON Pipeline
# ===============================

def process_pdf(pdf_path):
    images = pdf_to_images(pdf_path, max_pages=10)
    all_rows = []

    for page_idx, img in enumerate(images):
        print(f"Sayfa {page_idx+1}/{len(images)} işleniyor...")
        raw_text, page_confidence = ocr_page(img)

        paragraphs = split_paragraphs(raw_text)

        for i, para in enumerate(paragraphs):
            clean = ollama_clean(para)
            vec = embed(clean)

            row = {
                "id": f"{PDF_NAME}_p{page_idx+1}_parag_{i+1}",
                "embedding": vec,
                "document": clean,
                "metadata": {
                    "source_pdf": PDF_NAME,
                    "source_page": page_idx + 1,
                    "paragraph_index": i + 1,
                    "ocr_confidence": page_confidence
                }
            }

            all_rows.append(row)

    return all_rows


# ===============================
# RUN
# ===============================

if __name__ == "__main__":
    pdf_path = "data/1_Turk_istiklal_harbi_mondros_mutarekesi_tatbikat.pdf"

    data = process_pdf(pdf_path)

    with open("output.json", "w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\nTamamlandı → output.jsonl (RAG uyumlu)")
