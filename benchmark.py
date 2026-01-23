import os
import re
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
# Sayfa Numarası Tespiti
# ===============================

def detect_page_number(text):
    """Basılı sayfa numarasını tespit et. Bulunamazsa 'unknown' döndür."""
    lines = text.strip().split('\n')
    candidates = lines[:3] + lines[-3:] if len(lines) > 6 else lines

    for line in candidates:
        line = line.strip()
        if re.fullmatch(r'\d{1,4}', line):
            num = int(line)
            if 1 <= num <= 1500 and not (1800 <= num <= 2100):
                return num
    return "unknown"


# ===============================
# Paragraf Filtreleme
# ===============================

NON_PARAGRAPH_PATTERNS = [
    r'^(İÇİNDEKİLER|ÖNSÖZ|KAYNAKÇA|KISALTMALAR|EK\s*\d*)$',
    r'^(BÖLÜM|KISIM|FASIL)\s*[\dIVXLC]+',
    r'^ISBN[\s:\-]*[\dX\-]+',
    r'^\d{4}$',  # yıl
    r'^(Sayfa|Page)\s*\d+',
    r'^\.\.\.\s*\d+$',  # içindekiler satırı
]

def is_valid_paragraph(text):
    """Gerçek paragraf mı kontrol et (en az 2 cümle, anlatı içermeli)."""
    text = text.strip()
    if len(text) < 50:
        return False

    # Non-paragraph pattern kontrolü
    for pattern in NON_PARAGRAPH_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return False

    # Tamamen büyük harf = başlık
    if text.isupper() and len(text) < 100:
        return False

    # En az 2 cümle (nokta sayısı)
    sentence_count = len(re.findall(r'[.!?]', text))
    if sentence_count < 2:
        return False

    return True


def split_paragraphs(text):
    """Metni paragraflara böl ve filtrele."""
    blocks = []
    for p in text.split("\n\n"):
        p = ' '.join(p.split())  # normalize whitespace
        if is_valid_paragraph(p):
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
        printed_page = detect_page_number(raw_text)

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
                    "source_page": printed_page,
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
