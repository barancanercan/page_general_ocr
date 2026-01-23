import os
import re
import json
import requests
import pytesseract
import pypdfium2 as pdfium
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = ""   # GPU tamamen kapalı

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
# Paragraf Sınıflandırma (LLM)
# ===============================

CLASSIFY_PROMPT = """Aşağıdaki metni sınıflandır. Sadece şu etiketlerden BİRİNİ yaz:

REAL_PARAGRAPH - Olay, bilgi veya açıklama içeren gerçek paragraf
HEADING - Başlık veya bölüm adı
METADATA - Yayınevi, ISBN, tarih, yazar bilgisi
JUNK - İçindekiler, sayfa numarası, anlamsız metin

Sadece etiketi yaz, başka bir şey yazma.

Metin:
{text}"""


def classify_text(text):
    """LLM ile metin sınıflandır."""
    if len(text.strip()) < 20:
        return "JUNK"

    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": CLASSIFY_PROMPT.format(text=text)}],
        "stream": False,
        "options": {"temperature": 0}
    }
    try:
        r = requests.post(OLLAMA_CHAT, json=payload, timeout=60)
        r.raise_for_status()
        result = r.json()["message"]["content"].strip().upper()
        if "REAL_PARAGRAPH" in result:
            return "REAL_PARAGRAPH"
        elif "HEADING" in result:
            return "HEADING"
        elif "METADATA" in result:
            return "METADATA"
        return "JUNK"
    except:
        return "JUNK"


def split_paragraphs(text):
    """Metni paragraflara böl, LLM ile sınıflandır, sadece REAL_PARAGRAPH döndür."""
    blocks = []
    for p in text.split("\n\n"):
        p = ' '.join(p.split())
        if len(p) > 30 and classify_text(p) == "REAL_PARAGRAPH":
            blocks.append(p)
    return blocks


# ===============================
# Askeri Birlik Çıkarımı (LLM)
# ===============================

DIVISION_PROMPT = """Metindeki askeri birlikleri bul ve normalize et.

Birlik türleri: Tümen, Alay, Tugay, Kolordu, Fırka

Kurallar:
- OCR hatalarını düzelt (Tümenii → Tümeni)
- Yazıyla yazılmış sayıları rakama çevir (Dördüncü → 4.)
- "nci/ncu/ncı" eklerini kaldır (9 ncu → 9.)
- Fırka → Tümen olarak normalize et

Format: Her birliği yeni satırda yaz. Birlik yoksa BOŞ yaz.

Örnek çıktı:
24. Piyade Tümeni
9. Piyade Tümeni

Metin:
{text}"""


def extract_divisions(text):
    """LLM ile askeri birlikleri çıkar ve normalize et."""
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": DIVISION_PROMPT.format(text=text)}],
        "stream": False,
        "options": {"temperature": 0}
    }
    try:
        r = requests.post(OLLAMA_CHAT, json=payload, timeout=60)
        r.raise_for_status()
        result = r.json()["message"]["content"].strip()
        if "BOŞ" in result.upper() or len(result) < 3:
            return []
        divisions = [d.strip() for d in result.split('\n') if d.strip()]
        return divisions
    except Exception:
        return []


# ===============================
# Ollama → Metin Temizleyici
# ===============================

SYSTEM_PROMPT = """Sen bir OCR post-processing motorusun. Görevin OCR'dan geçmiş Türkçe akademik/askeri metinleri düzeltmek.

KURALLAR:
1. Yalnızca OCR kaynaklı hataları düzelt (yanlış karakter, eksik harf, birleşik kelimeler)
2. Türkçe imla ve noktalama kurallarını uygula
3. Gereksiz satır kırılımlarını ve tire ile bölünmüş kelimeleri birleştir
4. Orijinal anlam ve içeriği ASLA değiştirme
5. Yeni bilgi, yorum veya açıklama EKLEME
6. Sadece düzeltilmiş metni döndür, başka hiçbir şey yazma

YASAK:
- "İşte düzeltilmiş metin" gibi girişler
- Madde işaretleri veya listeler oluşturma
- Özet veya açıklama ekleme
- Metni yeniden yazma veya genişletme"""

USER_PROMPT_TEMPLATE = """Aşağıdaki OCR metnini düzelt:

{text}"""


def ollama_clean(text):
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(text=text)}
        ],
        "stream": False,
        "options": {"temperature": 0.1, "top_p": 0.9}
    }

    r = requests.post(OLLAMA_CHAT, json=payload, timeout=300)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


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
# PDF → JSONL Pipeline
# ===============================

def process_pdf(pdf_path):
    book_id = os.path.basename(pdf_path).rsplit('.', 1)[0]
    images = pdf_to_images(pdf_path, max_pages=10)
    all_rows = []
    global_para_idx = 0

    for page_idx, img in enumerate(images):
        print(f"[{page_idx+1}/{len(images)}] OCR işleniyor...")
        raw_text, page_confidence = ocr_page(img)
        printed_page = detect_page_number(raw_text)

        paragraphs = split_paragraphs(raw_text)

        for para in paragraphs:
            global_para_idx += 1
            clean = ollama_clean(para)
            divisions = extract_divisions(clean)
            vec = embed(clean)

            row = {
                "id": f"parag_{global_para_idx}",
                "embedding": vec,
                "document": clean,
                "metadata": {
                    "division": divisions,
                    "confidence": page_confidence,
                    "source_page": printed_page,
                    "book_id": book_id,
                    "paragraph_index": global_para_idx
                }
            }
            all_rows.append(row)

    return all_rows


# ===============================
# RUN
# ===============================

if __name__ == "__main__":
    import sys

    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "data/1_Turk_istiklal_harbi_mondros_mutarekesi_tatbikat.pdf"
    output_path = pdf_path.rsplit('.', 1)[0] + ".jsonl"

    print(f"Girdi: {pdf_path}")
    print(f"Çıktı: {output_path}\n")

    data = process_pdf(pdf_path)

    with open(output_path, "w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n✓ Tamamlandı: {len(data)} paragraf → {output_path}")
