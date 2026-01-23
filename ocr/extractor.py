import re
import pytesseract
import pypdfium2 as pdfium
from config.settings import OCR_LANG, OCR_DPI, MAX_PAGES


def pdf_to_images(path, dpi=OCR_DPI, max_pages=MAX_PAGES):
    pdf = pdfium.PdfDocument(path)
    images = []
    for i in range(min(len(pdf), max_pages)):
        page = pdf.get_page(i)
        bitmap = page.render(scale=dpi / 72)
        images.append(bitmap.to_pil())
    return images


def ocr_page(img):
    text = pytesseract.image_to_string(img, lang=OCR_LANG)
    data = pytesseract.image_to_data(img, lang=OCR_LANG, output_type=pytesseract.Output.DICT)
    confidences = [int(c) for c in data['conf'] if int(c) > 0]
    avg_conf = sum(confidences) / len(confidences) / 100.0 if confidences else 0.0
    return text, round(avg_conf, 3)


def detect_page_number(text):
    lines = text.strip().split('\n')
    candidates = lines[:3] + lines[-3:] if len(lines) > 6 else lines
    for line in candidates:
        line = line.strip()
        if re.fullmatch(r'\d{1,4}', line):
            num = int(line)
            if 1 <= num <= 1500 and not (1800 <= num <= 2100):
                return num
    return "unknown"
