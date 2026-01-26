"""PDF to Text extraction with OCR"""

import re
import pytesseract
import pypdfium2 as pdfium
from PIL import ImageEnhance, ImageFilter
from concurrent.futures import ThreadPoolExecutor
from config.settings import OCR_LANG, OCR_DPI, WORKERS


def pdf_to_images(path, dpi=OCR_DPI):
    """Convert PDF pages to images."""
    pdf = pdfium.PdfDocument(path)
    images = []
    for i in range(len(pdf)):
        page = pdf.get_page(i)
        bitmap = page.render(scale=dpi / 72)
        images.append(bitmap.to_pil())
    return images


def preprocess(img):
    """Enhance image for better OCR."""
    if img.mode != 'L':
        img = img.convert('L')
    img = ImageEnhance.Contrast(img).enhance(1.5)
    img = img.filter(ImageFilter.MedianFilter(3))
    return img


def ocr_page(img):
    """OCR single page, return (text, confidence)."""
    img = preprocess(img)
    text = pytesseract.image_to_string(img, lang=OCR_LANG)
    data = pytesseract.image_to_data(img, lang=OCR_LANG, output_type=pytesseract.Output.DICT)
    confs = [int(c) for c in data['conf'] if int(c) > 0]
    conf = sum(confs) / len(confs) / 100 if confs else 0.0
    return text, round(conf, 3)


def ocr_parallel(images):
    """OCR multiple pages in parallel."""
    results = [None] * len(images)

    def process(args):
        idx, img = args
        return idx, ocr_page(img)

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        for idx, result in executor.map(process, enumerate(images)):
            results[idx] = result
            print(f"\rOCR: {idx+1}/{len(images)}", end="", flush=True)

    print()
    return results


def detect_page_number(text):
    """Extract page number from text."""
    lines = text.strip().split('\n')
    candidates = lines[:10] + lines[-10:]

    patterns = [
        r'^[—\-–]\s*(\d{1,4})\s*[—\-–]$',
        r'^[—\-–](\d{1,4})[—\-–]$',
        r'^\((\d{1,4})\)$',
        r'^(\d{1,4})$',
    ]

    for line in candidates:
        line = line.strip()
        if len(line) > 15:
            continue
        for pattern in patterns:
            match = re.fullmatch(pattern, line)
            if match:
                num = int(match.group(1))
                if 1 <= num <= 2000:
                    return num
    return "unknown"
