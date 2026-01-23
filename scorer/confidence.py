import re


def calc_ocr_confidence(raw_confidence: float) -> float:
    """OCR confidence (Tesseract'tan gelen)."""
    return round(max(0.0, min(1.0, raw_confidence)), 3)


def calc_para_quality(text: str) -> float:
    """Paragraf kalitesi: uzunluk, cümle yapısı, bütünlük."""
    score = 0.0

    # Uzunluk puanı (200+ karakter ideal)
    if len(text) >= 200:
        score += 0.4
    elif len(text) >= 100:
        score += 0.2

    # Cümle sonu işareti sayısı
    sentences = len(re.findall(r'[.!?]', text))
    if sentences >= 3:
        score += 0.3
    elif sentences >= 1:
        score += 0.15

    # Kelime çeşitliliği
    words = text.split()
    if len(words) >= 30:
        score += 0.3
    elif len(words) >= 15:
        score += 0.15

    return round(min(1.0, score), 3)


def calc_entity_certainty(divisions: list, text: str) -> float:
    """Entity extraction güvenilirliği."""
    if not divisions:
        return 1.0  # Birlik yoksa kesin (yanlış pozitif riski yok)

    score = 0.0
    for div in divisions:
        # Division metinde geçiyor mu kontrol et
        if any(part.lower() in text.lower() for part in div.split()):
            score += 1.0

    return round(min(1.0, score / len(divisions)), 3)


def calc_final_confidence(ocr_conf: float, para_qual: float, entity_cert: float) -> float:
    """Final confidence: 0.4*ocr + 0.3*para + 0.3*entity"""
    return round(0.4 * ocr_conf + 0.3 * para_qual + 0.3 * entity_cert, 3)
