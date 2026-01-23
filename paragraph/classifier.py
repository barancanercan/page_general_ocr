import re

# JUNK patternleri (regex ile hızlı filtreleme)
JUNK_PATTERNS = [
    r'^[—\-–\s]*\d{1,4}[—\-–\s]*$',          # sayfa numarası
    r'^[\d\.\s]+$',                            # sadece sayı
    r'^\[?\d+\]$',                             # dipnot referansı
    r'^(İÇİNDEKİLER|ÖNSÖZ|KAYNAKÇA|INDEX)$',  # bölüm başlıkları
]

# HEADING patternleri
HEADING_PATTERNS = [
    r'^[A-ZÇĞİÖŞÜ\s]{10,}$',                  # tamamen büyük harf
    r'^(BÖLÜM|KISIM|FASIL)\s*[\dIVXLC]+',     # bölüm numarası
    r'^[IVX]+\.\s+[A-ZÇĞİÖŞÜ]',              # Romen rakamı başlık
]


def classify_text(text):
    """Hızlı regex tabanlı sınıflandırma."""
    text = text.strip()

    if len(text) < 30:
        return "JUNK"

    # JUNK kontrolü
    for pattern in JUNK_PATTERNS:
        if re.match(pattern, text, re.IGNORECASE):
            return "JUNK"

    # HEADING kontrolü
    for pattern in HEADING_PATTERNS:
        if re.match(pattern, text):
            return "HEADING"

    # En az bir cümle sonu işareti olmalı
    if not re.search(r'[.!?]', text):
        return "HEADING"

    return "REAL_PARAGRAPH"


MIN_PARAGRAPH_LENGTH = 200  # rules.md: en az 200 karakter


def split_paragraphs(text):
    """Metni paragraflara böl ve filtrele."""
    text = re.sub(r'\n{2,}', '\n\n', text)
    blocks = []

    for p in text.split("\n\n"):
        p = ' '.join(p.split())
        classification = classify_text(p)

        # REAL_PARAGRAPH ve minimum uzunluk
        if classification == "REAL_PARAGRAPH" and len(p) >= MIN_PARAGRAPH_LENGTH:
            blocks.append(p)
        # 2+ cümle varsa kısa da olsa kabul et
        elif classification == "REAL_PARAGRAPH" and len(re.findall(r'[.!?]', p)) >= 2:
            blocks.append(p)

    return blocks
