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


def split_paragraphs(text):
    """Metni paragraflara böl ve filtrele."""
    # Tek \n'leri de paragraf ayırıcı olarak kullan (eğer ardından boş satır varsa)
    text = re.sub(r'\n{2,}', '\n\n', text)
    blocks = []

    for p in text.split("\n\n"):
        p = ' '.join(p.split())
        if len(p) > 50 and classify_text(p) == "REAL_PARAGRAPH":
            blocks.append(p)

    return blocks
