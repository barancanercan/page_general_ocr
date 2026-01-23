import re
import requests
from config.settings import OLLAMA_CHAT, LLM_MODEL

# Askeri birlik regex pattern'leri (pre-filter)
UNIT_PATTERNS = [
    r'\d+\.?\s*(?:Piyade\s+)?(?:Tümen|Tümeni|Tümenii)',
    r'\d+\.?\s*(?:Piyade\s+)?(?:Kolordu|Kolordusu)',
    r'\d+\.?\s*(?:Piyade\s+)?(?:Tugay|Tugayı)',
    r'\d+\.?\s*(?:Piyade\s+)?(?:Alay|Alayı)',
    r'\d+\.?\s*(?:Piyade\s+)?(?:Fırka|Fırkası)',
    r'\d+\s*(?:nci|ncı|ncu|üncü|inci)\s+(?:Tümen|Kolordu|Tugay|Alay|Fırka)',
    r'(?:Birinci|İkinci|Üçüncü|Dördüncü|Beşinci|Altıncı|Yedinci|Sekizinci|Dokuzuncu|Onuncu)\s+(?:Tümen|Kolordu|Fırka)',
]

UNIT_REGEX = re.compile('|'.join(UNIT_PATTERNS), re.IGNORECASE)


def has_military_units(text):
    """Metinde askeri birlik var mı kontrol et (hızlı regex)."""
    return bool(UNIT_REGEX.search(text))


NORMALIZE_PROMPT = """Aşağıdaki metindeki askeri birlikleri normalize et.

KURALLAR:
- Sadece METİNDE AÇIKÇA GEÇEN birlikleri yaz
- OCR hatalarını düzelt (Tümenii → Tümeni)
- Sayı + birim formatı kullan (24. Piyade Tümeni)
- Fırka → Tümen olarak normalize et
- Her birliği yeni satırda yaz
- Metinde birlik YOKSA sadece BOŞ yaz

YASAK:
- Metinde olmayan birlik ekleme
- Açıklama veya yorum yazma

Metin:
{text}"""


def extract_divisions(text):
    """Regex pre-filter + LLM normalize ile askeri birlik çıkar."""
    # Pre-filter: regex ile birlik var mı kontrol et
    if not has_military_units(text):
        return []

    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": NORMALIZE_PROMPT.format(text=text)}],
        "stream": False,
        "options": {"temperature": 0}
    }

    try:
        r = requests.post(OLLAMA_CHAT, json=payload, timeout=60)
        r.raise_for_status()
        result = r.json()["message"]["content"].strip()

        # BOŞ veya çok kısa yanıt
        if "BOŞ" in result.upper() or len(result) < 5:
            return []

        # Satırlara böl ve filtrele
        divisions = []
        for line in result.split('\n'):
            line = line.strip()
            # Sadece birlik formatına uyan satırları al
            if re.search(r'\d+\.?\s*(?:Piyade\s+)?(?:Tümen|Kolordu|Tugay|Alay)', line, re.IGNORECASE):
                divisions.append(line)

        return divisions

    except Exception:
        return []
