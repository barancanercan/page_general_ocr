import re
import requests
from config.settings import OLLAMA_CHAT, LLM_MODEL

# Askeri birlik regex pattern'leri (pre-filter)
UNIT_PATTERNS = [
    # Sayısal formatlar
    r'\d+\.?\s*(?:Piyade\s+)?(?:Tümen|Tümeni|Tümenii|Tümenin)',
    r'\d+\.?\s*(?:Süvari\s+)?(?:Tümen|Tümeni)',
    r'\d+\.?\s*(?:Piyade\s+)?(?:Kolordu|Kolordusu|Kolordunun)',
    r'\d+\.?\s*(?:Piyade\s+)?(?:Tugay|Tugayı|Tugayın)',
    r'\d+\.?\s*(?:Piyade\s+)?(?:Alay|Alayı|Alayın)',
    r'\d+\.?\s*(?:Piyade\s+)?(?:Fırka|Fırkası|Fırkanın)',
    # Ordu formatları
    r'\d+\.?\s*(?:Ordu|Ordusu|Ordunun)',
    r'(?:Yıldırım|Kafkas|Şark|Garp)\s+(?:Ordu|Ordusu|Orduları)',
    # nci/ncı formatları
    r'\d+\s*(?:nci|ncı|ncu|üncü|inci|ıncı)\s+(?:Tümen|Kolordu|Tugay|Alay|Fırka|Ordu)',
    # Yazıyla yazılmış
    r'(?:Birinci|İkinci|Üçüncü|Dördüncü|Beşinci|Altıncı|Yedinci|Sekizinci|Dokuzuncu|Onuncu)\s+(?:Tümen|Kolordu|Fırka|Ordu)',
]

UNIT_REGEX = re.compile('|'.join(UNIT_PATTERNS), re.IGNORECASE)


def has_military_units(text):
    """Metinde askeri birlik var mı kontrol et (hızlı regex)."""
    return bool(UNIT_REGEX.search(text))


NORMALIZE_PROMPT = """Metindeki askeri birlikleri listele.

FORMAT: Her satırda SADECE birlik adı (örn: "3. Ordu", "9. Kolordu", "24. Piyade Tümeni")
Birlik yoksa: BOŞ

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
        unit_pattern = re.compile(
            r'(\d+\.?\s*(?:Piyade\s+|Süvari\s+)?(?:Tümen|Tümeni|Kolordu|Kolordusu|Tugay|Tugayı|Alay|Alayı|Ordu|Ordusu))',
            re.IGNORECASE
        )
        special_pattern = re.compile(
            r'((?:Yıldırım|Kafkas|Şark|Garp)\s+(?:Ordu|Ordusu|Kuvvet|Kuvvetleri))',
            re.IGNORECASE
        )

        for line in result.split('\n'):
            line = line.strip()
            # Çok uzun satırlar birlik adı olamaz
            if len(line) > 50:
                continue

            # Birlik adını regex ile çıkar
            match = unit_pattern.search(line)
            if match:
                divisions.append(match.group(1))
                continue

            match = special_pattern.search(line)
            if match:
                divisions.append(match.group(1))

        # Tekrarları kaldır
        return list(dict.fromkeys(divisions))

    except Exception:
        return []
