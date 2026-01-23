import requests
from config.settings import OLLAMA_CHAT, LLM_MODEL

DIVISION_PROMPT = """Metindeki askeri birlikleri bul ve normalize et.

Birlik türleri: Tümen, Alay, Tugay, Kolordu, Fırka

Kurallar:
- OCR hatalarını düzelt (Tümenii → Tümeni)
- Yazıyla yazılmış sayıları rakama çevir (Dördüncü → 4.)
- "nci/ncu/ncı" eklerini kaldır (9 ncu → 9.)
- Fırka → Tümen olarak normalize et

Format: Her birliği yeni satırda yaz. Birlik yoksa BOŞ yaz.

Metin:
{text}"""


def extract_divisions(text):
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
        return [d.strip() for d in result.split('\n') if d.strip()]
    except Exception:
        return []
