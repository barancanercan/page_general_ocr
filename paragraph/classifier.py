import requests
from config.settings import OLLAMA_CHAT, LLM_MODEL

CLASSIFY_PROMPT = """Aşağıdaki metni sınıflandır. Sadece şu etiketlerden BİRİNİ yaz:

REAL_PARAGRAPH - Olay, bilgi veya açıklama içeren gerçek paragraf
HEADING - Başlık veya bölüm adı
METADATA - Yayınevi, ISBN, tarih, yazar bilgisi
JUNK - İçindekiler, sayfa numarası, anlamsız metin

Sadece etiketi yaz, başka bir şey yazma.

Metin:
{text}"""


def classify_text(text):
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
    except Exception:
        return "JUNK"


def split_paragraphs(text):
    blocks = []
    for p in text.split("\n\n"):
        p = ' '.join(p.split())
        if len(p) > 30 and classify_text(p) == "REAL_PARAGRAPH":
            blocks.append(p)
    return blocks
