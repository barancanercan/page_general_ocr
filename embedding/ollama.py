import re
import requests
from config.settings import OLLAMA_CHAT, OLLAMA_EMBED, LLM_MODEL, EMBED_MODEL


def fix_hyphenation(text):
    """Tire ile bölünmüş kelimeleri birleştir."""
    text = re.sub(r'(\w+)[­\-]\s*\n\s*(\w+)', r'\1\2', text)
    text = text.replace('\xad', '')
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# LLM hallucination pattern'leri
HALLUCINATION_PATTERNS = [
    r'^[*\s]*İşte düzeltilmiş metin[:\s]*',
    r'^[*\s]*Düzeltilmiş metin[:\s]*',
    r'^[*\s]*Aşağıda düzeltilmiş metin[:\s]*',
    r'^[*\s]*Yeni bilgi[:\s]*.*$',
    r'^[*\s]*Not[:\s]*.*$',
    r'^[*\s]*Açıklama[:\s]*.*$',
    r'\*\*Düzeltilmiş Metin:\*\*\s*',
    r'\*\*Eklem.*?\*\*.*$',
]


def remove_hallucinations(text):
    """LLM çıktısındaki hallucination'ları temizle."""
    for pattern in HALLUCINATION_PATTERNS:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
    # Başta/sonda kalan boşluk ve yıldız temizle
    text = re.sub(r'^[\s\*\-]+', '', text)
    text = re.sub(r'[\s\*\-]+$', '', text)
    return text.strip()


SYSTEM_PROMPT = """Sen bir OCR post-processing motorusun. Görevin OCR'dan geçmiş Türkçe akademik/askeri metinleri düzeltmek.

KURALLAR:
1. Yalnızca OCR kaynaklı hataları düzelt
2. Türkçe imla ve noktalama kurallarını uygula
3. Gereksiz satır kırılımlarını ve tire ile bölünmüş kelimeleri birleştir
4. Orijinal anlam ve içeriği ASLA değiştirme
5. Yeni bilgi, yorum veya açıklama EKLEME
6. Sadece düzeltilmiş metni döndür

YASAK:
- "İşte düzeltilmiş metin" gibi girişler
- Madde işaretleri veya listeler oluşturma
- Özet veya açıklama ekleme"""


def clean_text(text):
    # Önce regex ile tire-bölünme düzelt
    text = fix_hyphenation(text)

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Aşağıdaki OCR metnini düzelt:\n\n{text}"}
        ],
        "stream": False,
        "options": {"temperature": 0.1, "top_p": 0.9}
    }
    r = requests.post(OLLAMA_CHAT, json=payload, timeout=300)
    r.raise_for_status()
    result = r.json()["message"]["content"].strip()

    # Hallucination temizle
    result = remove_hallucinations(result)
    return result


def embed(text):
    r = requests.post(OLLAMA_EMBED, json={"model": EMBED_MODEL, "prompt": text}, timeout=120)
    r.raise_for_status()
    return r.json()["embedding"]
