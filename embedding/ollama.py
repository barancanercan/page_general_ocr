import requests
from config.settings import OLLAMA_CHAT, OLLAMA_EMBED, LLM_MODEL, EMBED_MODEL

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
    return r.json()["message"]["content"].strip()


def embed(text):
    r = requests.post(OLLAMA_EMBED, json={"model": EMBED_MODEL, "prompt": text}, timeout=120)
    r.raise_for_status()
    return r.json()["embedding"]
