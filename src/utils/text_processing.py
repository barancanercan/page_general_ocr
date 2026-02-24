import re
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

TURKISH_CHARS = r'a-zA-ZçğıöşüÇĞİÖŞÜ'

# Modül düzeyinde derlenmiş regex patternleri
_REPEAT_LINE = re.compile(r'^(.+)$(\n\1)+', re.MULTILINE)
_REPEAT_SENTENCE = re.compile(r'(.+?)\s+\1\s+\1', re.IGNORECASE)
_HYPHEN_PATTERN = re.compile(r'([a-zA-ZçÇğĞıİöÖşŞüÜ])-\s*\n\s*([a-zA-ZçÇğĞıİöÖşŞüÜ])')
_DASH_PATTERN = re.compile(r'([a-zA-ZçÇğĞıİöÖşŞüÜ])[–—]\s*\n\s*([a-zA-ZçÇğĞıİöÖşŞüÜ])')

def calculate_confidence(text: str) -> float:
    """Calculates a simple confidence score based on text length and Turkish characters."""
    if not text:
        return 0.0
    score = min(1.0, len(text) / 2000)
    if re.search(r'[çğıöşüÇĞİÖŞÜ]', text):
        score = min(1.0, score + 0.1)
    return round(score, 3)

def detect_tail_repetition(text: str, min_phrase_len: int = 2, min_repeats: int = 3) -> Tuple[str, bool, int, str]:
    """
    OCR'ın paragraf sonunda takılıp kaldığı durumları tespit eder.
    Örnek: "yardımcı olur yardımcı olur yardımcı olur yardımcı olur"

    Metnin SON kısmına odaklanır ve iteratif temizleme yapar.
    En az 3 ardışık tekrar gerektirir (doğal tekrarları yok sayar).

    Returns: (cleaned_text, has_repetition, repeat_count, repeated_phrase)
    """
    if not text:
        return "", False, 0, ""

    current_text = text
    total_has_repetition = False
    total_repetition_count = 0
    last_repeated_phrase = ""

    # Iteratif temizleme - tekrar kalmayıncaya kadar devam et
    max_iterations = 10  # Sonsuz döngüyü önle
    for iteration in range(max_iterations):
        words = current_text.split()
        total_words = len(words)

        if total_words < 10:
            break

        # Son 150 kelimeyi kontrol et (daha geniş pencere - uzun tekrarlar için)
        tail_size = min(150, total_words)
        tail_start = total_words - tail_size
        tail_words = words[tail_start:]

        has_repetition = False
        repetition_count = 0
        repeated_phrase = ""
        best_cut_point = len(tail_words)

        # 2-20 kelimelik phrase'leri kontrol et (uzun tekrarlar için genişletildi)
        max_phrase_len = min(20, len(tail_words) // min_repeats)
        for phrase_len in range(min_phrase_len, max_phrase_len + 1):
            i = 0
            while i <= len(tail_words) - phrase_len * min_repeats:
                phrase = ' '.join(tail_words[i:i + phrase_len]).lower()

                # Çok kısa veya anlamsız phrase'leri atla
                if len(phrase) < 10:
                    i += 1
                    continue

                # Bu phrase kaç kez ardışık tekrar ediyor?
                repeat_count = 1
                j = i + phrase_len
                while j + phrase_len <= len(tail_words):
                    next_phrase = ' '.join(tail_words[j:j + phrase_len]).lower()
                    if next_phrase == phrase:
                        repeat_count += 1
                        j += phrase_len
                    else:
                        break

                # En az min_repeats ardışık tekrar varsa bu gerçek bir OCR hatası
                if repeat_count >= min_repeats:
                    has_repetition = True
                    repeated_phrase = phrase
                    cut_at = i + phrase_len  # Sadece ilk tekrarı tut
                    if cut_at < best_cut_point:
                        best_cut_point = cut_at
                        repetition_count = repeat_count - 1
                    break

                i += 1

            if has_repetition:
                break

        if has_repetition:
            total_has_repetition = True
            total_repetition_count += repetition_count
            last_repeated_phrase = repeated_phrase
            # Temizle ve tekrar kontrol et
            clean_words = words[:tail_start] + tail_words[:best_cut_point]
            current_text = ' '.join(clean_words)
        else:
            # Tekrar yok, çık
            break

    return current_text, total_has_repetition, total_repetition_count, last_repeated_phrase


def detect_phrase_repetition(text: str, min_phrase_len: int = 2, max_repeats: int = 4) -> Tuple[str, bool, int]:
    """
    Eski fonksiyon - geriye uyumluluk için.
    Yeni detect_tail_repetition fonksiyonunu çağırır.
    """
    cleaned, has_rep, count, _ = detect_tail_repetition(text, min_phrase_len, max_repeats)
    return cleaned, has_rep, count


def detect_tail_repetition_detailed(text: str, min_phrase_len: int = 2, min_repeats: int = 4) -> dict:
    """
    Detaylı tekrar analizi - debug ve raporlama için.
    """
    cleaned, has_rep, count, phrase = detect_tail_repetition(text, min_phrase_len, min_repeats)

    # Son 100 karakteri al (kuyruk önizlemesi)
    tail_preview = text[-150:] if len(text) > 150 else text

    return {
        "has_repetition": has_rep,
        "repeat_count": count,
        "repeated_phrase": phrase,
        "tail_preview": tail_preview,
        "cleaned_text": cleaned
    }

def detect_and_remove_repetitions(text: str) -> Tuple[str, bool]:
    """Detects and removes repetitive patterns in text."""
    if not text:
        return "", False

    original_len = len(text)
    cleaned_text = text
    has_repetition = False

    # 1. Phrase-level tekrar tespiti (yeni)
    cleaned_text, phrase_rep, rep_count = detect_phrase_repetition(cleaned_text)
    if phrase_rep:
        has_repetition = True
        logger.warning(f"Phrase repetition detected: {rep_count} repeated phrases removed")

    # 2. Line-based repetition
    lines = cleaned_text.split('\n')
    unique_lines = []
    seen_lines = set()

    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            unique_lines.append(line)
            continue

        if line_stripped in seen_lines:
            has_repetition = True
            continue
        seen_lines.add(line_stripped)
        unique_lines.append(line)

    cleaned_text = '\n'.join(unique_lines)

    # 3. Sentence-based repetition
    sentences = re.split(r'[.!?]+\s*', cleaned_text)
    unique_sentences = []
    seen_sentences = set()

    for sent in sentences:
        sent_stripped = sent.strip()
        if not sent_stripped or len(sent_stripped) < 20:
            unique_sentences.append(sent)
            continue

        if sent_stripped in seen_sentences:
            has_repetition = True
            continue
        seen_sentences.add(sent_stripped)
        unique_sentences.append(sent)

    cleaned_text = '. '.join([s for s in unique_sentences if s.strip()])
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
    cleaned_text = cleaned_text.strip()

    if has_repetition:
        logger.debug(f"Repetition cleanup: {original_len} -> {len(cleaned_text)} chars")

    return cleaned_text, has_repetition

def post_process_text(text: str) -> str:
    """Post-processes OCR text: removes repetitions and fixes formatting."""
    if not text:
        return text
    
    cleaned, _ = detect_and_remove_repetitions(text)
    
    # Fix hyphenation and Turkish characters
    tc = TURKISH_CHARS
    cleaned = re.sub(rf'([{tc}])-\s*\n\s*([{tc}])', r'\1\2', cleaned)
    cleaned = re.sub(rf'([{tc}])[–—]\s*\n\s*([{tc}])', r'\1\2', cleaned)
    
    cleaned = cleaned.replace('\u00AD', '')
    cleaned = re.sub(r' +', ' ', cleaned)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    
    return cleaned.strip()

def _is_junk(text: str) -> bool:
    """Filters out junk blocks like page numbers or footnotes."""
    stripped = text.strip()
    if not stripped:
        return True
    if re.fullmatch(r'\d{1,4}', stripped):
        return True
    if re.fullmatch(r'\d{1,3}[.)]\s*', stripped):
        return True
    if len(stripped) < 10 and not re.search(r'[a-zA-ZçğıöşüÇĞİÖŞÜ]{3,}', stripped):
        return True
    return False

def split_paragraphs(text: str, min_length: int = 200) -> List[str]:
    """Splits OCR text into paragraphs."""
    if not text or not text.strip():
        return []

    raw_blocks = re.split(r'\n\s*\n', text)
    blocks = [b.strip() for b in raw_blocks if not _is_junk(b)]

    if not blocks:
        return []

    merged = []
    buffer = ""

    for block in blocks:
        if buffer:
            buffer = buffer + "\n\n" + block
        else:
            buffer = block

        if len(buffer) >= min_length:
            merged.append(buffer)
            buffer = ""

    if buffer:
        if merged:
            merged[-1] = merged[-1] + "\n\n" + buffer
        else:
            merged.append(buffer)

    return merged
