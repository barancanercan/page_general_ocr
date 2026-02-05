import re
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

TURKISH_CHARS = r'a-zA-ZçğıöşüÇĞİÖŞÜ'

def calculate_confidence(text: str) -> float:
    """Calculates a simple confidence score based on text length and Turkish characters."""
    if not text:
        return 0.0
    score = min(1.0, len(text) / 2000)
    if re.search(r'[çğıöşüÇĞİÖŞÜ]', text):
        score = min(1.0, score + 0.1)
    return round(score, 3)

def detect_and_remove_repetitions(text: str) -> Tuple[str, bool]:
    """Detects and removes repetitive patterns in text."""
    if not text:
        return text, False
    
    original_len = len(text)
    cleaned_text = text
    has_repetition = False
    
    # Line-based repetition
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
    
    # Sentence-based repetition
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
