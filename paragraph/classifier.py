"""Paragraph extraction and classification"""

import re

MIN_LENGTH = 200

JUNK_PATTERNS = [
    r'^[—\-–\s]*\d{1,4}[—\-–\s]*$',
    r'^[\d\.\s]+$',
    r'^\[?\d+\]$',
]


def is_valid(text):
    """Check if text is a valid paragraph."""
    text = text.strip()

    if len(text) < 30:
        return False

    for pattern in JUNK_PATTERNS:
        if re.match(pattern, text):
            return False

    # Must have sentence ending or be long enough
    has_sentence = bool(re.search(r'[.!?]', text))
    return len(text) >= MIN_LENGTH or (has_sentence and len(text) >= 100)


def extract(text):
    """Extract valid paragraphs from text."""
    text = re.sub(r'\n{2,}', '\n\n', text)
    paragraphs = []

    for block in text.split("\n\n"):
        block = ' '.join(block.split())
        if is_valid(block):
            paragraphs.append(block)

    return paragraphs
