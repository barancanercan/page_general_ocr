import re
from typing import List

# More robust and flexible Turkish military unit patterns
# This version is designed to be less strict and capture more variations.
_PATTERNS = [
    # Handles "3. tümen", "3'üncü tümen", "3 ncü tümen", "3 üncü tümen" etc.
    r'\d{1,3}\s*[\'’`\.]?\s*(?:[uüiı]?nc[uüiı])?\s*[Tt]ümen',
    r'\d{1,3}\s*[\'’`\.]?\s*(?:[uüiı]?nc[uüiı])?\s*[Kk]olordu',
    r'\d{1,3}\s*[\'’`\.]?\s*(?:[uüiı]?nc[uüiı])?\s*[Oo]rdu(?!\s+[Gg]rubu)',
    r'\d{1,3}\s*[\'’`\.]?\s*(?:[uüiı]?nc[uüiı])?\s*[Aa]lay',
    r'\d{1,3}\s*[\'’`\.]?\s*(?:[uüiı]?nc[uüiı])?\s*[Tt]ugay',
    
    # Roman numerals for Corps
    r'[IVXLC]+\s*\.\s*[Kk]olordu',

    # Specific named groups
    r'[A-ZÇĞİÖŞÜ][a-zçğıöşü]+\s+[Oo]rdular[ıi]\s+[Gg]rubu',
    r'\d{1,3}\s*\.\s*[Oo]rdu\s+[Gg]rubu',

    # Other units
    r'[Ss]uvari\s+[Tt]ümen[iı]?',
    r'[Ss]uvari\s+[Aa]lay[ıi]?',
    r'[Tt]opçu\s+[Aa]lay[ıi]?',
    r'[Mm]üstahkem\s+[Mm]evki',
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in _PATTERNS]

def extract_units(text: str) -> List[str]:
    """Extracts Turkish military unit names from text, case-insensitively."""
    if not text:
        return []

    found = set()
    for pattern in _COMPILED:
        for match in pattern.finditer(text):
            # Normalize whitespace and strip any trailing punctuation
            unit = re.sub(r'\s+', ' ', match.group().strip())
            unit = unit.rstrip('.,:;')
            found.add(unit)

    return sorted(list(found))
