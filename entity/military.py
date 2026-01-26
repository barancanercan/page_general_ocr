"""Military unit patterns"""

UNIT_PATTERNS = [
    r'\d+\.?\s*(?:Piyade\s+)?(?:Tümen|Tümeni)',
    r'\d+\.?\s*(?:Kolordu|Kolordusu)',
    r'\d+\.?\s*(?:Tugay|Tugayı)',
    r'\d+\.?\s*(?:Alay|Alayı)',
    r'\d+\.?\s*(?:Ordu|Ordusu)',
    r'(?:Yıldırım|Kafkas|Şark|Garp)\s+(?:Ordu|Ordusu)',
]