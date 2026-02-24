import re

# Türkçe sıra ekleri (inci, üncü, nci, ncü, vs.)
_ORDINAL_PATTERN = r"['\']?\s*(?:inci|ıncı|uncu|üncü|nci|ncı|ncu|ncü|üncü|ıncı)"

# Birlik tipleri
_UNIT_TYPES = r'(?:Tümen|Kolordu|Ordu|Alay|Tugay|Tabur|Bölük|Batarya|Süvari|Piyade)'


def tr_capitalize(word: str) -> str:
    """Türkçe'ye özgü büyük harf dönüştürme."""
    if not word:
        return word
    turkish_upper = {
        'a': 'A', 'b': 'B', 'c': 'C', 'ç': 'Ç', 'd': 'D', 'e': 'E',
        'f': 'F', 'g': 'G', 'ğ': 'Ğ', 'h': 'H', 'ı': 'I', 'i': 'İ',
        'j': 'J', 'k': 'K', 'l': 'L', 'm': 'M', 'n': 'N', 'ö': 'Ö',
        'p': 'P', 'r': 'R', 's': 'S', 'ş': 'Ş', 't': 'T', 'u': 'U',
        'ü': 'Ü', 'v': 'V', 'y': 'Y', 'z': 'Z'
    }
    return turkish_upper.get(word[0], word[0].upper()) + word[1:] if len(word) > 1 else word[0].upper()


def normalize_unit_name(raw_name: str) -> str:
    """
    Ham birlik ismini standart bir formata dönüştürür.
    Örn: "3 ncü Tümen" -> "3. Tümen"
         "57 nci Alay" -> "57. Alay"
         "1. Kol." -> "1. Kolordu"
         "111. ORDU" -> "111. Ordu"
         "ı. Kolordu" -> "1. Kolordu" (OCR hatası: ı -> 1)
         "i. Kolordu" -> "1. Kolordu" (OCR hatası: i -> 1)
         "1nci Ordu" -> "1. Ordu" (OCR: arada boşluk yok)
         "3ncü Tümen" -> "3. Tümen" (OCR: nokta yok)
    """
    if not raw_name:
        return ""

    # 0. Büyük/küçük harf normalizasyonu - önce hepsini küçük yap
    name = raw_name.strip().lower()
    # Fazla boşlukları sil
    name = re.sub(r'\s+', ' ', name)

    # 0.1. OCR hataları: "nci" -> "nci", "ncı" normalizasyonu
    # "1nci" -> "1 nci" (rakam ile yazı arasında boşluk yok)
    name = re.sub(r'(\d)([a-zA-ZçğıöşüÇĞİÖŞÜ]{2,})', r'\1 \2', name)
    
    # 0.2. "nci", "ncı", "ncü", "ncu" -> "nci" standardizasyonu
    name = name.replace('ı', 'i').replace('ı', 'i')
    name = re.sub(r'nc[ıi]', 'nci', name)
    name = re.sub(r'ünc', 'unci', name)

    # 1. OCR hatalarını düzelt: "ı." veya "i." bazen "1." demek
    # "ı. Kolordu", "i. Kolordu" -> "1. Kolordu"
    # Birlik tipinin hemen önüne geliyorsa OCR hatası olur
    name = re.sub(rf'^[ıi]\.(\s+(?:{_UNIT_TYPES}))', r'1.\1', name, flags=re.IGNORECASE)
    name = re.sub(rf'(\s)[ıi]\.(\s+(?:{_UNIT_TYPES}))', r'\g<1>1.\2', name, flags=re.IGNORECASE)

    # "1 ncü", "1 nci", "1 ncı" vb. -> "1." (sıra eklerini kaldır)
    # Alternatif olarak, sıra eki patternini genişlet
    _ORDINAL_WITH_SPACE = r"\s+(?:inci|ıncı|uncu|üncü|nci|ncı|ncu|ncü|unci|uncı)"
    name = re.sub(rf'(\d+){_ORDINAL_WITH_SPACE}', r'\1.', name, flags=re.IGNORECASE)
    
    # 2a. Sıra eklerini kaldır ve nokta ekle
    # "3 ncü", "57 inci", "1'inci" -> "3.", "57.", "1."
    name = re.sub(rf'(\d+){_ORDINAL_PATTERN}', r'\1.', name, flags=re.IGNORECASE)

    # 2a-2. Nokta sonrası bağımsız sıra ekini kaldır
    # "57. nci Tümen", "57. Ncü Tümen" -> "57. Tümen"
    _STANDALONE_ORDINAL = r'\b(?:inci|ıncı|uncu|üncü|nci|ncı|ncu|ncü|unci|uncı)\b'
    name = re.sub(rf'(\d+\.)\s*{_STANDALONE_ORDINAL}\s+', r'\1 ', name, flags=re.IGNORECASE)

    # 2b. Ek olmadan "57 Alay" -> "57. Alay"
    name = re.sub(rf'(\d+)\s+({_UNIT_TYPES})', r'\1. \2', name, flags=re.IGNORECASE)

    # 2c. Çoklu noktaları temizle "57.." -> "57."
    name = re.sub(r'\.{2,}', '.', name)

    # 2d. Nokta sonrası boşluk garantisi "57.Alay" -> "57. Alay"
    name = re.sub(rf'(\d+\.)\s*({_UNIT_TYPES})', r'\1 \2', name, flags=re.IGNORECASE)

    # 3. Yaygın Kısaltmaları Açma
    # Kol. -> Kolordu
    name = re.sub(r'\bKol\.', 'Kolordu', name, flags=re.IGNORECASE)
    name = re.sub(r'\bKol(\s|$)', 'Kolordu ', name, flags=re.IGNORECASE)
    
    # Tüm. -> Tümen
    name = re.sub(r'\bTüm\.', 'Tümen', name, flags=re.IGNORECASE)
    
    # Al. -> Alay
    name = re.sub(r'\bAl\.', 'Alay', name, flags=re.IGNORECASE)
    
    # Süv. -> Süvari
    name = re.sub(r'\bSüv\.', 'Süvari', name, flags=re.IGNORECASE)
    
    # Piy. -> Piyade
    name = re.sub(r'\bPiy\.', 'Piyade', name, flags=re.IGNORECASE)

    # 4. Tamlamaları Standartlaştırma (Tümeni -> Tümen)
    # Genellikle aramalarda yalın hal daha iyidir, ancak "3. Tümen" ile "3. Tümeni" aynı şeydir.
    # Biz yalın hale getirelim.
    name = re.sub(r'\bTümeni\b', 'Tümen', name, flags=re.IGNORECASE)
    name = re.sub(r'\bKolordusu\b', 'Kolordu', name, flags=re.IGNORECASE)
    name = re.sub(r'\bAlayı\b', 'Alay', name, flags=re.IGNORECASE)
    name = re.sub(r'\bTaburu\b', 'Tabur', name, flags=re.IGNORECASE)
    
    # 5. Baş Harfleri Büyüt
    # Türkçe karakter sorunu olmaması için basit title() yerine capitalize mantığı
    words = name.split()
    capitalized_words = []
    for w in words:
        if w:
            # Nokta ile bitiyorsa (örn: 3.) dokunma, değilse baş harfi büyüt
            if w[0].isdigit() and w.endswith('.'):
                capitalized_words.append(w)
            else:
                capitalized_words.append(tr_capitalize(w))
                
    final_name = " ".join(capitalized_words)
    
    # 6. Özel Düzeltmeler
    final_name = final_name.replace("Ve", "ve")
    
    return final_name
