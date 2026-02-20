import re

# Türkçe sıra ekleri (inci, üncü, nci, ncü, vs.)
_ORDINAL_PATTERN = r"['\']?\s*(?:inci|ıncı|uncu|üncü|nci|ncı|ncu|ncü)"

# Birlik tipleri
_UNIT_TYPES = r'(?:Tümen|Kolordu|Ordu|Alay|Tugay|Tabur|Bölük|Batarya)'


def normalize_unit_name(raw_name: str) -> str:
    """
    Ham birlik ismini standart bir formata dönüştürür.
    Örn: "3 ncü Tümen" -> "3. Tümen"
         "57 nci Alay" -> "57. Alay"
         "1. Kol." -> "1. Kolordu"
    """
    if not raw_name:
        return ""
    
    # 1. Temizlik
    name = raw_name.strip()
    # Fazla boşlukları sil
    name = re.sub(r'\s+', ' ', name)
    
    # 2a. Sıra eklerini kaldır ve nokta ekle
    # "3 ncü", "57 inci", "1'inci" -> "3.", "57.", "1."
    name = re.sub(rf'(\d+){_ORDINAL_PATTERN}', r'\1.', name, flags=re.IGNORECASE)

    # 2a-2. Nokta sonrası bağımsız sıra ekini kaldır
    # "57. nci Tümen", "57. Ncü Tümen" -> "57. Tümen"
    _STANDALONE_ORDINAL = r'\b(?:inci|ıncı|uncu|üncü|nci|ncı|ncu|ncü)\b'
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
                capitalized_words.append(w.capitalize())
                
    final_name = " ".join(capitalized_words)
    
    # 6. Özel Düzeltmeler
    final_name = final_name.replace("Ve", "ve")
    
    return final_name
