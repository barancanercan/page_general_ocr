import re

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
    
    # 2. Sayı Suffixlerini Düzeltme (ncı, nci, uncu, üncü -> .)
    # "3 ncü" -> "3."
    name = re.sub(r'(\d+)\s*[\'’]?[ncvuü][cç][ıiuü]', r'\1.', name, flags=re.IGNORECASE)
    
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
