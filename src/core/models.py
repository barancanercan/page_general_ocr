from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime


class ToDictMixin:
    """Dataclass'lar için to_dict metodu."""
    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import fields
        return {f.name: getattr(self, f.name) for f in fields(self) if getattr(self, f.name) is not None}


@dataclass
class Paragraph(ToDictMixin):
    """
    Bir paragrafı temsil eden veri modeli.
    
    Attributes:
        paragraph_id: Paragrafın benzersiz tanımlayıcısı
        text: Paragrafın metin içeriği
        book_title: Paragrafın ait olduğu kitap başlığı
        page_num: Sayfa numarası
        paragraph_index: Sayfadaki paragraf sırası
        page_paragraph_count: Sayfadaki toplam paragraf sayısı
        military_units: Metinde bulunan askeri birimler listesi
        confidence: OCR güven skoru (0.0 - 1.0 arası)
        model_used: OCR işlemi için kullanılan model adı
    """
    paragraph_id: str
    text: str
    book_title: str
    page_num: int
    paragraph_index: int
    page_paragraph_count: Optional[int] = None
    military_units: List[str] = field(default_factory=list)
    confidence: Optional[float] = None
    model_used: Optional[str] = None


@dataclass
class PageResult(ToDictMixin):
    """
    Bir sayfa işleme sonucunu temsil eden veri modeli.
    
    Attributes:
        page_num: Sayfa numarası
        text: OCR sonucu elde edilen metin
        confidence: OCR güven skoru (0.0 - 1.0 arası)
        processing_time: İşleme süresi (saniye)
        model_used: OCR işlemi için kullanılan model adı
        success: İşleme başarılı mı
        error_message: Hata durumunda hata mesajı
    """
    page_num: int
    text: str
    confidence: Optional[float] = None
    processing_time: Optional[float] = None
    model_used: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class ProcessingStats(ToDictMixin):
    """
    OCR işleme istatistiklerini temsil eden veri modeli.
    
    Attributes:
        total_pages: Toplam sayfa sayısı
        processed_pages: İşlenen sayfa sayısı
        successful_pages: Başarılı işlenen sayfa sayısı
        failed_pages: Başarısız işlenen sayfa sayısı
        total_time: Toplam işleme süresi (saniye)
        total_characters: Toplam karakter sayısı
        average_confidence: Ortalama güven skoru
        fallback_count: Fallback model kullanım sayısı
    """
    total_pages: int = 0
    processed_pages: int = 0
    successful_pages: int = 0
    failed_pages: int = 0
    total_time: float = 0.0
    total_characters: int = 0
    average_confidence: float = 0.0
    fallback_count: int = 0
