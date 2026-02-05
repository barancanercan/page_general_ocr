from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from datetime import datetime

@dataclass
class Paragraph:
    paragraph_id: str
    text: str
    book_title: str
    page_num: int
    paragraph_index: int
    page_paragraph_count: int = 0
    military_units: List[str] = field(default_factory=list)
    confidence: float = 0.0
    model_used: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class PageResult:
    page_num: int
    text: str
    confidence: float
    processing_time: float
    model_used: str
    success: bool = True
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class ProcessingStats:
    total_pages: int = 0
    processed_pages: int = 0
    successful_pages: int = 0
    failed_pages: int = 0
    total_time: float = 0.0
    total_characters: int = 0
    average_confidence: float = 0.0
    fallback_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
