"""Data models"""

from dataclasses import dataclass, field
from typing import List, Union


@dataclass
class Paragraph:
    book_id: str
    index: int
    text: str
    embedding: List[float]
    page: Union[int, str]
    units: List[str] = field(default_factory=list)
    confidence: float = 0.0

    @property
    def id(self):
        page = self.page if self.page != "unknown" else "x"
        return f"{self.book_id}_p{page}_{self.index:03d}"

    def to_dict(self):
        return {
            "id": self.id,
            "embedding": self.embedding,
            "document": self.text,
            "metadata": {
                "book_id": self.book_id,
                "index": self.index,
                "page": self.page,
                "units": self.units,
                "confidence": self.confidence
            }
        }
