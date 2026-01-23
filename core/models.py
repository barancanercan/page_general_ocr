from dataclasses import dataclass, field
from typing import List, Union


def generate_id(book_id: str, source_page: Union[int, str], paragraph_index: int) -> str:
    """Generate canonical ID: bookid_p<page>_para<index>"""
    page = source_page if source_page != "unknown" else "x"
    return f"{book_id}_p{page}_para{paragraph_index:03d}"


@dataclass
class Paragraph:
    book_id: str
    paragraph_index: int
    document: str
    embedding: List[float]
    source_page: Union[int, str]
    division: List[str] = field(default_factory=list)
    ocr_confidence: float = 0.0
    para_quality: float = 0.0
    entity_certainty: float = 0.0
    confidence: float = 0.0

    @property
    def id(self) -> str:
        return generate_id(self.book_id, self.source_page, self.paragraph_index)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "embedding": self.embedding,
            "document": self.document,
            "metadata": {
                "book_id": self.book_id,
                "paragraph_index": self.paragraph_index,
                "source_page": self.source_page,
                "division": self.division,
                "confidence": self.confidence,
                "ocr_confidence": self.ocr_confidence,
                "para_quality": self.para_quality,
                "entity_certainty": self.entity_certainty
            }
        }
