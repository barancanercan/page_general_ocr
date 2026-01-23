from dataclasses import dataclass, field, asdict
from typing import List, Union


def generate_id(book_id: str, paragraph_index: int) -> str:
    """Generate canonical ID: bookid_p0005"""
    return f"{book_id}_p{paragraph_index:04d}"


@dataclass
class Paragraph:
    book_id: str
    paragraph_index: int
    document: str
    embedding: List[float]
    source_page: Union[int, str]
    division: List[str] = field(default_factory=list)
    confidence: float = 0.0

    @property
    def id(self) -> str:
        return generate_id(self.book_id, self.paragraph_index)

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
                "confidence": self.confidence
            }
        }
