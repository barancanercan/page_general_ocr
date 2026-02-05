import logging
from typing import List, Optional, Dict, Any
from src.services.vector_db_service import VectorDBService

logger = logging.getLogger(__name__)

class SupervisorAgent:
    """Agent responsible for supervising and reporting on the system state."""

    def __init__(self):
        self.vector_db = VectorDBService()

    def get_system_status(self) -> str:
        stats = self.vector_db.get_collection_stats()
        books = self.vector_db.get_ingested_books()
        
        info = f"System Status Report:\n"
        info += f"---------------------\n"
        info += f"Total Paragraphs: {stats['points_count']}\n"
        info += f"Total Vectors: {stats['vectors_count']}\n"
        info += f"Ingested Books ({len(books)}):\n"
        if books:
            info += "\n".join(f"  - {b}" for b in books)
        else:
            info += "  (No books ingested yet)"
        
        return info

    def get_ingested_books(self) -> List[str]:
        return self.vector_db.get_ingested_books()

    def get_paragraphs_with_units(self, book_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        # This logic was previously in vectordb.py, moving it here or keeping it in service
        # Since it's a specific query, let's implement it via the service
        client = self.vector_db.get_client()
        self.vector_db.ensure_collection()
        
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        from src.config import settings

        all_paragraphs = []
        offset = None

        scroll_filter = None
        if book_filter:
            scroll_filter = Filter(
                must=[FieldCondition(key="book_title", match=MatchValue(value=book_filter))]
            )

        while True:
            result = client.scroll(
                collection_name=settings.COLLECTION_NAME,
                scroll_filter=scroll_filter,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            points, next_offset = result
            for point in points:
                payload = point.payload
                units = payload.get("military_units", [])
                if units:
                    all_paragraphs.append(payload)
            if next_offset is None:
                break
            offset = next_offset

        return all_paragraphs
