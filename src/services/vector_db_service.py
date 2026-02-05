import logging
import uuid
import threading
from typing import List, Optional, Dict, Any
from qdrant_client import QdrantClient, models

from src.config import settings
from src.core.models import Paragraph

logger = logging.getLogger(__name__)

class VectorDBService:
    _client = None
    _client_lock = threading.Lock()

    @classmethod
    def get_client(cls) -> QdrantClient:
        if cls._client is None:
            with cls._client_lock:
                if cls._client is None: # Double-check locking
                    logger.info(f"Connecting to Qdrant at {settings.QDRANT_PATH}")
                    cls._client = QdrantClient(path=str(settings.QDRANT_PATH))
        return cls._client

    @classmethod
    def ensure_collection(cls):
        client = cls.get_client()
        try:
            collections = [c.name for c in client.get_collections().collections]
            if settings.COLLECTION_NAME not in collections:
                client.create_collection(
                    collection_name=settings.COLLECTION_NAME,
                    vectors_config=models.VectorParams(size=settings.EMBED_DIM, distance=models.Distance.COSINE),
                )
                logger.info(f"Created collection: {settings.COLLECTION_NAME}")
        except Exception as e:
            logger.error(f"Failed to ensure collection: {e}")


    @classmethod
    def upsert_paragraphs(cls, paragraphs: List[Paragraph], vectors: List[List[float]]):
        client = cls.get_client()
        cls.ensure_collection()

        points = []
        for para, vec in zip(paragraphs, vectors):
            point = models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload=para.to_dict(),
            )
            points.append(point)

        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            client.upsert(collection_name=settings.COLLECTION_NAME, points=batch)
        
        logger.info(f"Upserted {len(points)} paragraphs to Qdrant")

    @classmethod
    def is_book_ingested(cls, book_title: str) -> bool:
        client = cls.get_client()
        cls.ensure_collection()
        
        result, _ = client.scroll(
            collection_name=settings.COLLECTION_NAME,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="book_title", match=models.MatchValue(value=book_title))]
            ),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        return len(result) > 0

    @classmethod
    def hybrid_search(
        cls, 
        query_vector: List[float], 
        entities: List[str],
        top_k: int = 5, 
        book_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Performs a hybrid search using both semantic vector search and metadata filtering.
        """
        client = cls.get_client()
        cls.ensure_collection()

        base_filters = []
        if book_filter:
            base_filters.append(models.FieldCondition(key="book_title", match=models.MatchValue(value=book_filter)))

        # 1. Semantic Search
        semantic_results = client.query_points(
            collection_name=settings.COLLECTION_NAME,
            query=query_vector,
            query_filter=models.Filter(must=base_filters) if base_filters else None,
            limit=top_k,
            with_payload=True,
        ).points

        # 2. Entity Search
        entity_results = []
        if entities:
            entity_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="military_units",
                        match=models.MatchAny(any=entities),
                    )
                ] + base_filters
            )
            entity_results = client.query_points(
                collection_name=settings.COLLECTION_NAME,
                query=query_vector,
                query_filter=entity_filter,
                limit=top_k,
                with_payload=True,
            ).points

        # 3. Combine and de-duplicate
        combined_results = {}
        for point in semantic_results + entity_results:
            # Use paragraph_id for de-duplication
            para_id = point.payload.get('paragraph_id')
            if para_id and para_id not in combined_results:
                hit = dict(point.payload)
                hit["score"] = point.score
                combined_results[para_id] = hit
        
        sorted_hits = sorted(combined_results.values(), key=lambda x: x['score'], reverse=True)
        
        logger.info(f"Hybrid search: Found {len(semantic_results)} semantic, {len(entity_results)} entity results. Combined to {len(sorted_hits)} unique results.")
        
        return sorted_hits[:top_k]

    @classmethod
    def get_ingested_books(cls) -> List[str]:
        client = cls.get_client()
        cls.ensure_collection()
        
        books = set()
        offset = None
        while True:
            result, next_offset = client.scroll(
                collection_name=settings.COLLECTION_NAME,
                limit=100,
                offset=offset,
                with_payload=["book_title"],
                with_vectors=False,
            )
            for point in result:
                books.add(point.payload.get("book_title", ""))
            if next_offset is None:
                break
            offset = next_offset
        return sorted(list(books))

    @classmethod
    def get_collection_stats(cls) -> Dict[str, Any]:
        client = cls.get_client()
        try:
            info = client.get_collection(settings.COLLECTION_NAME)
            return {
                "vectors_count": info.vectors_count or 0,
                "points_count": info.points_count or 0,
            }
        except Exception:
            return {"vectors_count": 0, "points_count": 0}
