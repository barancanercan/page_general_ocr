import logging
import uuid
import threading
from typing import List, Optional, Dict, Any, Union
from qdrant_client import QdrantClient, models

from src.config import settings
from src.core.models import Paragraph

logger = logging.getLogger(__name__)

_collection_checked = set()

class VectorDBService:
    _client = None
    _client_lock = threading.Lock()
    _initialized = False

    @classmethod
    def get_client(cls) -> QdrantClient:
        # Eger client zaten varsa, onu kullan
        if cls._client is not None:
            return cls._client

        # Streamlit context'inde ise, _client'in set edilmesini bekle
        # (init_qdrant tarafindan set edilecek)
        if cls._initialized:
            raise RuntimeError("Qdrant client not initialized. Call init_qdrant first.")

        with cls._client_lock:
            if cls._client is None:
                cls._init_client()
        return cls._client

    @classmethod
    def _init_client(cls):
        """Internal: client olustur"""
        import os
        logger.info(f"Connecting to Qdrant at {settings.QDRANT_PATH}")

        # Lock dosyasini temizle
        lock_file = settings.QDRANT_PATH / ".lock"
        if lock_file.exists():
            try:
                os.remove(lock_file)
            except:
                pass

        # Eski flock dosyasini da temizle
        flock_file = settings.QDRANT_PATH / ".qdrant_flock"
        if flock_file.exists():
            try:
                os.remove(flock_file)
            except:
                pass

        cls._client = QdrantClient(path=str(settings.QDRANT_PATH))
        cls._initialized = True

    @classmethod
    def set_client(cls, client: QdrantClient):
        """Harici client enjekte et (streamlit icin)"""
        cls._client = client
        cls._initialized = True

    @classmethod
    def ensure_collection(cls, force: bool = False):
        global _collection_checked
        
        if not force and settings.COLLECTION_NAME in _collection_checked:
            return
        
        client = cls.get_client()
        try:
            collections = [c.name for c in client.get_collections().collections]
            if settings.COLLECTION_NAME not in collections:
                client.create_collection(
                    collection_name=settings.COLLECTION_NAME,
                    vectors_config=models.VectorParams(size=settings.EMBED_DIM, distance=models.Distance.COSINE),
                )
                logger.info(f"Created collection: {settings.COLLECTION_NAME}")
            _collection_checked.add(settings.COLLECTION_NAME)
        except Exception as e:
            logger.error(f"Failed to ensure collection: {e}")


    @classmethod
    def upsert_paragraphs(cls, paragraphs: List[Paragraph], vectors: List[List[float]]):
        client = cls.get_client()
        cls.ensure_collection()

        points = []
        for para, vec in zip(paragraphs, vectors):
            payload = para.to_dict()
            if "military_units" not in payload:
                payload["military_units"] = []
            
            point = models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload=payload,
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
        book_filter: Optional[str] = None,
        unit_filter: Optional[Union[str, List[str]]] = None
    ) -> List[Dict[str, Any]]:
        client = cls.get_client()
        cls.ensure_collection()

        must_filters = []
        
        if book_filter and book_filter != "Tüm Kitaplar":
            must_filters.append(models.FieldCondition(key="book_title", match=models.MatchValue(value=book_filter)))
            
        if unit_filter and unit_filter != "Tüm Birlikler":
            # Eğer unit_filter bir liste ise (varyasyonlar), MatchAny kullan
            if isinstance(unit_filter, list):
                must_filters.append(models.FieldCondition(key="military_units", match=models.MatchAny(any=unit_filter)))
            else:
                must_filters.append(models.FieldCondition(key="military_units", match=models.MatchValue(value=unit_filter)))

        query_filter = models.Filter(must=must_filters) if must_filters else None

        semantic_results = client.query_points(
            collection_name=settings.COLLECTION_NAME,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        ).points

        entity_results = []
        if entities and (not unit_filter or unit_filter == "Tüm Birlikler"):
            entity_filter_cond = models.Filter(
                must=[
                    models.FieldCondition(
                        key="military_units",
                        match=models.MatchAny(any=entities),
                    )
                ] + (must_filters if must_filters else [])
            )
            entity_results = client.query_points(
                collection_name=settings.COLLECTION_NAME,
                query=query_vector,
                query_filter=entity_filter_cond,
                limit=top_k,
                with_payload=True,
            ).points

        combined_results = {}
        for point in semantic_results + entity_results:
            para_id = point.payload.get('paragraph_id')
            if not para_id:
                para_id = str(point.id)
                
            if para_id not in combined_results:
                hit = dict(point.payload)
                hit["score"] = point.score
                combined_results[para_id] = hit
        
        sorted_hits = sorted(combined_results.values(), key=lambda x: x['score'], reverse=True)
        logger.info(f"Search (Unit: {unit_filter}, Book: {book_filter}): Found {len(sorted_hits)} results.")
        return sorted_hits[:top_k]

    @classmethod
    def browse_paragraphs(
        cls, 
        book_filter: Optional[str] = None, 
        unit_filter: Optional[Union[str, List[str]]] = None, 
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        client = cls.get_client()
        cls.ensure_collection()

        must_filters = []
        
        if book_filter and book_filter != "Tüm Kitaplar":
            must_filters.append(models.FieldCondition(key="book_title", match=models.MatchValue(value=book_filter)))
            
        if unit_filter and unit_filter != "Tüm Birlikler":
            if isinstance(unit_filter, list):
                must_filters.append(models.FieldCondition(key="military_units", match=models.MatchAny(any=unit_filter)))
            else:
                must_filters.append(models.FieldCondition(key="military_units", match=models.MatchValue(value=unit_filter)))

        result, _ = client.scroll(
            collection_name=settings.COLLECTION_NAME,
            scroll_filter=models.Filter(must=must_filters) if must_filters else None,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        
        paragraphs = []
        for point in result:
            payload = point.payload
            units = payload.get("military_units", [])
            if isinstance(units, list):
                units_str = ", ".join(units)
            else:
                units_str = str(units)

            paragraphs.append({
                "Kitap": payload.get("book_title", "Bilinmiyor"),
                "Sayfa": payload.get("page_num", "?"),
                "Birlikler": units_str,
                "Metin": payload.get("text", "")
            })
            
        return paragraphs

    @classmethod
    def get_ingested_books(cls) -> List[str]:
        client = cls.get_client()
        cls.ensure_collection()
        
        books = set()
        offset = None
        while True:
            result, next_offset = client.scroll(
                collection_name=settings.COLLECTION_NAME,
                limit=500,
                offset=offset,
                with_payload=["book_title"],
                with_vectors=False,
            )
            if not result:
                break
            for point in result:
                books.add(point.payload.get("book_title", ""))
            if next_offset is None:
                break
            offset = next_offset
        return sorted(list(books))

    @classmethod
    def get_all_units(cls) -> List[str]:
        client = cls.get_client()
        cls.ensure_collection()
        
        units = set()
        offset = None
        while True:
            result, next_offset = client.scroll(
                collection_name=settings.COLLECTION_NAME,
                limit=500,
                offset=offset,
                with_payload=["military_units"],
                with_vectors=False,
            )
            if not result:
                break
            for point in result:
                point_units = point.payload.get("military_units", [])
                if isinstance(point_units, list):
                    for u in point_units:
                        if u: units.add(u)
            if next_offset is None:
                break
            offset = next_offset
        return sorted(list(units))

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
