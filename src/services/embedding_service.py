import logging
import threading
from typing import List, Tuple
from sentence_transformers import SentenceTransformer, CrossEncoder
from src.config import settings

logger = logging.getLogger(__name__)

class EmbeddingService:
    _embed_model = None
    _rerank_model = None
    _lock = threading.Lock()

    @classmethod
    def get_embed_model(cls):
        if cls._embed_model is None:
            with cls._lock:
                if cls._embed_model is None:
                    logger.info(f"Loading embedding model: {settings.EMBED_MODEL}")
                    cls._embed_model = SentenceTransformer(settings.EMBED_MODEL)
        return cls._embed_model

    @classmethod
    def get_rerank_model(cls):
        if cls._rerank_model is None:
            with cls._lock:
                if cls._rerank_model is None:
                    logger.info(f"Loading re-ranking model: {settings.RERANK_MODEL}")
                    cls._rerank_model = CrossEncoder(settings.RERANK_MODEL)
        return cls._rerank_model

    @classmethod
    def embed_texts(cls, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        model = cls.get_embed_model()
        embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return embeddings.tolist()

    @classmethod
    def embed_query(cls, text: str) -> List[float]:
        model = cls.get_embed_model()
        embedding = model.encode(text, show_progress_bar=False, convert_to_numpy=True)
        return embedding.tolist()

    @classmethod
    def rerank(cls, query: str, documents: List[str]) -> List[float]:
        """
        Re-ranks a list of documents based on their relevance to the query.
        Returns a list of scores corresponding to the documents.
        """
        if not documents:
            return []
        
        model = cls.get_rerank_model()
        # Create pairs of (query, document)
        pairs = [[query, doc] for doc in documents]
        scores = model.predict(pairs)
        return scores.tolist()
