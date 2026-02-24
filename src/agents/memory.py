import logging
import json
import os
import threading
import re
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any
from collections import deque

logger = logging.getLogger(__name__)

CORPUS_FILE = "data/memory/military_corpus.json"
ONTOLOGY_FILE = "data/memory/military_ontology.json"
DECISIONS_FILE = "data/memory/military_decisions.json"
MICRO_DECISIONS_FILE = "data/memory/micro_decisions.json"

# Optimized configuration constants
DEFAULT_SUMMARY_INTERVAL = 8  # Increased from 5 for better context accumulation
MAX_MESSAGE_BUFFER = 100  # Prevent unbounded memory growth
MAX_CONTEXT_LENGTH = 300  # Content truncation length


class ConversationMemory:
    """
    Short-term conversation memory with optimized summarization interval.
    Maintains conversation history and periodic summaries for context preservation.
    """

    def __init__(self, summary_interval: int = DEFAULT_SUMMARY_INTERVAL):
        """
        Initialize conversation memory.

        Args:
            summary_interval: Number of messages before auto-summarization
                             Optimal range: 7-10 (default: 8)
        """
        self.summary_interval = max(3, min(summary_interval, 15))  # Clamp between 3-15
        self.messages = deque(maxlen=min(self.summary_interval * 3, MAX_MESSAGE_BUFFER))
        self.summary = ""
        self.session_id = None
        self._message_count = 0  # Track total messages processed

    def add_message(self, role: str, content: str):
        """Add a message with validation."""
        if not content or not content.strip():
            return

        self.messages.append({
            "role": role,
            "content": content.strip(),
            "timestamp": datetime.now().isoformat()
        })
        self._message_count += 1

    def should_summarize(self) -> bool:
        """Check if summarization is needed."""
        return len(self.messages) >= self.summary_interval

    def get_recent_context(self, max_messages: Optional[int] = None) -> str:
        """
        Get recent context with optional message limit.

        Args:
            max_messages: Maximum number of recent messages to include

        Returns:
            Formatted context string
        """
        if not self.messages:
            return ""

        context_parts = []

        # Always include summary if available
        if self.summary:
            context_parts.append(f"[Önceki Konuşma Özeti]: {self.summary}")

        # Get recent messages (optimized limit)
        limit = max_messages or self.summary_interval
        recent = list(self.messages)[-limit:]

        for msg in recent:
            role = "Kullanıcı" if msg["role"] == "user" else "Asistan"
            content = msg["content"]

            # Truncate long content for context window efficiency
            if len(content) > MAX_CONTEXT_LENGTH:
                content = content[:MAX_CONTEXT_LENGTH] + "..."

            context_parts.append(f"{role}: {content}")

        return "\n".join(context_parts)

    def update_summary(self, summary: str):
        """
        Update conversation summary.

        Args:
            summary: New summary text
        """
        if summary:
            self.summary = summary
            logger.info(f"Conversation summary updated ({len(summary)} chars)")

    def get_message_count(self) -> int:
        """Get total number of messages processed."""
        return self._message_count

    def clear(self):
        """Clear all messages and summary."""
        self.messages.clear()
        self.summary = ""
        self._message_count = 0


class GlobalMemory:
    """
    Thread-safe global memory manager for multiple sessions.
    Provides singleton-like access with proper synchronization.
    """

    _instance = None
    _memory: Dict[str, ConversationMemory] = {}
    _lock = threading.RLock()  # Use RLock for reentrant safety
    _session_timestamps: Dict[str, datetime] = {}
    _max_sessions = 100  # Prevent unbounded session growth

    @classmethod
    def get_memory(cls, session_id: str = "default") -> ConversationMemory:
        """
        Get or create conversation memory for a session.

        Args:
            session_id: Unique session identifier

        Returns:
            ConversationMemory instance for the session

        Raises:
            ValueError: If session_id is invalid
        """
        if not session_id or not isinstance(session_id, str):
            raise ValueError("Invalid session_id: must be non-empty string")

        with cls._lock:
            # Check max sessions limit
            if (len(cls._memory) >= cls._max_sessions and
                session_id not in cls._memory):
                # Clean up oldest session
                oldest_session = min(
                    cls._session_timestamps.items(),
                    key=lambda x: x[1]
                )[0]
                cls._memory.pop(oldest_session, None)
                cls._session_timestamps.pop(oldest_session, None)
                logger.warning(f"Removed oldest session {oldest_session} to stay within limits")

            if session_id not in cls._memory:
                memory = ConversationMemory(summary_interval=DEFAULT_SUMMARY_INTERVAL)
                memory.session_id = session_id
                cls._memory[session_id] = memory
                cls._session_timestamps[session_id] = datetime.now()
                logger.debug(f"Created new session: {session_id}")
            else:
                # Update access timestamp
                cls._session_timestamps[session_id] = datetime.now()

            return cls._memory[session_id]

    @classmethod
    def clear_session(cls, session_id: str = "default"):
        """
        Clear memory for a specific session.

        Args:
            session_id: Session identifier to clear
        """
        with cls._lock:
            if session_id in cls._memory:
                cls._memory[session_id].clear()
                cls._session_timestamps.pop(session_id, None)
                logger.debug(f"Cleared session: {session_id}")

    @classmethod
    def clear_all(cls):
        """Clear all sessions."""
        with cls._lock:
            cls._memory.clear()
            cls._session_timestamps.clear()
            logger.info("Cleared all sessions")

    @classmethod
    def get_session_count(cls) -> int:
        """Get number of active sessions."""
        with cls._lock:
            return len(cls._memory)

    @classmethod
    def get_session_ids(cls) -> List[str]:
        """Get list of all session IDs."""
        with cls._lock:
            return list(cls._memory.keys())


class LongTermMemory:
    """
    Long-term memory system for military and historical corpus.
    Stores previously discussed topics and general military history knowledge.
    Optimized for fast search and relevant context retrieval.
    """

    def __init__(self, corpus_path: str = CORPUS_FILE):
        """
        Initialize long-term memory.

        Args:
            corpus_path: Path to corpus JSON file
        """
        self.corpus_path = corpus_path
        self.corpus: Dict[str, Any] = {
            "topics": {},
            "entities": {},
            "history": [],
            "metadata": {}
        }
        # Search cache for performance optimization
        self._search_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_hit_count = 0
        self._load_corpus()
    
    def _load_corpus(self):
        """Corpus ve Ontology dosyalarını yükle."""
        self.corpus = {"topics": {}, "entities": {}, "ontology": {}, "history": [], "metadata": {}}
        
        if os.path.exists(self.corpus_path):
            try:
                with open(self.corpus_path, 'r', encoding='utf-8') as f:
                    corpus_data = json.load(f)
                    self.corpus.update(corpus_data)
                logger.info(f"Corpus loaded: {len(self.corpus.get('topics', {}))} topics, {len(self.corpus.get('entities', {}))} entities")
            except Exception as e:
                logger.warning(f"Failed to load corpus: {e}")
        
        if os.path.exists(ONTOLOGY_FILE):
            try:
                with open(ONTOLOGY_FILE, 'r', encoding='utf-8') as f:
                    ontology_data = json.load(f)
                    self.corpus["ontology"] = ontology_data
                logger.info("Military ontology loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load ontology: {e}")
        
        if os.path.exists(DECISIONS_FILE):
            try:
                with open(DECISIONS_FILE, 'r', encoding='utf-8') as f:
                    decisions_data = json.load(f)
                    self.corpus["decisions"] = decisions_data
                logger.info("Military decisions dataset loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load decisions: {e}")
        
        if os.path.exists(MICRO_DECISIONS_FILE):
            try:
                with open(MICRO_DECISIONS_FILE, 'r', encoding='utf-8') as f:
                    micro_data = json.load(f)
                    self.corpus["micro_decisions"] = micro_data
                logger.info("Micro decisions dataset loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load micro decisions: {e}")
        
        if not self.corpus.get("metadata"):
            self._init_empty_corpus()
    
    def _init_empty_corpus(self):
        """Boş corpus başlat."""
        now = datetime.now().isoformat()
        self.corpus = {
            "topics": {},
            "entities": {},
            "history": [],
            "metadata": {
                "created": now,
                "last_updated": now,
                "version": "1.0"
            }
        }
    
    def _save_corpus(self):
        """Corpus dosyasını kaydet."""
        try:
            os.makedirs(os.path.dirname(self.corpus_path), exist_ok=True)
            with open(self.corpus_path, 'w', encoding='utf-8') as f:
                json.dump(self.corpus, f, ensure_ascii=False, indent=2)
            logger.info("Corpus saved successfully")
        except Exception as e:
            logger.error(f"Failed to save corpus: {e}")
    
    def add_topic(self, topic: str, content: str, keywords: List[str], sources: List[str] = None):
        """
        Yeni bir konu ekle.
        
        Args:
            topic: Konu başlığı (örn: "1. İnönü Muharebesi")
            content: Detaylı bilgi
            keywords: Anahtar kelimeler listesi
            sources: Kaynaklar
        """
        self.corpus["topics"][topic] = {
            "content": content,
            "keywords": keywords,
            "sources": sources or [],
            "last_accessed": "2026-02-23"
        }
        self._update_metadata()
        self._save_corpus()
    
    def add_entity(self, entity_type: str, name: str, data: Dict[str, Any]):
        """
        Bir varlık ekle (birlik, komutan, savaş vb.)
        
        Args:
            entity_type: Tür (örn: "birlik", "komutan", "cephane")
            name: İsim
            data: Detaylı bilgi
        """
        if entity_type not in self.corpus["entities"]:
            self.corpus["entities"][entity_type] = {}
        
        self.corpus["entities"][entity_type][name] = {
            **data,
            "last_updated": "2026-02-23"
        }
        self._update_metadata()
        self._save_corpus()
    
    def _normalize_query(self, query: str) -> set:
        """
        Normalize query to searchable words (optimized).

        Args:
            query: Raw query string

        Returns:
            Set of normalized query words
        """
        # More efficient normalization using regex
        normalized = re.sub(r'[?.,!;:\s]+', ' ', query.lower()).strip()
        return set(normalized.split()) if normalized else set()

    def search(self, query: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Search corpus for information related to query.

        Args:
            query: Search query string
            use_cache: Enable result caching for identical queries

        Returns:
            Dictionary containing relevant topics, entities, and history
        """
        # Cache check for identical queries (performance optimization)
        if use_cache and query in self._search_cache:
            self._cache_hit_count += 1
            return self._search_cache[query].copy()

        query_words = self._normalize_query(query)

        if not query_words:
            return {
                "topics": [],
                "entities": {},
                "battle_patterns": [],
                "timeline": [],
                "advanced_analysis": [],
                "ontology": [],
                "micro_decisions": []
            }

        results = {
            "topics": [],
            "entities": {},
            "battle_patterns": [],
            "timeline": [],
            "advanced_analysis": [],
            "ontology": [],
            "micro_decisions": []
        }

        # Search topics with improved scoring
        for topic_key, data in self.corpus.get("topics", {}).items():
            score = self._calculate_topic_score(data, query_words, topic_key)

            if score > 0:
                results["topics"].append({
                    **data,
                    "topic_key": topic_key,
                    "score": score
                })

        # Search entities with optimized matching
        for entity_type, entities in self.corpus.get("entities", {}).items():
            if isinstance(entities, dict):
                for name, data in entities.items():
                    name_lower = name.lower() if isinstance(name, str) else ""
                    content_str = str(data).lower()

                    if any(w in name_lower for w in query_words) or any(w in content_str for w in query_words):
                        if entity_type not in results["entities"]:
                            results["entities"][entity_type] = []

                        if isinstance(data, dict):
                            results["entities"][entity_type].append({**data, "name": name})
                        else:
                            results["entities"][entity_type].append({
                                "value": data,
                                "name": name
                            })

        # Search battle patterns and analysis
        self._search_patterns(query_words, results)

        # Search ontology with optimized nested search
        self._search_ontology(query_words, results)

        # Search micro-decisions
        self._search_micro_decisions(query_words, results)

        # Sort by relevance
        results["topics"].sort(key=lambda x: x.get("score", 0), reverse=True)

        # Cache results
        if use_cache:
            self._search_cache[query] = results.copy()
            # Limit cache size
            if len(self._search_cache) > 1000:
                # Remove oldest entry
                oldest = next(iter(self._search_cache))
                del self._search_cache[oldest]

        return results

    def _calculate_topic_score(self, data: Dict, query_words: set, topic_key: str) -> int:
        """
        Calculate relevance score for a topic.

        Args:
            data: Topic data
            query_words: Set of query words
            topic_key: Topic identifier

        Returns:
            Relevance score
        """
        score = 0
        title = data.get("title", topic_key).lower()
        summary = data.get("summary", "").lower()
        description = data.get("description", "").lower()
        date_range = data.get("date_range", "").lower()

        # Scoring weights (optimized for relevance)
        if any(w in title for w in query_words):
            score += 15  # Title match is most relevant
        if any(w in summary for w in query_words):
            score += 10  # Summary match is important
        if any(w in description for w in query_words):
            score += 8  # Description match
        if any(w in date_range for w in query_words):
            score += 5  # Date match

        # Check keywords
        keywords_flat = []
        if "keywords" in data:
            keywords_flat.extend([k.lower() for k in data["keywords"]])
        if "strategic_impact" in data:
            keywords_flat.extend([s.lower() for s in data["strategic_impact"]])

        if any(any(w in kw for w in query_words) for kw in keywords_flat):
            score += 7  # Keyword match

        return score

    def _search_patterns(self, query_words: set, results: Dict):
        """Search battle patterns and advanced analysis."""
        for pattern_list, key in [
            (self.corpus.get("battle_patterns", {}), "battle_patterns"),
            (self.corpus.get("advanced_analysis", {}), "advanced_analysis")
        ]:
            if isinstance(pattern_list, dict):
                for pattern_name, pattern_data in pattern_list.items():
                    pattern_name_lower = str(pattern_name).lower()
                    pattern_data_lower = str(pattern_data).lower()

                    if (any(w in pattern_name_lower for w in query_words) or
                            any(w in pattern_data_lower for w in query_words)):
                        results[key].append({
                            "name": pattern_name,
                            "data": pattern_data
                        })

    def _search_ontology(self, query_words: set, results: Dict):
        """Search ontology with optimized nested search."""
        ontology_data = self.corpus.get("ontology", {})
        if "turkish_military_ontology" not in ontology_data:
            return

        ontology = ontology_data["turkish_military_ontology"]

        def search_nested_ontology(obj, path="", depth=0):
            """Recursively search nested ontology with depth limit."""
            if depth > 10:  # Prevent infinite recursion
                return []

            found = []
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_path = f"{path}.{key}" if path else key
                    key_lower = str(key).lower()

                    if any(w in key_lower for w in query_words):
                        found.append({
                            "path": current_path,
                            "key": key,
                            "value": value
                        })

                    if isinstance(value, str):
                        value_lower = value.lower()
                        if any(w in value_lower for w in query_words):
                            found.append({
                                "path": current_path,
                                "key": key,
                                "value": value
                            })
                    elif isinstance(value, (dict, list)):
                        found.extend(
                            search_nested_ontology(value, current_path, depth + 1)
                        )

            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    found.extend(
                        search_nested_ontology(item, f"{path}[{i}]", depth + 1)
                    )

            return found

        for section_name, section_data in ontology.items():
            if section_name == "metadata":
                continue
            search_results = search_nested_ontology(section_data, section_name)
            if search_results:
                results["ontology"].extend(search_results[:5])

    def _search_micro_decisions(self, query_words: set, results: Dict):
        """Search micro-decisions database."""
        micro_data = self.corpus.get("micro_decisions", {})
        if "micro_decisions" not in micro_data:
            return

        for md in micro_data.get("micro_decisions", []):
            situation = str(md.get("situation", "")).lower()
            decision = str(md.get("decision", "")).lower()

            if (any(w in situation for w in query_words) or
                    any(w in decision for w in query_words)):
                results["micro_decisions"].append(md)
    
    def get_context_for_query(self, query: str) -> str:
        """
        Query için uzun vadeli hafızadan ilgili context döndür.
        """
        results = self.search(query)
        
        context_parts = []
        
        if results["topics"]:
            context_parts.append("=== İLGİLİ KONULAR ===")
            for topic_data in results["topics"][:3]:
                topic = topic_data.get("topic", "")
                content = topic_data.get("content", "")[:500]
                context_parts.append(f"\n[{topic}]\n{content}\n")
        
        if results["entities"]:
            context_parts.append("\n=== İLGİLİ VARLIKLAR ===")
            for entity_type, entities in results["entities"].items():
                context_parts.append(f"\n[{entity_type.upper()}]")
                for entity in entities[:3]:
                    name = entity.get("name", "")
                    info = {k: v for k, v in entity.items() if k != "name"}
                    if info:
                        info_str = str(info)[:300]
                        context_parts.append(f"  - {name}: {info_str}")
                    else:
                        context_parts.append(f"  - {name}")
        
        if results["battle_patterns"]:
            context_parts.append("\n=== SAVAŞ DOKTRİNLERİ ===")
            for pattern in results["battle_patterns"][:3]:
                name = pattern.get("name", "")
                data = pattern.get("data", "")
                context_parts.append(f"  - {name}: {str(data)[:200]}")
        
        if results["advanced_analysis"]:
            context_parts.append("\n=== İLERİ DÜZEY ANALİZ ===")
            for analysis in results["advanced_analysis"][:2]:
                name = analysis.get("name", "")
                data = analysis.get("data", "")
                context_parts.append(f"  - {name}: {str(data)[:200]}")
        
        if results["ontology"]:
            context_parts.append("\n=== ASKERİ ONTOLOJİ ===")
            for ont in results["ontology"][:5]:
                path = ont.get("path", "")
                key = ont.get("key", "")
                value = ont.get("value", "")
                value_str = str(value)[:200] if value else ""
                context_parts.append(f"  • {path}: {value_str}")
        
        if results["micro_decisions"]:
            context_parts.append("\n=== MİKRO KARARLAR ===")
            for md in results["micro_decisions"][:3]:
                situation = md.get("situation", "")
                decision = md.get("decision", "")
                reasoning = md.get("reasoning", "")
                context_parts.append(f"  • Durum: {situation[:100]}")
                context_parts.append(f"    Karar: {decision}")
                context_parts.append(f"    Gerekçe: {reasoning[:150]}")
        
        if not context_parts:
            return "Uzun vadeli hafızada bu konuyla ilgili bilgi bulunamadı."
        
        return "\n".join(context_parts)
    
    def _update_metadata(self):
        """Metadata güncelle."""
        self.corpus["metadata"]["last_updated"] = datetime.now().isoformat()
        if "version" not in self.corpus["metadata"]:
            self.corpus["metadata"]["version"] = "1.0"
    
    def get_all_topics(self) -> List[str]:
        """Tüm konuları listele."""
        return list(self.corpus.get("topics", {}).keys())
    
    def get_all_entities(self, entity_type: str = None) -> Dict[str, Any]:
        """Tüm varlıkları veya belirli tipteki varlıkları getir."""
        if entity_type:
            return self.corpus.get("entities", {}).get(entity_type, {})
        return self.corpus.get("entities", {})


_long_term_memory = None

def get_long_term_memory() -> LongTermMemory:
    """LongTermMemory singleton."""
    global _long_term_memory
    if _long_term_memory is None:
        _long_term_memory = LongTermMemory()
    return _long_term_memory


class QueryClassifier:
    """
    Analyzes user queries and determines their type.
    Optimized for accurate classification with improved keyword matching.
    """

    QUESTION_TYPES = {
        "factual": {
            "keywords": [
                "nerede", "ne zaman", "kim", "kaç", "hangisi", "nedir",
                "var mı", "bulundu", "savaştı", "katıldı", "geçmiş",
                "tarih", "olay", "yer", "tarihç"
            ],
            "patterns": ["nerede", "ne zaman", "kim", "kaç", "nedir", "hangi"],
            "weight": 1.0
        },
        "analytical": {
            "keywords": [
                "ne yapılmalı", "nasıl", "hangi strateji", "hangi taktik",
                "öneri", "değerlendir", "analiz et", "incelemek", "araştır",
                "mekanizm", "işlem", "süreç", "açıkla"
            ],
            "patterns": ["nasıl", "ne yapılmalı", "strateji", "taktik"],
            "weight": 1.2
        },
        "causal": {
            "keywords": [
                "neden", "niçin", "bu yüzden", "sonucu", "sebebi",
                "gerekçe", "nedeni", "etken", "faktör", "sebep"
            ],
            "patterns": ["neden", "niçin", "sebep"],
            "weight": 1.1
        },
        "counterfactual": {
            "keywords": [
                "eğer", "olsaydı", "söyleydi", "şöyle olsaydı",
                "ne olurdu", "ya da", "başka", "öteki"
            ],
            "patterns": ["olsaydı", "eğer", "ne olurdu"],
            "weight": 1.3
        },
        "comparative": {
            "keywords": [
                "fark", "karşılaştır", "hangisi daha", "avantaj",
                "dezavantaj", "üstünlük", "benzer", "farklı", "kıyasla"
            ],
            "patterns": ["fark", "karşılaştır", "hangisi daha", "benzer"],
            "weight": 1.15
        },
        "historical": {
            "keywords": [
                "listele", "sırala", "numarala", "detaylı", "tüm",
                "hepsini", "dönem", "era", "çağ", "yüzyıl"
            ],
            "patterns": ["listele", "sırala", "tüm"],
            "weight": 1.0
        }
    }

    @classmethod
    def classify(cls, query: str) -> Dict[str, Any]:
        """
        Classify query type and return detailed classification.

        Args:
            query: User query string

        Returns:
            Dictionary with type, confidence, scores, and sub_types
        """
        query_lower = query.lower()

        scores = {}
        for qtype, config in cls.QUESTION_TYPES.items():
            score = 0

            # Check keywords (exact substring matching for accuracy)
            for keyword in config["keywords"]:
                if keyword in query_lower:
                    score += 2

            # Check patterns (optimized pattern matching)
            for pattern in config["patterns"]:
                if pattern in query_lower:
                    score += 3

            # Apply type weight
            if score > 0:
                scores[qtype] = score * config["weight"]

        # Default fallback
        if not scores:
            return {
                "type": "factual",
                "confidence": 0.5,
                "scores": {},
                "sub_types": []
            }

        # Find primary type
        primary_type = max(scores, key=scores.get)
        total_score = sum(scores.values())
        confidence = min(scores[primary_type] / total_score, 1.0) if total_score > 0 else 0.5

        # Detect sub-types
        sub_types = cls._extract_sub_types(query_lower, primary_type)

        return {
            "type": primary_type,
            "confidence": confidence,
            "scores": scores,
            "sub_types": sub_types
        }

    @classmethod
    def _extract_sub_types(cls, query_lower: str, primary_type: str) -> List[str]:
        """
        Extract sub-types from query.

        Args:
            query_lower: Lowercase query
            primary_type: Primary question type

        Returns:
            List of sub-types
        """
        sub_types = []

        # Unit-specific sub-type detection
        unit_keywords = ["tümen", "alay", "kolordu", "birlik", "piyade", "süvari", "topçu"]
        if any(k in query_lower for k in unit_keywords):
            sub_types.append("unit_specific")

        # Operational sub-type detection
        operational_keywords = ["savaştı", "katıldı", "yer aldı", "çatışma", "muharebe", "harekat"]
        if any(k in query_lower for k in operational_keywords):
            sub_types.append("operational")

        # Strategic sub-type detection
        strategic_keywords = ["strateji", "komuta", "komutan", "doktrin", "doktrinu", "stratejik"]
        if any(k in query_lower for k in strategic_keywords):
            sub_types.append("strategic")

        # Tactical sub-type detection
        tactical_keywords = ["taktik", "taktikal", "maneuvra", "manevra", "hareketi", "harekete"]
        if any(k in query_lower for k in tactical_keywords):
            sub_types.append("tactical")

        return sub_types


class DecisionEngine:
    """
    Analyzes military situations and generates decision recommendations.
    Multi-layered decision system with doctrine-based rules.
    """

    DECISION_RULES = {
        "savunma_doktrini": {
            "condition": {
                "enemy_strength": ["superior", "high"],
                "ammo_level": ["low", "critical"]
            },
            "decision": "防守 (Savunma)",
            "priority": 1,
            "options": [
                {
                    "type": "mevzi_savunması",
                    "description": "Önceden belirlenmiş mevzilerde savunma",
                    "use_if": ["terrain_favorable", "time_needed"]
                },
                {
                    "type": "hareketli_savunma",
                    "description": "Hareketi koruma altında gerçekleştiren savunma",
                    "use_if": ["terrain_open", "mobility_available"]
                },
                {
                    "type": "geciktirme",
                    "description": "Düşmanı geciktirme yoluyla zaman kazanma",
                    "use_if": ["delaying_acceptable", "buy_time"]
                }
            ],
            "reasoning_template": [
                "Düşman kuvvetleri üstün ({enemy_strength})",
                "Mühimmat durumu: {ammo_level}",
                "Bu koşullarda en uygun seçenek: {decision}"
            ]
        },
        "taarruz_doktrini": {
            "condition": {
                "enemy_strength": ["inferior", "low"],
                "ammo_level": ["adequate", "high"],
                "surprise_possible": [True]
            },
            "decision": "Taarruz",
            "priority": 1,
            "options": [
                {
                    "type": "cephe_taarruzu",
                    "description": "Düşman cephesine doğrudan saldırı",
                    "use_if": ["flanks_vulnerable"]
                },
                {
                    "type": "kuşatma",
                    "description": "Düşmanı dört taraftan sarmalama",
                    "use_if": ["enemy_encircled", "reinforcements_far"]
                },
                {
                    "type": "yarma_harekati",
                    "description": "Zayıf noktadan sızma",
                    "use_if": ["weak_point_identified"]
                }
            ],
            "reasoning_template": [
                "Düşman zayıf: {enemy_strength}",
                "Mühimmat yeterli: {ammo_level}",
                "Sürpriz mümkün: {surprise_possible}",
                "Önerilen: {decision}"
            ]
        },
        "geri_cekilme": {
            "condition": {
                "losses": ["high", "critical"],
                "reinforcements": ["unavailable", "delayed"],
                "position": ["unsustainable"]
            },
            "decision": "Geri Çekilme",
            "priority": 2,
            "options": [
                {
                    "type": "düzenli_geri",
                    "description": "Organize bir şekilde geri çekilme",
                    "use_if": ["time_available"]
                },
                {
                    "type": "acil_geri",
                    "description": "Acil geri çekilme",
                    "use_if": ["imminent_threat"]
                },
                {
                    "type": "sahte_geri",
                    "description": "Düşmanı çekmeye çalışan sahte geri çekilme",
                    "use_if": ["trap_setup_possible"]
                }
            ],
            "reasoning_template": [
                "Kayıplar yüksek: {losses}",
                "Takviye durumu: {reinforcements}",
                "Pozisyon sürdürülemez: {position}",
                "Karar: {decision}"
            ]
        }
    }

    @classmethod
    def analyze(
        cls,
        query: str,
        query_type: str,
        context_data: List[Dict] = None
    ) -> Dict[str, Any]:
        """
        Analyze situation and generate decision recommendations.

        Args:
            query: User query
            query_type: Query classification type
            context_data: Optional context from retrieved documents

        Returns:
            Analysis results with decisions and reasoning
        """
        requires_decision = query_type in [
            "analytical",
            "causal",
            "counterfactual"
        ]

        result = {
            "requires_decision": requires_decision,
            "query_type": query_type,
            "analysis": {},
            "decisions": [],
            "reasoning": []
        }

        if not requires_decision:
            return result

        query_lower = query.lower()

        # Extract context from documents
        entity_context = cls._extract_context(context_data, query_lower)
        result["analysis"]["entity_context"] = entity_context
        result["analysis"]["query_focus"] = cls._extract_focus(query_lower)

        # Apply decision rules
        for rule_name, rule in cls.DECISION_RULES.items():
            if cls._check_rule_conditions(rule, entity_context):
                decision_result = {
                    "rule_name": rule_name,
                    "decision": rule["decision"],
                    "priority": rule.get("priority", 0),
                    "options": rule["options"],
                    "reasoning_template": rule["reasoning_template"]
                }
                result["decisions"].append(decision_result)

                # Generate reasoning for this rule
                reasoning_lines = cls._generate_reasoning_lines(
                    rule,
                    entity_context
                )
                result["reasoning"].extend(reasoning_lines)

        # Generate context-aware decision options
        if any(
            word in query_lower
            for word in ["ne yapılmalı", "öneri", "tavsiye", "strateji"]
        ):
            decisions = cls._generate_decision_options(query_lower, entity_context)
            result["decisions"].extend(decisions)
            result["reasoning"].extend(
                cls._build_reasoning(decisions, entity_context)
            )

        # Add causal analysis if needed
        if "neden" in query_lower or "sebep" in query_lower:
            causal_reasoning = cls._analyze_causal_factors(entity_context)
            result["reasoning"].extend(causal_reasoning)

        return result

    @classmethod
    def _extract_context(
        cls,
        context_data: Optional[List[Dict]],
        query_lower: str
    ) -> Dict[str, Any]:
        """
        Extract military context from document data.

        Args:
            context_data: Retrieved document data
            query_lower: Lowercase query

        Returns:
            Context dictionary with extracted information
        """
        entity_context = {}

        if not context_data:
            return entity_context

        # Scan first 10 documents for context clues
        for item in context_data[:10]:
            text = item.get("Metin", "").lower()
            units = item.get("Birlikler", "").lower()

            # Movement detection
            if any(w in text for w in ["geri", "çekil", "geriç"]):
                entity_context["recent_movement"] = "geri_cekilme"

            # Action detection
            if any(w in text for w in ["taarruz", "saldır", "hücum"]):
                entity_context["recent_action"] = "taarruz"
            elif any(w in text for w in ["savun", "mevzi", "savın"]):
                entity_context["recent_action"] = "savunma"

            # Unit strength detection
            if any(w in text for w in ["kayıp", "zayiat", "hasar"]):
                entity_context["losses"] = "high"

        return entity_context

    @classmethod
    def _check_rule_conditions(cls, rule: Dict, context: Dict) -> bool:
        """
        Check if all rule conditions are met.

        Args:
            rule: Decision rule
            context: Entity context

        Returns:
            True if all conditions are met
        """
        for key, expected_values in rule.get("condition", {}).items():
            actual_value = context.get(key)
            if actual_value not in expected_values:
                return False
        return True

    @classmethod
    def _generate_reasoning_lines(cls, rule: Dict, context: Dict) -> List[str]:
        """
        Generate reasoning lines from rule template.

        Args:
            rule: Decision rule
            context: Entity context

        Returns:
            List of reasoning lines
        """
        reasoning = []
        for template in rule.get("reasoning_template", []):
            reasoning_line = template
            for key, expected_values in rule.get("condition", {}).items():
                actual_value = context.get(key, "bilinmiyor")
                reasoning_line = reasoning_line.replace(
                    f"{{{key}}}",
                    str(actual_value)
                )
            reasoning_line = reasoning_line.replace(
                "{decision}",
                rule.get("decision", "")
            )
            reasoning.append(f"• {reasoning_line}")
        return reasoning

    @classmethod
    def _extract_focus(cls, query_lower: str) -> str:
        """
        Extract query focus area.

        Args:
            query_lower: Lowercase query

        Returns:
            Focus area string
        """
        unit_keywords = [
            "tümen", "alay", "kolordu", "birlik",
            "piyade", "süvari", "topçu"
        ]
        if any(w in query_lower for w in unit_keywords):
            return "unit_tactical"

        operational_keywords = [
            "cephe", "savaş", "muharebe", "çatışma",
            "harekat", "operasyon"
        ]
        if any(w in query_lower for w in operational_keywords):
            return "operational"

        strategic_keywords = [
            "komutan", "karar", "strateji", "doktrin",
            "stratejik", "komuta"
        ]
        if any(w in query_lower for w in strategic_keywords):
            return "strategic"

        return "general"

    @classmethod
    def _generate_decision_options(cls, query_lower: str, context: Dict) -> List[Dict]:
        """
        Generate context-aware decision options.

        Args:
            query_lower: Lowercase query
            context: Entity context

        Returns:
            List of decision options
        """
        options = []

        if "savun" in query_lower or "savunma" in query_lower:
            options.append({
                "type": "防守 (Savunma)",
                "sub_options": [
                    "Mevzi savunması",
                    "Hareketli savunma",
                    "Geciktirme harekatı"
                ],
                "doctrine_ref": "Alan savunması doktrini",
                "priority": 1
            })

        if "taarruz" in query_lower or "saldır" in query_lower:
            options.append({
                "type": "Taarruz",
                "sub_options": [
                    "Cephe taarruzu",
                    "Kuşatma",
                    "Yarma harekatı"
                ],
                "doctrine_ref": "Taarruz doktrini",
                "priority": 1
            })

        if not options:
            options.append({
                "type": "Durum Değerlendirmesi",
                "sub_options": [
                    "Ek istihbarat gerekli",
                    "Karar için veri yetersiz"
                ],
                "doctrine_ref": "Kara Kuvvetleri Talimatname",
                "priority": 2
            })

        return options

    @classmethod
    def _build_reasoning(cls, decisions: List[Dict], context: Dict) -> List[str]:
        """
        Build detailed reasoning for decisions.

        Args:
            decisions: List of decisions
            context: Entity context

        Returns:
            List of reasoning statements
        """
        reasoning = []

        if context.get("recent_movement") == "geri_cekilme":
            reasoning.append(
                "• Son dönemde geri çekilme hareketleri tespit edilmiş - "
                "bu durumda savunma öncelikli olmalı"
            )

        if context.get("recent_action") == "taarruz":
            reasoning.append(
                "• Son harekette taarruz eylemi görülmüş - "
                "momentum değerlendirilmeli"
            )

        for decision in decisions:
            decision_type = decision.get("type", "")

            if decision_type == "防守 (Savunma)":
                reasoning.append(
                    "• Savunma doktrinine göre: "
                    "Mevzi savunması veya hareketli savunma tercih edilmeli"
                )
                reasoning.append(
                    "• Amaç: Düşmanı yıpratma ve zaman kazanma"
                )

            elif decision_type == "Taarruz":
                reasoning.append(
                    "• Taarruz doktrinine göre: "
                    "Zayıf noktaya yoğunlaşma ve süratli takip"
                )
                reasoning.append(
                    "• Amaç: Düşmanı imha etmek veya saf dışı bırakmak"
                )

        return reasoning

    @classmethod
    def _analyze_causal_factors(cls, context: Dict) -> List[str]:
        """
        Analyze and list causal factors for the situation.

        Args:
            context: Entity context

        Returns:
            List of causal analysis statements
        """
        reasoning = []

        if context.get("recent_movement"):
            reasoning.append(
                f"• Yakın hareketi etkileme sebebi: {context['recent_movement']}"
            )

        if context.get("losses") == "high":
            reasoning.append(
                "• Yüksek kayıplar stratejik ve taktik karar almayı zorunlu kılmakta"
            )

        if not reasoning:
            reasoning.append(
                "• Olası nedenler: Düşman hareketi, coğrafi faktörler, "
                "iklim koşulları veya istihbarat verisi analiz edilmeli"
            )

        return reasoning
