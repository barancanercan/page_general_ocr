import logging
import json
import os
import threading
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any
from collections import deque

logger = logging.getLogger(__name__)

CORPUS_FILE = "data/memory/military_corpus.json"
ONTOLOGY_FILE = "data/memory/military_ontology.json"
DECISIONS_FILE = "data/memory/military_decisions.json"
MICRO_DECISIONS_FILE = "data/memory/micro_decisions.json"


class ConversationMemory:
    
    def __init__(self, summary_interval: int = 5):
        self.summary_interval = summary_interval
        self.messages = deque(maxlen=summary_interval * 2)
        self.summary = ""
        self.session_id = None
    
    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
    
    def should_summarize(self) -> bool:
        return len(self.messages) >= self.summary_interval
    
    def get_recent_context(self) -> str:
        if not self.messages:
            return ""
        
        context_parts = []
        if self.summary:
            context_parts.append(f"[Önceki Konuşma Özeti]: {self.summary}")
        
        recent = list(self.messages)[-self.summary_interval:]
        for msg in recent:
            role = "Kullanıcı" if msg["role"] == "user" else "Asistan"
            content = msg["content"][:200] + "..." if len(msg["content"]) > 200 else msg["content"]
            context_parts.append(f"{role}: {content}")
        
        return "\n".join(context_parts)
    
    def update_summary(self, summary: str):
        self.summary = summary
        logger.info(f"Conversation summary updated: {summary[:100]}...")
    
    def clear(self):
        self.messages.clear()
        self.summary = ""


class GlobalMemory:
    _instance = None
    _memory: Dict[str, ConversationMemory] = {}
    _lock = threading.Lock()
    
    @classmethod
    def get_memory(cls, session_id: str = "default") -> ConversationMemory:
        with cls._lock:
            if session_id not in cls._memory:
                cls._memory[session_id] = ConversationMemory(summary_interval=5)
            return cls._memory[session_id]
    
    @classmethod
    def clear_session(cls, session_id: str = "default"):
        if session_id in cls._memory:
            cls._memory[session_id].clear()
    
    @classmethod
    def clear_all(cls):
        cls._memory.clear()


class LongTermMemory:
    """
    Uzun vadeli hafıza - Askeri ve tarihi corpus.
    Bu corpus, daha önce konuşulan konuları ve genel askeri tarih bilgilerini saklar.
    """
    
    def __init__(self, corpus_path: str = CORPUS_FILE):
        self.corpus_path = corpus_path
        self.corpus: Dict[str, Any] = {"topics": {}, "entities": {}, "history": []}
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
    
    def search(self, query: str) -> Dict[str, Any]:
        """
        Query ile ilgili bilgileri bul.
        
        Args:
            query: Arama sorgusu
            
        Returns:
            Dict containing relevant topics, entities, and history
        """
        query_lower = query.lower()
        query_words = set(query_lower.replace('?', ' ').replace('.', ' ').replace(',', ' ').split())
        
        results = {
            "topics": [],
            "entities": {},
            "battle_patterns": [],
            "timeline": [],
            "advanced_analysis": [],
            "ontology": [],
            "micro_decisions": []
        }
        
        for topic_key, data in self.corpus.get("topics", {}).items():
            title = data.get("title", topic_key).lower()
            summary = data.get("summary", "").lower()
            description = data.get("description", "").lower()
            date_range = data.get("date_range", "").lower()
            
            keywords_flat = []
            if "keywords" in data:
                keywords_flat.extend([k.lower() for k in data["keywords"]])
            if "strategic_impact" in data:
                keywords_flat.extend([s.lower() for s in data["strategic_impact"]])
            
            score = 0
            if any(w in title for w in query_words):
                score += 15
            if any(w in summary for w in query_words):
                score += 10
            if any(w in description for w in query_words):
                score += 8
            if any(w in date_range for w in query_words):
                score += 5
            if any(any(w in kw for w in query_words) for kw in keywords_flat):
                score += 7
            
            if score > 0:
                results["topics"].append({**data, "topic_key": topic_key, "score": score})
        
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
                            results["entities"][entity_type].append({"value": data, "name": name})
        
        for pattern_list, key in [(self.corpus.get("battle_patterns", []), "battle_patterns"), 
                                    (self.corpus.get("advanced_analysis", {}), "advanced_analysis")]:
            if key == "battle_patterns" and isinstance(pattern_list, dict):
                for pattern_name, pattern_data in pattern_list.items():
                    if any(w in str(pattern_name).lower() for w in query_words) or any(w in str(pattern_data).lower() for w in query_words):
                        results["battle_patterns"].append({"name": pattern_name, "data": pattern_data})
            elif key == "advanced_analysis" and isinstance(pattern_list, dict):
                for analysis_name, analysis_data in pattern_list.items():
                    if any(w in str(analysis_name).lower() for w in query_words) or any(w in str(analysis_data).lower() for w in query_words):
                        results["advanced_analysis"].append({"name": analysis_name, "data": analysis_data})
        
        ontology_data = self.corpus.get("ontology", {})
        if "turkish_military_ontology" in ontology_data:
            ontology = ontology_data["turkish_military_ontology"]
            
            def search_nested_ontology(obj, path=""):
                found = []
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        current_path = f"{path}.{key}" if path else key
                        key_lower = str(key).lower()
                        if any(w in key_lower for w in query_words):
                            found.append({"path": current_path, "key": key, "value": value})
                        if isinstance(value, str):
                            value_lower = value.lower()
                            if any(w in value_lower for w in query_words):
                                found.append({"path": current_path, "key": key, "value": value})
                        elif isinstance(value, (dict, list)):
                            found.extend(search_nested_ontology(value, current_path))
                elif isinstance(obj, list):
                    for i, item in enumerate(obj):
                        found.extend(search_nested_ontology(item, f"{path}[{i}]"))
                return found
            
            for section_name, section_data in ontology.items():
                if section_name == "metadata":
                    continue
                search_results = search_nested_ontology(section_data, section_name)
                if search_results:
                    results["ontology"].extend(search_results[:5])
        
        micro_data = self.corpus.get("micro_decisions", {})
        if "micro_decisions" in micro_data:
            query_words = query_lower.split()
            for md in micro_data.get("micro_decisions", []):
                situation = str(md.get("situation", "")).lower()
                decision = str(md.get("decision", "")).lower()
                if any(w in situation for w in query_words) or any(w in decision for w in query_words):
                    results["micro_decisions"].append(md)
        
        results["topics"].sort(key=lambda x: x.get("score", 0), reverse=True)
        
        return results
    
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
    Kullanıcı sorgusunu analiz eder ve tipini belirler.
    """
    
    QUESTION_TYPES = {
        "factual": {
            "keywords": ["nerede", "ne zaman", "kim", "kaç", "hangisi", "nedir", "var mı", "bulundu", "savaştı", "katıldı"],
            "patterns": ["* nerede", "* ne zaman", "* kim", "* kaç", "* nedir"]
        },
        "analytical": {
            "keywords": ["ne yapılmalı", "nasıl", "hangi strateji", "hangi taktik", "öneri", "değerlendir", "analiz et"],
            "patterns": ["* ne yapılmalı", "* nasıl", "* hangi strateji"]
        },
        "causal": {
            "keywords": ["neden", "niçin", "bu yüzden", "sonucu", "sebebi", "gerekçe"],
            "patterns": ["* neden", "* niçin"]
        },
        "counterfactual": {
            "keywords": ["eğer", "olsaydı", "söyleydi", "şöyle olsaydı", "ne olurdu"],
            "patterns": ["* olsaydı", "* söyleydi"]
        },
        "comparative": {
            "keywords": ["fark", "karşılaştır", "hangisi daha", "avantaj", "dezavantaj", "üstünlük"],
            "patterns": ["* fark", "* karşılaştır", "hangisi daha"]
        },
        "listorical": {
            "keywords": ["listele", "sırala", "numarala", "detaylı", "tüm", "hepsini"],
            "patterns": ["* listele", "* sırala"]
        }
    }
    
    @classmethod
    def classify(cls, query: str) -> Dict[str, Any]:
        """
        Sorgunun tipini ve özelliklerini döndürür.
        """
        query_lower = query.lower()
        
        scores = {}
        for qtype, config in cls.QUESTION_TYPES.items():
            score = 0
            
            for keyword in config["keywords"]:
                if keyword in query_lower:
                    score += 2
            
            for pattern in config["patterns"]:
                if pattern.replace("*", "") in query_lower:
                    score += 3
            
            if score > 0:
                scores[qtype] = score
        
        if not scores:
            return {"type": "factual", "confidence": 0.5, "sub_types": []}
        
        primary_type = max(scores, key=scores.get)
        confidence = scores[primary_type] / sum(scores.values())
        
        sub_types = []
        if primary_type == "factual":
            if any(k in query_lower for k in ["tümen", "alay", "kolordu", "birlik"]):
                sub_types.append("unit_specific")
            if any(k in query_lower for k in ["savaştı", "katıldı", "yer aldı"]):
                sub_types.append("operational")
        
        return {
            "type": primary_type,
            "confidence": min(confidence, 1.0),
            "scores": scores,
            "sub_types": sub_types
        }


class DecisionEngine:
    """
    Askeri durumları analiz eder ve karar önerileri üretir.
    Decision Engine - Çok Katmanlı Düşünme Sistemi
    """
    
    DECISION_RULES = {
        "savunma_doktrini": {
            "condition": {
                "enemy_strength": ["superior", "high"],
                "ammo_level": ["low", "critical"]
            },
            "decision": "防守 (Savunma)",
            "options": [
                {"type": "mevzi_savunması", "use_if": ["terrain_favorable", "time_needed"]},
                {"type": "hareketli_savunma", "use_if": ["terrain_open", "mobility_available"]},
                {"type": "geciktirme", "use_if": ["delaying_acceptable", "buy_time"]}
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
            "options": [
                {"type": "cephe_taarruzu", "use_if": ["flanks_vulnerable"]},
                {"type": "kuşatma", "use_if": ["enemy_encircled", "reinforcements_far"]},
                {"type": "yarma_harekati", "use_if": ["weak_point_identified"]}
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
            "options": [
                {"type": "düzenli_geri", "use_if": ["time_available"]},
                {"type": "acil_geri", "use_if": ["imminent_threat"]},
                {"type": "sahte_geri", "use_if": ["trap_setup_possible"]}
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
    def analyze(cls, query: str, query_type: str, context_data: List[Dict] = None) -> Dict[str, Any]:
        """
        Durumu analiz eder ve karar önerileri üretir.
        """
        result = {
            "requires_decision": query_type in ["analytical", "causal", "counterfactual"],
            "query_type": query_type,
            "analysis": {},
            "decisions": [],
            "reasoning": []
        }
        
        if not result["requires_decision"]:
            return result
        
        query_lower = query.lower()
        
        entity_context = {}
        if context_data:
            for item in context_data[:10]:
                text = item.get("Metin", "").lower()
                units = item.get("Birlikler", "").lower()
                
                if "geri" in text or "çekil" in text:
                    entity_context["recent_movement"] = "geri_cekilme"
                if "taarruz" in text or "saldır" in text:
                    entity_context["recent_action"] = "taarruz"
                if "savun" in text or "mevzi" in text:
                    entity_context["recent_action"] = "savunma"
        
        result["analysis"]["entity_context"] = entity_context
        result["analysis"]["query_focus"] = cls._extract_focus(query_lower)
        
        for rule_name, rule in cls.DECISION_RULES.items():
            conditions_met = True
            for key, expected_values in rule["condition"].items():
                actual_value = entity_context.get(key)
                if actual_value not in expected_values:
                    conditions_met = False
                    break
            
            if conditions_met:
                result["decisions"].append({
                    "rule_name": rule_name,
                    "decision": rule["decision"],
                    "options": rule["options"],
                    "reasoning_template": rule["reasoning_template"]
                })
                
                for template in rule["reasoning_template"]:
                    reasoning_line = template
                    for key, expected_values in rule["condition"].items():
                        actual_value = entity_context.get(key, "bilinmiyor")
                        reasoning_line = reasoning_line.replace(f"{{{key}}}", str(actual_value))
                    reasoning_line = reasoning_line.replace("{decision}", rule["decision"])
                    result["reasoning"].append(f"• {reasoning_line}")
        
        if any(word in query_lower for word in ["ne yapılmalı", "öneri", "tavsiye", "strateji"]):
            decisions = cls._generate_decision_options(query_lower, entity_context)
            result["decisions"].extend(decisions)
            result["reasoning"].extend(cls._build_reasoning(decisions, entity_context))
        
        if "neden" in query_lower or "sebep" in query_lower:
            result["reasoning"].append("Olası nedenler analiz edilmeli.")
        
        return result
    
    @classmethod
    def _extract_focus(cls, query_lower: str) -> str:
        """Sorgunun odak noktasını çıkarır."""
        if any(w in query_lower for w in ["tümen", "alay", "kolordu", "birlik"]):
            return "unit_tactical"
        if any(w in query_lower for w in ["cephe", "savaş", "muharebe"]):
            return "operational"
        if any(w in query_lower for w in ["komutan", "karar", "strateji"]):
            return "strategic"
        return "general"
    
    @classmethod
    def _generate_decision_options(cls, query_lower: str, context: Dict) -> List[Dict]:
        """Duruma göre karar seçenekleri üretir."""
        options = []
        
        if "savun" in query_lower or "savunma" in query_lower:
            options.append({
                "type": "防守 (Savunma)",
                "sub_options": ["Mevzi savunması", "Hareketli savunma", "Geciktirme harekatı"],
                "doctrine_ref": "Alan savunması doktrini"
            })
        
        if "taarruz" in query_lower or "saldır" in query_lower:
            options.append({
                "type": "Taarruz",
                "sub_options": ["Cephe taarruzu", "Kuşatma", "Yarma harekatı"],
                "doctrine_ref": "Taarruz doktrini"
            })
        
        if not options:
            options.append({
                "type": "Durum Değerlendirmesi",
                "sub_options": ["Ek istihbarat gerekli", "Karar için veri yetersiz"],
                "doctrine_ref": "Kara Kuvvetleri Talimatname"
            })
        
        return options
    
    @classmethod
    def _build_reasoning(cls, decisions: List[Dict], context: Dict) -> List[str]:
        """Karar için gerekçe oluşturur."""
        reasoning = []
        
        if context.get("recent_movement") == "geri_cekilme":
            reasoning.append("• Son dönemde geri çekilme hareketleri tespit edilmiş - bu durumda savunma öncelikli olmalı")
        
        if context.get("recent_action") == "taarruz":
            reasoning.append("• Son harekette taarruz eylemi görülmüş - momentum değerlendirilmeli")
        
        for decision in decisions:
            decision_type = decision.get("type", "")
            if decision_type == "防守 (Savunma)":
                reasoning.append("• Savunma doktrinine göre: Mevzi savunması veya hareketli savunma tercih edilmeli")
                reasoning.append("• Amaç: Düşmanı yıpratma ve zaman kazanma")
            elif decision_type == "Taarruz":
                reasoning.append("• Taarruz doktrinine göre: Zayıf noktaya yoğunlaşma ve süratli takip")
                reasoning.append("• Amaç: Düşmanı imha")
        
        return reasoning
