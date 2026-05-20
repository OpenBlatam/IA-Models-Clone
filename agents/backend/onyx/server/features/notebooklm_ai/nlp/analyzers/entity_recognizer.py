from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import asyncio
import re
import time
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, OrderedDict
import structlog
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Reconocedor de Entidades Ultra-Optimizado - NotebookLM AI
🧩 Reconocimiento avanzado de entidades para producción, multilingüe, extensible
"""


logger = structlog.get_logger()

# Cache LRU thread-safe
class LRUCache:
    def __init__(self, maxsize: int = 1000):
        
    """__init__ function."""
self.maxsize = maxsize
        self.cache = OrderedDict()
        self.lock = threading.Lock()
    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            if key in self.cache:
                value = self.cache.pop(key)
                self.cache[key] = value
                return value
            return None
    def put(self, key: str, value: Any):
        
    """put function."""
with self.lock:
            if key in self.cache:
                self.cache.pop(key)
            elif len(self.cache) >= self.maxsize:
                self.cache.popitem(last=False)
            self.cache[key] = value
    def clear(self) -> Any:
        with self.lock:
            self.cache.clear()

@dataclass
class EntityConfig:
    # Patrones y tipos
    entity_types: List[str] = field(default_factory=lambda: [
        "person", "organization", "location", "date", "email", "url", "phone", "money"])
    custom_patterns: Dict[str, str] = field(default_factory=dict)
    # ML y extensión
    use_ml: bool = False
    ml_model_path: str = ""
    # Cache y rendimiento
    enable_caching: bool = True
    cache_maxsize: int = 1000
    batch_size: int = 50
    max_workers: int = 4
    # Idiomas soportados
    supported_languages: List[str] = field(default_factory=lambda: ["es", "en", "fr", "de", "it", "pt"])
    default_language: str = "es"

class EntityRecognizer:
    def __init__(self, config: EntityConfig = None):
        
    """__init__ function."""
self.config = config or EntityConfig()
        self.stats = defaultdict(int)
        self.cache = LRUCache(self.config.cache_maxsize) if self.config.enable_caching else None
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        # Patrones regex multilingües
        self.patterns = self._default_patterns()
        self.patterns.update(self.config.custom_patterns)
    def _default_patterns(self) -> Dict[str, str]:
        return {
            "person": r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',
            "organization": r'\b[A-Z][a-z]+ (Inc|Corp|LLC|Ltd|S\.A\.|S\.L\.|Company|Organization)\b',
            "location": r'\b[A-Z][a-z]+ (Street|Avenue|Road|City|Country|State|Province)\b',
            "date": r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b|\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "url": r'https?://[^\s]+|www\.[^\s]+',
            "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b|\+\d{1,3}[-.]?\d{1,4}[-.]?\d{1,4}[-.]?\d{1,4}\b',
            "money": r'\$\d+(?:,\d{3})*(?:\.\d{2})?\b|\b\d+(?:,\d{3})*(?:\.\d{2})?\s*(dollars?|euros?|pesos?)\b'
        }
    def _generate_cache_key(self, text: str, language: str) -> str:
        content = f"{text}:{language}:{','.join(self.config.entity_types)}"
        return hashlib.md5(content.encode()).hexdigest()
    def _detect_language(self, text: str) -> str:
        # Simple detection (could be improved)
        if any(ord(c) > 128 for c in text):
            return "es"
        return "en"
    async def extract(self, text: str, language: str = "auto") -> Dict[str, Any]:
        start_time = time.time()
        try:
            if language == "auto":
                language = self._detect_language(text)
            if self.cache:
                cache_key = self._generate_cache_key(text, language)
                cached_result = self.cache.get(cache_key)
                if cached_result:
                    self.stats["cache_hits"] += 1
                    return cached_result
            entities = await self._extract_entities(text, language)
            duration = time.time() - start_time
            self.stats["total_requests"] += 1
            self.stats["total_processing_time"] += duration
            result = {
                "entities": entities,
                "entity_count": sum(len(v) for v in entities.values()),
                "language": language,
                "processing_time_ms": duration * 1000,
                "timestamp": time.time()
            }
            if self.cache:
                self.cache.put(self._generate_cache_key(text, language), result)
            return result
        except Exception as e:
            self.stats["errors"] += 1
            logger.error("Error extrayendo entidades", error=str(e), text=text[:100])
            raise
    async def _extract_entities(self, text: str, language: str) -> Dict[str, List[str]]:
        entities = defaultdict(list)
        for entity_type in self.config.entity_types:
            pattern = self.patterns.get(entity_type)
            if not pattern:
                continue
            matches = re.findall(pattern, text)
            if matches:
                # Flatten tuples if regex returns them
                if isinstance(matches[0], tuple):
                    matches = [m[0] if m else "" for m in matches]
                entities[entity_type].extend(matches)
        # ML-based NER (placeholder for future extension)
        if self.config.use_ml:
            # TODO: Integrate spaCy, transformers, etc.
            pass
        return dict(entities)
    async def batch_extract(self, texts: List[str], language: str = "auto") -> List[Dict[str, Any]]:
        tasks = [self.extract(text, language) for text in texts]
        return await asyncio.gather(*tasks, return_exceptions=True)
    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_requests": self.stats["total_requests"],
            "cache_hits": self.stats.get("cache_hits", 0),
            "errors": self.stats.get("errors", 0),
            "cache_size": len(self.cache.cache) if self.cache else 0,
            "cache_hit_rate": self.stats.get("cache_hits", 0) / max(self.stats["total_requests"], 1)
        }
    def clear_cache(self) -> Any:
        if self.cache:
            self.cache.clear()
    async def health_check(self) -> Dict[str, Any]:
        test_text = "John Doe works at Acme Corp. Contact: john@example.com, +1-555-123-4567, https://acme.com"
        try:
            result = await self.extract(test_text, "en")
            return {
                "status": "healthy",
                "test_result": {
                    "entity_count": result["entity_count"],
                    "processing_time_ms": result["processing_time_ms"]
                },
                "stats": self.get_stats()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "stats": self.get_stats()
            }
def get_entity_recognizer(config: EntityConfig = None) -> EntityRecognizer:
    return EntityRecognizer(config) 