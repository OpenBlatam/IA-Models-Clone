from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import asyncio
import time
import hashlib
import numpy as np
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from collections import defaultdict, OrderedDict
import structlog
import threading
from concurrent.futures import ThreadPoolExecutor
                from sentence_transformers import SentenceTransformer
        import hashlib
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Motor de Embeddings Avanzado - NotebookLM AI
🔗 Generación de embeddings vectoriales con múltiples modelos
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
class EmbeddingConfig:
    """Configuración del motor de embeddings."""
    # Modelo
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    model_type: str = "sentence-transformers"  # sentence-transformers, openai, custom
    
    # Configuración
    max_length: int = 512
    normalize: bool = True
    device: str = "cpu"  # cpu, cuda, mps
    
    # Cache y rendimiento
    enable_caching: bool = True
    cache_maxsize: int = 1000
    batch_size: int = 32
    max_workers: int = 4
    
    # Idiomas soportados
    supported_languages: List[str] = field(default_factory=lambda: ["es", "en", "fr", "de", "it", "pt"])
    default_language: str = "es"

class EmbeddingEngine:
    """Motor de embeddings avanzado."""
    
    def __init__(self, config: EmbeddingConfig = None):
        
    """__init__ function."""
self.config = config or EmbeddingConfig()
        self.stats = defaultdict(int)
        self.cache = LRUCache(self.config.cache_maxsize) if self.config.enable_caching else None
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        
        # Modelo placeholder (se cargará bajo demanda)
        self.model = None
        self.tokenizer = None
        
        # Modelos predefinidos
        self.model_registry = {
            "sentence-transformers": {
                "all-MiniLM-L6-v2": {"dim": 384, "max_length": 256},
                "all-mpnet-base-v2": {"dim": 768, "max_length": 384},
                "paraphrase-multilingual-MiniLM-L12-v2": {"dim": 384, "max_length": 128}
            },
            "openai": {
                "text-embedding-ada-002": {"dim": 1536, "max_length": 8191}
            }
        }
    
    def _generate_cache_key(self, text: str, model_name: str) -> str:
        """Genera clave única para el cache."""
        content = f"{text}:{model_name}:{self.config.max_length}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _load_model(self) -> Any:
        """Carga el modelo de embeddings bajo demanda."""
        if self.model is not None:
            return
        
        try:
            if self.config.model_type == "sentence-transformers":
                self.model = SentenceTransformer(self.config.model_name, device=self.config.device)
                logger.info(f"Modelo cargado: {self.config.model_name}")
            else:
                # Placeholder para otros tipos de modelos
                logger.warning(f"Tipo de modelo no soportado: {self.config.model_type}")
        except ImportError:
            logger.warning("sentence-transformers no disponible, usando embeddings simulados")
            self.model = None
    
    async def embed(self, text: str, language: str = "auto") -> Dict[str, Any]:
        """Genera embeddings para el texto."""
        start_time = time.time()
        
        try:
            # Verificar cache
            if self.cache:
                cache_key = self._generate_cache_key(text, self.config.model_name)
                cached_result = self.cache.get(cache_key)
                if cached_result:
                    self.stats["cache_hits"] += 1
                    return cached_result
            
            # Cargar modelo si es necesario
            self._load_model()
            
            # Generar embeddings
            if self.model is not None:
                embeddings = await self._generate_embeddings(text)
            else:
                # Fallback: embeddings simulados
                embeddings = await self._generate_fallback_embeddings(text)
            
            duration = time.time() - start_time
            self.stats["total_requests"] += 1
            self.stats["total_processing_time"] += duration
            
            result = {
                "text": text,
                "embeddings": embeddings,
                "dimension": len(embeddings),
                "model": self.config.model_name,
                "language": language,
                "processing_time_ms": duration * 1000,
                "timestamp": time.time()
            }
            
            # Guardar en cache
            if self.cache:
                cache_key = self._generate_cache_key(text, self.config.model_name)
                self.cache.put(cache_key, result)
            
            return result
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error("Error generando embeddings", error=str(e), text=text[:100])
            raise
    
    async def _generate_embeddings(self, text: str) -> List[float]:
        """Genera embeddings usando el modelo cargado."""
        if self.model is None:
            return await self._generate_fallback_embeddings(text)
        
        # Ejecutar en thread pool para evitar bloqueo
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            self.executor, 
            self._embed_sync, 
            text
        )
        
        if self.config.normalize:
            embeddings = self._normalize_embeddings(embeddings)
        
        return embeddings.tolist()
    
    def _embed_sync(self, text: str) -> np.ndarray:
        """Genera embeddings de forma síncrona."""
        return self.model.encode(text, max_length=self.config.max_length)
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normaliza los embeddings."""
        norm = np.linalg.norm(embeddings)
        if norm > 0:
            return embeddings / norm
        return embeddings
    
    async def _generate_fallback_embeddings(self, text: str) -> List[float]:
        """Genera embeddings simulados como fallback."""
        # Embeddings simulados basados en características del texto
        
        # Hash del texto para generar vector consistente
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        # Generar vector de 384 dimensiones (como all-MiniLM-L6-v2)
        np.random.seed(int(text_hash[:8], 16))
        embeddings = np.random.normal(0, 1, 384)
        
        # Normalizar
        norm = np.linalg.norm(embeddings)
        if norm > 0:
            embeddings = embeddings / norm
        
        return embeddings.tolist()
    
    async def batch_embed(self, texts: List[str], language: str = "auto") -> List[Dict[str, Any]]:
        """Genera embeddings para múltiples textos en lote."""
        tasks = [self.embed(text, language) for text in texts]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def similarity(self, text1: str, text2: str, language: str = "auto") -> Dict[str, Any]:
        """Calcula la similitud entre dos textos."""
        emb1 = await self.embed(text1, language)
        emb2 = await self.embed(text2, language)
        
        # Calcular similitud coseno
        vec1 = np.array(emb1["embeddings"])
        vec2 = np.array(emb2["embeddings"])
        
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        
        return {
            "text1": text1,
            "text2": text2,
            "similarity": float(similarity),
            "embeddings1": emb1,
            "embeddings2": emb2
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del motor de embeddings."""
        return {
            "total_requests": self.stats["total_requests"],
            "cache_hits": self.stats.get("cache_hits", 0),
            "errors": self.stats.get("errors", 0),
            "cache_size": len(self.cache.cache) if self.cache else 0,
            "cache_hit_rate": self.stats.get("cache_hits", 0) / max(self.stats["total_requests"], 1),
            "model": self.config.model_name,
            "model_loaded": self.model is not None
        }
    
    def clear_cache(self) -> Any:
        """Limpia el cache."""
        if self.cache:
            self.cache.clear()
    
    async def health_check(self) -> Dict[str, Any]:
        """Verifica la salud del motor de embeddings."""
        test_text = "This is a test text for embedding generation."
        
        try:
            result = await self.embed(test_text, "en")
            return {
                "status": "healthy",
                "test_result": {
                    "dimension": result["dimension"],
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

def get_embedding_engine(config: EmbeddingConfig = None) -> EmbeddingEngine:
    """Función factory para obtener una instancia del motor de embeddings."""
    return EmbeddingEngine(config) 