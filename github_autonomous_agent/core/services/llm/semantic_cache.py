"""
Semantic Caching para LLMs.

Caching inteligente basado en embeddings para encontrar
respuestas similares incluso cuando el prompt no es idéntico.
"""

import json
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import os
import numpy as np

from config.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class CachedItem:
    """Item en el cache semántico."""
    key: str
    prompt: str
    response: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = None
    created_at: datetime = None
    access_count: int = 0
    last_accessed: datetime = None
    
    def __post_init__(self):
        """Inicializar valores por defecto."""
        if self.metadata is None:
            self.metadata = {}
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.last_accessed is None:
            self.last_accessed = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "key": self.key,
            "prompt": self.prompt,
            "response": self.response,
            "embedding": self.embedding,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None
        }


class SemanticCache:
    """
    Cache semántico para respuestas LLM.
    
    Usa embeddings para encontrar respuestas similares incluso
    cuando el prompt no es exactamente igual.
    
    Características:
    - Búsqueda por similitud semántica
    - TTL configurable
    - LRU eviction
    - Metadata personalizable
    """
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        similarity_threshold: float = 0.85,
        max_size: int = 1000,
        ttl_seconds: Optional[int] = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Inicializar cache semántico.
        
        Args:
            storage_path: Ruta para persistencia (opcional)
            similarity_threshold: Umbral de similitud (0-1)
            max_size: Tamaño máximo del cache
            ttl_seconds: TTL en segundos (None = sin expiración)
            embedding_model: Modelo para generar embeddings
        """
        self.storage_path = storage_path or "data/semantic_cache"
        self.similarity_threshold = similarity_threshold
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.embedding_model = embedding_model
        
        self.cache: Dict[str, CachedItem] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
        
        # Crear directorio si no existe
        os.makedirs(self.storage_path, exist_ok=True)
        
        # Cargar cache desde disco
        self._load_cache()
        
        # Inicializar modelo de embeddings (lazy loading)
        self._embedding_model = None
    
    def _get_embedding_model(self):
        """Obtener modelo de embeddings (lazy loading)."""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedding_model = SentenceTransformer(self.embedding_model)
                logger.info(f"Modelo de embeddings cargado: {self.embedding_model}")
            except ImportError:
                logger.warning(
                    "sentence-transformers no disponible. "
                    "Usando hash simple para similitud."
                )
                self._embedding_model = "simple"
        return self._embedding_model
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """
        Generar embedding para un texto.
        
        Args:
            text: Texto a procesar
            
        Returns:
            Embedding como numpy array
        """
        model = self._get_embedding_model()
        
        if model == "simple":
            # Fallback: usar hash simple (no es semántico, pero funciona)
            hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
            # Convertir a vector de dimensión fija
            return np.array([hash_val % 1000] * 384) / 1000.0
        
        try:
            embedding = model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            logger.error(f"Error generando embedding: {e}")
            # Fallback
            hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
            return np.array([hash_val % 1000] * 384) / 1000.0
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calcular similitud coseno entre dos vectores.
        
        Args:
            vec1: Primer vector
            vec2: Segundo vector
            
        Returns:
            Similitud (0-1)
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def get(
        self,
        prompt: str,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> Optional[Tuple[str, float]]:
        """
        Buscar respuesta en cache por similitud semántica.
        
        Args:
            prompt: Prompt a buscar
            metadata_filter: Filtrar por metadata (opcional)
            
        Returns:
            Tupla (respuesta, similitud) o None si no se encuentra
        """
        # Generar embedding del prompt
        prompt_embedding = self._generate_embedding(prompt)
        
        best_match = None
        best_similarity = 0.0
        
        # Limpiar items expirados
        self._cleanup_expired()
        
        # Buscar en cache
        for key, item in self.cache.items():
            # Filtrar por metadata si se proporciona
            if metadata_filter:
                if not all(
                    item.metadata.get(k) == v
                    for k, v in metadata_filter.items()
                ):
                    continue
            
            # Calcular similitud
            if key in self.embeddings:
                similarity = self._cosine_similarity(
                    prompt_embedding,
                    self.embeddings[key]
                )
            else:
                # Si no hay embedding, generar uno
                item_embedding = self._generate_embedding(item.prompt)
                self.embeddings[key] = item_embedding
                similarity = self._cosine_similarity(prompt_embedding, item_embedding)
            
            if similarity > best_similarity and similarity >= self.similarity_threshold:
                best_similarity = similarity
                best_match = (item.response, similarity)
        
        if best_match:
            # Actualizar estadísticas
            for key, item in self.cache.items():
                if item.response == best_match[0]:
                    item.access_count += 1
                    item.last_accessed = datetime.now()
                    break
            
            logger.debug(f"Cache hit semántico: similitud {best_similarity:.3f}")
            return best_match
        
        return None
    
    def set(
        self,
        prompt: str,
        response: str,
        metadata: Optional[Dict[str, Any]] = None,
        key: Optional[str] = None
    ) -> str:
        """
        Guardar respuesta en cache.
        
        Args:
            prompt: Prompt
            response: Respuesta
            metadata: Metadatos adicionales
            key: Clave personalizada (opcional)
            
        Returns:
            Clave del item guardado
        """
        if key is None:
            key = hashlib.md5(
                f"{prompt}{response}{datetime.now().isoformat()}".encode()
            ).hexdigest()[:16]
        
        # Limpiar si el cache está lleno
        if len(self.cache) >= self.max_size:
            self._evict_lru()
        
        # Generar embedding
        embedding = self._generate_embedding(prompt)
        
        # Crear item
        item = CachedItem(
            key=key,
            prompt=prompt,
            response=response,
            embedding=embedding.tolist(),
            metadata=metadata or {}
        )
        
        self.cache[key] = item
        self.embeddings[key] = embedding
        
        # Guardar en disco
        self._save_item(key, item)
        
        logger.debug(f"Item guardado en cache semántico: {key}")
        return key
    
    def delete(self, key: str) -> bool:
        """
        Eliminar item del cache.
        
        Args:
            key: Clave del item
            
        Returns:
            True si se eliminó correctamente
        """
        if key in self.cache:
            del self.cache[key]
        if key in self.embeddings:
            del self.embeddings[key]
        
        # Eliminar de disco
        file_path = os.path.join(self.storage_path, f"item_{key}.json")
        if os.path.exists(file_path):
            os.remove(file_path)
        
        return True
    
    def clear(self) -> None:
        """Limpiar todo el cache."""
        self.cache.clear()
        self.embeddings.clear()
        
        # Limpiar disco
        if os.path.exists(self.storage_path):
            for filename in os.listdir(self.storage_path):
                if filename.startswith("item_") and filename.endswith(".json"):
                    os.remove(os.path.join(self.storage_path, filename))
        
        logger.info("Cache semántico limpiado")
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del cache."""
        total_items = len(self.cache)
        total_accesses = sum(item.access_count for item in self.cache.values())
        
        return {
            "total_items": total_items,
            "max_size": self.max_size,
            "usage_percent": (total_items / self.max_size) * 100 if self.max_size > 0 else 0,
            "total_accesses": total_accesses,
            "avg_accesses_per_item": total_accesses / total_items if total_items > 0 else 0,
            "similarity_threshold": self.similarity_threshold,
            "ttl_seconds": self.ttl_seconds
        }
    
    def _cleanup_expired(self) -> None:
        """Limpiar items expirados."""
        if self.ttl_seconds is None:
            return
        
        now = datetime.now()
        expired_keys = [
            key for key, item in self.cache.items()
            if (now - item.created_at).total_seconds() > self.ttl_seconds
        ]
        
        for key in expired_keys:
            self.delete(key)
    
    def _evict_lru(self) -> None:
        """Eliminar item menos usado recientemente (LRU)."""
        if not self.cache:
            return
        
        # Encontrar item con menor last_accessed
        lru_key = min(
            self.cache.keys(),
            key=lambda k: self.cache[k].last_accessed
        )
        
        self.delete(lru_key)
        logger.debug(f"Item LRU eliminado: {lru_key}")
    
    def _save_item(self, key: str, item: CachedItem) -> None:
        """Guardar item en disco."""
        file_path = os.path.join(self.storage_path, f"item_{key}.json")
        with open(file_path, 'w') as f:
            json.dump(item.to_dict(), f, indent=2, default=str)
    
    def _load_cache(self) -> None:
        """Cargar cache desde disco."""
        if not os.path.exists(self.storage_path):
            return
        
        for filename in os.listdir(self.storage_path):
            if filename.startswith("item_") and filename.endswith(".json"):
                file_path = os.path.join(self.storage_path, filename)
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    item = CachedItem(
                        key=data["key"],
                        prompt=data["prompt"],
                        response=data["response"],
                        embedding=data.get("embedding"),
                        metadata=data.get("metadata", {}),
                        access_count=data.get("access_count", 0)
                    )
                    
                    if data.get("created_at"):
                        item.created_at = datetime.fromisoformat(data["created_at"])
                    if data.get("last_accessed"):
                        item.last_accessed = datetime.fromisoformat(data["last_accessed"])
                    
                    self.cache[item.key] = item
                    
                    # Cargar embedding
                    if item.embedding:
                        self.embeddings[item.key] = np.array(item.embedding)
                except Exception as e:
                    logger.error(f"Error cargando item desde {filename}: {e}")


def get_semantic_cache(
    storage_path: Optional[str] = None,
    similarity_threshold: float = 0.85,
    max_size: int = 1000,
    ttl_seconds: Optional[int] = None
) -> SemanticCache:
    """Factory function para obtener instancia singleton del cache."""
    cache_key = f"{storage_path}_{similarity_threshold}_{max_size}_{ttl_seconds}"
    if not hasattr(get_semantic_cache, "_instances"):
        get_semantic_cache._instances = {}
    
    if cache_key not in get_semantic_cache._instances:
        get_semantic_cache._instances[cache_key] = SemanticCache(
            storage_path=storage_path,
            similarity_threshold=similarity_threshold,
            max_size=max_size,
            ttl_seconds=ttl_seconds
        )
    
    return get_semantic_cache._instances[cache_key]



