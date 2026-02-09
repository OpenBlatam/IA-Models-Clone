"""
Model Caching System - Sistema de caché para modelos
=====================================================
"""

import logging
import torch
import hashlib
import pickle
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import os

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Entrada de caché"""
    key: str
    prediction: Any
    timestamp: datetime = field(default_factory=datetime.now)
    hit_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModelCache:
    """Sistema de caché para modelos"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: Optional[int] = None, cache_dir: Optional[str] = None):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache_dir = cache_dir
        self.cache: Dict[str, CacheEntry] = {}
        self.hit_count = 0
        self.miss_count = 0
        
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
    
    def _generate_key(self, input_data: Any) -> str:
        """Genera clave de caché"""
        if isinstance(input_data, torch.Tensor):
            data_bytes = input_data.cpu().numpy().tobytes()
        elif isinstance(input_data, dict):
            data_bytes = pickle.dumps(input_data)
        else:
            data_bytes = pickle.dumps(input_data)
        
        return hashlib.md5(data_bytes).hexdigest()
    
    def get(self, input_data: Any) -> Optional[Any]:
        """Obtiene predicción del caché"""
        key = self._generate_key(input_data)
        
        if key in self.cache:
            entry = self.cache[key]
            
            # Verificar TTL
            if self.ttl_seconds and (datetime.now() - entry.timestamp).total_seconds() > self.ttl_seconds:
                del self.cache[key]
                self.miss_count += 1
                return None
            
            entry.hit_count += 1
            self.hit_count += 1
            logger.debug(f"Cache hit: {key}")
            return entry.prediction
        
        self.miss_count += 1
        logger.debug(f"Cache miss: {key}")
        return None
    
    def set(self, input_data: Any, prediction: Any, metadata: Optional[Dict[str, Any]] = None):
        """Almacena predicción en caché"""
        key = self._generate_key(input_data)
        
        # Verificar tamaño máximo
        if len(self.cache) >= self.max_size:
            self._evict_oldest()
        
        entry = CacheEntry(
            key=key,
            prediction=prediction,
            metadata=metadata or {}
        )
        
        self.cache[key] = entry
        
        # Guardar en disco si cache_dir está configurado
        if self.cache_dir:
            self._save_to_disk(key, entry)
    
    def _evict_oldest(self):
        """Elimina entrada más antigua"""
        if not self.cache:
            return
        
        oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k].timestamp)
        del self.cache[oldest_key]
        logger.debug(f"Evicted cache entry: {oldest_key}")
    
    def _save_to_disk(self, key: str, entry: CacheEntry):
        """Guarda entrada en disco"""
        if not self.cache_dir:
            return
        
        try:
            cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
            with open(cache_file, 'wb') as f:
                pickle.dump(entry, f)
        except Exception as e:
            logger.warning(f"Error guardando caché en disco: {e}")
    
    def _load_from_disk(self, key: str) -> Optional[CacheEntry]:
        """Carga entrada desde disco"""
        if not self.cache_dir:
            return None
        
        try:
            cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.warning(f"Error cargando caché desde disco: {e}")
        
        return None
    
    def clear(self):
        """Limpia el caché"""
        self.cache.clear()
        self.hit_count = 0
        self.miss_count = 0
        logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del caché"""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0
        
        return {
            "cache_size": len(self.cache),
            "max_size": self.max_size,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate,
            "total_requests": total_requests
        }




