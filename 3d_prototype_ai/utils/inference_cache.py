"""
Inference Cache - Caché avanzado para inferencia
=================================================
Caching inteligente de predicciones con invalidación
"""

import logging
import torch
import hashlib
import json
from typing import Dict, Any, Optional, Tuple
from collections import OrderedDict
from datetime import datetime, timedelta
import pickle

logger = logging.getLogger(__name__)


class InferenceCache:
    """Caché avanzado para inferencia"""
    
    def __init__(
        self,
        max_size: int = 10000,
        ttl_seconds: Optional[int] = None,
        eviction_policy: str = "lru"
    ):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.eviction_policy = eviction_policy
        
        if eviction_policy == "lru":
            self.cache: OrderedDict = OrderedDict()
        else:
            self.cache: Dict[str, Dict[str, Any]] = {}
        
        self.hit_count = 0
        self.miss_count = 0
    
    def _generate_key(self, input_data: Any, model_id: str) -> str:
        """Genera clave de cache"""
        # Serializar input
        if isinstance(input_data, torch.Tensor):
            input_str = str(input_data.cpu().numpy().tobytes())
        elif isinstance(input_data, dict):
            input_str = json.dumps(input_data, sort_keys=True)
        else:
            input_str = str(input_data)
        
        # Hash
        key_data = f"{model_id}_{input_str}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, input_data: Any, model_id: str) -> Optional[Any]:
        """Obtiene de cache"""
        key = self._generate_key(input_data, model_id)
        
        if key not in self.cache:
            self.miss_count += 1
            return None
        
        cached_item = self.cache[key]
        
        # Verificar TTL
        if self.ttl_seconds:
            cached_time = datetime.fromisoformat(cached_item["timestamp"])
            if datetime.now() - cached_time > timedelta(seconds=self.ttl_seconds):
                del self.cache[key]
                self.miss_count += 1
                return None
        
        # Mover al final (LRU)
        if self.eviction_policy == "lru":
            self.cache.move_to_end(key)
        
        self.hit_count += 1
        return cached_item["result"]
    
    def set(self, input_data: Any, model_id: str, result: Any):
        """Guarda en cache"""
        key = self._generate_key(input_data, model_id)
        
        # Evict si es necesario
        if len(self.cache) >= self.max_size:
            if self.eviction_policy == "lru":
                self.cache.popitem(last=False)  # Eliminar más antiguo
            else:
                # Eliminar aleatorio
                first_key = next(iter(self.cache))
                del self.cache[first_key]
        
        # Guardar
        cache_item = {
            "result": result,
            "timestamp": datetime.now().isoformat(),
            "model_id": model_id
        }
        
        if self.eviction_policy == "lru":
            self.cache[key] = cache_item
            self.cache.move_to_end(key)
        else:
            self.cache[key] = cache_item
    
    def clear(self):
        """Limpia cache"""
        self.cache.clear()
        self.hit_count = 0
        self.miss_count = 0
        logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de cache"""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0.0
        
        return {
            "cache_size": len(self.cache),
            "max_size": self.max_size,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate,
            "total_requests": total_requests
        }




