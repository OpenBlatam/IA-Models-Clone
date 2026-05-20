"""
Intelligent Cache - Sistema de cache inteligente
=================================================
"""

import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from collections import OrderedDict
import hashlib
import json

logger = logging.getLogger(__name__)


class IntelligentCache:
    """Sistema de cache inteligente con predicción"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.access_patterns: Dict[str, List[datetime]] = defaultdict(list)
        self.prediction_model: Optional[Callable] = None
    
    def get(self, key: str) -> Optional[Any]:
        """Obtiene valor del cache"""
        if key in self.cache:
            # Mover al final (LRU)
            value = self.cache.pop(key)
            self.cache[key] = value
            
            # Registrar acceso
            self.access_patterns[key].append(datetime.now())
            
            # Mantener solo últimas 100 accesos
            if len(self.access_patterns[key]) > 100:
                self.access_patterns[key] = self.access_patterns[key][-100:]
            
            return value["value"]
        
        return None
    
    def set(self, key: str, value: Any, ttl_seconds: int = 3600,
           priority: int = 0):
        """Almacena valor en cache"""
        expires_at = datetime.now() + timedelta(seconds=ttl_seconds)
        
        cache_entry = {
            "value": value,
            "expires_at": expires_at.isoformat(),
            "priority": priority,
            "created_at": datetime.now().isoformat(),
            "access_count": 0
        }
        
        # Si existe, actualizar
        if key in self.cache:
            cache_entry["access_count"] = self.cache[key].get("access_count", 0)
            del self.cache[key]
        
        self.cache[key] = cache_entry
        
        # Si excede tamaño máximo, eliminar menos usado
        if len(self.cache) > self.max_size:
            self._evict_least_used()
        
        logger.debug(f"Valor almacenado en cache: {key}")
    
    def _evict_least_used(self):
        """Elimina entrada menos usada"""
        # Ordenar por prioridad y acceso
        sorted_items = sorted(
            self.cache.items(),
            key=lambda x: (x[1]["priority"], -x[1]["access_count"])
        )
        
        # Eliminar el primero (menor prioridad/uso)
        if sorted_items:
            key_to_remove = sorted_items[0][0]
            del self.cache[key_to_remove]
            logger.debug(f"Entrada eliminada del cache: {key_to_remove}")
    
    def predict_and_prefetch(self, current_key: str):
        """Predice y precarga valores"""
        # Analizar patrones de acceso
        similar_keys = self._find_similar_access_patterns(current_key)
        
        for key in similar_keys[:3]:  # Precargar top 3
            if key not in self.cache:
                # En producción, esto llamaría a la función para generar el valor
                logger.debug(f"Precargando: {key}")
    
    def _find_similar_access_patterns(self, key: str) -> List[str]:
        """Encuentra claves con patrones de acceso similares"""
        current_pattern = self.access_patterns.get(key, [])
        
        if not current_pattern:
            return []
        
        # Buscar claves accedidas después de esta
        similar = []
        for other_key, pattern in self.access_patterns.items():
            if other_key != key and pattern:
                # Verificar si hay correlación temporal
                if len(pattern) > 0:
                    similar.append(other_key)
        
        return similar[:10]  # Top 10
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del cache"""
        total_accesses = sum(len(pattern) for pattern in self.access_patterns.values())
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "usage_percent": (len(self.cache) / self.max_size * 100) if self.max_size > 0 else 0,
            "total_keys": len(self.cache),
            "total_accesses": total_accesses,
            "most_accessed": self._get_most_accessed_keys(10)
        }
    
    def _get_most_accessed_keys(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Obtiene claves más accedidas"""
        key_access = [
            {
                "key": key,
                "access_count": len(pattern),
                "last_access": pattern[-1].isoformat() if pattern else None
            }
            for key, pattern in self.access_patterns.items()
        ]
        
        key_access.sort(key=lambda x: x["access_count"], reverse=True)
        return key_access[:limit]
    
    def clear_expired(self):
        """Limpia entradas expiradas"""
        now = datetime.now()
        expired_keys = [
            key for key, entry in self.cache.items()
            if datetime.fromisoformat(entry["expires_at"]) < now
        ]
        
        for key in expired_keys:
            del self.cache[key]
        
        if expired_keys:
            logger.info(f"Eliminadas {len(expired_keys)} entradas expiradas")




