"""
KV Cache - Cache de Key-Value
==============================

Sistema de cache KV para acelerar generación autoregresiva.
"""

import logging
import torch
from typing import Optional, Dict, Any, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


class KVCache:
    """Cache de Key-Value para atención"""
    
    def __init__(self, max_cache_size: int = 1000):
        """
        Inicializar KV cache
        
        Args:
            max_cache_size: Tamaño máximo del cache
        """
        self.cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.max_cache_size = max_cache_size
        self.hits = 0
        self.misses = 0
        logger.info("KV Cache inicializado")
    
    def get(self, cache_key: str) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Obtener del cache
        
        Args:
            cache_key: Clave del cache
            
        Returns:
            Tuple (keys, values) o None
        """
        if cache_key in self.cache:
            self.hits += 1
            return self.cache[cache_key]
        
        self.misses += 1
        return None
    
    def set(
        self,
        cache_key: str,
        keys: torch.Tensor,
        values: torch.Tensor
    ):
        """
        Guardar en cache
        
        Args:
            cache_key: Clave del cache
            keys: Tensor de keys
            values: Tensor de values
        """
        if len(self.cache) >= self.max_cache_size:
            # Remover entrada más antigua (FIFO)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[cache_key] = (keys.detach().clone(), values.detach().clone())
    
    def clear(self):
        """Limpiar cache"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del cache"""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0.0
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "cache_size": len(self.cache)
        }


class CachedGeneration:
    """Generación con cache KV"""
    
    def __init__(self, model: torch.nn.Module, kv_cache: Optional[KVCache] = None):
        """
        Inicializar generación con cache
        
        Args:
            model: Modelo a usar
            kv_cache: Cache KV (opcional)
        """
        self.model = model
        self.kv_cache = kv_cache or KVCache()
    
    def generate_with_cache(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        use_cache: bool = True
    ) -> torch.Tensor:
        """
        Generar con cache KV
        
        Args:
            input_ids: IDs de input
            max_length: Longitud máxima
            use_cache: Usar cache
            
        Returns:
            Secuencia generada
        """
        generated = input_ids.clone()
        
        for step in range(max_length - input_ids.size(1)):
            # Crear clave de cache
            cache_key = f"{input_ids.sum().item()}_{step}"
            
            # Intentar obtener del cache
            cached_kv = None
            if use_cache:
                cached_kv = self.kv_cache.get(cache_key)
            
            # Forward pass
            with torch.no_grad():
                if cached_kv:
                    # Usar cache
                    outputs = self.model(
                        input_ids=generated[:, -1:],
                        past_key_values=cached_kv
                    )
                else:
                    # Forward completo
                    outputs = self.model(input_ids=generated)
                    
                    # Guardar en cache si es posible
                    if use_cache and hasattr(outputs, "past_key_values"):
                        self.kv_cache.set(cache_key, *outputs.past_key_values)
                
                # Obtener siguiente token
                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)
        
        return generated




