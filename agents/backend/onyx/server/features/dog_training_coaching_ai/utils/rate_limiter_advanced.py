"""
Advanced Rate Limiter
=====================
Rate limiter avanzado con múltiples estrategias.
"""

from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import asyncio
import hashlib

from .logger import get_logger

logger = get_logger(__name__)


class MultiStrategyRateLimiter:
    """Rate limiter con múltiples estrategias."""
    
    def __init__(self):
        self.strategies: Dict[str, Dict] = {}
        self._lock = asyncio.Lock()
    
    def add_strategy(
        self,
        name: str,
        max_requests: int,
        window_seconds: float,
        algorithm: str = "sliding_window"
    ):
        """
        Agregar estrategia de rate limiting.
        
        Args:
            name: Nombre de la estrategia
            max_requests: Máximo de requests
            window_seconds: Ventana de tiempo
            algorithm: Algoritmo (sliding_window, token_bucket, fixed_window)
        """
        self.strategies[name] = {
            "max_requests": max_requests,
            "window_seconds": window_seconds,
            "algorithm": algorithm,
            "requests": defaultdict(list),
            "tokens": defaultdict(lambda: max_requests) if algorithm == "token_bucket" else None
        }
    
    async def is_allowed(
        self,
        key: str,
        strategy_name: str
    ) -> Tuple[bool, Optional[float], Optional[str]]:
        """
        Verificar si request está permitida.
        
        Args:
            key: Clave única
            strategy_name: Nombre de la estrategia
            
        Returns:
            Tupla (permitido, tiempo_restante, mensaje)
        """
        if strategy_name not in self.strategies:
            return (True, None, "Strategy not found, allowing request")
        
        strategy = self.strategies[strategy_name]
        algorithm = strategy["algorithm"]
        
        async with self._lock:
            if algorithm == "sliding_window":
                return await self._check_sliding_window(key, strategy)
            elif algorithm == "token_bucket":
                return await self._check_token_bucket(key, strategy)
            elif algorithm == "fixed_window":
                return await self._check_fixed_window(key, strategy)
            else:
                return (True, None, "Unknown algorithm, allowing request")
    
    async def _check_sliding_window(
        self,
        key: str,
        strategy: Dict
    ) -> Tuple[bool, Optional[float], Optional[str]]:
        """Verificar usando sliding window."""
        now = datetime.now()
        cutoff = now - timedelta(seconds=strategy["window_seconds"])
        
        requests = strategy["requests"][key]
        requests[:] = [req_time for req_time in requests if req_time > cutoff]
        
        if len(requests) < strategy["max_requests"]:
            requests.append(now)
            return (True, None, "Request allowed")
        
        oldest = min(requests)
        wait_time = (oldest + timedelta(seconds=strategy["window_seconds"]) - now).total_seconds()
        
        return (False, max(0, wait_time), "Rate limit exceeded")
    
    async def _check_token_bucket(
        self,
        key: str,
        strategy: Dict
    ) -> Tuple[bool, Optional[float], Optional[str]]:
        """Verificar usando token bucket."""
        # Implementación simplificada
        tokens = strategy["tokens"][key]
        
        if tokens > 0:
            strategy["tokens"][key] = tokens - 1
            return (True, None, "Request allowed")
        
        # Calcular tiempo hasta que se recargue un token
        wait_time = strategy["window_seconds"] / strategy["max_requests"]
        return (False, wait_time, "Rate limit exceeded")
    
    async def _check_fixed_window(
        self,
        key: str,
        strategy: Dict
    ) -> Tuple[bool, Optional[float], Optional[str]]:
        """Verificar usando fixed window."""
        now = datetime.now()
        window_start = now.replace(second=0, microsecond=0)
        
        requests = strategy["requests"][key]
        # Filtrar requests de la ventana actual
        current_window_requests = [
            req_time for req_time in requests
            if req_time >= window_start
        ]
        
        if len(current_window_requests) < strategy["max_requests"]:
            current_window_requests.append(now)
            strategy["requests"][key] = current_window_requests
            return (True, None, "Request allowed")
        
        # Calcular tiempo hasta la siguiente ventana
        next_window = window_start + timedelta(minutes=1)
        wait_time = (next_window - now).total_seconds()
        
        return (False, max(0, wait_time), "Rate limit exceeded")


class DistributedRateLimiter:
    """Rate limiter para entornos distribuidos (simplificado)."""
    
    def __init__(self, redis_client=None):
        """
        Inicializar rate limiter distribuido.
        
        Args:
            redis_client: Cliente Redis (opcional)
        """
        self.redis_client = redis_client
        self.local_cache: Dict[str, Dict] = {}
    
    async def is_allowed(
        self,
        key: str,
        max_requests: int,
        window_seconds: float
    ) -> Tuple[bool, Optional[float]]:
        """
        Verificar si request está permitida.
        
        Args:
            key: Clave única
            max_requests: Máximo de requests
            window_seconds: Ventana de tiempo
            
        Returns:
            Tupla (permitido, tiempo_restante)
        """
        if self.redis_client:
            # Usar Redis para rate limiting distribuido
            try:
                # Implementación simplificada - usar Redis INCR con TTL
                current = await self.redis_client.get(f"ratelimit:{key}")
                if current and int(current) >= max_requests:
                    ttl = await self.redis_client.ttl(f"ratelimit:{key}")
                    return (False, ttl)
                
                pipe = self.redis_client.pipeline()
                pipe.incr(f"ratelimit:{key}")
                pipe.expire(f"ratelimit:{key}", int(window_seconds))
                await pipe.execute()
                return (True, None)
            except Exception as e:
                logger.warning(f"Redis rate limiting failed, falling back to local: {e}")
        
        # Fallback a cache local
        return await self._local_check(key, max_requests, window_seconds)
    
    async def _local_check(
        self,
        key: str,
        max_requests: int,
        window_seconds: float
    ) -> Tuple[bool, Optional[float]]:
        """Verificar usando cache local."""
        now = datetime.now()
        
        if key not in self.local_cache:
            self.local_cache[key] = {
                "requests": [],
                "window_start": now
            }
        
        cache_entry = self.local_cache[key]
        cutoff = now - timedelta(seconds=window_seconds)
        
        cache_entry["requests"] = [
            req_time for req_time in cache_entry["requests"]
            if req_time > cutoff
        ]
        
        if len(cache_entry["requests"]) < max_requests:
            cache_entry["requests"].append(now)
            return (True, None)
        
        oldest = min(cache_entry["requests"])
        wait_time = (oldest + timedelta(seconds=window_seconds) - now).total_seconds()
        return (False, max(0, wait_time))

