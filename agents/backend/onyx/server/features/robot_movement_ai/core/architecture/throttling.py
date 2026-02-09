"""
Sistema de throttling avanzado para Robot Movement AI v2.0
Throttling con múltiples algoritmos y ventanas de tiempo
"""

from typing import Optional, Dict, Tuple, Any
from datetime import datetime, timedelta
from collections import defaultdict
import time

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None


class ThrottleAlgorithm:
    """Algoritmos de throttling disponibles"""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"


class Throttler:
    """Throttler con múltiples algoritmos"""
    
    def __init__(
        self,
        algorithm: str = ThrottleAlgorithm.SLIDING_WINDOW,
        redis_client: Optional[Any] = None
    ):
        """
        Inicializar throttler
        
        Args:
            algorithm: Algoritmo a usar
            redis_client: Cliente Redis (opcional, para throttling distribuido)
        """
        self.algorithm = algorithm
        self.redis = redis_client
        self.local_cache: Dict[str, list] = defaultdict(list)
    
    def is_allowed(
        self,
        key: str,
        max_requests: int,
        window_seconds: int
    ) -> Tuple[bool, Optional[int], Optional[float]]:
        """
        Verificar si se permite la petición
        
        Args:
            key: Clave única (IP, user_id, etc.)
            max_requests: Número máximo de peticiones
            window_seconds: Ventana de tiempo en segundos
            
        Returns:
            Tuple de (allowed, remaining, reset_time)
        """
        if self.algorithm == ThrottleAlgorithm.FIXED_WINDOW:
            return self._fixed_window(key, max_requests, window_seconds)
        elif self.algorithm == ThrottleAlgorithm.SLIDING_WINDOW:
            return self._sliding_window(key, max_requests, window_seconds)
        elif self.algorithm == ThrottleAlgorithm.TOKEN_BUCKET:
            return self._token_bucket(key, max_requests, window_seconds)
        else:
            return self._sliding_window(key, max_requests, window_seconds)
    
    def _fixed_window(
        self,
        key: str,
        max_requests: int,
        window_seconds: int
    ) -> Tuple[bool, Optional[int], Optional[float]]:
        """Fixed window algorithm"""
        now = time.time()
        window_start = int(now / window_seconds) * window_seconds
        window_key = f"{key}:{window_start}"
        
        if self.redis:
            # Usar Redis para distribución
            count = self.redis.incr(window_key)
            if count == 1:
                self.redis.expire(window_key, window_seconds)
            
            remaining = max(0, max_requests - count)
            reset_time = window_start + window_seconds
            return (count <= max_requests, remaining, reset_time)
        else:
            # Usar cache local
            if window_key not in self.local_cache:
                self.local_cache[window_key] = []
            
            self.local_cache[window_key].append(now)
            count = len(self.local_cache[window_key])
            
            # Limpiar ventanas antiguas
            self._cleanup_old_windows(window_seconds)
            
            remaining = max(0, max_requests - count)
            reset_time = window_start + window_seconds
            return (count <= max_requests, remaining, reset_time)
    
    def _sliding_window(
        self,
        key: str,
        max_requests: int,
        window_seconds: int
    ) -> Tuple[bool, Optional[int], Optional[float]]:
        """Sliding window algorithm"""
        now = time.time()
        window_start = now - window_seconds
        
        if self.redis:
            # Usar Redis sorted set
            redis_key = f"throttle:{key}"
            pipe = self.redis.pipeline()
            pipe.zremrangebyscore(redis_key, 0, window_start)
            pipe.zcard(redis_key)
            pipe.zadd(redis_key, {str(now): now})
            pipe.expire(redis_key, window_seconds)
            
            results = pipe.execute()
            count = results[1] + 1  # +1 porque agregamos la actual
            
            remaining = max(0, max_requests - count)
            reset_time = now + window_seconds
            return (count <= max_requests, remaining, reset_time)
        else:
            # Usar cache local
            if key not in self.local_cache:
                self.local_cache[key] = []
            
            # Limpiar entradas antiguas
            self.local_cache[key] = [
                t for t in self.local_cache[key] if t > window_start
            ]
            
            self.local_cache[key].append(now)
            count = len(self.local_cache[key])
            
            remaining = max(0, max_requests - count)
            reset_time = now + window_seconds
            return (count <= max_requests, remaining, reset_time)
    
    def _token_bucket(
        self,
        key: str,
        capacity: int,
        refill_rate: float
    ) -> Tuple[bool, Optional[int], Optional[float]]:
        """Token bucket algorithm"""
        now = time.time()
        redis_key = f"token_bucket:{key}"
        
        if self.redis:
            pipe = self.redis.pipeline()
            pipe.hgetall(redis_key)
            result = pipe.execute()[0]
            
            if not result:
                tokens = capacity - 1
                self.redis.hset(redis_key, mapping={
                    "tokens": tokens,
                    "last_refill": now
                })
                self.redis.expire(redis_key, int(capacity / refill_rate) + 1)
                return (True, int(tokens), now + (capacity / refill_rate))
            
            tokens = float(result.get(b"tokens", capacity))
            last_refill = float(result.get(b"last_refill", now))
            
            elapsed = now - last_refill
            tokens_to_add = elapsed * refill_rate
            tokens = min(capacity, tokens + tokens_to_add)
            
            if tokens >= 1:
                tokens -= 1
                self.redis.hset(redis_key, mapping={
                    "tokens": tokens,
                    "last_refill": now
                })
                return (True, int(tokens), now + ((capacity - tokens) / refill_rate))
            else:
                return (False, 0, now + ((1 - tokens) / refill_rate))
        else:
            # Implementación local simplificada
            return (True, capacity - 1, now + 1.0)
    
    def _cleanup_old_windows(self, window_seconds: int):
        """Limpiar ventanas antiguas del cache local"""
        now = time.time()
        cutoff = now - (window_seconds * 2)
        
        keys_to_remove = [
            key for key in self.local_cache.keys()
            if key.split(":")[-1] and float(key.split(":")[-1]) < cutoff
        ]
        
        for key in keys_to_remove:
            del self.local_cache[key]
    
    def reset(self, key: str):
        """Resetear contador para una clave"""
        if self.redis:
            self.redis.delete(f"throttle:{key}")
            self.redis.delete(f"token_bucket:{key}")
        else:
            self.local_cache.pop(key, None)


def create_throttler(
    algorithm: str = ThrottleAlgorithm.SLIDING_WINDOW,
    redis_client: Optional[Any] = None
) -> Throttler:
    """Crear instancia de throttler"""
    return Throttler(algorithm=algorithm, redis_client=redis_client)


