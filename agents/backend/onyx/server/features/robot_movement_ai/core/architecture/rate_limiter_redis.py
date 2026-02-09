"""
Rate Limiter distribuido con Redis para Robot Movement AI v2.0
Token bucket y sliding window algorithms
"""

from typing import Optional
from datetime import datetime, timedelta
import time

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None


class RedisRateLimiter:
    """Rate limiter distribuido usando Redis"""
    
    def __init__(
        self,
        redis_client,
        key_prefix: str = "rate_limit"
    ):
        """
        Inicializar rate limiter
        
        Args:
            redis_client: Cliente Redis
            key_prefix: Prefijo para claves en Redis
        """
        if not REDIS_AVAILABLE:
            raise ImportError("redis package is required")
        
        self.redis = redis_client
        self.key_prefix = key_prefix
    
    def is_allowed(
        self,
        identifier: str,
        max_requests: int,
        window_seconds: int
    ) -> tuple[bool, Optional[int]]:
        """
        Verificar si se permite la petición (Sliding Window)
        
        Args:
            identifier: Identificador único (IP, user_id, etc.)
            max_requests: Número máximo de peticiones
            window_seconds: Ventana de tiempo en segundos
            
        Returns:
            Tuple de (allowed, remaining_requests)
        """
        key = f"{self.key_prefix}:{identifier}"
        now = time.time()
        window_start = now - window_seconds
        
        # Usar pipeline para operaciones atómicas
        pipe = self.redis.pipeline()
        
        # Eliminar entradas antiguas
        pipe.zremrangebyscore(key, 0, window_start)
        
        # Contar entradas actuales
        pipe.zcard(key)
        
        # Agregar entrada actual
        pipe.zadd(key, {str(now): now})
        
        # Establecer expiración
        pipe.expire(key, window_seconds)
        
        results = pipe.execute()
        current_count = results[1]
        
        if current_count < max_requests:
            # Agregar entrada (ya se agregó en pipeline)
            remaining = max_requests - current_count - 1
            return True, remaining
        else:
            return False, 0
    
    def token_bucket(
        self,
        identifier: str,
        capacity: int,
        refill_rate: float
    ) -> tuple[bool, Optional[float]]:
        """
        Token Bucket algorithm
        
        Args:
            identifier: Identificador único
            capacity: Capacidad del bucket
            refill_rate: Tokens por segundo
            
        Returns:
            Tuple de (allowed, tokens_remaining)
        """
        key = f"{self.key_prefix}:bucket:{identifier}"
        now = time.time()
        
        pipe = self.redis.pipeline()
        
        # Obtener estado actual
        pipe.hgetall(key)
        pipe.hset(key, "last_refill", now)
        
        result = pipe.execute()[0]
        
        if not result:
            # Inicializar bucket
            tokens = capacity - 1
            self.redis.hset(key, mapping={
                "tokens": tokens,
                "last_refill": now
            })
            self.redis.expire(key, int(capacity / refill_rate) + 1)
            return True, tokens
        
        tokens = float(result.get(b"tokens", capacity))
        last_refill = float(result.get(b"last_refill", now))
        
        # Calcular tokens a agregar
        elapsed = now - last_refill
        tokens_to_add = elapsed * refill_rate
        tokens = min(capacity, tokens + tokens_to_add)
        
        if tokens >= 1:
            tokens -= 1
            self.redis.hset(key, mapping={
                "tokens": tokens,
                "last_refill": now
            })
            return True, tokens
        else:
            return False, tokens
    
    def reset(self, identifier: str):
        """Resetear contador para un identificador"""
        keys = [
            f"{self.key_prefix}:{identifier}",
            f"{self.key_prefix}:bucket:{identifier}"
        ]
        for key in keys:
            self.redis.delete(key)


def get_redis_rate_limiter(redis_client=None) -> Optional[RedisRateLimiter]:
    """Obtener instancia de rate limiter Redis"""
    if not REDIS_AVAILABLE:
        return None
    
    if redis_client is None:
        from core.architecture.redis_cache import get_redis_cache
        cache = get_redis_cache()
        if cache:
            redis_client = cache._get_client()
        else:
            return None
    
    if redis_client:
        return RedisRateLimiter(redis_client)
    return None




