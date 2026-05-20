"""
Rate Limiter con Redis - Rate limiting distribuido
==================================================

Rate limiting usando Redis para entornos distribuidos.
"""

import os
import time
import logging
from typing import Optional, Dict, Any
from fastapi import Request, HTTPException, status

logger = logging.getLogger(__name__)

# Redis client (lazy initialization)
_redis_client: Optional[Any] = None


def get_redis_client() -> Optional[Any]:
    """Obtener cliente Redis."""
    global _redis_client
    
    if _redis_client is not None:
        return _redis_client
    
    try:
        import redis
        
        redis_url = os.getenv("REDIS_URL") or os.getenv("REDIS_ENDPOINT")
        if not redis_url:
            logger.warning("Redis URL not configured, rate limiting disabled")
            return None
        
        # Parsear URL
        if redis_url.startswith("redis://") or redis_url.startswith("rediss://"):
            _redis_client = redis.from_url(redis_url, decode_responses=True)
        else:
            # Asumir formato host:port
            host, port = redis_url.split(":") if ":" in redis_url else (redis_url, 6379)
            _redis_client = redis.Redis(
                host=host,
                port=int(port),
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
        
        # Test connection
        _redis_client.ping()
        logger.info("Redis client connected for rate limiting")
        
        return _redis_client
    
    except ImportError:
        logger.warning("redis not installed, rate limiting disabled")
        return None
    except Exception as e:
        logger.warning(f"Failed to connect to Redis: {e}")
        return None


class RedisRateLimiter:
    """
    Rate limiter usando Redis (Token Bucket Algorithm).
    
    Soporta rate limiting distribuido para múltiples instancias.
    """
    
    def __init__(
        self,
        max_requests: int = 100,
        window_seconds: float = 60.0,
        key_prefix: str = "rate_limit"
    ):
        """
        Inicializar rate limiter.
        
        Args:
            max_requests: Número máximo de requests por ventana.
            window_seconds: Duración de la ventana en segundos.
            key_prefix: Prefijo para keys en Redis.
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.key_prefix = key_prefix
        self.redis = get_redis_client()
    
    def _get_key(self, identifier: str) -> str:
        """Obtener key de Redis para un identificador."""
        return f"{self.key_prefix}:{identifier}"
    
    async def is_allowed(self, identifier: str) -> tuple[bool, Dict[str, Any]]:
        """
        Verificar si un request está permitido.
        
        Args:
            identifier: Identificador único (IP, user_id, etc.).
        
        Returns:
            Tuple (allowed, info) donde info contiene:
            - allowed: Si el request está permitido
            - remaining: Requests restantes
            - reset_time: Tiempo de reset
        """
        if not self.redis:
            # Si Redis no está disponible, permitir (fallback)
            return True, {
                "remaining": self.max_requests,
                "reset_time": int(time.time() + self.window_seconds)
            }
        
        key = self._get_key(identifier)
        now = time.time()
        window_start = now - self.window_seconds
        
        try:
            # Usar pipeline para operaciones atómicas
            pipe = self.redis.pipeline()
            
            # Remover entradas antiguas
            pipe.zremrangebyscore(key, 0, window_start)
            
            # Contar requests en la ventana
            pipe.zcard(key)
            
            # Agregar request actual
            pipe.zadd(key, {str(now): now})
            
            # Expirar key después de la ventana
            pipe.expire(key, int(self.window_seconds) + 1)
            
            results = pipe.execute()
            count = results[1]
            
            allowed = count < self.max_requests
            
            if allowed:
                # Agregar el request actual
                self.redis.zadd(key, {str(now): now})
            
            remaining = max(0, self.max_requests - count - (1 if allowed else 0))
            reset_time = int(now + self.window_seconds)
            
            return allowed, {
                "remaining": remaining,
                "reset_time": reset_time,
                "limit": self.max_requests
            }
        
        except Exception as e:
            logger.error(f"Error in rate limiter: {e}")
            # En caso de error, permitir (fail open)
            return True, {
                "remaining": self.max_requests,
                "reset_time": int(time.time() + self.window_seconds)
            }
    
    async def get_rate_limit_info(self, identifier: str) -> Dict[str, Any]:
        """Obtener información de rate limit sin consumir un request."""
        if not self.redis:
            return {
                "remaining": self.max_requests,
                "limit": self.max_requests,
                "reset_time": int(time.time() + self.window_seconds)
            }
        
        key = self._get_key(identifier)
        now = time.time()
        window_start = now - self.window_seconds
        
        try:
            # Remover entradas antiguas y contar
            count = self.redis.zremrangebyscore(key, 0, window_start)
            count = self.redis.zcard(key)
            
            remaining = max(0, self.max_requests - count)
            reset_time = int(now + self.window_seconds)
            
            return {
                "remaining": remaining,
                "limit": self.max_requests,
                "reset_time": reset_time
            }
        
        except Exception as e:
            logger.error(f"Error getting rate limit info: {e}")
            return {
                "remaining": self.max_requests,
                "limit": self.max_requests,
                "reset_time": int(time.time() + self.window_seconds)
            }


def get_client_identifier(request: Request) -> str:
    """
    Obtener identificador único del cliente.
    
    Prioridad:
    1. API Key (si está presente)
    2. User ID (si está autenticado)
    3. IP address
    """
    # API Key
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return f"api_key:{api_key}"
    
    # User ID (si está autenticado)
    if hasattr(request.state, "user"):
        return f"user:{request.state.user.username}"
    
    # IP address
    client_ip = request.client.host if request.client else "unknown"
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        client_ip = forwarded_for.split(",")[0].strip()
    
    return f"ip:{client_ip}"


async def rate_limit_check(
    request: Request,
    max_requests: int = 100,
    window_seconds: float = 60.0
) -> None:
    """
    Dependency para verificar rate limit.
    
    Raises:
        HTTPException: Si se excede el rate limit.
    """
    limiter = RedisRateLimiter(max_requests, window_seconds)
    identifier = get_client_identifier(request)
    
    allowed, info = await limiter.is_allowed(identifier)
    
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers={
                "X-RateLimit-Limit": str(info["limit"]),
                "X-RateLimit-Remaining": str(info["remaining"]),
                "X-RateLimit-Reset": str(info["reset_time"]),
                "Retry-After": str(int(info["reset_time"] - time.time()))
            }
        )
    
    # Agregar headers de rate limit
    request.state.rate_limit_info = info




