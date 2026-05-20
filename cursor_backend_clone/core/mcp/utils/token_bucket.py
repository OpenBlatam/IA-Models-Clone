"""
MCP Token Bucket - Rate Limiting con Token Bucket
==================================================

Implementación de rate limiting usando algoritmo Token Bucket.
"""

import time
import asyncio
import logging
from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class TokenBucket:
    """Token bucket para rate limiting"""
    capacity: float
    refill_rate: float
    tokens: float
    last_refill: float
    
    def __init__(self, capacity: float, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()
    
    def refill(self):
        """Recargar tokens"""
        now = time.time()
        elapsed = now - self.last_refill
        tokens_to_add = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now
    
    def consume(self, tokens: float = 1.0) -> bool:
        """Consumir tokens"""
        self.refill()
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False
    
    def available_tokens(self) -> float:
        """Obtener tokens disponibles"""
        self.refill()
        return self.tokens
    
    def time_until_next_token(self) -> float:
        """Tiempo hasta el próximo token disponible"""
        if self.tokens >= 1.0:
            return 0.0
        return (1.0 - self.tokens) / self.refill_rate


class TokenBucketRateLimiter:
    """Rate limiter usando Token Bucket algorithm"""
    
    def __init__(
        self,
        capacity: float = 100.0,
        refill_rate: float = 1.0,
        tokens_per_request: float = 1.0
    ):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens_per_request = tokens_per_request
        self._buckets: Dict[str, TokenBucket] = {}
        self._lock = asyncio.Lock()
    
    async def is_allowed(self, client_id: str = "default", tokens: Optional[float] = None) -> bool:
        """Verificar si el request está permitido"""
        async with self._lock:
            if client_id not in self._buckets:
                self._buckets[client_id] = TokenBucket(
                    capacity=self.capacity,
                    refill_rate=self.refill_rate
                )
            
            bucket = self._buckets[client_id]
            tokens_to_consume = tokens or self.tokens_per_request
            return bucket.consume(tokens_to_consume)
    
    async def get_retry_after(self, client_id: str = "default") -> float:
        """Obtener tiempo hasta el próximo request permitido"""
        async with self._lock:
            if client_id not in self._buckets:
                return 0.0
            
            bucket = self._buckets[client_id]
            return bucket.time_until_next_token()
    
    async def reset(self, client_id: Optional[str] = None):
        """Resetear bucket(s)"""
        async with self._lock:
            if client_id:
                if client_id in self._buckets:
                    del self._buckets[client_id]
            else:
                self._buckets.clear()
    
    async def get_stats(self, client_id: str = "default") -> Dict[str, Any]:
        """Obtener estadísticas del bucket"""
        async with self._lock:
            if client_id not in self._buckets:
                return {
                    "available_tokens": self.capacity,
                    "capacity": self.capacity,
                    "refill_rate": self.refill_rate
                }
            
            bucket = self._buckets[client_id]
            return {
                "available_tokens": bucket.available_tokens(),
                "capacity": bucket.capacity,
                "refill_rate": bucket.refill_rate,
                "time_until_next_token": bucket.time_until_next_token()
            }

