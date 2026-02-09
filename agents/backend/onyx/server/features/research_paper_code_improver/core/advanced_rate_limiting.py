"""
Advanced Rate Limiting - Rate limiting avanzado con múltiples algoritmos
=========================================================================
"""

import logging
import time
from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque

logger = logging.getLogger(__name__)


class RateLimitAlgorithm(Enum):
    """Algoritmos de rate limiting"""
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"


@dataclass
class RateLimitRule:
    """Regla de rate limiting"""
    identifier: str  # user_id, ip, etc.
    algorithm: RateLimitAlgorithm
    limit: int  # Número de requests permitidos
    window_seconds: int  # Ventana de tiempo en segundos
    burst: Optional[int] = None  # Para token bucket
    refill_rate: Optional[float] = None  # Para token bucket


@dataclass
class RateLimitResult:
    """Resultado de rate limiting"""
    allowed: bool
    limit: int
    remaining: int
    reset_time: datetime
    retry_after: Optional[float] = None  # Segundos hasta el próximo intento


class TokenBucket:
    """Token Bucket para rate limiting"""
    
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate  # tokens por segundo
        self.tokens = float(capacity)
        self.last_refill = time.time()
    
    def consume(self, tokens: int = 1) -> bool:
        """Consume tokens"""
        self._refill()
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False
    
    def _refill(self):
        """Refill tokens"""
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now
    
    def get_retry_after(self) -> float:
        """Calcula tiempo hasta que haya tokens disponibles"""
        if self.tokens >= 1:
            return 0.0
        needed = 1 - self.tokens
        return needed / self.refill_rate


class LeakyBucket:
    """Leaky Bucket para rate limiting"""
    
    def __init__(self, capacity: int, leak_rate: float):
        self.capacity = capacity
        self.leak_rate = leak_rate  # requests por segundo que salen
        self.queue = deque()
        self.last_leak = time.time()
    
    def add(self) -> bool:
        """Agrega un request"""
        self._leak()
        if len(self.queue) < self.capacity:
            self.queue.append(time.time())
            return True
        return False
    
    def _leak(self):
        """Aplica leak"""
        now = time.time()
        elapsed = now - self.last_leak
        leak_count = int(elapsed * self.leak_rate)
        
        for _ in range(min(leak_count, len(self.queue))):
            self.queue.popleft()
        
        self.last_leak = now
    
    def get_retry_after(self) -> float:
        """Calcula tiempo hasta que haya espacio"""
        if len(self.queue) < self.capacity:
            return 0.0
        oldest = self.queue[0]
        return (1.0 / self.leak_rate) - (time.time() - oldest)


class SlidingWindow:
    """Sliding Window para rate limiting"""
    
    def __init__(self, limit: int, window_seconds: int):
        self.limit = limit
        self.window_seconds = window_seconds
        self.requests = deque()
    
    def add(self) -> bool:
        """Agrega un request"""
        now = time.time()
        cutoff = now - self.window_seconds
        
        # Remover requests antiguos
        while self.requests and self.requests[0] < cutoff:
            self.requests.popleft()
        
        if len(self.requests) < self.limit:
            self.requests.append(now)
            return True
        return False
    
    def get_retry_after(self) -> float:
        """Calcula tiempo hasta que expire el request más antiguo"""
        if len(self.requests) < self.limit:
            return 0.0
        oldest = self.requests[0]
        return self.window_seconds - (time.time() - oldest)


class FixedWindow:
    """Fixed Window para rate limiting"""
    
    def __init__(self, limit: int, window_seconds: int):
        self.limit = limit
        self.window_seconds = window_seconds
        self.count = 0
        self.window_start = time.time()
    
    def add(self) -> bool:
        """Agrega un request"""
        now = time.time()
        
        # Resetear ventana si expiró
        if now - self.window_start >= self.window_seconds:
            self.count = 0
            self.window_start = now
        
        if self.count < self.limit:
            self.count += 1
            return True
        return False
    
    def get_retry_after(self) -> float:
        """Calcula tiempo hasta el próximo reset"""
        now = time.time()
        elapsed = now - self.window_start
        return max(0, self.window_seconds - elapsed)


class AdvancedRateLimiter:
    """Rate limiter avanzado con múltiples algoritmos"""
    
    def __init__(self):
        self.rules: Dict[str, RateLimitRule] = {}
        self.buckets: Dict[str, Any] = {}  # identifier -> bucket/window
    
    def add_rule(self, rule: RateLimitRule):
        """Agrega una regla de rate limiting"""
        self.rules[rule.identifier] = rule
        
        # Crear bucket/window según algoritmo
        if rule.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
            capacity = rule.burst or rule.limit
            refill_rate = rule.refill_rate or (rule.limit / rule.window_seconds)
            self.buckets[rule.identifier] = TokenBucket(capacity, refill_rate)
        
        elif rule.algorithm == RateLimitAlgorithm.LEAKY_BUCKET:
            leak_rate = rule.limit / rule.window_seconds
            self.buckets[rule.identifier] = LeakyBucket(rule.limit, leak_rate)
        
        elif rule.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
            self.buckets[rule.identifier] = SlidingWindow(rule.limit, rule.window_seconds)
        
        elif rule.algorithm == RateLimitAlgorithm.FIXED_WINDOW:
            self.buckets[rule.identifier] = FixedWindow(rule.limit, rule.window_seconds)
    
    def check_rate_limit(
        self,
        identifier: str,
        default_limit: Optional[int] = None,
        default_window: Optional[int] = None
    ) -> RateLimitResult:
        """Verifica rate limit"""
        rule = self.rules.get(identifier)
        
        if not rule:
            # Usar defaults si no hay regla específica
            if default_limit and default_window:
                rule = RateLimitRule(
                    identifier=identifier,
                    algorithm=RateLimitAlgorithm.SLIDING_WINDOW,
                    limit=default_limit,
                    window_seconds=default_window
                )
                self.add_rule(rule)
            else:
                # Sin límite
                return RateLimitResult(
                    allowed=True,
                    limit=0,
                    remaining=0,
                    reset_time=datetime.now()
                )
        
        bucket = self.buckets.get(identifier)
        if not bucket:
            return RateLimitResult(
                allowed=True,
                limit=rule.limit,
                remaining=rule.limit,
                reset_time=datetime.now() + timedelta(seconds=rule.window_seconds)
            )
        
        # Verificar según algoritmo
        allowed = False
        remaining = 0
        
        if isinstance(bucket, TokenBucket):
            allowed = bucket.consume()
            remaining = int(bucket.tokens)
        elif isinstance(bucket, (LeakyBucket, SlidingWindow, FixedWindow)):
            allowed = bucket.add()
            if isinstance(bucket, SlidingWindow):
                remaining = rule.limit - len(bucket.requests)
            elif isinstance(bucket, FixedWindow):
                remaining = rule.limit - bucket.count
            else:
                remaining = rule.limit - len(bucket.queue)
        
        reset_time = datetime.now() + timedelta(seconds=rule.window_seconds)
        retry_after = None
        
        if not allowed:
            if hasattr(bucket, 'get_retry_after'):
                retry_after = bucket.get_retry_after()
        
        return RateLimitResult(
            allowed=allowed,
            limit=rule.limit,
            remaining=max(0, remaining),
            reset_time=reset_time,
            retry_after=retry_after
        )
    
    def get_rate_limit_info(self, identifier: str) -> Optional[Dict[str, Any]]:
        """Obtiene información de rate limit"""
        rule = self.rules.get(identifier)
        if not rule:
            return None
        
        bucket = self.buckets.get(identifier)
        remaining = 0
        
        if isinstance(bucket, TokenBucket):
            remaining = int(bucket.tokens)
        elif isinstance(bucket, SlidingWindow):
            remaining = rule.limit - len(bucket.requests)
        elif isinstance(bucket, FixedWindow):
            remaining = rule.limit - bucket.count
        elif isinstance(bucket, LeakyBucket):
            remaining = rule.limit - len(bucket.queue)
        
        return {
            "identifier": identifier,
            "algorithm": rule.algorithm.value,
            "limit": rule.limit,
            "window_seconds": rule.window_seconds,
            "remaining": max(0, remaining),
            "used": rule.limit - max(0, remaining)
        }




