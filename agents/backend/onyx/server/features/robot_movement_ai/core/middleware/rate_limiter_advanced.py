"""
Advanced Rate Limiter System
=============================

Sistema avanzado de rate limiting con múltiples estrategias.
"""

import logging
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class RateLimitStrategy(Enum):
    """Estrategia de rate limiting."""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"


@dataclass
class RateLimitRule:
    """Regla de rate limiting."""
    rule_id: str
    key: str  # Clave para identificar el límite (usuario, IP, etc.)
    limit: int  # Número máximo de requests
    window: float  # Ventana de tiempo en segundos
    strategy: RateLimitStrategy = RateLimitStrategy.FIXED_WINDOW
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RateLimitResult:
    """Resultado de rate limiting."""
    allowed: bool
    limit: int
    remaining: int
    reset_at: float
    retry_after: Optional[float] = None


class AdvancedRateLimiter:
    """
    Rate limiter avanzado.
    
    Gestiona rate limiting con múltiples estrategias.
    """
    
    def __init__(self):
        """Inicializar rate limiter."""
        self.rules: Dict[str, RateLimitRule] = {}
        self.counters: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.tokens: Dict[str, Dict[str, Any]] = defaultdict(dict)  # Para token bucket
        self.windows: Dict[str, deque] = defaultdict(deque)  # Para sliding window
    
    def add_rule(
        self,
        rule_id: str,
        key: str,
        limit: int,
        window: float,
        strategy: RateLimitStrategy = RateLimitStrategy.FIXED_WINDOW,
        metadata: Optional[Dict[str, Any]] = None
    ) -> RateLimitRule:
        """
        Agregar regla de rate limiting.
        
        Args:
            rule_id: ID único de la regla
            key: Clave para identificar el límite
            limit: Límite de requests
            window: Ventana de tiempo en segundos
            strategy: Estrategia
            metadata: Metadata adicional
            
        Returns:
            Regla creada
        """
        rule = RateLimitRule(
            rule_id=rule_id,
            key=key,
            limit=limit,
            window=window,
            strategy=strategy,
            metadata=metadata or {}
        )
        
        self.rules[rule_id] = rule
        logger.info(f"Added rate limit rule: {rule_id} ({strategy.value})")
        
        return rule
    
    def check_limit(
        self,
        rule_id: str,
        identifier: str
    ) -> RateLimitResult:
        """
        Verificar límite.
        
        Args:
            rule_id: ID de la regla
            identifier: Identificador (usuario, IP, etc.)
            
        Returns:
            Resultado del rate limiting
        """
        if rule_id not in self.rules:
            # Sin límite
            return RateLimitResult(
                allowed=True,
                limit=0,
                remaining=0,
                reset_at=time.time() + 3600
            )
        
        rule = self.rules[rule_id]
        key = f"{rule.key}:{identifier}"
        now = time.time()
        
        if rule.strategy == RateLimitStrategy.FIXED_WINDOW:
            return self._check_fixed_window(rule, key, now)
        elif rule.strategy == RateLimitStrategy.SLIDING_WINDOW:
            return self._check_sliding_window(rule, key, now)
        elif rule.strategy == RateLimitStrategy.TOKEN_BUCKET:
            return self._check_token_bucket(rule, key, now)
        elif rule.strategy == RateLimitStrategy.LEAKY_BUCKET:
            return self._check_leaky_bucket(rule, key, now)
        
        return RateLimitResult(
            allowed=True,
            limit=rule.limit,
            remaining=rule.limit,
            reset_at=now + rule.window
        )
    
    def _check_fixed_window(
        self,
        rule: RateLimitRule,
        key: str,
        now: float
    ) -> RateLimitResult:
        """Verificar límite con ventana fija."""
        counter_key = f"{key}:window"
        window_start = int(now / rule.window) * rule.window
        
        if counter_key not in self.counters[key]:
            self.counters[key][counter_key] = {
                "count": 0,
                "window_start": window_start
            }
        
        counter = self.counters[key][counter_key]
        
        # Resetear si nueva ventana
        if counter["window_start"] < window_start:
            counter["count"] = 0
            counter["window_start"] = window_start
        
        count = counter["count"]
        allowed = count < rule.limit
        
        if allowed:
            counter["count"] += 1
            count += 1
        
        return RateLimitResult(
            allowed=allowed,
            limit=rule.limit,
            remaining=max(0, rule.limit - count),
            reset_at=window_start + rule.window,
            retry_after=(window_start + rule.window - now) if not allowed else None
        )
    
    def _check_sliding_window(
        self,
        rule: RateLimitRule,
        key: str,
        now: float
    ) -> RateLimitResult:
        """Verificar límite con ventana deslizante."""
        window = self.windows[key]
        cutoff = now - rule.window
        
        # Limpiar timestamps antiguos
        while window and window[0] < cutoff:
            window.popleft()
        
        count = len(window)
        allowed = count < rule.limit
        
        if allowed:
            window.append(now)
            count += 1
        
        return RateLimitResult(
            allowed=allowed,
            limit=rule.limit,
            remaining=max(0, rule.limit - count),
            reset_at=now + rule.window,
            retry_after=(window[0] + rule.window - now) if not allowed and window else None
        )
    
    def _check_token_bucket(
        self,
        rule: RateLimitRule,
        key: str,
        now: float
    ) -> RateLimitResult:
        """Verificar límite con token bucket."""
        if key not in self.tokens:
            self.tokens[key] = {
                "tokens": rule.limit,
                "last_refill": now
            }
        
        bucket = self.tokens[key]
        tokens = bucket["tokens"]
        last_refill = bucket["last_refill"]
        
        # Refill tokens
        elapsed = now - last_refill
        refill_rate = rule.limit / rule.window
        tokens = min(rule.limit, tokens + elapsed * refill_rate)
        bucket["tokens"] = tokens
        bucket["last_refill"] = now
        
        allowed = tokens >= 1.0
        
        if allowed:
            bucket["tokens"] = tokens - 1.0
            tokens -= 1.0
        
        return RateLimitResult(
            allowed=allowed,
            limit=rule.limit,
            remaining=int(tokens),
            reset_at=now + rule.window,
            retry_after=((1.0 - tokens) / refill_rate) if not allowed else None
        )
    
    def _check_leaky_bucket(
        self,
        rule: RateLimitRule,
        key: str,
        now: float
    ) -> RateLimitResult:
        """Verificar límite con leaky bucket."""
        if key not in self.tokens:
            self.tokens[key] = {
                "level": 0,
                "last_leak": now
            }
        
        bucket = self.tokens[key]
        level = bucket["level"]
        last_leak = bucket["last_leak"]
        
        # Leak tokens
        elapsed = now - last_leak
        leak_rate = rule.limit / rule.window
        level = max(0, level - elapsed * leak_rate)
        bucket["level"] = level
        bucket["last_leak"] = now
        
        allowed = level < rule.limit
        
        if allowed:
            bucket["level"] = level + 1
            level += 1
        
        return RateLimitResult(
            allowed=allowed,
            limit=rule.limit,
            remaining=max(0, rule.limit - int(level)),
            reset_at=now + rule.window,
            retry_after=((level - rule.limit + 1) / leak_rate) if not allowed else None
        )
    
    def get_rule(self, rule_id: str) -> Optional[RateLimitRule]:
        """Obtener regla por ID."""
        return self.rules.get(rule_id)
    
    def list_rules(self) -> List[RateLimitRule]:
        """Listar todas las reglas."""
        return list(self.rules.values())


# Instancia global
_advanced_rate_limiter: Optional[AdvancedRateLimiter] = None


def get_advanced_rate_limiter() -> AdvancedRateLimiter:
    """Obtener instancia global del rate limiter avanzado."""
    global _advanced_rate_limiter
    if _advanced_rate_limiter is None:
        _advanced_rate_limiter = AdvancedRateLimiter()
    return _advanced_rate_limiter






