"""
Advanced Rate Limiting Service - Rate limiting avanzado
=======================================================

Sistema de rate limiting avanzado con múltiples estrategias.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum

logger = logging.getLogger(__name__)


class RateLimitStrategy(str, Enum):
    """Estrategias de rate limiting"""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"


@dataclass
class RateLimitRule:
    """Regla de rate limiting"""
    identifier: str  # user_id, ip, etc.
    strategy: RateLimitStrategy
    limit: int
    window_seconds: int
    current_count: int = 0
    reset_at: datetime = field(default_factory=datetime.now)
    tokens: int = 0  # Para token bucket


@dataclass
class RateLimitResult:
    """Resultado de rate limiting"""
    allowed: bool
    limit: int
    remaining: int
    reset_at: datetime
    retry_after: Optional[int] = None  # segundos


class AdvancedRateLimitingService:
    """Servicio de rate limiting avanzado"""
    
    def __init__(self):
        """Inicializar servicio"""
        self.rules: Dict[str, RateLimitRule] = {}
        self.request_history: Dict[str, List[datetime]] = defaultdict(list)
        logger.info("AdvancedRateLimitingService initialized")
    
    def create_rate_limit(
        self,
        identifier: str,
        strategy: RateLimitStrategy,
        limit: int,
        window_seconds: int
    ) -> RateLimitRule:
        """Crear regla de rate limiting"""
        rule = RateLimitRule(
            identifier=identifier,
            strategy=strategy,
            limit=limit,
            window_seconds=window_seconds,
            tokens=limit if strategy == RateLimitStrategy.TOKEN_BUCKET else 0,
            reset_at=datetime.now() + timedelta(seconds=window_seconds),
        )
        
        self.rules[identifier] = rule
        
        logger.info(f"Rate limit created for {identifier}: {limit}/{window_seconds}s")
        return rule
    
    def check_rate_limit(
        self,
        identifier: str,
        endpoint: Optional[str] = None
    ) -> RateLimitResult:
        """Verificar rate limit"""
        rule_key = f"{identifier}:{endpoint}" if endpoint else identifier
        rule = self.rules.get(rule_key)
        
        if not rule:
            # Regla por defecto
            rule = self.create_rate_limit(
                identifier=rule_key,
                strategy=RateLimitStrategy.FIXED_WINDOW,
                limit=100,
                window_seconds=60,
            )
        
        now = datetime.now()
        
        # Verificar si necesita reset
        if now >= rule.reset_at:
            self._reset_rule(rule)
        
        # Aplicar estrategia
        if rule.strategy == RateLimitStrategy.FIXED_WINDOW:
            return self._check_fixed_window(rule, now)
        elif rule.strategy == RateLimitStrategy.SLIDING_WINDOW:
            return self._check_sliding_window(rule, identifier, now)
        elif rule.strategy == RateLimitStrategy.TOKEN_BUCKET:
            return self._check_token_bucket(rule, now)
        elif rule.strategy == RateLimitStrategy.LEAKY_BUCKET:
            return self._check_leaky_bucket(rule, now)
        
        # Default: permitir
        return RateLimitResult(
            allowed=True,
            limit=rule.limit,
            remaining=rule.limit - rule.current_count,
            reset_at=rule.reset_at,
        )
    
    def _reset_rule(self, rule: RateLimitRule):
        """Resetear regla"""
        rule.current_count = 0
        rule.reset_at = datetime.now() + timedelta(seconds=rule.window_seconds)
        if rule.strategy == RateLimitStrategy.TOKEN_BUCKET:
            rule.tokens = rule.limit
    
    def _check_fixed_window(self, rule: RateLimitRule, now: datetime) -> RateLimitResult:
        """Verificar fixed window"""
        if rule.current_count < rule.limit:
            rule.current_count += 1
            return RateLimitResult(
                allowed=True,
                limit=rule.limit,
                remaining=rule.limit - rule.current_count,
                reset_at=rule.reset_at,
            )
        else:
            retry_after = int((rule.reset_at - now).total_seconds())
            return RateLimitResult(
                allowed=False,
                limit=rule.limit,
                remaining=0,
                reset_at=rule.reset_at,
                retry_after=retry_after,
            )
    
    def _check_sliding_window(
        self,
        rule: RateLimitRule,
        identifier: str,
        now: datetime
    ) -> RateLimitResult:
        """Verificar sliding window"""
        window_start = now - timedelta(seconds=rule.window_seconds)
        
        # Limpiar requests fuera de la ventana
        history = self.request_history[identifier]
        history[:] = [req_time for req_time in history if req_time >= window_start]
        
        if len(history) < rule.limit:
            history.append(now)
            rule.current_count = len(history)
            return RateLimitResult(
                allowed=True,
                limit=rule.limit,
                remaining=rule.limit - len(history),
                reset_at=now + timedelta(seconds=rule.window_seconds),
            )
        else:
            oldest_request = min(history)
            retry_after = int((oldest_request + timedelta(seconds=rule.window_seconds) - now).total_seconds())
            return RateLimitResult(
                allowed=False,
                limit=rule.limit,
                remaining=0,
                reset_at=oldest_request + timedelta(seconds=rule.window_seconds),
                retry_after=retry_after,
            )
    
    def _check_token_bucket(self, rule: RateLimitRule, now: datetime) -> RateLimitResult:
        """Verificar token bucket"""
        # Recargar tokens (simplificado: 1 token por segundo)
        time_passed = (now - (rule.reset_at - timedelta(seconds=rule.window_seconds))).total_seconds()
        tokens_to_add = int(time_passed)
        rule.tokens = min(rule.limit, rule.tokens + tokens_to_add)
        
        if rule.tokens > 0:
            rule.tokens -= 1
            return RateLimitResult(
                allowed=True,
                limit=rule.limit,
                remaining=int(rule.tokens),
                reset_at=rule.reset_at,
            )
        else:
            return RateLimitResult(
                allowed=False,
                limit=rule.limit,
                remaining=0,
                reset_at=rule.reset_at,
                retry_after=1,
            )
    
    def _check_leaky_bucket(self, rule: RateLimitRule, now: datetime) -> RateLimitResult:
        """Verificar leaky bucket"""
        # Similar a token bucket pero con tasa de fuga
        if rule.current_count < rule.limit:
            rule.current_count += 1
            return RateLimitResult(
                allowed=True,
                limit=rule.limit,
                remaining=rule.limit - rule.current_count,
                reset_at=rule.reset_at,
            )
        else:
            return RateLimitResult(
                allowed=False,
                limit=rule.limit,
                remaining=0,
                reset_at=rule.reset_at,
                retry_after=1,
            )
    
    def get_rate_limit_status(self, identifier: str) -> Dict[str, Any]:
        """Obtener estado de rate limit"""
        rule = self.rules.get(identifier)
        if not rule:
            return {"identifier": identifier, "rule_exists": False}
        
        return {
            "identifier": identifier,
            "strategy": rule.strategy.value,
            "limit": rule.limit,
            "current_count": rule.current_count,
            "remaining": rule.limit - rule.current_count,
            "window_seconds": rule.window_seconds,
            "reset_at": rule.reset_at.isoformat(),
        }




