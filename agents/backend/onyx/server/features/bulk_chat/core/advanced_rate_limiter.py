"""
Advanced Rate Limiter - Rate Limiter Avanzado
=============================================

Sistema avanzado de rate limiting con múltiples estrategias, sliding window y análisis de patrones.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
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
    identifier: str  # IP, user_id, etc.
    strategy: RateLimitStrategy
    max_requests: int
    window_seconds: int
    burst_size: Optional[int] = None
    tokens_per_second: Optional[float] = None
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RateLimitState:
    """Estado de rate limiting."""
    identifier: str
    requests: deque = field(default_factory=lambda: deque())
    tokens: float = 0.0
    last_refill: datetime = field(default_factory=datetime.now)
    blocked_until: Optional[datetime] = None


class AdvancedRateLimiter:
    """Rate limiter avanzado."""
    
    def __init__(self):
        self.rules: Dict[str, RateLimitRule] = {}
        self.states: Dict[str, RateLimitState] = {}
        self.violations: List[Dict[str, Any]] = []
        self._lock = asyncio.Lock()
    
    def create_rule(
        self,
        rule_id: str,
        identifier: str,
        strategy: RateLimitStrategy,
        max_requests: int,
        window_seconds: int,
        burst_size: Optional[int] = None,
        tokens_per_second: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Crear regla de rate limiting."""
        rule = RateLimitRule(
            rule_id=rule_id,
            identifier=identifier,
            strategy=strategy,
            max_requests=max_requests,
            window_seconds=window_seconds,
            burst_size=burst_size,
            tokens_per_second=tokens_per_second,
            metadata=metadata or {},
        )
        
        async def save_rule():
            async with self._lock:
                self.rules[rule_id] = rule
                
                # Inicializar estado
                if identifier not in self.states:
                    self.states[identifier] = RateLimitState(identifier=identifier)
        
        asyncio.create_task(save_rule())
        
        logger.info(f"Created rate limit rule: {rule_id} for {identifier}")
        return rule_id
    
    async def check_rate_limit(
        self,
        identifier: str,
        rule_id: Optional[str] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Verificar rate limit."""
        async with self._lock:
            # Buscar regla
            rule = None
            if rule_id:
                rule = self.rules.get(rule_id)
            else:
                # Buscar primera regla para el identificador
                rule = next((r for r in self.rules.values() if r.identifier == identifier), None)
            
            if not rule or not rule.enabled:
                return True, {"allowed": True, "remaining": float('inf')}
            
            # Obtener o crear estado
            state_key = f"{rule.identifier}_{rule.rule_id}"
            if state_key not in self.states:
                self.states[state_key] = RateLimitState(identifier=identifier)
            
            state = self.states[state_key]
            
            # Verificar si está bloqueado
            if state.blocked_until and datetime.now() < state.blocked_until:
                return False, {
                    "allowed": False,
                    "blocked_until": state.blocked_until.isoformat(),
                    "reason": "temporarily_blocked",
                }
            
            # Aplicar estrategia
            if rule.strategy == RateLimitStrategy.FIXED_WINDOW:
                allowed, info = self._check_fixed_window(state, rule)
            elif rule.strategy == RateLimitStrategy.SLIDING_WINDOW:
                allowed, info = self._check_sliding_window(state, rule)
            elif rule.strategy == RateLimitStrategy.TOKEN_BUCKET:
                allowed, info = self._check_token_bucket(state, rule)
            elif rule.strategy == RateLimitStrategy.LEAKY_BUCKET:
                allowed, info = self._check_leaky_bucket(state, rule)
            else:
                allowed, info = True, {"allowed": True}
            
            if not allowed:
                # Registrar violación
                violation = {
                    "identifier": identifier,
                    "rule_id": rule_id,
                    "timestamp": datetime.now().isoformat(),
                    "rule": rule_id,
                }
                self.violations.append(violation)
                if len(self.violations) > 10000:
                    self.violations.pop(0)
            
            return allowed, info
    
    def _check_fixed_window(
        self,
        state: RateLimitState,
        rule: RateLimitRule,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Verificar fixed window."""
        now = datetime.now()
        window_start = now - timedelta(seconds=rule.window_seconds)
        
        # Limpiar requests fuera de la ventana
        while state.requests and state.requests[0] < window_start:
            state.requests.popleft()
        
        if len(state.requests) >= rule.max_requests:
            return False, {
                "allowed": False,
                "remaining": 0,
                "reset_at": (state.requests[0] + timedelta(seconds=rule.window_seconds)).isoformat(),
            }
        
        state.requests.append(now)
        return True, {
            "allowed": True,
            "remaining": rule.max_requests - len(state.requests),
        }
    
    def _check_sliding_window(
        self,
        state: RateLimitState,
        rule: RateLimitRule,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Verificar sliding window."""
        now = datetime.now()
        window_start = now - timedelta(seconds=rule.window_seconds)
        
        # Limpiar requests fuera de la ventana
        while state.requests and state.requests[0] < window_start:
            state.requests.popleft()
        
        if len(state.requests) >= rule.max_requests:
            return False, {
                "allowed": False,
                "remaining": 0,
                "reset_at": (state.requests[0] + timedelta(seconds=rule.window_seconds)).isoformat(),
            }
        
        state.requests.append(now)
        return True, {
            "allowed": True,
            "remaining": rule.max_requests - len(state.requests),
        }
    
    def _check_token_bucket(
        self,
        state: RateLimitState,
        rule: RateLimitRule,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Verificar token bucket."""
        now = datetime.now()
        
        # Refill tokens
        if rule.tokens_per_second:
            elapsed = (now - state.last_refill).total_seconds()
            tokens_to_add = elapsed * rule.tokens_per_second
            max_tokens = rule.max_requests
            state.tokens = min(state.tokens + tokens_to_add, max_tokens)
            state.last_refill = now
        
        if state.tokens >= 1.0:
            state.tokens -= 1.0
            return True, {
                "allowed": True,
                "remaining": int(state.tokens),
            }
        else:
            return False, {
                "allowed": False,
                "remaining": 0,
                "reset_at": (now + timedelta(seconds=1.0 / rule.tokens_per_second if rule.tokens_per_second else 1.0)).isoformat(),
            }
    
    def _check_leaky_bucket(
        self,
        state: RateLimitState,
        rule: RateLimitRule,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Verificar leaky bucket."""
        now = datetime.now()
        
        # Similar a token bucket pero con tasa de fuga constante
        if rule.tokens_per_second:
            elapsed = (now - state.last_refill).total_seconds()
            # Tokens se "fugan" a una tasa constante
            tokens_to_add = elapsed * rule.tokens_per_second
            max_tokens = rule.max_requests
            state.tokens = min(state.tokens + tokens_to_add, max_tokens)
            state.last_refill = now
        
        if state.tokens >= 1.0:
            state.tokens -= 1.0
            return True, {
                "allowed": True,
                "remaining": int(state.tokens),
            }
        else:
            return False, {
                "allowed": False,
                "remaining": 0,
            }
    
    async def block_identifier(
        self,
        identifier: str,
        duration_seconds: int,
        reason: str = "manual",
    ):
        """Bloquear identificador temporalmente."""
        async with self._lock:
            state_key = f"{identifier}_blocked"
            if state_key not in self.states:
                self.states[state_key] = RateLimitState(identifier=identifier)
            
            self.states[state_key].blocked_until = datetime.now() + timedelta(seconds=duration_seconds)
    
    def get_violations(
        self,
        identifier: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Obtener violaciones."""
        violations = self.violations
        
        if identifier:
            violations = [v for v in violations if v.get("identifier") == identifier]
        
        violations.sort(key=lambda v: v.get("timestamp", ""), reverse=True)
        return violations[:limit]
    
    def get_rate_limiter_summary(self) -> Dict[str, Any]:
        """Obtener resumen del rate limiter."""
        by_strategy: Dict[str, int] = defaultdict(int)
        
        for rule in self.rules.values():
            by_strategy[rule.strategy.value] += 1
        
        return {
            "total_rules": len(self.rules),
            "rules_by_strategy": dict(by_strategy),
            "total_states": len(self.states),
            "total_violations": len(self.violations),
        }














