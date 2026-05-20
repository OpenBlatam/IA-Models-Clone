"""
Rate Limiter V2 - Limitador de Tasa Avanzado
=============================================

Sistema avanzado de rate limiting con múltiples algoritmos, sliding windows y distributed support.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import time

logger = logging.getLogger(__name__)


class RateLimitAlgorithm(Enum):
    """Algoritmo de rate limiting."""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"
    SLIDING_LOG = "sliding_log"


@dataclass
class RateLimitRule:
    """Regla de rate limiting."""
    rule_id: str
    identifier: str
    algorithm: RateLimitAlgorithm
    limit: int
    window_seconds: float
    tokens: Optional[int] = None  # Para token bucket
    refill_rate: Optional[float] = None  # Para token bucket
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RateLimitState:
    """Estado de rate limiting."""
    identifier: str
    requests: deque = field(default_factory=deque)
    tokens: float = 0.0
    last_refill: datetime = field(default_factory=datetime.now)
    blocked_until: Optional[datetime] = None


class RateLimiterV2:
    """Limitador de tasa avanzado."""
    
    def __init__(self):
        self.rules: Dict[str, RateLimitRule] = {}
        self.states: Dict[str, RateLimitState] = {}
        self.block_history: deque = deque(maxlen=10000)
        self._lock = asyncio.Lock()
    
    def add_rule(
        self,
        rule_id: str,
        identifier: str,
        algorithm: RateLimitAlgorithm,
        limit: int,
        window_seconds: float,
        tokens: Optional[int] = None,
        refill_rate: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Agregar regla de rate limiting."""
        rule = RateLimitRule(
            rule_id=rule_id,
            identifier=identifier,
            algorithm=algorithm,
            limit=limit,
            window_seconds=window_seconds,
            tokens=tokens,
            refill_rate=refill_rate,
            metadata=metadata or {},
        )
        
        async def save_rule():
            async with self._lock:
                self.rules[rule_id] = rule
        
        asyncio.create_task(save_rule())
        
        logger.info(f"Added rate limit rule: {rule_id}")
        return rule_id
    
    async def check_rate_limit(
        self,
        identifier: str,
        rule_id: Optional[str] = None,
    ) -> tuple[bool, Optional[Dict[str, Any]]]:
        """Verificar rate limit."""
        # Buscar regla aplicable
        rule = None
        if rule_id:
            rule = self.rules.get(rule_id)
        else:
            # Buscar primera regla que coincida con identifier
            for r in self.rules.values():
                if r.identifier == identifier or r.identifier == "*":
                    rule = r
                    break
        
        if not rule or not rule.enabled:
            return True, None
        
        # Obtener o crear estado
        state_key = f"{rule.rule_id}_{identifier}"
        async with self._lock:
            if state_key not in self.states:
                self.states[state_key] = RateLimitState(
                    identifier=identifier,
                    tokens=rule.tokens or rule.limit,
                )
            state = self.states[state_key]
        
        # Verificar si está bloqueado
        if state.blocked_until and datetime.now() < state.blocked_until:
            remaining = (state.blocked_until - datetime.now()).total_seconds()
            return False, {
                "allowed": False,
                "reason": "rate_limit_exceeded",
                "retry_after": remaining,
                "rule_id": rule.rule_id,
            }
        
        # Evaluar según algoritmo
        allowed = await self._evaluate_algorithm(rule, state, identifier)
        
        if not allowed:
            # Bloquear por un período
            state.blocked_until = datetime.now() + timedelta(seconds=rule.window_seconds)
            
            async with self._lock:
                self.block_history.append({
                    "identifier": identifier,
                    "rule_id": rule.rule_id,
                    "timestamp": datetime.now().isoformat(),
                })
            
            return False, {
                "allowed": False,
                "reason": "rate_limit_exceeded",
                "retry_after": rule.window_seconds,
                "rule_id": rule.rule_id,
            }
        
        return True, {
            "allowed": True,
            "remaining": self._calculate_remaining(rule, state),
            "rule_id": rule.rule_id,
        }
    
    async def _evaluate_algorithm(
        self,
        rule: RateLimitRule,
        state: RateLimitState,
        identifier: str,
    ) -> bool:
        """Evaluar algoritmo de rate limiting."""
        now = datetime.now()
        
        if rule.algorithm == RateLimitAlgorithm.FIXED_WINDOW:
            # Limpiar requests fuera de la ventana
            window_start = now - timedelta(seconds=rule.window_seconds)
            while state.requests and state.requests[0] < window_start:
                state.requests.popleft()
            
            if len(state.requests) >= rule.limit:
                return False
            
            state.requests.append(now)
            return True
        
        elif rule.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
            # Limpiar requests fuera de la ventana
            window_start = now - timedelta(seconds=rule.window_seconds)
            while state.requests and state.requests[0] < window_start:
                state.requests.popleft()
            
            if len(state.requests) >= rule.limit:
                return False
            
            state.requests.append(now)
            return True
        
        elif rule.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
            # Refill tokens
            if rule.refill_rate:
                time_passed = (now - state.last_refill).total_seconds()
                tokens_to_add = time_passed * rule.refill_rate
                state.tokens = min(rule.tokens or rule.limit, state.tokens + tokens_to_add)
                state.last_refill = now
            
            if state.tokens < 1:
                return False
            
            state.tokens -= 1
            return True
        
        elif rule.algorithm == RateLimitAlgorithm.LEAKY_BUCKET:
            # Similar a token bucket pero con leak rate
            if rule.refill_rate:
                time_passed = (now - state.last_refill).total_seconds()
                tokens_to_remove = time_passed * rule.refill_rate
                state.tokens = max(0, state.tokens - tokens_to_remove)
                state.last_refill = now
            
            if len(state.requests) >= rule.limit:
                return False
            
            state.requests.append(now)
            return True
        
        elif rule.algorithm == RateLimitAlgorithm.SLIDING_LOG:
            # Log de todos los requests
            window_start = now - timedelta(seconds=rule.window_seconds)
            while state.requests and state.requests[0] < window_start:
                state.requests.popleft()
            
            if len(state.requests) >= rule.limit:
                return False
            
            state.requests.append(now)
            return True
        
        return False
    
    def _calculate_remaining(self, rule: RateLimitRule, state: RateLimitState) -> int:
        """Calcular requests restantes."""
        if rule.algorithm in [RateLimitAlgorithm.FIXED_WINDOW, RateLimitAlgorithm.SLIDING_WINDOW, RateLimitAlgorithm.SLIDING_LOG]:
            return max(0, rule.limit - len(state.requests))
        elif rule.algorithm in [RateLimitAlgorithm.TOKEN_BUCKET, RateLimitAlgorithm.LEAKY_BUCKET]:
            return int(state.tokens)
        return 0
    
    def get_rate_limit_status(self, identifier: str, rule_id: Optional[str] = None) -> Dict[str, Any]:
        """Obtener estado de rate limiting."""
        rule = None
        if rule_id:
            rule = self.rules.get(rule_id)
        else:
            for r in self.rules.values():
                if r.identifier == identifier or r.identifier == "*":
                    rule = r
                    break
        
        if not rule:
            return {"status": "no_rule"}
        
        state_key = f"{rule.rule_id}_{identifier}"
        state = self.states.get(state_key)
        
        if not state:
            return {
                "status": "allowed",
                "remaining": rule.limit,
                "rule_id": rule.rule_id,
            }
        
        remaining = self._calculate_remaining(rule, state)
        blocked = state.blocked_until and datetime.now() < state.blocked_until
        
        return {
            "status": "blocked" if blocked else "allowed",
            "remaining": remaining,
            "blocked_until": state.blocked_until.isoformat() if state.blocked_until else None,
            "rule_id": rule.rule_id,
        }
    
    def get_block_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Obtener historial de bloqueos."""
        return list(self.block_history)[-limit:]
    
    def get_rate_limiter_v2_summary(self) -> Dict[str, Any]:
        """Obtener resumen del limitador."""
        by_algorithm: Dict[str, int] = defaultdict(int)
        
        for rule in self.rules.values():
            by_algorithm[rule.algorithm.value] += 1
        
        return {
            "total_rules": len(self.rules),
            "active_rules": len([r for r in self.rules.values() if r.enabled]),
            "rules_by_algorithm": dict(by_algorithm),
            "total_states": len(self.states),
            "total_blocks": len(self.block_history),
        }


