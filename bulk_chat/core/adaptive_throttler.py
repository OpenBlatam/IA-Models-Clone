"""
Adaptive Throttler - Limitador Adaptativo
==========================================

Sistema de throttling adaptativo que ajusta límites dinámicamente basado en condiciones del sistema.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import statistics

logger = logging.getLogger(__name__)


class ThrottleStrategy(Enum):
    """Estrategia de throttling."""
    ADAPTIVE = "adaptive"
    FIXED = "fixed"
    GRADUAL = "gradual"
    EXPONENTIAL = "exponential"


@dataclass
class ThrottleRule:
    """Regla de throttling."""
    rule_id: str
    identifier: str
    base_limit: int
    strategy: ThrottleStrategy
    min_limit: int = 1
    max_limit: int = 1000
    adjustment_factor: float = 0.1
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ThrottleMetric:
    """Métrica para throttling."""
    timestamp: datetime
    success_rate: float = 1.0
    error_rate: float = 0.0
    response_time: float = 0.0
    queue_size: int = 0
    system_load: float = 0.0


class AdaptiveThrottler:
    """Limitador adaptativo."""
    
    def __init__(self):
        self.rules: Dict[str, ThrottleRule] = {}
        self.current_limits: Dict[str, int] = {}
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.block_history: deque = deque(maxlen=10000)
        self._lock = asyncio.Lock()
    
    def add_rule(
        self,
        rule_id: str,
        identifier: str,
        base_limit: int,
        strategy: ThrottleStrategy = ThrottleStrategy.ADAPTIVE,
        min_limit: int = 1,
        max_limit: int = 1000,
        adjustment_factor: float = 0.1,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Agregar regla de throttling."""
        rule = ThrottleRule(
            rule_id=rule_id,
            identifier=identifier,
            base_limit=base_limit,
            strategy=strategy,
            min_limit=min_limit,
            max_limit=max_limit,
            adjustment_factor=adjustment_factor,
            metadata=metadata or {},
        )
        
        async def save_rule():
            async with self._lock:
                self.rules[rule_id] = rule
                self.current_limits[f"{rule_id}_{identifier}"] = base_limit
        
        asyncio.create_task(save_rule())
        
        logger.info(f"Added throttle rule: {rule_id}")
        return rule_id
    
    def record_metric(
        self,
        rule_id: str,
        success_rate: Optional[float] = None,
        error_rate: Optional[float] = None,
        response_time: Optional[float] = None,
        queue_size: Optional[int] = None,
        system_load: Optional[float] = None,
    ):
        """Registrar métrica para throttling."""
        metric = ThrottleMetric(
            timestamp=datetime.now(),
            success_rate=success_rate or 1.0,
            error_rate=error_rate or 0.0,
            response_time=response_time or 0.0,
            queue_size=queue_size or 0,
            system_load=system_load or 0.0,
        )
        
        async def save_metric():
            async with self._lock:
                self.metrics[rule_id].append(metric)
        
        asyncio.create_task(save_metric())
        
        # Ajustar límite basado en métricas
        asyncio.create_task(self._adjust_limit(rule_id))
    
    async def _adjust_limit(self, rule_id: str):
        """Ajustar límite basado en métricas."""
        rule = self.rules.get(rule_id)
        if not rule or not rule.enabled:
            return
        
        metric_history = self.metrics.get(rule_id, deque())
        if not metric_history:
            return
        
        # Calcular métricas promedio
        recent = list(metric_history)[-10:]
        if not recent:
            return
        
        avg_success_rate = statistics.mean([m.success_rate for m in recent])
        avg_error_rate = statistics.mean([m.error_rate for m in recent])
        avg_response_time = statistics.mean([m.response_time for m in recent])
        avg_queue_size = statistics.mean([m.queue_size for m in recent])
        avg_system_load = statistics.mean([m.system_load for m in recent])
        
        # Obtener límite actual
        limit_key = f"{rule_id}_{rule.identifier}"
        current_limit = self.current_limits.get(limit_key, rule.base_limit)
        new_limit = current_limit
        
        if rule.strategy == ThrottleStrategy.ADAPTIVE:
            # Ajustar basado en múltiples factores
            if avg_error_rate > 0.1 or avg_system_load > 0.8:
                # Reducir límite
                adjustment = -rule.adjustment_factor * current_limit
            elif avg_success_rate > 0.95 and avg_system_load < 0.5:
                # Aumentar límite
                adjustment = rule.adjustment_factor * current_limit
            else:
                adjustment = 0
            
            new_limit = int(current_limit + adjustment)
        
        elif rule.strategy == ThrottleStrategy.GRADUAL:
            # Ajuste gradual basado en error rate
            if avg_error_rate > 0.05:
                new_limit = int(current_limit * (1 - rule.adjustment_factor))
            elif avg_error_rate < 0.01:
                new_limit = int(current_limit * (1 + rule.adjustment_factor))
        
        elif rule.strategy == ThrottleStrategy.EXPONENTIAL:
            # Ajuste exponencial
            if avg_error_rate > 0.1:
                new_limit = int(current_limit * 0.5)
            elif avg_error_rate < 0.01:
                new_limit = int(current_limit * 1.5)
        
        # Aplicar límites min/max
        new_limit = max(rule.min_limit, min(rule.max_limit, new_limit))
        
        async with self._lock:
            self.current_limits[limit_key] = new_limit
        
        if new_limit != current_limit:
            logger.info(f"Adjusted throttle limit for {rule_id}: {current_limit} -> {new_limit}")
    
    async def check_throttle(
        self,
        rule_id: str,
        identifier: Optional[str] = None,
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Verificar throttling."""
        rule = self.rules.get(rule_id)
        if not rule or not rule.enabled:
            return True, None
        
        id_key = identifier or rule.identifier
        limit_key = f"{rule_id}_{id_key}"
        current_limit = self.current_limits.get(limit_key, rule.base_limit)
        
        # Obtener conteo actual (simplificado - en producción usaría contador real)
        current_count = 0  # En producción, obtendría de contador real
        
        if current_count >= current_limit:
            async with self._lock:
                self.block_history.append({
                    "rule_id": rule_id,
                    "identifier": id_key,
                    "timestamp": datetime.now().isoformat(),
                    "limit": current_limit,
                })
            
            return False, {
                "allowed": False,
                "reason": "throttle_limit_exceeded",
                "current_limit": current_limit,
                "rule_id": rule_id,
            }
        
        return True, {
            "allowed": True,
            "remaining": current_limit - current_count,
            "current_limit": current_limit,
        }
    
    def get_throttle_status(self, rule_id: str) -> Dict[str, Any]:
        """Obtener estado de throttling."""
        rule = self.rules.get(rule_id)
        if not rule:
            return {"status": "no_rule"}
        
        limit_key = f"{rule_id}_{rule.identifier}"
        current_limit = self.current_limits.get(limit_key, rule.base_limit)
        
        return {
            "rule_id": rule_id,
            "current_limit": current_limit,
            "base_limit": rule.base_limit,
            "min_limit": rule.min_limit,
            "max_limit": rule.max_limit,
            "strategy": rule.strategy.value,
        }
    
    def get_adaptive_throttler_summary(self) -> Dict[str, Any]:
        """Obtener resumen del limitador."""
        return {
            "total_rules": len(self.rules),
            "active_rules": len([r for r in self.rules.values() if r.enabled]),
            "total_blocks": len(self.block_history),
        }

