"""
Backpressure Manager - Gestor de Backpressure
==============================================

Sistema de gestión de backpressure para prevenir sobrecarga y mantener estabilidad del sistema.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import statistics

logger = logging.getLogger(__name__)


class BackpressureLevel(Enum):
    """Nivel de backpressure."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class BackpressureMetric:
    """Métrica de backpressure."""
    timestamp: datetime
    queue_size: int = 0
    processing_rate: float = 0.0
    arrival_rate: float = 0.0
    error_rate: float = 0.0
    latency: float = 0.0
    system_load: float = 0.0


@dataclass
class BackpressureRule:
    """Regla de backpressure."""
    rule_id: str
    component_id: str
    queue_threshold: int = 100
    error_rate_threshold: float = 0.1
    latency_threshold: float = 2.0
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class BackpressureManager:
    """Gestor de backpressure."""
    
    def __init__(self):
        self.rules: Dict[str, BackpressureRule] = {}
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.current_levels: Dict[str, BackpressureLevel] = {}
        self.backpressure_history: deque = deque(maxlen=10000)
        self._lock = asyncio.Lock()
    
    def add_rule(
        self,
        rule_id: str,
        component_id: str,
        queue_threshold: int = 100,
        error_rate_threshold: float = 0.1,
        latency_threshold: float = 2.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Agregar regla de backpressure."""
        rule = BackpressureRule(
            rule_id=rule_id,
            component_id=component_id,
            queue_threshold=queue_threshold,
            error_rate_threshold=error_rate_threshold,
            latency_threshold=latency_threshold,
            metadata=metadata or {},
        )
        
        async def save_rule():
            async with self._lock:
                self.rules[rule_id] = rule
                self.current_levels[component_id] = BackpressureLevel.NONE
        
        asyncio.create_task(save_rule())
        
        logger.info(f"Added backpressure rule: {rule_id}")
        return rule_id
    
    def record_metric(
        self,
        component_id: str,
        queue_size: Optional[int] = None,
        processing_rate: Optional[float] = None,
        arrival_rate: Optional[float] = None,
        error_rate: Optional[float] = None,
        latency: Optional[float] = None,
        system_load: Optional[float] = None,
    ):
        """Registrar métrica."""
        metric = BackpressureMetric(
            timestamp=datetime.now(),
            queue_size=queue_size or 0,
            processing_rate=processing_rate or 0.0,
            arrival_rate=arrival_rate or 0.0,
            error_rate=error_rate or 0.0,
            latency=latency or 0.0,
            system_load=system_load or 0.0,
        )
        
        async def save_metric():
            async with self._lock:
                self.metrics[component_id].append(metric)
        
        asyncio.create_task(save_metric())
        
        # Evaluar backpressure
        asyncio.create_task(self._evaluate_backpressure(component_id))
    
    async def _evaluate_backpressure(self, component_id: str):
        """Evaluar nivel de backpressure."""
        rules = [r for r in self.rules.values() if r.component_id == component_id and r.enabled]
        if not rules:
            return
        
        rule = rules[0]  # Usar primera regla
        metric_history = self.metrics.get(component_id, deque())
        
        if not metric_history:
            return
        
        latest = metric_history[-1]
        
        # Determinar nivel de backpressure
        level = BackpressureLevel.NONE
        
        # Calcular score de backpressure
        queue_ratio = latest.queue_size / rule.queue_threshold if rule.queue_threshold > 0 else 0
        error_ratio = latest.error_rate / rule.error_rate_threshold if rule.error_rate_threshold > 0 else 0
        latency_ratio = latest.latency / rule.latency_threshold if rule.latency_threshold > 0 else 0
        
        max_ratio = max(queue_ratio, error_ratio, latency_ratio)
        
        if max_ratio >= 2.0 or latest.system_load > 0.95:
            level = BackpressureLevel.CRITICAL
        elif max_ratio >= 1.5 or latest.system_load > 0.85:
            level = BackpressureLevel.HIGH
        elif max_ratio >= 1.0 or latest.system_load > 0.70:
            level = BackpressureLevel.MEDIUM
        elif max_ratio >= 0.5 or latest.system_load > 0.50:
            level = BackpressureLevel.LOW
        
        # Actualizar nivel
        old_level = self.current_levels.get(component_id, BackpressureLevel.NONE)
        
        async with self._lock:
            self.current_levels[component_id] = level
        
        if level != old_level:
            async with self._lock:
                self.backpressure_history.append({
                    "component_id": component_id,
                    "old_level": old_level.value,
                    "new_level": level.value,
                    "timestamp": datetime.now().isoformat(),
                    "queue_size": latest.queue_size,
                    "error_rate": latest.error_rate,
                    "latency": latest.latency,
                })
            
            logger.warning(f"Backpressure level changed for {component_id}: {old_level.value} -> {level.value}")
    
    def get_backpressure_level(self, component_id: str) -> BackpressureLevel:
        """Obtener nivel de backpressure."""
        return self.current_levels.get(component_id, BackpressureLevel.NONE)
    
    def should_accept_request(self, component_id: str) -> bool:
        """Verificar si se debe aceptar petición."""
        level = self.get_backpressure_level(component_id)
        
        # Rechazar solo en niveles críticos o altos
        return level not in [BackpressureLevel.CRITICAL, BackpressureLevel.HIGH]
    
    def get_backpressure_status(self, component_id: Optional[str] = None) -> Dict[str, Any]:
        """Obtener estado de backpressure."""
        if component_id:
            level = self.get_backpressure_level(component_id)
            metric_history = self.metrics.get(component_id, deque())
            
            latest = metric_history[-1] if metric_history else None
            
            return {
                "component_id": component_id,
                "level": level.value,
                "queue_size": latest.queue_size if latest else 0,
                "error_rate": latest.error_rate if latest else 0.0,
                "latency": latest.latency if latest else 0.0,
            }
        else:
            return {
                "components": {
                    cid: {
                        "level": level.value,
                    }
                    for cid, level in self.current_levels.items()
                },
                "total_rules": len(self.rules),
            }
    
    def get_backpressure_history(self, component_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Obtener historial de backpressure."""
        history = list(self.backpressure_history)
        
        if component_id:
            history = [h for h in history if h["component_id"] == component_id]
        
        history.sort(key=lambda h: h["timestamp"], reverse=True)
        return history[:limit]
    
    def get_backpressure_manager_summary(self) -> Dict[str, Any]:
        """Obtener resumen del gestor."""
        by_level: Dict[str, int] = defaultdict(int)
        
        for level in self.current_levels.values():
            by_level[level.value] += 1
        
        return {
            "total_rules": len(self.rules),
            "components_by_level": dict(by_level),
            "total_history": len(self.backpressure_history),
        }


