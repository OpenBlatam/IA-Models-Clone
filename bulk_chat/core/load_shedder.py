"""
Load Shedder - Sistema de Descarga de Carga
===========================================

Sistema de descarga inteligente de carga para prevenir sobrecarga del sistema.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class SheddingStrategy(Enum):
    """Estrategia de descarga."""
    RANDOM = "random"
    FIFO = "fifo"
    LIFO = "lifo"
    PRIORITY = "priority"
    ADAPTIVE = "adaptive"


class RequestPriority(Enum):
    """Prioridad de petición."""
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    BEST_EFFORT = "best_effort"


@dataclass
class LoadMetric:
    """Métrica de carga."""
    timestamp: datetime
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    request_rate: float = 0.0
    response_time: float = 0.0
    queue_size: int = 0
    error_rate: float = 0.0


@dataclass
class SheddingRule:
    """Regla de descarga."""
    rule_id: str
    metric_name: str
    threshold: float
    shedding_percentage: float
    strategy: SheddingStrategy
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SheddingEvent:
    """Evento de descarga."""
    event_id: str
    timestamp: datetime
    rule_id: str
    requests_shed: int
    reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class LoadShedder:
    """Sistema de descarga de carga."""
    
    def __init__(
        self,
        cpu_threshold: float = 0.8,
        memory_threshold: float = 0.8,
        request_rate_threshold: float = 1000.0,
        response_time_threshold: float = 2.0,
    ):
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.request_rate_threshold = request_rate_threshold
        self.response_time_threshold = response_time_threshold
        
        self.load_metrics: deque = deque(maxlen=1000)
        self.shedding_rules: List[SheddingRule] = []
        self.shedding_history: deque = deque(maxlen=10000)
        self.request_queue: deque = deque()
        self.active_requests: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        self._shedding_active = False
    
    def record_load_metric(
        self,
        cpu_usage: Optional[float] = None,
        memory_usage: Optional[float] = None,
        request_rate: Optional[float] = None,
        response_time: Optional[float] = None,
        queue_size: Optional[int] = None,
        error_rate: Optional[float] = None,
    ):
        """Registrar métrica de carga."""
        metric = LoadMetric(
            timestamp=datetime.now(),
            cpu_usage=cpu_usage or 0.0,
            memory_usage=memory_usage or 0.0,
            request_rate=request_rate or 0.0,
            response_time=response_time or 0.0,
            queue_size=queue_size or 0,
            error_rate=error_rate or 0.0,
        )
        
        async def save_metric():
            async with self._lock:
                self.load_metrics.append(metric)
                
                # Evaluar si se necesita descarga
                asyncio.create_task(self._evaluate_shedding())
        
        asyncio.create_task(save_metric())
    
    async def _evaluate_shedding(self):
        """Evaluar si se necesita descarga."""
        if not self.load_metrics:
            return
        
        latest_metric = self.load_metrics[-1]
        
        # Verificar umbrales
        needs_shedding = False
        reason = ""
        
        if latest_metric.cpu_usage >= self.cpu_threshold:
            needs_shedding = True
            reason = f"High CPU usage: {latest_metric.cpu_usage:.2%}"
        
        elif latest_metric.memory_usage >= self.memory_threshold:
            needs_shedding = True
            reason = f"High memory usage: {latest_metric.memory_usage:.2%}"
        
        elif latest_metric.request_rate >= self.request_rate_threshold:
            needs_shedding = True
            reason = f"High request rate: {latest_metric.request_rate:.2f} req/s"
        
        elif latest_metric.response_time >= self.response_time_threshold:
            needs_shedding = True
            reason = f"High response time: {latest_metric.response_time:.2f}s"
        
        elif latest_metric.error_rate > 0.1:
            needs_shedding = True
            reason = f"High error rate: {latest_metric.error_rate:.2%}"
        
        if needs_shedding:
            # Calcular porcentaje de descarga
            shedding_percentage = self._calculate_shedding_percentage(latest_metric)
            
            # Aplicar descarga
            await self._apply_shedding(shedding_percentage, reason)
    
    def _calculate_shedding_percentage(self, metric: LoadMetric) -> float:
        """Calcular porcentaje de descarga."""
        # Basado en qué tan sobrecargado está el sistema
        max_overload = max(
            metric.cpu_usage / self.cpu_threshold if self.cpu_threshold > 0 else 0,
            metric.memory_usage / self.memory_threshold if self.memory_threshold > 0 else 0,
            metric.request_rate / self.request_rate_threshold if self.request_rate_threshold > 0 else 0,
            metric.response_time / self.response_time_threshold if self.response_time_threshold > 0 else 0,
        )
        
        # Descargar más si hay más sobrecarga
        if max_overload > 1.5:
            return 0.5  # Descargar 50%
        elif max_overload > 1.2:
            return 0.3  # Descargar 30%
        elif max_overload > 1.0:
            return 0.1  # Descargar 10%
        
        return 0.0
    
    async def _apply_shedding(self, percentage: float, reason: str):
        """Aplicar descarga."""
        if percentage <= 0:
            return
        
        # Buscar regla aplicable
        applicable_rule = next(
            (r for r in self.shedding_rules if r.enabled),
            None
        )
        
        if not applicable_rule:
            # Crear regla temporal
            applicable_rule = SheddingRule(
                rule_id="temp",
                metric_name="overload",
                threshold=1.0,
                shedding_percentage=percentage,
                strategy=SheddingStrategy.PRIORITY,
            )
        
        # Descargar peticiones
        requests_to_shed = int(len(self.request_queue) * percentage)
        shed_requests = []
        
        async with self._lock:
            if applicable_rule.strategy == SheddingStrategy.PRIORITY:
                # Descargar primero las de menor prioridad
                sorted_queue = sorted(
                    self.request_queue,
                    key=lambda r: r.get("priority", RequestPriority.NORMAL.value),
                )
                shed_requests = sorted_queue[:requests_to_shed]
            
            elif applicable_rule.strategy == SheddingStrategy.FIFO:
                # Descargar las más antiguas
                shed_requests = list(self.request_queue)[:requests_to_shed]
            
            elif applicable_rule.strategy == SheddingStrategy.LIFO:
                # Descargar las más recientes
                shed_requests = list(self.request_queue)[-requests_to_shed:]
            
            elif applicable_rule.strategy == SheddingStrategy.RANDOM:
                # Descargar aleatoriamente
                import random
                shed_requests = random.sample(list(self.request_queue), min(requests_to_shed, len(self.request_queue)))
            
            # Remover de cola
            for req in shed_requests:
                if req in self.request_queue:
                    self.request_queue.remove(req)
        
        # Registrar evento
        event = SheddingEvent(
            event_id=f"shed_{datetime.now().timestamp()}",
            timestamp=datetime.now(),
            rule_id=applicable_rule.rule_id,
            requests_shed=len(shed_requests),
            reason=reason,
        )
        
        async with self._lock:
            self.shedding_history.append(event)
        
        logger.warning(f"Load shedding applied: {len(shed_requests)} requests shed. Reason: {reason}")
    
    def add_shedding_rule(
        self,
        rule_id: str,
        metric_name: str,
        threshold: float,
        shedding_percentage: float,
        strategy: SheddingStrategy,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Agregar regla de descarga."""
        rule = SheddingRule(
            rule_id=rule_id,
            metric_name=metric_name,
            threshold=threshold,
            shedding_percentage=shedding_percentage,
            strategy=strategy,
            metadata=metadata or {},
        )
        
        async def save_rule():
            async with self._lock:
                self.shedding_rules.append(rule)
        
        asyncio.create_task(save_rule())
        
        logger.info(f"Added shedding rule: {rule_id}")
        return rule_id
    
    async def should_accept_request(self, request_id: str, priority: RequestPriority = RequestPriority.NORMAL) -> bool:
        """Verificar si se debe aceptar petición."""
        if not self.load_metrics:
            return True
        
        latest_metric = self.load_metrics[-1]
        
        # Rechazar si está sobrecargado y la petición no es crítica
        if priority == RequestPriority.CRITICAL:
            return True
        
        if latest_metric.cpu_usage >= self.cpu_threshold * 1.2:
            return False
        
        if latest_metric.memory_usage >= self.memory_threshold * 1.2:
            return False
        
        return True
    
    def get_load_statistics(self, window_minutes: int = 5) -> Dict[str, Any]:
        """Obtener estadísticas de carga."""
        if not self.load_metrics:
            return {}
        
        window_start = datetime.now() - timedelta(minutes=window_minutes)
        recent_metrics = [
            m for m in self.load_metrics
            if m.timestamp >= window_start
        ]
        
        if not recent_metrics:
            return {}
        
        return {
            "avg_cpu": sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics),
            "avg_memory": sum(m.memory_usage for m in recent_metrics) / len(recent_metrics),
            "avg_request_rate": sum(m.request_rate for m in recent_metrics) / len(recent_metrics),
            "avg_response_time": sum(m.response_time for m in recent_metrics) / len(recent_metrics),
            "avg_error_rate": sum(m.error_rate for m in recent_metrics) / len(recent_metrics),
            "current_queue_size": len(self.request_queue),
        }
    
    def get_shedding_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Obtener historial de descarga."""
        history = list(self.shedding_history)[-limit:]
        
        return [
            {
                "event_id": e.event_id,
                "timestamp": e.timestamp.isoformat(),
                "rule_id": e.rule_id,
                "requests_shed": e.requests_shed,
                "reason": e.reason,
            }
            for e in history
        ]
    
    def get_load_shedder_summary(self) -> Dict[str, Any]:
        """Obtener resumen del shedder."""
        return {
            "shedding_active": self._shedding_active,
            "total_rules": len(self.shedding_rules),
            "active_rules": len([r for r in self.shedding_rules if r.enabled]),
            "total_shedding_events": len(self.shedding_history),
            "current_queue_size": len(self.request_queue),
            "total_load_metrics": len(self.load_metrics),
        }


