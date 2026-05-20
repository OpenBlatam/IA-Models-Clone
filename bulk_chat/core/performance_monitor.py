"""
Performance Monitor - Monitor de Rendimiento Avanzado
=======================================================

Sistema avanzado de monitoreo de rendimiento con análisis de latencias, throughput y recursos.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import statistics
import time

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Tipo de métrica."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class PerformanceMetric:
    """Métrica de rendimiento."""
    metric_name: str
    metric_type: MetricType
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceSnapshot:
    """Snapshot de rendimiento."""
    snapshot_id: str
    timestamp: datetime
    metrics: Dict[str, float]
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    throughput: float = 0.0
    error_rate: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceMonitor:
    """Monitor de rendimiento avanzado."""
    
    def __init__(self):
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.latency_samples: deque = deque(maxlen=100000)
        self.snapshots: deque = deque(maxlen=10000)
        self.active_operations: Dict[str, float] = {}
        self._lock = asyncio.Lock()
        self._monitoring_active = False
    
    def record_metric(
        self,
        metric_name: str,
        value: float,
        metric_type: MetricType = MetricType.GAUGE,
        labels: Optional[Dict[str, str]] = None,
        timestamp: Optional[datetime] = None,
    ):
        """Registrar métrica."""
        timestamp = timestamp or datetime.now()
        
        metric = PerformanceMetric(
            metric_name=metric_name,
            metric_type=metric_type,
            value=value,
            timestamp=timestamp,
            labels=labels or {},
        )
        
        async def save_metric():
            async with self._lock:
                self.metrics[metric_name].append(metric)
        
        asyncio.create_task(save_metric())
    
    def record_latency(
        self,
        operation_name: str,
        latency_seconds: float,
    ):
        """Registrar latencia."""
        async def save_latency():
            async with self._lock:
                self.latency_samples.append({
                    "operation": operation_name,
                    "latency": latency_seconds,
                    "timestamp": datetime.now(),
                })
        
        asyncio.create_task(save_latency())
    
    async def measure_operation(
        self,
        operation_name: str,
        operation: Callable,
        *args,
        **kwargs,
    ) -> Any:
        """Medir tiempo de operación."""
        start_time = time.time()
        
        try:
            if asyncio.iscoroutinefunction(operation):
                result = await operation(*args, **kwargs)
            else:
                result = operation(*args, **kwargs)
            
            latency = time.time() - start_time
            self.record_latency(operation_name, latency)
            
            return result
        
        except Exception as e:
            latency = time.time() - start_time
            self.record_latency(operation_name, latency)
            raise
    
    def create_snapshot(self) -> str:
        """Crear snapshot de rendimiento."""
        snapshot_id = f"snapshot_{datetime.now().timestamp()}"
        timestamp = datetime.now()
        
        # Calcular métricas agregadas
        latencies = [s["latency"] for s in self.latency_samples]
        
        latency_p50 = 0.0
        latency_p95 = 0.0
        latency_p99 = 0.0
        
        if latencies:
            sorted_latencies = sorted(latencies)
            n = len(sorted_latencies)
            latency_p50 = sorted_latencies[int(n * 0.50)]
            latency_p95 = sorted_latencies[int(n * 0.95)] if n > 1 else sorted_latencies[-1]
            latency_p99 = sorted_latencies[int(n * 0.99)] if n > 1 else sorted_latencies[-1]
        
        # Calcular throughput (operaciones por segundo en última ventana)
        window_start = timestamp - timedelta(seconds=60)
        recent_operations = [
            s for s in self.latency_samples
            if s["timestamp"] >= window_start
        ]
        throughput = len(recent_operations) / 60.0 if recent_operations else 0.0
        
        # Calcular error rate (simplificado)
        error_rate = 0.0
        
        # Agregar métricas actuales
        current_metrics = {}
        for metric_name, metric_history in self.metrics.items():
            if metric_history:
                recent = list(metric_history)[-10:]
                if recent:
                    current_metrics[metric_name] = statistics.mean([m.value for m in recent])
        
        snapshot = PerformanceSnapshot(
            snapshot_id=snapshot_id,
            timestamp=timestamp,
            metrics=current_metrics,
            latency_p50=latency_p50,
            latency_p95=latency_p95,
            latency_p99=latency_p99,
            throughput=throughput,
            error_rate=error_rate,
        )
        
        async def save_snapshot():
            async with self._lock:
                self.snapshots.append(snapshot)
        
        asyncio.create_task(save_snapshot())
        
        logger.debug(f"Created performance snapshot: {snapshot_id}")
        return snapshot_id
    
    def get_performance_summary(
        self,
        window_minutes: int = 5,
    ) -> Dict[str, Any]:
        """Obtener resumen de rendimiento."""
        window_start = datetime.now() - timedelta(minutes=window_minutes)
        
        # Filtrar muestras recientes
        recent_latencies = [
            s["latency"] for s in self.latency_samples
            if s["timestamp"] >= window_start
        ]
        
        if not recent_latencies:
            return {
                "window_minutes": window_minutes,
                "total_operations": 0,
                "latency_p50": 0.0,
                "latency_p95": 0.0,
                "latency_p99": 0.0,
                "throughput": 0.0,
            }
        
        sorted_latencies = sorted(recent_latencies)
        n = len(sorted_latencies)
        
        return {
            "window_minutes": window_minutes,
            "total_operations": n,
            "latency_avg": statistics.mean(recent_latencies),
            "latency_min": min(recent_latencies),
            "latency_max": max(recent_latencies),
            "latency_p50": sorted_latencies[int(n * 0.50)],
            "latency_p95": sorted_latencies[int(n * 0.95)] if n > 1 else sorted_latencies[-1],
            "latency_p99": sorted_latencies[int(n * 0.99)] if n > 1 else sorted_latencies[-1],
            "throughput": n / (window_minutes * 60.0),
        }
    
    def get_metric_history(
        self,
        metric_name: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Obtener historial de métrica."""
        history = self.metrics.get(metric_name, deque())
        
        return [
            {
                "value": m.value,
                "timestamp": m.timestamp.isoformat(),
                "labels": m.labels,
            }
            for m in list(history)[-limit:]
        ]
    
    def get_performance_monitor_summary(self) -> Dict[str, Any]:
        """Obtener resumen del monitor."""
        return {
            "monitoring_active": self._monitoring_active,
            "total_metrics": len(self.metrics),
            "total_latency_samples": len(self.latency_samples),
            "total_snapshots": len(self.snapshots),
            "active_operations": len(self.active_operations),
        }
