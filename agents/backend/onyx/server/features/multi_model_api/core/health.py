"""
Health monitoring and metrics for multi-model API
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque

from .models import ModelRegistry
from .cache import EnhancedCache

logger = logging.getLogger(__name__)


@dataclass
class HealthMetrics:
    """Health metrics container"""
    timestamp: datetime
    cache_hit_rate: float
    cache_size: int
    models_available: int
    models_total: int
    avg_latency_ms: float
    error_rate: float
    circuit_breakers_open: int


class HealthMonitor:
    """Monitor system health and metrics"""
    
    def __init__(
        self,
        registry: ModelRegistry,
        cache: EnhancedCache,
        check_interval: int = 60
    ):
        self.registry = registry
        self.cache = cache
        self.check_interval = check_interval
        self.metrics_history: deque = deque(maxlen=100)
        self.is_monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
    
    async def start_monitoring(self):
        """Start background health monitoring"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Health monitoring started")
    
    async def stop_monitoring(self):
        """Stop health monitoring"""
        self.is_monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Health monitoring stopped")
    
    async def _monitor_loop(self):
        """Background monitoring loop"""
        while self.is_monitoring:
            try:
                metrics = await self.collect_metrics()
                self.metrics_history.append(metrics)
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitoring: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def collect_metrics(self) -> HealthMetrics:
        """Collect current health metrics"""
        cache_stats = await self.cache.get_stats()
        
        total_calls = 0
        total_errors = 0
        total_latency = 0.0
        circuit_breakers_open = 0
        
        for model_meta in self.registry.models.values():
            total_calls += model_meta.call_count
            total_errors += model_meta.error_count
            total_latency += model_meta.total_latency_ms
            
            if model_meta.circuit_breaker.state == "open":
                circuit_breakers_open += 1
        
        avg_latency = total_latency / total_calls if total_calls > 0 else 0.0
        error_rate = (total_errors / total_calls * 100) if total_calls > 0 else 0.0
        
        return HealthMetrics(
            timestamp=datetime.now(),
            cache_hit_rate=cache_stats.get("hit_rate", 0.0),
            cache_size=cache_stats.get("l1_size", 0),
            models_available=len(self.registry.get_available_models()),
            models_total=len(self.registry.models),
            avg_latency_ms=round(avg_latency, 2),
            error_rate=round(error_rate, 2),
            circuit_breakers_open=circuit_breakers_open
        )
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get current health status"""
        metrics = await self.collect_metrics()
        
        is_healthy = (
            metrics.models_available > 0 and
            metrics.error_rate < 10.0 and
            metrics.circuit_breakers_open < metrics.models_total / 2
        )
        
        return {
            "status": "healthy" if is_healthy else "degraded",
            "metrics": {
                "cache_hit_rate": metrics.cache_hit_rate,
                "cache_size": metrics.cache_size,
                "models_available": metrics.models_available,
                "models_total": metrics.models_total,
                "avg_latency_ms": metrics.avg_latency_ms,
                "error_rate": metrics.error_rate,
                "circuit_breakers_open": metrics.circuit_breakers_open
            },
            "timestamp": metrics.timestamp.isoformat()
        }
    
    def get_metrics_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent metrics history"""
        return [
            {
                "timestamp": m.timestamp.isoformat(),
                "cache_hit_rate": m.cache_hit_rate,
                "avg_latency_ms": m.avg_latency_ms,
                "error_rate": m.error_rate,
                "circuit_breakers_open": m.circuit_breakers_open
            }
            for m in list(self.metrics_history)[-limit:]
        ]


# Global health monitor instance
_health_monitor: Optional[HealthMonitor] = None


def get_health_monitor(
    registry: ModelRegistry,
    cache: EnhancedCache
) -> HealthMonitor:
    """Get or create health monitor instance"""
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = HealthMonitor(registry, cache)
    return _health_monitor

