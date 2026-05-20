"""
Network Analyzer - Analizador de Red
=====================================

Sistema de análisis de red con monitoreo de tráfico, latencia y detección de problemas.
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


class NetworkEventType(Enum):
    """Tipo de evento de red."""
    CONNECT = "connect"
    DISCONNECT = "disconnect"
    TIMEOUT = "timeout"
    ERROR = "error"
    SLOW_RESPONSE = "slow_response"


@dataclass
class NetworkMetric:
    """Métrica de red."""
    metric_id: str
    endpoint: str
    latency: float
    bytes_sent: int = 0
    bytes_received: int = 0
    status_code: Optional[int] = None
    timestamp: datetime = field(default_factory=datetime.now)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NetworkEvent:
    """Evento de red."""
    event_id: str
    event_type: NetworkEventType
    endpoint: str
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)


class NetworkAnalyzer:
    """Analizador de red."""
    
    def __init__(self, history_size: int = 10000):
        self.history_size = history_size
        self.metrics: deque = deque(maxlen=history_size)
        self.events: List[NetworkEvent] = []
        self.endpoint_stats: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
    
    def record_metric(
        self,
        endpoint: str,
        latency: float,
        bytes_sent: int = 0,
        bytes_received: int = 0,
        status_code: Optional[int] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Registrar métrica de red."""
        metric_id = f"metric_{endpoint}_{datetime.now().timestamp()}"
        
        metric = NetworkMetric(
            metric_id=metric_id,
            endpoint=endpoint,
            latency=latency,
            bytes_sent=bytes_sent,
            bytes_received=bytes_received,
            status_code=status_code,
            error=error,
            metadata=metadata or {},
        )
        
        self.metrics.append(metric)
        
        # Actualizar estadísticas de endpoint
        asyncio.create_task(self._update_endpoint_stats(endpoint, metric))
        
        # Detectar eventos
        if error:
            self._record_event(NetworkEventType.ERROR, endpoint, {"error": error})
        elif latency > 5.0:  # 5 segundos
            self._record_event(NetworkEventType.SLOW_RESPONSE, endpoint, {"latency": latency})
        
        return metric_id
    
    async def _update_endpoint_stats(self, endpoint: str, metric: NetworkMetric):
        """Actualizar estadísticas de endpoint."""
        async with self._lock:
            if endpoint not in self.endpoint_stats:
                self.endpoint_stats[endpoint] = {
                    "total_requests": 0,
                    "total_latency": 0.0,
                    "avg_latency": 0.0,
                    "min_latency": float('inf'),
                    "max_latency": 0.0,
                    "total_bytes_sent": 0,
                    "total_bytes_received": 0,
                    "error_count": 0,
                    "latencies": deque(maxlen=1000),
                }
            
            stats = self.endpoint_stats[endpoint]
            stats["total_requests"] += 1
            stats["total_latency"] += metric.latency
            stats["avg_latency"] = stats["total_latency"] / stats["total_requests"]
            stats["min_latency"] = min(stats["min_latency"], metric.latency)
            stats["max_latency"] = max(stats["max_latency"], metric.latency)
            stats["total_bytes_sent"] += metric.bytes_sent
            stats["total_bytes_received"] += metric.bytes_received
            
            if metric.error:
                stats["error_count"] += 1
            
            stats["latencies"].append(metric.latency)
    
    def _record_event(
        self,
        event_type: NetworkEventType,
        endpoint: str,
        details: Dict[str, Any],
    ):
        """Registrar evento de red."""
        event_id = f"event_{endpoint}_{datetime.now().timestamp()}"
        
        event = NetworkEvent(
            event_id=event_id,
            event_type=event_type,
            endpoint=endpoint,
            timestamp=datetime.now(),
            details=details,
        )
        
        self.events.append(event)
        
        # Mantener solo últimos 1000 eventos
        if len(self.events) > 1000:
            self.events.pop(0)
    
    def get_endpoint_stats(self, endpoint: str) -> Optional[Dict[str, Any]]:
        """Obtener estadísticas de endpoint."""
        stats = self.endpoint_stats.get(endpoint)
        if not stats:
            return None
        
        latencies = list(stats["latencies"])
        
        return {
            "endpoint": endpoint,
            "total_requests": stats["total_requests"],
            "avg_latency": stats["avg_latency"],
            "min_latency": stats["min_latency"] if stats["min_latency"] != float('inf') else 0.0,
            "max_latency": stats["max_latency"],
            "p95_latency": sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 20 else stats["max_latency"],
            "p99_latency": sorted(latencies)[int(len(latencies) * 0.99)] if len(latencies) > 100 else stats["max_latency"],
            "total_bytes_sent": stats["total_bytes_sent"],
            "total_bytes_received": stats["total_bytes_received"],
            "error_count": stats["error_count"],
            "error_rate": stats["error_count"] / stats["total_requests"] if stats["total_requests"] > 0 else 0.0,
        }
    
    def get_slow_endpoints(
        self,
        threshold: float = 1.0,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Obtener endpoints lentos."""
        slow = []
        
        for endpoint, stats in self.endpoint_stats.items():
            if stats["avg_latency"] > threshold:
                endpoint_stats = self.get_endpoint_stats(endpoint)
                if endpoint_stats:
                    slow.append(endpoint_stats)
        
        slow.sort(key=lambda x: x["avg_latency"], reverse=True)
        return slow[:limit]
    
    def get_network_events(
        self,
        event_type: Optional[NetworkEventType] = None,
        endpoint: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Obtener eventos de red."""
        events = self.events
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        if endpoint:
            events = [e for e in events if e.endpoint == endpoint]
        
        events.sort(key=lambda e: e.timestamp, reverse=True)
        
        return [
            {
                "event_id": e.event_id,
                "event_type": e.event_type.value,
                "endpoint": e.endpoint,
                "timestamp": e.timestamp.isoformat(),
                "details": e.details,
            }
            for e in events[:limit]
        ]
    
    def get_network_summary(self) -> Dict[str, Any]:
        """Obtener resumen de red."""
        total_requests = sum(s["total_requests"] for s in self.endpoint_stats.values())
        total_errors = sum(s["error_count"] for s in self.endpoint_stats.values())
        total_bytes_sent = sum(s["total_bytes_sent"] for s in self.endpoint_stats.values())
        total_bytes_received = sum(s["total_bytes_received"] for s in self.endpoint_stats.values())
        
        by_event_type: Dict[str, int] = defaultdict(int)
        for event in self.events:
            by_event_type[event.event_type.value] += 1
        
        return {
            "total_endpoints": len(self.endpoint_stats),
            "total_requests": total_requests,
            "total_errors": total_errors,
            "error_rate": total_errors / total_requests if total_requests > 0 else 0.0,
            "total_bytes_sent": total_bytes_sent,
            "total_bytes_received": total_bytes_received,
            "total_events": len(self.events),
            "events_by_type": dict(by_event_type),
        }















