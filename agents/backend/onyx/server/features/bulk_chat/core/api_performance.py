"""
API Performance - Rendimiento de API
====================================

Sistema de análisis de rendimiento de API con métricas detalladas, profiling y optimización.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)


class PerformanceMetric(Enum):
    """Métrica de rendimiento."""
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    CONCURRENT_REQUESTS = "concurrent_requests"


@dataclass
class APICall:
    """Llamada a API."""
    call_id: str
    endpoint: str
    method: str
    status_code: int
    response_time: float
    timestamp: datetime
    request_size: int = 0
    response_size: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EndpointPerformance:
    """Rendimiento de endpoint."""
    endpoint: str
    method: str
    total_calls: int = 0
    avg_response_time: float = 0.0
    p50_response_time: float = 0.0
    p95_response_time: float = 0.0
    p99_response_time: float = 0.0
    error_count: int = 0
    success_count: int = 0
    throughput: float = 0.0  # requests per second


class APIPerformance:
    """Analizador de rendimiento de API."""
    
    def __init__(self, max_calls: int = 100000):
        self.max_calls = max_calls
        self.api_calls: List[APICall] = []
        self.endpoint_performance: Dict[str, EndpointPerformance] = {}
        self._lock = asyncio.Lock()
    
    def record_api_call(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        response_time: float,
        request_size: int = 0,
        response_size: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Registrar llamada a API."""
        call_id = f"api_call_{datetime.now().timestamp()}"
        
        call = APICall(
            call_id=call_id,
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            response_time=response_time,
            timestamp=datetime.now(),
            request_size=request_size,
            response_size=response_size,
            metadata=metadata or {},
        )
        
        async with self._lock:
            self.api_calls.append(call)
            
            # Actualizar rendimiento de endpoint
            endpoint_key = f"{method} {endpoint}"
            if endpoint_key not in self.endpoint_performance:
                self.endpoint_performance[endpoint_key] = EndpointPerformance(
                    endpoint=endpoint,
                    method=method,
                )
            
            perf = self.endpoint_performance[endpoint_key]
            perf.total_calls += 1
            
            if status_code < 400:
                perf.success_count += 1
            else:
                perf.error_count += 1
            
            # Mantener solo últimos N llamadas
            if len(self.api_calls) > self.max_calls:
                self.api_calls.pop(0)
        
        # Actualizar estadísticas periódicamente
        asyncio.create_task(self._update_endpoint_stats(endpoint_key))
        
        return call_id
    
    async def _update_endpoint_stats(self, endpoint_key: str):
        """Actualizar estadísticas de endpoint."""
        perf = self.endpoint_performance.get(endpoint_key)
        if not perf:
            return
        
        # Obtener llamadas recientes de este endpoint
        recent_calls = [
            c for c in self.api_calls[-1000:]
            if f"{c.method} {c.endpoint}" == endpoint_key
        ]
        
        if not recent_calls:
            return
        
        response_times = [c.response_time for c in recent_calls]
        response_times.sort()
        
        async with self._lock:
            perf.avg_response_time = statistics.mean(response_times)
            
            if len(response_times) > 0:
                perf.p50_response_time = response_times[int(len(response_times) * 0.5)]
                perf.p95_response_time = response_times[int(len(response_times) * 0.95)] if len(response_times) > 20 else response_times[-1]
                perf.p99_response_time = response_times[int(len(response_times) * 0.99)] if len(response_times) > 100 else response_times[-1]
            
            # Calcular throughput (últimos 60 segundos)
            cutoff = datetime.now() - timedelta(seconds=60)
            recent_60s = [c for c in recent_calls if c.timestamp >= cutoff]
            perf.throughput = len(recent_60s) / 60.0 if recent_60s else 0.0
    
    def get_endpoint_performance(
        self,
        endpoint: Optional[str] = None,
        method: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Obtener rendimiento de endpoint(s)."""
        if endpoint and method:
            endpoint_key = f"{method} {endpoint}"
            perf = self.endpoint_performance.get(endpoint_key)
            if not perf:
                return {}
            
            return {
                "endpoint": perf.endpoint,
                "method": perf.method,
                "total_calls": perf.total_calls,
                "avg_response_time": perf.avg_response_time,
                "p50_response_time": perf.p50_response_time,
                "p95_response_time": perf.p95_response_time,
                "p99_response_time": perf.p99_response_time,
                "error_count": perf.error_count,
                "success_count": perf.success_count,
                "error_rate": perf.error_count / perf.total_calls if perf.total_calls > 0 else 0.0,
                "throughput": perf.throughput,
            }
        
        # Retornar todos los endpoints
        return {
            key: {
                "endpoint": perf.endpoint,
                "method": perf.method,
                "total_calls": perf.total_calls,
                "avg_response_time": perf.avg_response_time,
                "p95_response_time": perf.p95_response_time,
                "error_rate": perf.error_count / perf.total_calls if perf.total_calls > 0 else 0.0,
                "throughput": perf.throughput,
            }
            for key, perf in self.endpoint_performance.items()
        }
    
    def get_slow_endpoints(
        self,
        threshold_ms: float = 1000.0,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Obtener endpoints lentos."""
        slow_endpoints = []
        
        for key, perf in self.endpoint_performance.items():
            if perf.avg_response_time > threshold_ms / 1000.0:
                slow_endpoints.append({
                    "endpoint": perf.endpoint,
                    "method": perf.method,
                    "avg_response_time": perf.avg_response_time,
                    "p95_response_time": perf.p95_response_time,
                    "total_calls": perf.total_calls,
                })
        
        slow_endpoints.sort(key=lambda x: x["avg_response_time"], reverse=True)
        return slow_endpoints[:limit]
    
    def get_error_endpoints(
        self,
        min_error_rate: float = 0.05,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Obtener endpoints con errores."""
        error_endpoints = []
        
        for key, perf in self.endpoint_performance.items():
            if perf.total_calls > 0:
                error_rate = perf.error_count / perf.total_calls
                if error_rate >= min_error_rate:
                    error_endpoints.append({
                        "endpoint": perf.endpoint,
                        "method": perf.method,
                        "error_rate": error_rate,
                        "error_count": perf.error_count,
                        "total_calls": perf.total_calls,
                    })
        
        error_endpoints.sort(key=lambda x: x["error_rate"], reverse=True)
        return error_endpoints[:limit]
    
    def get_api_performance_summary(self) -> Dict[str, Any]:
        """Obtener resumen de rendimiento."""
        total_calls = sum(p.total_calls for p in self.endpoint_performance.values())
        total_errors = sum(p.error_count for p in self.endpoint_performance.values())
        
        avg_response_times = [
            p.avg_response_time for p in self.endpoint_performance.values()
            if p.total_calls > 0
        ]
        
        return {
            "total_endpoints": len(self.endpoint_performance),
            "total_api_calls": total_calls,
            "total_errors": total_errors,
            "overall_error_rate": total_errors / total_calls if total_calls > 0 else 0.0,
            "avg_response_time": statistics.mean(avg_response_times) if avg_response_times else 0.0,
            "slow_endpoints_count": len([
                p for p in self.endpoint_performance.values()
                if p.avg_response_time > 1.0
            ]),
        }
















