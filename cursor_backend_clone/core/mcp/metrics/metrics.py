"""
MCP Metrics - Métricas específicas para MCP Server
===================================================

Recopila métricas específicas del servidor MCP.
"""

import time
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class MCPServerMetrics:
    """Métricas específicas del servidor MCP"""
    
    def __init__(self):
        self.request_count = 0
        self.error_count = 0
        self.websocket_connections = 0
        self.commands_executed = 0
        self.commands_failed = 0
        self.response_times: deque = deque(maxlen=1000)
        self.error_types: Dict[str, int] = defaultdict(int)
        self.client_requests: Dict[str, int] = defaultdict(int)
        self.endpoint_requests: Dict[str, int] = defaultdict(int)
        self.endpoint_response_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.start_time = datetime.now()
        self.last_request_time: Optional[datetime] = None
    
    def record_request(
        self,
        client_id: str,
        response_time: float,
        success: bool = True,
        endpoint: Optional[str] = None
    ):
        """Registrar una request"""
        self.request_count += 1
        self.client_requests[client_id] += 1
        self.last_request_time = datetime.now()
        self.response_times.append(response_time)
        
        if endpoint:
            self.endpoint_requests[endpoint] += 1
            self.endpoint_response_times[endpoint].append(response_time)
        
        if success:
            self.commands_executed += 1
        else:
            self.error_count += 1
            self.commands_failed += 1
    
    def record_error(self, error_type: str):
        """Registrar un error"""
        self.error_count += 1
        self.error_types[error_type] += 1
    
    def set_websocket_connections(self, count: int):
        """Establecer número de conexiones WebSocket"""
        self.websocket_connections = count
    
    def _calculate_percentiles(self, values: deque) -> Dict[str, float]:
        """Calcular percentiles de una lista de valores"""
        if not values:
            return {}
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        def percentile(p: float) -> float:
            index = int((p / 100) * n)
            if index >= n:
                index = n - 1
            return sorted_values[index]
        
        return {
            "p50": round(percentile(50) * 1000, 2),
            "p75": round(percentile(75) * 1000, 2),
            "p90": round(percentile(90) * 1000, 2),
            "p95": round(percentile(95) * 1000, 2),
            "p99": round(percentile(99) * 1000, 2),
            "min": round(min(sorted_values) * 1000, 2),
            "max": round(max(sorted_values) * 1000, 2)
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas"""
        uptime = (datetime.now() - self.start_time).total_seconds()
        avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
        
        response_time_percentiles = self._calculate_percentiles(self.response_times)
        
        endpoint_stats = {}
        for endpoint, times in self.endpoint_response_times.items():
            if times:
                endpoint_stats[endpoint] = {
                    "request_count": self.endpoint_requests.get(endpoint, 0),
                    "avg_response_time_ms": round(sum(times) / len(times) * 1000, 2),
                    "percentiles": self._calculate_percentiles(times)
                }
        
        return {
            "uptime_seconds": int(uptime),
            "total_requests": self.request_count,
            "total_errors": self.error_count,
            "commands_executed": self.commands_executed,
            "commands_failed": self.commands_failed,
            "websocket_connections": self.websocket_connections,
            "average_response_time_ms": round(avg_response_time * 1000, 2),
            "response_time_percentiles": response_time_percentiles,
            "requests_per_second": round(self.request_count / uptime, 2) if uptime > 0 else 0,
            "error_rate": round(self.error_count / self.request_count, 4) if self.request_count > 0 else 0,
            "error_types": dict(self.error_types),
            "top_clients": dict(sorted(
                self.client_requests.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]),
            "endpoint_stats": endpoint_stats,
            "last_request_time": self.last_request_time.isoformat() if self.last_request_time else None
        }

