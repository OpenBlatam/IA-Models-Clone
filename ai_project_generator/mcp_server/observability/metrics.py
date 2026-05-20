"""
MCP Metrics - Métricas Prometheus para MCP
===========================================
"""

from typing import Dict, Any, Optional
from prometheus_client import Counter, Histogram, Gauge


class MCPMetrics:
    """
    Métricas Prometheus para MCP
    
    Registra:
    - Latencia de operaciones
    - Tamaño de contexto
    - Errores
    - Requests por recurso
    """
    
    def __init__(self):
        # Contadores
        self.mcp_requests_total = Counter(
            "mcp_requests_total",
            "Total MCP requests",
            ["resource_id", "operation", "status"]
        )
        
        self.mcp_errors_total = Counter(
            "mcp_errors_total",
            "Total MCP errors",
            ["error_type", "resource_id"]
        )
        
        # Histogramas
        self.mcp_latency_seconds = Histogram(
            "mcp_latency_seconds",
            "MCP operation latency",
            ["operation", "resource_id"],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
        )
        
        self.mcp_context_size = Histogram(
            "mcp_context_size_tokens",
            "MCP context size in tokens",
            ["resource_id"],
            buckets=[100, 500, 1000, 2000, 5000, 10000, 20000]
        )
        
        # Gauges
        self.mcp_active_requests = Gauge(
            "mcp_active_requests",
            "Active MCP requests",
            ["resource_id"]
        )
        
        self.mcp_resources_count = Gauge(
            "mcp_resources_count",
            "Number of registered MCP resources"
        )
    
    def record(self, metric_name: str, value: float, **labels):
        """
        Registra una métrica genérica
        
        Args:
            metric_name: Nombre de la métrica
            value: Valor
            **labels: Labels
        """
        # Implementación genérica - puede extenderse
        pass
    
    def record_latency(self, operation: str, duration: float, resource_id: Optional[str] = None):
        """
        Registra latencia de operación
        
        Args:
            operation: Nombre de la operación
            duration: Duración en segundos
            resource_id: ID del recurso (opcional)
        """
        self.mcp_latency_seconds.labels(
            operation=operation,
            resource_id=resource_id or "unknown"
        ).observe(duration)
    
    def record_context_size(self, size: int, resource_id: Optional[str] = None):
        """
        Registra tamaño de contexto
        
        Args:
            size: Tamaño en tokens
            resource_id: ID del recurso (opcional)
        """
        self.mcp_context_size.labels(
            resource_id=resource_id or "unknown"
        ).observe(size)
    
    def record_error(self, error_type: str, **labels):
        """
        Registra un error
        
        Args:
            error_type: Tipo de error
            **labels: Labels adicionales
        """
        resource_id = labels.get("resource_id", "unknown")
        self.mcp_errors_total.labels(
            error_type=error_type,
            resource_id=resource_id
        ).inc()
    
    def record_request(self, resource_id: str, operation: str, status: str = "success"):
        """
        Registra un request
        
        Args:
            resource_id: ID del recurso
            operation: Operación realizada
            status: Estado (success, error)
        """
        self.mcp_requests_total.labels(
            resource_id=resource_id,
            operation=operation,
            status=status
        ).inc()
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Obtiene resumen de métricas
        
        Returns:
            Diccionario con resumen
        """
        return {
            "metrics_enabled": True,
            "counters": {
                "requests_total": "mcp_requests_total",
                "errors_total": "mcp_errors_total",
            },
            "histograms": {
                "latency_seconds": "mcp_latency_seconds",
                "context_size_tokens": "mcp_context_size_tokens",
            },
            "gauges": {
                "active_requests": "mcp_active_requests",
                "resources_count": "mcp_resources_count",
            },
        }

