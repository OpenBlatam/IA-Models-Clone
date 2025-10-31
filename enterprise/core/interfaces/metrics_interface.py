from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from abc import ABC, abstractmethod
from typing import Dict, Any
from ..entities.metrics import MetricsData
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Metrics Service Interface
========================

Abstract interface for metrics collection and reporting.
"""



class IMetricsService(ABC):
    """Abstract interface for metrics operations."""
    
    @abstractmethod
    def record_request(self, method: str, endpoint: str, status: int, duration: float):
        """Record a request metric."""
        pass
    
    @abstractmethod
    def record_error(self, error_type: str, endpoint: str):
        """Record an error metric."""
        pass
    
    @abstractmethod
    def record_cache_operation(self, operation: str, result: str):
        """Record a cache operation metric."""
        pass
    
    @abstractmethod
    def update_active_connections(self, count: int):
        """Update active connections count."""
        pass
    
    @abstractmethod
    def update_circuit_breaker_state(self, service: str, state: str):
        """Update circuit breaker state."""
        pass
    
    @abstractmethod
    def get_metrics_data(self) -> MetricsData:
        """Get current metrics data."""
        pass
    
    @abstractmethod
    def get_prometheus_metrics(self) -> str:
        """Get metrics in Prometheus format."""
        pass
    
    @abstractmethod
    def add_custom_metric(self, name: str, value: Any, labels: Dict[str, str] = None):
        """Add a custom metric."""
        pass 