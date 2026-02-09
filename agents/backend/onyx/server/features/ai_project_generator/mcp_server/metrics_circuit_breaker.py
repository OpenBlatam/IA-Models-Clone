"""
MCP Metrics Circuit Breaker - Circuit breaker con métricas
===========================================================
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from enum import Enum

from .circuit_breaker import CircuitBreaker, CircuitState

logger = logging.getLogger(__name__)


class CircuitBreakerMetrics:
    """Métricas de circuit breaker"""
    
    def __init__(self):
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.circuit_open_count = 0
        self.circuit_close_count = 0
        self.last_state_change: Optional[datetime] = None
        self.state_history: List[tuple[datetime, CircuitState]] = []
    
    def record_request(self, success: bool):
        """Registra un request"""
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
    
    def record_state_change(self, new_state: CircuitState):
        """Registra cambio de estado"""
        if new_state == CircuitState.OPEN:
            self.circuit_open_count += 1
        elif new_state == CircuitState.CLOSED:
            self.circuit_close_count += 1
        
        self.last_state_change = datetime.utcnow()
        self.state_history.append((datetime.utcnow(), new_state))
        
        # Mantener solo últimos 100 cambios
        if len(self.state_history) > 100:
            self.state_history = self.state_history[-100:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas"""
        success_rate = (
            self.successful_requests / self.total_requests * 100
            if self.total_requests > 0 else 0
        )
        
        failure_rate = (
            self.failed_requests / self.total_requests * 100
            if self.total_requests > 0 else 0
        )
        
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate_percent": success_rate,
            "failure_rate_percent": failure_rate,
            "circuit_open_count": self.circuit_open_count,
            "circuit_close_count": self.circuit_close_count,
            "last_state_change": self.last_state_change.isoformat() if self.last_state_change else None,
            "recent_state_changes": [
                {"timestamp": ts.isoformat(), "state": state.value}
                for ts, state in self.state_history[-10:]
            ],
        }


class MetricsCircuitBreaker(CircuitBreaker):
    """
    Circuit breaker con métricas
    
    Extiende CircuitBreaker con tracking de métricas.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics = CircuitBreakerMetrics()
        self._original_state = self.state
    
    def _on_state_change(self, new_state: CircuitState):
        """Callback cuando cambia el estado"""
        super()._on_state_change(new_state)
        self.metrics.record_state_change(new_state)
    
    async def call(self, func, *args, **kwargs):
        """
        Ejecuta función con circuit breaker y métricas
        
        Args:
            func: Función a ejecutar
            *args: Argumentos
            **kwargs: Keyword arguments
            
        Returns:
            Resultado de la función
        """
        try:
            result = await super().call(func, *args, **kwargs)
            self.metrics.record_request(success=True)
            return result
        except Exception as e:
            self.metrics.record_request(success=False)
            raise
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Obtiene métricas del circuit breaker
        
        Returns:
            Diccionario con métricas
        """
        base_stats = super().get_stats()
        metrics_stats = self.metrics.get_stats()
        
        return {
            **base_stats,
            "metrics": metrics_stats,
        }

