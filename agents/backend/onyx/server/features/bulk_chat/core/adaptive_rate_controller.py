"""
Adaptive Rate Controller - Controlador de Tasa Adaptativo
==========================================================

Sistema de control de tasa adaptativo que ajusta dinámicamente los límites basado en condiciones del sistema.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque, defaultdict
import statistics

logger = logging.getLogger(__name__)


class RateAdjustmentStrategy(Enum):
    """Estrategia de ajuste de tasa."""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    AUTO = "auto"


@dataclass
class RateLimit:
    """Límite de tasa."""
    identifier: str
    current_rate: float
    base_rate: float
    min_rate: float
    max_rate: float
    adjustment_factor: float = 1.0
    last_adjustment: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RateMetric:
    """Métrica de tasa."""
    timestamp: datetime
    request_count: int
    success_count: int
    error_count: int
    avg_response_time: float
    system_load: Optional[float] = None


class AdaptiveRateController:
    """Controlador de tasa adaptativo."""
    
    def __init__(
        self,
        base_rate: float = 100.0,
        min_rate: float = 10.0,
        max_rate: float = 1000.0,
        adjustment_window: int = 60,
        strategy: RateAdjustmentStrategy = RateAdjustmentStrategy.AUTO,
    ):
        self.base_rate = base_rate
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.adjustment_window = adjustment_window
        self.strategy = strategy
        
        self.rate_limits: Dict[str, RateLimit] = {}
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.adjustment_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._lock = asyncio.Lock()
    
    def register_identifier(
        self,
        identifier: str,
        base_rate: Optional[float] = None,
        min_rate: Optional[float] = None,
        max_rate: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Registrar identificador para control de tasa."""
        rate_limit = RateLimit(
            identifier=identifier,
            current_rate=base_rate or self.base_rate,
            base_rate=base_rate or self.base_rate,
            min_rate=min_rate or self.min_rate,
            max_rate=max_rate or self.max_rate,
            metadata=metadata or {},
        )
        
        async def save_rate_limit():
            async with self._lock:
                self.rate_limits[identifier] = rate_limit
        
        asyncio.create_task(save_rate_limit())
        
        logger.info(f"Registered rate limit for identifier: {identifier}")
        return identifier
    
    async def record_request(
        self,
        identifier: str,
        success: bool,
        response_time: float,
        system_load: Optional[float] = None,
    ):
        """Registrar petición."""
        metric = RateMetric(
            timestamp=datetime.now(),
            request_count=1,
            success_count=1 if success else 0,
            error_count=0 if success else 1,
            avg_response_time=response_time,
            system_load=system_load,
        )
        
        async with self._lock:
            if identifier not in self.metrics:
                self.metrics[identifier] = deque(maxlen=1000)
            
            self.metrics[identifier].append(metric)
            
            # Auto-ajustar si es necesario
            if len(self.metrics[identifier]) >= 10:
                asyncio.create_task(self._auto_adjust_rate(identifier))
    
    async def _auto_adjust_rate(self, identifier: str):
        """Ajustar tasa automáticamente."""
        rate_limit = self.rate_limits.get(identifier)
        if not rate_limit:
            return
        
        metrics_list = list(self.metrics[identifier])
        if len(metrics_list) < 10:
            return
        
        # Calcular métricas recientes
        recent_metrics = metrics_list[-self.adjustment_window:]
        
        total_requests = sum(m.request_count for m in recent_metrics)
        total_success = sum(m.success_count for m in recent_metrics)
        total_errors = sum(m.error_count for m in recent_metrics)
        
        success_rate = total_success / total_requests if total_requests > 0 else 1.0
        error_rate = total_errors / total_requests if total_requests > 0 else 0.0
        avg_response_time = statistics.mean([m.avg_response_time for m in recent_metrics])
        
        # Calcular system load promedio
        load_values = [m.system_load for m in recent_metrics if m.system_load is not None]
        avg_load = statistics.mean(load_values) if load_values else 0.5
        
        # Determinar ajuste
        adjustment_factor = 1.0
        
        if self.strategy == RateAdjustmentStrategy.AUTO:
            # Ajustar basado en múltiples factores
            if error_rate > 0.1:  # Más del 10% de errores
                adjustment_factor = 0.8  # Reducir 20%
            elif error_rate > 0.05:  # Más del 5% de errores
                adjustment_factor = 0.9  # Reducir 10%
            elif success_rate > 0.95 and avg_response_time < 0.5:  # Muy bien
                adjustment_factor = 1.1  # Aumentar 10%
            
            # Ajustar por system load
            if avg_load > 0.8:  # Alta carga
                adjustment_factor *= 0.85  # Reducir más
            elif avg_load < 0.3:  # Baja carga
                adjustment_factor *= 1.05  # Aumentar más
        
        elif self.strategy == RateAdjustmentStrategy.CONSERVATIVE:
            if error_rate > 0.05:
                adjustment_factor = 0.9
            elif success_rate > 0.98:
                adjustment_factor = 1.05
        
        elif self.strategy == RateAdjustmentStrategy.MODERATE:
            if error_rate > 0.1:
                adjustment_factor = 0.85
            elif success_rate > 0.95:
                adjustment_factor = 1.1
        
        elif self.strategy == RateAdjustmentStrategy.AGGRESSIVE:
            if error_rate > 0.15:
                adjustment_factor = 0.7
            elif success_rate > 0.9:
                adjustment_factor = 1.2
        
        # Aplicar ajuste
        new_rate = rate_limit.current_rate * adjustment_factor
        new_rate = max(rate_limit.min_rate, min(rate_limit.max_rate, new_rate))
        
        async with self._lock:
            old_rate = rate_limit.current_rate
            rate_limit.current_rate = new_rate
            rate_limit.adjustment_factor = adjustment_factor
            rate_limit.last_adjustment = datetime.now()
            
            # Guardar historial
            self.adjustment_history[identifier].append({
                "timestamp": datetime.now().isoformat(),
                "old_rate": old_rate,
                "new_rate": new_rate,
                "adjustment_factor": adjustment_factor,
                "success_rate": success_rate,
                "error_rate": error_rate,
                "avg_response_time": avg_response_time,
                "system_load": avg_load,
            })
            
            # Limitar historial
            if len(self.adjustment_history[identifier]) > 1000:
                self.adjustment_history[identifier].pop(0)
        
        logger.info(f"Adjusted rate for {identifier}: {old_rate:.2f} -> {new_rate:.2f} (factor: {adjustment_factor:.2f})")
    
    async def check_rate_limit(self, identifier: str) -> tuple:
        """Verificar si se puede procesar petición."""
        rate_limit = self.rate_limits.get(identifier)
        if not rate_limit:
            # Permitir si no hay límite configurado
            return True, {"allowed": True, "current_rate": self.base_rate}
        
        # Verificar tasa actual
        metrics_list = list(self.metrics[identifier])
        if not metrics_list:
            return True, {"allowed": True, "current_rate": rate_limit.current_rate}
        
        # Contar peticiones en ventana
        window_start = datetime.now() - timedelta(seconds=1)
        recent_requests = [
            m for m in metrics_list
            if m.timestamp >= window_start
        ]
        
        request_count = sum(m.request_count for m in recent_requests)
        allowed = request_count < rate_limit.current_rate
        
        return allowed, {
            "allowed": allowed,
            "current_rate": rate_limit.current_rate,
            "base_rate": rate_limit.base_rate,
            "requests_in_window": request_count,
            "adjustment_factor": rate_limit.adjustment_factor,
        }
    
    def get_rate_limit(self, identifier: str) -> Optional[Dict[str, Any]]:
        """Obtener límite de tasa."""
        rate_limit = self.rate_limits.get(identifier)
        if not rate_limit:
            return None
        
        metrics_list = list(self.metrics[identifier])
        recent_metrics = metrics_list[-60:] if len(metrics_list) > 60 else metrics_list
        
        total_requests = sum(m.request_count for m in recent_metrics) if recent_metrics else 0
        total_success = sum(m.success_count for m in recent_metrics) if recent_metrics else 0
        total_errors = sum(m.error_count for m in recent_metrics) if recent_metrics else 0
        avg_response_time = statistics.mean([m.avg_response_time for m in recent_metrics]) if recent_metrics else 0.0
        
        return {
            "identifier": identifier,
            "current_rate": rate_limit.current_rate,
            "base_rate": rate_limit.base_rate,
            "min_rate": rate_limit.min_rate,
            "max_rate": rate_limit.max_rate,
            "adjustment_factor": rate_limit.adjustment_factor,
            "last_adjustment": rate_limit.last_adjustment.isoformat() if rate_limit.last_adjustment else None,
            "recent_stats": {
                "total_requests": total_requests,
                "success_count": total_success,
                "error_count": total_errors,
                "success_rate": total_success / total_requests if total_requests > 0 else 0.0,
                "avg_response_time": avg_response_time,
            },
        }
    
    def get_adjustment_history(self, identifier: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Obtener historial de ajustes."""
        history = self.adjustment_history.get(identifier, [])
        return history[-limit:]
    
    def get_adaptive_rate_controller_summary(self) -> Dict[str, Any]:
        """Obtener resumen del controlador."""
        return {
            "total_identifiers": len(self.rate_limits),
            "base_rate": self.base_rate,
            "min_rate": self.min_rate,
            "max_rate": self.max_rate,
            "strategy": self.strategy.value,
            "adjustment_window": self.adjustment_window,
        }

