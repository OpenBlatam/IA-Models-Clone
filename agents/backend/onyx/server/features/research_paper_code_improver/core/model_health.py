"""
Model Health Checker - Verificador de salud de modelos
========================================================
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .base_classes import BaseManager, BaseConfig
from .common_utils import (
    get_device, move_to_device, check_model_health as check_basic_health,
    measure_inference_time, calculate_model_size
)
from .constants import LATENCY_THRESHOLD_MS

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Estado de salud"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Resultado de health check"""
    component: str
    status: HealthStatus
    message: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario"""
        return {
            "component": self.component,
            "status": self.status.value,
            "message": self.message,
            "metrics": self.metrics,
            "timestamp": self.timestamp.isoformat()
        }


class ModelHealthChecker(BaseManager):
    """Verificador de salud de modelos"""
    
    def __init__(self, config: Optional[BaseConfig] = None):
        super().__init__(config or BaseConfig())
        self.health_results: List[HealthCheckResult] = []
        self.thresholds: Dict[str, Dict[str, float]] = {
            "latency": {"warning": LATENCY_THRESHOLD_MS, "critical": LATENCY_THRESHOLD_MS * 5},
            "error_rate": {"warning": 0.05, "critical": 0.1},
            "memory": {"warning": 80.0, "critical": 95.0},
            "cpu": {"warning": 80.0, "critical": 95.0}
        }
    
    def check_model_health(
        self,
        model: nn.Module,
        example_input: torch.Tensor,
        device: Optional[str] = None
    ) -> List[HealthCheckResult]:
        """Verifica salud del modelo usando utilidades compartidas"""
        results = []
        device_obj = get_device(device)
        
        # Check 1: Basic health usando utilidades compartidas
        basic_health = check_basic_health(model, str(device_obj))
        if not basic_health["has_parameters"]:
            results.append(HealthCheckResult(
                component="model_parameters",
                status=HealthStatus.CRITICAL,
                message="Modelo no tiene parámetros"
            ))
            return results
        
        if not basic_health["device_compatible"]:
            results.append(HealthCheckResult(
                component="model_loading",
                status=HealthStatus.CRITICAL,
                message="Modelo no compatible con dispositivo"
            ))
            return results
        
        results.append(HealthCheckResult(
            component="model_loading",
            status=HealthStatus.HEALTHY,
            message="Modelo cargado correctamente"
        ))
        
        # Check 2: Inference usando utilidades compartidas
        try:
            model = model.to(device_obj)
            model.eval()
            
            # Medir latencia usando utilidades compartidas
            latency_ms = measure_inference_time(
                model, example_input, num_runs=10, warmup=3, device=str(device_obj)
            )
            
            status = self._get_status_from_threshold("latency", latency_ms)
            results.append(HealthCheckResult(
                component="inference",
                status=status,
                message=f"Latencia: {latency_ms:.2f}ms",
                metrics={"latency_ms": latency_ms}
            ))
        except Exception as e:
            results.append(HealthCheckResult(
                component="inference",
                status=HealthStatus.CRITICAL,
                message=f"Error en inferencia: {e}"
            ))
        
        # Check 3: Memory usando utilidades compartidas
        if torch.cuda.is_available():
            try:
                # Calcular tamaño del modelo usando utilidades compartidas
                model_size_mb = calculate_model_size(model)
                
                memory_allocated = torch.cuda.memory_allocated(device_obj) / 1024**2
                memory_reserved = torch.cuda.memory_reserved(device_obj) / 1024**2
                memory_total = torch.cuda.get_device_properties(device_obj).total_memory / 1024**2
                memory_percent = (memory_reserved / memory_total) * 100
                
                status = self._get_status_from_threshold("memory", memory_percent)
                results.append(HealthCheckResult(
                    component="memory",
                    status=status,
                    message=f"Memoria: {memory_percent:.1f}%",
                    metrics={
                    "model_size_mb": model_size_mb,
                    "allocated_mb": memory_allocated,
                    "reserved_mb": memory_reserved,
                    "total_mb": memory_total,
                    "percent": memory_percent
                    }
                ))
            except Exception as e:
                results.append(HealthCheckResult(
                    component="memory",
                    status=HealthStatus.WARNING,
                    message=f"Error verificando memoria: {e}"
                ))
        
        # Check 4: Model parameters usando utilidades compartidas
        try:
            from .common_utils import count_parameters
            param_info = count_parameters(model)
            total_params = param_info["total"]
            trainable_params = param_info["trainable"]
            
            # Verificar NaN/Inf en parámetros
            has_nan = False
            has_inf = False
            for param in model.parameters():
                if torch.isnan(param).any():
                    has_nan = True
                if torch.isinf(param).any():
                    has_inf = True
            
            status = HealthStatus.HEALTHY
            message = f"Parámetros: {total_params:,} total, {trainable_params:,} entrenables"
            
            if has_nan or has_inf:
                status = HealthStatus.CRITICAL
                message += f" - {'NaN' if has_nan else ''} {'Inf' if has_inf else ''} detectado"
            
            results.append(HealthCheckResult(
                component="parameters",
                status=status,
                message=message,
                metrics={
                    "total_params": total_params,
                    "trainable_params": trainable_params,
                    "has_nan": has_nan,
                    "has_inf": has_inf
                }
            ))
        except Exception as e:
            results.append(HealthCheckResult(
                component="parameters",
                status=HealthStatus.WARNING,
                message=f"Error verificando parámetros: {e}"
            ))
        
        self.health_results.extend(results)
        self.log_event("health_check", {"total_checks": len(results)})
        return results
    
    def _get_status_from_threshold(self, metric: str, value: float) -> HealthStatus:
        """Obtiene estado basado en umbrales"""
        if metric not in self.thresholds:
            return HealthStatus.UNKNOWN
        
        thresholds = self.thresholds[metric]
        if value >= thresholds.get("critical", float('inf')):
            return HealthStatus.CRITICAL
        elif value >= thresholds.get("warning", float('inf')):
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Obtiene resumen de salud"""
        if not self.health_results:
            return {"status": "unknown", "message": "No hay health checks realizados"}
        
        latest_results = self.health_results[-10:]  # Últimos 10
        
        statuses = [r.status for r in latest_results]
        if HealthStatus.CRITICAL in statuses:
            overall_status = HealthStatus.CRITICAL
        elif HealthStatus.WARNING in statuses:
            overall_status = HealthStatus.WARNING
        else:
            overall_status = HealthStatus.HEALTHY
        
        return {
            "overall_status": overall_status.value,
            "total_checks": len(self.health_results),
            "latest_checks": [r.to_dict() for r in latest_results]
        }

