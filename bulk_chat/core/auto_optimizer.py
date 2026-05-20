"""
Auto Optimizer - Optimizador Automático
======================================

Sistema de optimización automática con análisis de performance y ajuste dinámico de parámetros.
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


class OptimizationTarget(Enum):
    """Objetivo de optimización."""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    MEMORY = "memory"
    CPU = "cpu"
    COST = "cost"
    QUALITY = "quality"


@dataclass
class OptimizationResult:
    """Resultado de optimización."""
    optimization_id: str
    target: OptimizationTarget
    parameter_name: str
    old_value: Any
    new_value: Any
    improvement_percentage: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetric:
    """Métrica de performance."""
    metric_name: str
    value: float
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)


class AutoOptimizer:
    """Optimizador automático."""
    
    def __init__(self):
        self.optimizations: List[OptimizationResult] = []
        self.performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.parameter_ranges: Dict[str, Dict[str, Any]] = {}
        self.optimization_targets: Dict[str, OptimizationTarget] = {}
        self._lock = asyncio.Lock()
    
    def register_parameter(
        self,
        parameter_name: str,
        min_value: Any,
        max_value: Any,
        current_value: Any,
        target: OptimizationTarget,
        step: Optional[float] = None,
    ):
        """Registrar parámetro para optimización."""
        self.parameter_ranges[parameter_name] = {
            "min": min_value,
            "max": max_value,
            "current": current_value,
            "step": step,
        }
        self.optimization_targets[parameter_name] = target
        
        logger.info(f"Registered parameter for optimization: {parameter_name}")
    
    def record_performance(
        self,
        metric_name: str,
        value: float,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Registrar métrica de performance."""
        metric = PerformanceMetric(
            metric_name=metric_name,
            value=value,
            timestamp=datetime.now(),
            context=context or {},
        )
        
        self.performance_history[metric_name].append(metric)
        
        # Analizar y optimizar si es necesario
        asyncio.create_task(self._analyze_and_optimize(metric_name))
    
    async def _analyze_and_optimize(self, metric_name: str):
        """Analizar y optimizar."""
        history = self.performance_history.get(metric_name)
        if not history or len(history) < 100:
            return
        
        # Obtener valores recientes
        recent_values = [m.value for m in list(history)[-100:]]
        
        # Calcular tendencia
        if len(recent_values) < 10:
            return
        
        avg_recent = statistics.mean(recent_values[-10:])
        avg_older = statistics.mean(recent_values[-100:-10]) if len(recent_values) > 10 else avg_recent
        
        # Buscar parámetros relacionados
        for param_name, target in self.optimization_targets.items():
            if metric_name.startswith(param_name) or param_name in metric_name:
                await self._optimize_parameter(param_name, target, avg_recent, avg_older)
    
    async def _optimize_parameter(
        self,
        parameter_name: str,
        target: OptimizationTarget,
        current_avg: float,
        previous_avg: float,
    ):
        """Optimizar parámetro."""
        param_range = self.parameter_ranges.get(parameter_name)
        if not param_range:
            return
        
        improvement = 0.0
        new_value = param_range["current"]
        
        if target == OptimizationTarget.LATENCY:
            # Reducir latencia
            if current_avg > previous_avg * 1.1:  # Empeoró más del 10%
                # Reducir parámetro (ej. batch size, timeout)
                if isinstance(param_range["current"], (int, float)):
                    step = param_range.get("step", (param_range["max"] - param_range["min"]) * 0.1)
                    new_value = max(param_range["min"], param_range["current"] - step)
                    improvement = ((current_avg - previous_avg) / previous_avg) * 100
        
        elif target == OptimizationTarget.THROUGHPUT:
            # Aumentar throughput
            if current_avg < previous_avg * 0.9:  # Empeoró más del 10%
                if isinstance(param_range["current"], (int, float)):
                    step = param_range.get("step", (param_range["max"] - param_range["min"]) * 0.1)
                    new_value = min(param_range["max"], param_range["current"] + step)
                    improvement = ((previous_avg - current_avg) / previous_avg) * 100
        
        if new_value != param_range["current"] and improvement > 5.0:  # Mejora mínima del 5%
            old_value = param_range["current"]
            param_range["current"] = new_value
            
            optimization = OptimizationResult(
                optimization_id=f"opt_{parameter_name}_{datetime.now().timestamp()}",
                target=target,
                parameter_name=parameter_name,
                old_value=old_value,
                new_value=new_value,
                improvement_percentage=improvement,
                metadata={
                    "current_avg": current_avg,
                    "previous_avg": previous_avg,
                },
            )
            
            async with self._lock:
                self.optimizations.append(optimization)
                if len(self.optimizations) > 10000:
                    self.optimizations.pop(0)
            
            logger.info(f"Optimized {parameter_name}: {old_value} -> {new_value} ({improvement:.2f}% improvement)")
    
    def get_parameter_value(self, parameter_name: str) -> Optional[Any]:
        """Obtener valor actual de parámetro."""
        param_range = self.parameter_ranges.get(parameter_name)
        if not param_range:
            return None
        
        return param_range["current"]
    
    def get_optimizations(
        self,
        parameter_name: Optional[str] = None,
        target: Optional[OptimizationTarget] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Obtener optimizaciones."""
        optimizations = self.optimizations
        
        if parameter_name:
            optimizations = [o for o in optimizations if o.parameter_name == parameter_name]
        
        if target:
            optimizations = [o for o in optimizations if o.target == target]
        
        optimizations.sort(key=lambda o: o.timestamp, reverse=True)
        
        return [
            {
                "optimization_id": o.optimization_id,
                "target": o.target.value,
                "parameter_name": o.parameter_name,
                "old_value": o.old_value,
                "new_value": o.new_value,
                "improvement_percentage": o.improvement_percentage,
                "timestamp": o.timestamp.isoformat(),
                "metadata": o.metadata,
            }
            for o in optimizations[:limit]
        ]
    
    def get_auto_optimizer_summary(self) -> Dict[str, Any]:
        """Obtener resumen del optimizador."""
        by_target: Dict[str, int] = defaultdict(int)
        total_improvement = 0.0
        
        for opt in self.optimizations:
            by_target[opt.target.value] += 1
            total_improvement += opt.improvement_percentage
        
        return {
            "total_parameters": len(self.parameter_ranges),
            "total_optimizations": len(self.optimizations),
            "optimizations_by_target": dict(by_target),
            "avg_improvement": total_improvement / len(self.optimizations) if self.optimizations else 0.0,
        }
