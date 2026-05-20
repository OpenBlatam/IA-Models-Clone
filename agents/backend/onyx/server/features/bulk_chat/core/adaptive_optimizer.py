"""
Adaptive Optimizer - Optimizador Adaptativo
============================================

Sistema de optimización adaptativa que ajusta parámetros automáticamente basado en métricas de rendimiento.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
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
    ERROR_RATE = "error_rate"
    COST = "cost"
    HYBRID = "hybrid"


@dataclass
class OptimizationParameter:
    """Parámetro a optimizar."""
    parameter_id: str
    name: str
    current_value: float
    min_value: float
    max_value: float
    step: float = 0.1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationGoal:
    """Objetivo de optimización."""
    goal_id: str
    target: OptimizationTarget
    target_value: Optional[float] = None
    maximize: bool = True
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationResult:
    """Resultado de optimización."""
    result_id: str
    timestamp: datetime
    parameters: Dict[str, float]
    metrics: Dict[str, float]
    score: float
    improvement: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdaptiveOptimizer:
    """Optimizador adaptativo."""
    
    def __init__(self):
        self.parameters: Dict[str, OptimizationParameter] = {}
        self.goals: List[OptimizationGoal] = []
        self.results: deque = deque(maxlen=10000)
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._lock = asyncio.Lock()
        self._optimization_active = False
    
    def register_parameter(
        self,
        parameter_id: str,
        name: str,
        initial_value: float,
        min_value: float,
        max_value: float,
        step: float = 0.1,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Registrar parámetro a optimizar."""
        param = OptimizationParameter(
            parameter_id=parameter_id,
            name=name,
            current_value=initial_value,
            min_value=min_value,
            max_value=max_value,
            step=step,
            metadata=metadata or {},
        )
        
        async def save_parameter():
            async with self._lock:
                self.parameters[parameter_id] = param
        
        asyncio.create_task(save_parameter())
        
        logger.info(f"Registered optimization parameter: {parameter_id}")
        return parameter_id
    
    def add_goal(
        self,
        goal_id: str,
        target: OptimizationTarget,
        target_value: Optional[float] = None,
        maximize: bool = True,
        weight: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Agregar objetivo de optimización."""
        goal = OptimizationGoal(
            goal_id=goal_id,
            target=target,
            target_value=target_value,
            maximize=maximize,
            weight=weight,
            metadata=metadata or {},
        )
        
        async def save_goal():
            async with self._lock:
                self.goals.append(goal)
        
        asyncio.create_task(save_goal())
        
        logger.info(f"Added optimization goal: {goal_id}")
        return goal_id
    
    def record_metric(
        self,
        metric_name: str,
        value: float,
        timestamp: Optional[datetime] = None,
    ):
        """Registrar métrica."""
        timestamp = timestamp or datetime.now()
        
        async def save_metric():
            async with self._lock:
                self.metrics_history[metric_name].append({
                    "value": value,
                    "timestamp": timestamp,
                })
        
        asyncio.create_task(save_metric())
        
        # Trigger optimización si está activa
        if self._optimization_active:
            asyncio.create_task(self._check_and_optimize())
    
    async def _check_and_optimize(self):
        """Verificar y optimizar."""
        if not self.goals:
            return
        
        # Calcular score actual
        current_score = await self._calculate_score()
        
        # Intentar optimización
        best_params = None
        best_score = current_score
        best_improvement = 0.0
        
        # Buscar mejores parámetros
        for param_id, param in self.parameters.items():
            # Probar incremento
            new_value = min(param.max_value, param.current_value + param.step)
            if new_value != param.current_value:
                old_value = param.current_value
                param.current_value = new_value
                
                # Simular score (en producción, ejecutaría tests reales)
                new_score = await self._calculate_score()
                
                if new_score > best_score:
                    best_score = new_score
                    best_params = {param_id: new_value}
                    best_improvement = new_score - current_score
                
                param.current_value = old_value
            
            # Probar decremento
            new_value = max(param.min_value, param.current_value - param.step)
            if new_value != param.current_value:
                old_value = param.current_value
                param.current_value = new_value
                
                new_score = await self._calculate_score()
                
                if new_score > best_score:
                    best_score = new_score
                    best_params = {param_id: new_value}
                    best_improvement = new_score - current_score
                
                param.current_value = old_value
        
        # Aplicar mejores parámetros
        if best_params:
            async with self._lock:
                for param_id, value in best_params.items():
                    self.parameters[param_id].current_value = value
            
            # Guardar resultado
            result = OptimizationResult(
                result_id=f"opt_{datetime.now().timestamp()}",
                timestamp=datetime.now(),
                parameters=best_params.copy(),
                metrics={},
                score=best_score,
                improvement=best_improvement,
            )
            
            async with self._lock:
                self.results.append(result)
            
            logger.info(f"Optimization applied: {best_params}, improvement: {best_improvement:.4f}")
    
    async def _calculate_score(self) -> float:
        """Calcular score basado en objetivos."""
        if not self.goals:
            return 0.0
        
        total_score = 0.0
        total_weight = 0.0
        
        for goal in self.goals:
            metric_history = self.metrics_history.get(goal.target.value, deque())
            
            if not metric_history:
                continue
            
            # Calcular métrica actual
            recent_values = [m["value"] for m in list(metric_history)[-10:]]
            if not recent_values:
                continue
            
            current_metric = statistics.mean(recent_values)
            
            # Calcular score para este objetivo
            if goal.target_value:
                if goal.maximize:
                    score = min(1.0, current_metric / goal.target_value) if goal.target_value > 0 else 0.0
                else:
                    score = min(1.0, goal.target_value / current_metric) if current_metric > 0 else 0.0
            else:
                # Normalizar basado en historial
                if len(recent_values) > 1:
                    mean_val = statistics.mean(recent_values)
                    try:
                        std_val = statistics.stdev(recent_values)
                    except statistics.StatisticsError:
                        std_val = 0.0
                    if std_val > 0:
                        score = max(0.0, min(1.0, (current_metric - mean_val + std_val) / (2 * std_val)))
                    else:
                        score = 0.5
                else:
                    score = 0.5
            
            total_score += score * goal.weight
            total_weight += goal.weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def start_optimization(self):
        """Iniciar optimización."""
        self._optimization_active = True
        logger.info("Adaptive optimization started")
    
    def stop_optimization(self):
        """Detener optimización."""
        self._optimization_active = False
        logger.info("Adaptive optimization stopped")
    
    def get_parameter(self, parameter_id: str) -> Optional[Dict[str, Any]]:
        """Obtener información de parámetro."""
        param = self.parameters.get(parameter_id)
        if not param:
            return None
        
        return {
            "parameter_id": param.parameter_id,
            "name": param.name,
            "current_value": param.current_value,
            "min_value": param.min_value,
            "max_value": param.max_value,
            "step": param.step,
        }
    
    def get_optimization_results(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Obtener resultados de optimización."""
        results = list(self.results)[-limit:]
        
        return [
            {
                "result_id": r.result_id,
                "timestamp": r.timestamp.isoformat(),
                "parameters": r.parameters,
                "score": r.score,
                "improvement": r.improvement,
            }
            for r in results
        ]
    
    def get_adaptive_optimizer_summary(self) -> Dict[str, Any]:
        """Obtener resumen del optimizador."""
        return {
            "optimization_active": self._optimization_active,
            "total_parameters": len(self.parameters),
            "total_goals": len(self.goals),
            "total_results": len(self.results),
            "total_metrics": len(self.metrics_history),
        }

