"""
Intelligent Optimizer - Optimizador Inteligente
==============================================

Sistema de optimización inteligente con ML para mejorar rendimiento automáticamente.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)


@dataclass
class OptimizationSuggestion:
    """Sugerencia de optimización."""
    suggestion_id: str
    parameter: str
    current_value: Any
    suggested_value: Any
    expected_improvement: float
    confidence: float
    reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class IntelligentOptimizer:
    """Optimizador inteligente con ML."""
    
    def __init__(self):
        self.performance_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.optimizations: List[OptimizationSuggestion] = []
        self.applied_optimizations: Dict[str, Any] = {}
    
    async def record_performance(
        self,
        operation: str,
        parameters: Dict[str, Any],
        metrics: Dict[str, float],
    ):
        """
        Registrar rendimiento de operación.
        
        Args:
            operation: Nombre de la operación
            parameters: Parámetros utilizados
            metrics: Métricas de rendimiento
        """
        record = {
            "operation": operation,
            "parameters": parameters,
            "metrics": metrics,
            "timestamp": datetime.now(),
        }
        
        self.performance_history[operation].append(record)
        
        # Mantener solo últimos 1000 registros
        if len(self.performance_history[operation]) > 1000:
            self.performance_history[operation].pop(0)
        
        logger.debug(f"Recorded performance for {operation}")
    
    async def analyze_and_suggest(
        self,
        operation: str,
        current_parameters: Dict[str, Any],
    ) -> List[OptimizationSuggestion]:
        """
        Analizar y sugerir optimizaciones.
        
        Args:
            operation: Nombre de la operación
            current_parameters: Parámetros actuales
        
        Returns:
            Lista de sugerencias de optimización
        """
        history = self.performance_history.get(operation, [])
        
        if len(history) < 10:
            return []  # No hay suficientes datos
        
        suggestions = []
        
        # Analizar cada parámetro
        for param_name, param_value in current_parameters.items():
            if isinstance(param_value, (int, float)):
                suggestion = await self._analyze_parameter(
                    operation,
                    param_name,
                    param_value,
                    history,
                )
                if suggestion:
                    suggestions.append(suggestion)
        
        return suggestions
    
    async def _analyze_parameter(
        self,
        operation: str,
        param_name: str,
        current_value: Any,
        history: List[Dict[str, Any]],
    ) -> Optional[OptimizationSuggestion]:
        """Analizar parámetro específico."""
        # Agrupar por valor de parámetro
        value_performance: Dict[Any, List[float]] = defaultdict(list)
        
        for record in history:
            param_value = record["parameters"].get(param_name)
            if param_value is not None:
                # Usar métrica de throughput como medida de rendimiento
                throughput = record["metrics"].get("throughput", 0.0)
                value_performance[param_value].append(throughput)
        
        if not value_performance:
            return None
        
        # Calcular promedio por valor
        avg_performance = {
            value: statistics.mean(perfs)
            for value, perfs in value_performance.items()
        }
        
        # Encontrar mejor valor
        best_value = max(avg_performance.items(), key=lambda x: x[1])[0]
        best_performance = avg_performance[best_value]
        current_performance = avg_performance.get(current_value, 0.0)
        
        if best_performance <= current_performance:
            return None  # No hay mejora
        
        # Calcular mejora esperada
        improvement = ((best_performance - current_performance) / current_performance * 100) if current_performance > 0 else 0.0
        
        # Calcular confianza basada en número de muestras
        samples = len(value_performance[best_value])
        confidence = min(1.0, samples / 20.0)
        
        suggestion_id = f"opt_{operation}_{param_name}_{datetime.now().timestamp()}"
        
        suggestion = OptimizationSuggestion(
            suggestion_id=suggestion_id,
            parameter=param_name,
            current_value=current_value,
            suggested_value=best_value,
            expected_improvement=improvement,
            confidence=confidence,
            reason=f"Based on {samples} samples, {best_value} showed {improvement:.1f}% better performance",
            metadata={
                "samples": samples,
                "current_performance": current_performance,
                "best_performance": best_performance,
            },
        )
        
        self.optimizations.append(suggestion)
        return suggestion
    
    async def apply_optimization(
        self,
        suggestion_id: str,
    ) -> bool:
        """Aplicar optimización sugerida."""
        suggestion = next(
            (s for s in self.optimizations if s.suggestion_id == suggestion_id),
            None,
        )
        
        if not suggestion:
            return False
        
        self.applied_optimizations[suggestion.parameter] = suggestion.suggested_value
        
        logger.info(
            f"Applied optimization: {suggestion.parameter} = "
            f"{suggestion.suggested_value} (expected improvement: {suggestion.expected_improvement:.1f}%)"
        )
        
        return True
    
    def get_applied_optimizations(self) -> Dict[str, Any]:
        """Obtener optimizaciones aplicadas."""
        return dict(self.applied_optimizations)
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Obtener historial de optimizaciones."""
        return [
            {
                "suggestion_id": s.suggestion_id,
                "parameter": s.parameter,
                "current_value": s.current_value,
                "suggested_value": s.suggested_value,
                "expected_improvement": s.expected_improvement,
                "confidence": s.confidence,
                "reason": s.reason,
            }
            for s in self.optimizations[-100:]  # Últimas 100
        ]
















