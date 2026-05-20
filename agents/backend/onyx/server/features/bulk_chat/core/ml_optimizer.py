"""
ML Optimizer - Sistema de Machine Learning Avanzado
==================================================

Sistema de optimización basado en machine learning.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import math

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Resultado de optimización."""
    parameter_name: str
    optimal_value: float
    confidence: float
    improvement: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class MLOptimizer:
    """Optimizador basado en ML."""
    
    def __init__(self):
        self.performance_data: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.optimization_history: List[OptimizationResult] = []
    
    async def record_performance(
        self,
        parameters: Dict[str, float],
        performance_metric: float,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Registrar datos de rendimiento.
        
        Args:
            parameters: Parámetros utilizados
            performance_metric: Métrica de rendimiento (mayor es mejor)
            metadata: Metadatos adicionales
        """
        record = {
            "parameters": parameters,
            "performance": performance_metric,
            "timestamp": datetime.now(),
            "metadata": metadata or {},
        }
        
        # Almacenar para cada parámetro
        for param_name, param_value in parameters.items():
            self.performance_data[param_name].append({
                "value": param_value,
                "performance": performance_metric,
                "timestamp": datetime.now(),
            })
        
        logger.debug(f"Recorded performance: {performance_metric} with parameters {parameters}")
    
    async def optimize_parameter(
        self,
        parameter_name: str,
        min_value: float = 0.0,
        max_value: float = 1.0,
        step: float = 0.1,
    ) -> Optional[OptimizationResult]:
        """
        Optimizar parámetro usando datos históricos.
        
        Args:
            parameter_name: Nombre del parámetro
            min_value: Valor mínimo
            max_value: Valor máximo
            step: Paso de búsqueda
        
        Returns:
            Resultado de optimización
        """
        if parameter_name not in self.performance_data:
            logger.warning(f"No data for parameter: {parameter_name}")
            return None
        
        data = self.performance_data[parameter_name]
        
        if len(data) < 3:
            logger.warning(f"Insufficient data for optimization: {len(data)} points")
            return None
        
        # Agrupar por valor y calcular promedio de rendimiento
        value_performance: Dict[float, List[float]] = defaultdict(list)
        
        for record in data:
            value = round(record["value"] / step) * step  # Redondear al paso más cercano
            value = max(min_value, min(max_value, value))  # Asegurar rango
            value_performance[value].append(record["performance"])
        
        # Calcular promedio de rendimiento por valor
        avg_performance = {
            value: sum(perfs) / len(perfs)
            for value, perfs in value_performance.items()
        }
        
        # Encontrar valor óptimo
        if not avg_performance:
            return None
        
        optimal_value = max(avg_performance.items(), key=lambda x: x[1])[0]
        optimal_performance = avg_performance[optimal_value]
        
        # Calcular confianza basada en número de muestras
        samples_count = len(value_performance[optimal_value])
        confidence = min(1.0, samples_count / 10.0)  # Máximo 1.0 con 10+ muestras
        
        # Calcular mejora
        all_performances = [p for perfs in value_performance.values() for p in perfs]
        baseline = sum(all_performances) / len(all_performances) if all_performances else 0.0
        improvement = ((optimal_performance - baseline) / baseline * 100) if baseline > 0 else 0.0
        
        result = OptimizationResult(
            parameter_name=parameter_name,
            optimal_value=optimal_value,
            confidence=confidence,
            improvement=improvement,
            metadata={
                "samples": samples_count,
                "optimal_performance": optimal_performance,
                "baseline": baseline,
            },
        )
        
        self.optimization_history.append(result)
        
        logger.info(
            f"Optimized {parameter_name}: {optimal_value} "
            f"(improvement: {improvement:.2f}%, confidence: {confidence:.2f})"
        )
        
        return result
    
    async def predict_performance(
        self,
        parameter_name: str,
        value: float,
    ) -> Optional[float]:
        """
        Predecir rendimiento para un valor de parámetro.
        
        Args:
            parameter_name: Nombre del parámetro
            value: Valor del parámetro
        
        Returns:
            Rendimiento predicho
        """
        if parameter_name not in self.performance_data:
            return None
        
        data = self.performance_data[parameter_name]
        
        if len(data) < 2:
            return None
        
        # Interpolación simple basada en valores cercanos
        sorted_data = sorted(data, key=lambda x: abs(x["value"] - value))
        
        # Usar promedio de los 3 valores más cercanos
        closest = sorted_data[:3]
        if closest:
            predicted = sum(r["performance"] for r in closest) / len(closest)
            return predicted
        
        return None
    
    def get_optimization_history(
        self,
        parameter_name: Optional[str] = None,
    ) -> List[OptimizationResult]:
        """Obtener historial de optimizaciones."""
        results = self.optimization_history
        
        if parameter_name:
            results = [r for r in results if r.parameter_name == parameter_name]
        
        return results
































