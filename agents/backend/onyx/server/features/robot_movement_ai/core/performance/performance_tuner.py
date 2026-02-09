"""
Performance Tuner System
========================

Sistema de ajuste automático de performance.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class TuningParameter:
    """Parámetro de ajuste."""
    name: str
    current_value: Any
    optimal_value: Any
    range_min: Any
    range_max: Any
    step: Any
    impact: float = 0.0  # Impacto en performance (0-1)


@dataclass
class TuningResult:
    """Resultado de ajuste."""
    parameter_name: str
    old_value: Any
    new_value: Any
    performance_gain: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class PerformanceTuner:
    """
    Ajustador de performance.
    
    Ajusta automáticamente parámetros para optimizar performance.
    """
    
    def __init__(self):
        """Inicializar ajustador de performance."""
        self.parameters: Dict[str, TuningParameter] = {}
        self.tuning_history: List[TuningResult] = []
        self.baseline_performance: Optional[float] = None
    
    def register_parameter(
        self,
        name: str,
        current_value: Any,
        range_min: Any,
        range_max: Any,
        step: Any,
        optimal_value: Optional[Any] = None
    ) -> TuningParameter:
        """
        Registrar parámetro para ajuste.
        
        Args:
            name: Nombre del parámetro
            current_value: Valor actual
            range_min: Valor mínimo
            range_max: Valor máximo
            step: Paso de ajuste
            optimal_value: Valor óptimo conocido (opcional)
            
        Returns:
            Parámetro registrado
        """
        param = TuningParameter(
            name=name,
            current_value=current_value,
            optimal_value=optimal_value or current_value,
            range_min=range_min,
            range_max=range_max,
            step=step
        )
        
        self.parameters[name] = param
        logger.info(f"Registered tuning parameter: {name}")
        
        return param
    
    def tune_parameter(
        self,
        parameter_name: str,
        performance_metric: callable,
        iterations: int = 10
    ) -> TuningResult:
        """
        Ajustar parámetro.
        
        Args:
            parameter_name: Nombre del parámetro
            performance_metric: Función que retorna métrica de performance
            iterations: Número de iteraciones
            
        Returns:
            Resultado del ajuste
        """
        if parameter_name not in self.parameters:
            raise ValueError(f"Parameter not found: {parameter_name}")
        
        param = self.parameters[parameter_name]
        
        # Medir performance actual
        current_perf = performance_metric()
        
        # Probar diferentes valores
        best_value = param.current_value
        best_perf = current_perf
        
        test_values = self._generate_test_values(param, iterations)
        
        for test_value in test_values:
            # Aplicar valor temporalmente
            old_value = param.current_value
            param.current_value = test_value
            
            # Medir performance
            test_perf = performance_metric()
            
            # Restaurar valor
            param.current_value = old_value
            
            if test_perf > best_perf:
                best_value = test_value
                best_perf = test_perf
        
        # Aplicar mejor valor
        old_value = param.current_value
        param.current_value = best_value
        
        performance_gain = ((best_perf - current_perf) / current_perf * 100) if current_perf > 0 else 0.0
        
        result = TuningResult(
            parameter_name=parameter_name,
            old_value=old_value,
            new_value=best_value,
            performance_gain=performance_gain
        )
        
        self.tuning_history.append(result)
        logger.info(f"Tuned {parameter_name}: {old_value} -> {best_value} ({performance_gain:.2f}% gain)")
        
        return result
    
    def _generate_test_values(self, param: TuningParameter, iterations: int) -> List[Any]:
        """Generar valores de prueba."""
        import numpy as np
        
        if isinstance(param.range_min, (int, float)):
            # Valores numéricos
            if iterations <= 1:
                return [param.range_min, param.range_max]
            
            return np.linspace(
                param.range_min,
                param.range_max,
                iterations
            ).tolist()
        else:
            # Valores discretos
            return [param.range_min, param.range_max]
    
    def auto_tune_all(
        self,
        performance_metric: callable,
        max_iterations: int = 5
    ) -> List[TuningResult]:
        """
        Ajustar todos los parámetros automáticamente.
        
        Args:
            performance_metric: Función que retorna métrica de performance
            max_iterations: Máximo de iteraciones por parámetro
            
        Returns:
            Lista de resultados de ajuste
        """
        results = []
        
        for param_name in self.parameters.keys():
            try:
                result = self.tune_parameter(
                    param_name,
                    performance_metric,
                    iterations=max_iterations
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Error tuning {param_name}: {e}")
        
        return results
    
    def get_tuning_summary(self) -> Dict[str, Any]:
        """Obtener resumen de ajustes."""
        if not self.tuning_history:
            return {
                "total_tunings": 0,
                "average_gain": 0.0,
                "total_gain": 0.0
            }
        
        total_gain = sum(r.performance_gain for r in self.tuning_history)
        avg_gain = total_gain / len(self.tuning_history)
        
        return {
            "total_tunings": len(self.tuning_history),
            "average_gain": avg_gain,
            "total_gain": total_gain,
            "parameters": {
                name: {
                    "current_value": param.current_value,
                    "optimal_value": param.optimal_value,
                    "range": (param.range_min, param.range_max)
                }
                for name, param in self.parameters.items()
            }
        }


# Instancia global
_performance_tuner: Optional[PerformanceTuner] = None


def get_performance_tuner() -> PerformanceTuner:
    """Obtener instancia global del ajustador de performance."""
    global _performance_tuner
    if _performance_tuner is None:
        _performance_tuner = PerformanceTuner()
    return _performance_tuner






