"""
Auto Optimizer System
======================

Sistema de optimización automática basado en métricas.
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime

from ..analytics.analytics import get_analytics_engine
from .metrics import get_metrics_collector
from .dynamic_config import get_dynamic_config_manager

logger = logging.getLogger(__name__)


@dataclass
class OptimizationRule:
    """Regla de optimización."""
    name: str
    condition: str  # Ej: "p95_response_time > 1.0"
    action: str  # Ej: "increase_cache_size"
    parameters: Dict[str, Any]
    enabled: bool = True


class AutoOptimizer:
    """
    Optimizador automático.
    
    Optimiza el sistema automáticamente basado en métricas.
    """
    
    def __init__(self):
        """Inicializar optimizador automático."""
        self.rules: List[OptimizationRule] = []
        self.optimization_history: List[Dict[str, Any]] = []
        self.enabled = True
        
        # Reglas por defecto
        self._setup_default_rules()
    
    def _setup_default_rules(self) -> None:
        """Configurar reglas por defecto."""
        self.rules = [
            OptimizationRule(
                name="optimize_cache_on_slow_response",
                condition="p95_response_time > 1.0",
                action="increase_cache_size",
                parameters={"multiplier": 1.5}
            ),
            OptimizationRule(
                name="reduce_iterations_on_fast_response",
                condition="p95_response_time < 0.1",
                action="reduce_max_iterations",
                parameters={"reduction": 0.1}
            ),
            OptimizationRule(
                name="enable_profiling_on_high_error_rate",
                condition="error_rate > 0.05",
                action="enable_profiling",
                parameters={"enabled": True}
            )
        ]
    
    def add_rule(self, rule: OptimizationRule) -> None:
        """Agregar regla de optimización."""
        self.rules.append(rule)
        logger.info(f"Added optimization rule: {rule.name}")
    
    def evaluate_rules(self) -> List[Dict[str, Any]]:
        """
        Evaluar todas las reglas.
        
        Returns:
            Lista de optimizaciones aplicadas
        """
        if not self.enabled:
            return []
        
        analytics = get_analytics_engine()
        metrics = get_metrics_collector()
        config = get_dynamic_config_manager()
        
        # Generar reporte de performance
        report = analytics.generate_performance_report(period_hours=1)
        
        applied_optimizations = []
        
        for rule in self.rules:
            if not rule.enabled:
                continue
            
            try:
                # Evaluar condición
                if self._evaluate_condition(rule.condition, report, metrics):
                    # Aplicar acción
                    result = self._apply_action(rule.action, rule.parameters, config)
                    
                    optimization = {
                        "rule_name": rule.name,
                        "action": rule.action,
                        "parameters": rule.parameters,
                        "result": result,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    applied_optimizations.append(optimization)
                    self.optimization_history.append(optimization)
                    
                    logger.info(f"Applied optimization: {rule.name} -> {rule.action}")
            
            except Exception as e:
                logger.error(f"Error evaluating rule {rule.name}: {e}")
        
        return applied_optimizations
    
    def _evaluate_condition(
        self,
        condition: str,
        report: Any,
        metrics: Any
    ) -> bool:
        """
        Evaluar condición de regla.
        
        Args:
            condition: Condición a evaluar
            report: Reporte de performance
            metrics: Colector de métricas
            
        Returns:
            True si la condición se cumple
        """
        # Evaluar condiciones simples
        # En producción, usar un parser más robusto
        
        try:
            # Reemplazar variables
            condition = condition.replace("p95_response_time", str(report.p95_response_time))
            condition = condition.replace("error_rate", str(report.error_rate))
            condition = condition.replace("throughput", str(report.throughput))
            
            # Evaluar
            return eval(condition)
        except Exception as e:
            logger.error(f"Error evaluating condition: {e}")
            return False
    
    def _apply_action(
        self,
        action: str,
        parameters: Dict[str, Any],
        config: Any
    ) -> Dict[str, Any]:
        """
        Aplicar acción de optimización.
        
        Args:
            action: Acción a aplicar
            parameters: Parámetros de la acción
            config: Gestor de configuración
            
        Returns:
            Resultado de la acción
        """
        try:
            if action == "increase_cache_size":
                current_size = config.get("performance.cache_size", 128)
                new_size = int(current_size * parameters.get("multiplier", 1.5))
                config.set("performance.cache_size", new_size)
                return {"cache_size": new_size}
            
            elif action == "reduce_max_iterations":
                current_iter = config.get("optimization.max_iterations", 100)
                reduction = parameters.get("reduction", 0.1)
                new_iter = max(10, int(current_iter * (1 - reduction)))
                config.set("optimization.max_iterations", new_iter)
                return {"max_iterations": new_iter}
            
            elif action == "enable_profiling":
                enabled = parameters.get("enabled", True)
                config.set("performance.enable_profiling", enabled)
                return {"profiling_enabled": enabled}
            
            else:
                logger.warning(f"Unknown action: {action}")
                return {"error": f"Unknown action: {action}"}
        
        except Exception as e:
            logger.error(f"Error applying action {action}: {e}")
            return {"error": str(e)}
    
    def get_optimization_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Obtener historial de optimizaciones.
        
        Args:
            limit: Límite de resultados
            
        Returns:
            Historial de optimizaciones
        """
        return self.optimization_history[-limit:]
    
    def enable(self) -> None:
        """Habilitar optimizador automático."""
        self.enabled = True
        logger.info("Auto optimizer enabled")
    
    def disable(self) -> None:
        """Deshabilitar optimizador automático."""
        self.enabled = False
        logger.info("Auto optimizer disabled")


# Instancia global
_auto_optimizer: Optional[AutoOptimizer] = None


def get_auto_optimizer() -> AutoOptimizer:
    """Obtener instancia global del optimizador automático."""
    global _auto_optimizer
    if _auto_optimizer is None:
        _auto_optimizer = AutoOptimizer()
    return _auto_optimizer






