"""
Optimizaciones de Sistemas Autónomos para Routing.

Este módulo implementa capacidades de auto-optimización, auto-reparación
y auto-adaptación para sistemas de routing autónomos.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class AutonomyLevel(Enum):
    """Niveles de autonomía."""
    MANUAL = "manual"
    ASSISTED = "assisted"
    SEMI_AUTONOMOUS = "semi_autonomous"
    FULLY_AUTONOMOUS = "fully_autonomous"


class AdaptationStrategy(Enum):
    """Estrategias de adaptación."""
    REACTIVE = "reactive"
    PROACTIVE = "proactive"
    PREDICTIVE = "predictive"
    ADAPTIVE = "adaptive"


@dataclass
class SystemState:
    """Estado del sistema."""
    timestamp: float
    performance_metrics: Dict[str, float]
    resource_usage: Dict[str, float]
    error_rate: float
    health_score: float


@dataclass
class AdaptationAction:
    """Acción de adaptación."""
    action_type: str
    parameters: Dict[str, Any]
    expected_impact: float
    confidence: float
    timestamp: float = field(default_factory=time.time)


class SelfHealingSystem:
    """Sistema de auto-reparación."""
    
    def __init__(self):
        self.health_history: List[SystemState] = []
        self.recovery_actions: List[AdaptationAction] = []
        self.successful_recoveries = 0
        self.failed_recoveries = 0
        self.health_threshold = 0.7
    
    def monitor_health(self, metrics: Dict[str, float]) -> SystemState:
        """Monitorear salud del sistema."""
        state = SystemState(
            timestamp=time.time(),
            performance_metrics=metrics,
            resource_usage={},
            error_rate=metrics.get("error_rate", 0.0),
            health_score=self._calculate_health_score(metrics)
        )
        
        self.health_history.append(state)
        
        # Mantener solo últimos 100 estados
        if len(self.health_history) > 100:
            self.health_history.pop(0)
        
        return state
    
    def _calculate_health_score(self, metrics: Dict[str, float]) -> float:
        """Calcular puntuación de salud."""
        error_rate = metrics.get("error_rate", 0.0)
        latency = metrics.get("latency", 0.0)
        throughput = metrics.get("throughput", 0.0)
        
        # Normalizar métricas
        error_score = max(0, 1.0 - error_rate)
        latency_score = max(0, 1.0 - min(latency / 1000.0, 1.0))
        throughput_score = min(1.0, throughput / 1000.0)
        
        # Ponderar
        health = (error_score * 0.4 + latency_score * 0.3 + throughput_score * 0.3)
        return health
    
    def detect_anomaly(self, state: SystemState) -> bool:
        """Detectar anomalías."""
        if state.health_score < self.health_threshold:
            return True
        
        # Detectar cambios abruptos
        if len(self.health_history) >= 2:
            prev_state = self.health_history[-2]
            health_change = abs(state.health_score - prev_state.health_score)
            if health_change > 0.3:
                return True
        
        return False
    
    def recover(self, state: SystemState) -> Optional[AdaptationAction]:
        """Intentar recuperación automática."""
        if not self.detect_anomaly(state):
            return None
        
        # Determinar acción de recuperación
        if state.error_rate > 0.1:
            action = AdaptationAction(
                action_type="reset_connections",
                parameters={"timeout": 5.0},
                expected_impact=0.3,
                confidence=0.7
            )
        elif state.health_score < 0.5:
            action = AdaptationAction(
                action_type="scale_resources",
                parameters={"scale_factor": 1.5},
                expected_impact=0.5,
                confidence=0.8
            )
        else:
            action = AdaptationAction(
                action_type="optimize_cache",
                parameters={"clear_rate": 0.3},
                expected_impact=0.2,
                confidence=0.6
            )
        
        self.recovery_actions.append(action)
        
        # Simular recuperación
        if np.random.random() > 0.2:  # 80% éxito
            self.successful_recoveries += 1
            return action
        else:
            self.failed_recoveries += 1
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        return {
            "health_history_size": len(self.health_history),
            "recovery_actions": len(self.recovery_actions),
            "successful_recoveries": self.successful_recoveries,
            "failed_recoveries": self.failed_recoveries,
            "recovery_rate": self.successful_recoveries / max(
                self.successful_recoveries + self.failed_recoveries, 1
            ),
            "health_threshold": self.health_threshold
        }


class SelfOptimizingSystem:
    """Sistema de auto-optimización."""
    
    def __init__(self):
        self.optimization_history: List[Dict[str, Any]] = []
        self.parameter_space: Dict[str, Tuple[float, float]] = {}
        self.best_config: Dict[str, Any] = {}
        self.best_performance = 0.0
        self.optimization_iterations = 0
    
    def define_parameter_space(self, parameters: Dict[str, Tuple[float, float]]):
        """Definir espacio de parámetros."""
        self.parameter_space = parameters
    
    def optimize(self, performance_func: Callable[[Dict[str, Any]], float],
                 max_iterations: int = 10) -> Dict[str, Any]:
        """Optimizar parámetros."""
        if not self.parameter_space:
            return {}
        
        best_config = {}
        best_performance = float('-inf')
        
        for iteration in range(max_iterations):
            # Generar configuración aleatoria
            config = {}
            for param, (min_val, max_val) in self.parameter_space.items():
                config[param] = np.random.uniform(min_val, max_val)
            
            # Evaluar rendimiento
            try:
                performance = performance_func(config)
                
                if performance > best_performance:
                    best_performance = performance
                    best_config = config.copy()
                
                self.optimization_history.append({
                    "iteration": iteration,
                    "config": config,
                    "performance": performance
                })
            except Exception as e:
                logger.warning(f"Optimization iteration failed: {e}")
            
            self.optimization_iterations += 1
        
        if best_performance > self.best_performance:
            self.best_performance = best_performance
            self.best_config = best_config
        
        return best_config
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        return {
            "optimization_iterations": self.optimization_iterations,
            "best_performance": self.best_performance,
            "best_config": self.best_config,
            "parameter_space_size": len(self.parameter_space),
            "history_size": len(self.optimization_history)
        }


class AdaptiveRouting:
    """Sistema de routing adaptativo."""
    
    def __init__(self, strategy: AdaptationStrategy = AdaptationStrategy.ADAPTIVE):
        self.strategy = strategy
        self.adaptation_history: List[AdaptationAction] = []
        self.performance_baseline: Dict[str, float] = {}
        self.adaptation_count = 0
    
    def adapt(self, current_performance: Dict[str, float],
              target_performance: Dict[str, float]) -> Optional[AdaptationAction]:
        """Adaptar sistema según rendimiento."""
        self.adaptation_count += 1
        
        # Calcular desviación
        deviations = {}
        for key in current_performance:
            if key in target_performance:
                deviations[key] = target_performance[key] - current_performance[key]
        
        # Determinar acción según estrategia
        if self.strategy == AdaptationStrategy.REACTIVE:
            action = self._reactive_adaptation(deviations)
        elif self.strategy == AdaptationStrategy.PROACTIVE:
            action = self._proactive_adaptation(deviations)
        elif self.strategy == AdaptationStrategy.PREDICTIVE:
            action = self._predictive_adaptation(deviations)
        else:  # ADAPTIVE
            action = self._adaptive_adaptation(deviations)
        
        if action:
            self.adaptation_history.append(action)
        
        return action
    
    def _reactive_adaptation(self, deviations: Dict[str, float]) -> Optional[AdaptationAction]:
        """Adaptación reactiva."""
        max_dev = max(abs(v) for v in deviations.values()) if deviations else 0
        
        if max_dev > 0.2:
            return AdaptationAction(
                action_type="adjust_parameters",
                parameters={"adjustment": 0.1},
                expected_impact=0.2,
                confidence=0.6
            )
        return None
    
    def _proactive_adaptation(self, deviations: Dict[str, float]) -> Optional[AdaptationAction]:
        """Adaptación proactiva."""
        # Predecir problemas antes de que ocurran
        trend = self._calculate_trend()
        
        if trend < -0.1:
            return AdaptationAction(
                action_type="preventive_adjustment",
                parameters={"adjustment": 0.15},
                expected_impact=0.3,
                confidence=0.7
            )
        return None
    
    def _predictive_adaptation(self, deviations: Dict[str, float]) -> Optional[AdaptationAction]:
        """Adaptación predictiva."""
        # Usar ML para predecir
        predicted_deviation = self._predict_deviation()
        
        if predicted_deviation > 0.15:
            return AdaptationAction(
                action_type="predictive_adjustment",
                parameters={"adjustment": 0.12},
                expected_impact=0.25,
                confidence=0.75
            )
        return None
    
    def _adaptive_adaptation(self, deviations: Dict[str, float]) -> Optional[AdaptationAction]:
        """Adaptación adaptativa (combinación)."""
        # Combinar estrategias
        reactive = self._reactive_adaptation(deviations)
        proactive = self._proactive_adaptation(deviations)
        
        if proactive:
            return proactive
        return reactive
    
    def _calculate_trend(self) -> float:
        """Calcular tendencia de rendimiento."""
        if len(self.adaptation_history) < 2:
            return 0.0
        
        # Simplificado
        return -0.05
    
    def _predict_deviation(self) -> float:
        """Predecir desviación futura."""
        # Simplificado
        return 0.1
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        return {
            "strategy": self.strategy.value,
            "adaptation_count": self.adaptation_count,
            "adaptation_history_size": len(self.adaptation_history),
            "baseline_metrics": self.performance_baseline
        }


class AutonomousOptimizer:
    """Optimizador principal de sistemas autónomos."""
    
    def __init__(self, autonomy_level: AutonomyLevel = AutonomyLevel.FULLY_AUTONOMOUS,
                 adaptation_strategy: AdaptationStrategy = AdaptationStrategy.ADAPTIVE):
        self.autonomy_level = autonomy_level
        self.self_healing = SelfHealingSystem()
        self.self_optimizing = SelfOptimizingSystem()
        self.adaptive_routing = AdaptiveRouting(strategy=adaptation_strategy)
        self.autonomous_actions = 0
        self.manual_interventions = 0
    
    def monitor_and_adapt(self, metrics: Dict[str, float]) -> List[AdaptationAction]:
        """Monitorear y adaptar automáticamente."""
        actions = []
        
        if self.autonomy_level == AutonomyLevel.MANUAL:
            return actions
        
        # Auto-reparación
        state = self.self_healing.monitor_health(metrics)
        recovery_action = self.self_healing.recover(state)
        if recovery_action:
            actions.append(recovery_action)
            self.autonomous_actions += 1
        
        # Auto-optimización
        if self.autonomy_level in [AutonomyLevel.SEMI_AUTONOMOUS, AutonomyLevel.FULLY_AUTONOMOUS]:
            # Definir función de rendimiento
            def performance_func(config: Dict[str, Any]) -> float:
                # Simplificado
                return metrics.get("throughput", 0.0) * (1.0 - metrics.get("error_rate", 0.0))
            
            # Optimizar si es necesario
            if metrics.get("error_rate", 0.0) > 0.1:
                best_config = self.self_optimizing.optimize(performance_func, max_iterations=5)
                if best_config:
                    actions.append(AdaptationAction(
                        action_type="apply_optimization",
                        parameters=best_config,
                        expected_impact=0.4,
                        confidence=0.8
                    ))
                    self.autonomous_actions += 1
        
        # Adaptación
        if self.autonomy_level == AutonomyLevel.FULLY_AUTONOMOUS:
            target_performance = {
                "error_rate": 0.01,
                "latency": 100.0,
                "throughput": 1000.0
            }
            adaptation_action = self.adaptive_routing.adapt(metrics, target_performance)
            if adaptation_action:
                actions.append(adaptation_action)
                self.autonomous_actions += 1
        
        return actions
    
    def set_autonomy_level(self, level: AutonomyLevel):
        """Establecer nivel de autonomía."""
        self.autonomy_level = level
        logger.info(f"Autonomy level set to {level.value}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        return {
            "autonomy_level": self.autonomy_level.value,
            "autonomous_actions": self.autonomous_actions,
            "manual_interventions": self.manual_interventions,
            "self_healing": self.self_healing.get_stats(),
            "self_optimizing": self.self_optimizing.get_stats(),
            "adaptive_routing": self.adaptive_routing.get_stats()
        }


