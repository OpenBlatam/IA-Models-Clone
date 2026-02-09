"""
Energy Optimization System for Humanoid Devin Robot (Optimizado)
=================================================================

Sistema de optimización de energía para maximizar la eficiencia energética
del robot humanoide.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)


def ErrorCode(description: str):
    """Decorador para anotar excepciones con códigos de error y descripciones."""
    def decorator(cls):
        cls._error_description = description
        return cls
    return decorator


@ErrorCode(description="Error in energy optimization system")
class EnergyOptimizerError(Exception):
    """Excepción para errores de optimización de energía."""
    
    def __init__(self):
        """Initialize exception with description from @ErrorCode decorator."""
        message = getattr(self.__class__, '_error_description', "Error in energy optimization system")
        super().__init__(message)
        self.message = message


class EnergyOptimizer:
    """
    Sistema de optimización de energía para el robot humanoide.
    
    Monitorea y optimiza el consumo de energía.
    """
    
    def __init__(
        self,
        target_power_budget: Optional[float] = None,
        enable_power_limiting: bool = True
    ):
        """
        Inicializar optimizador de energía.
        
        Args:
            target_power_budget: Presupuesto de energía objetivo (Watts)
            enable_power_limiting: Habilitar limitación de potencia
        """
        if target_power_budget is not None:
            if not isinstance(target_power_budget, (int, float)) or target_power_budget <= 0:
                raise ValueError("target_power_budget must be a positive number")
        
        self.target_power_budget = target_power_budget
        self.enable_power_limiting = enable_power_limiting
        
        # Historial de consumo
        self.power_history: deque = deque(maxlen=1000)
        self.energy_consumption: Dict[str, float] = {}
        
        # Modelos de consumo
        self.power_models: Dict[str, Dict[str, float]] = {}
        
        # Estadísticas
        self.total_energy_consumed = 0.0
        self.peak_power = 0.0
        self.avg_power = 0.0
        
        logger.info(
            f"Energy optimizer initialized: "
            f"target_budget={target_power_budget}W, "
            f"power_limiting={enable_power_limiting}"
        )
    
    def record_power_consumption(
        self,
        component: str,
        power: float,
        duration: float = 1.0
    ) -> None:
        """
        Registrar consumo de energía.
        
        Args:
            component: Componente que consume energía
            power: Potencia en Watts
            duration: Duración en segundos
        """
        if not component or not isinstance(component, str):
            raise ValueError("component must be a non-empty string")
        if not np.isfinite(power) or power < 0:
            raise ValueError("power must be a non-negative finite number")
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("duration must be a positive finite number")
        
        energy = power * duration
        
        # Registrar en historial
        record = {
            "timestamp": datetime.now().isoformat(),
            "component": component,
            "power": float(power),
            "duration": float(duration),
            "energy": float(energy)
        }
        self.power_history.append(record)
        
        # Actualizar estadísticas
        self.total_energy_consumed += energy
        self.peak_power = max(self.peak_power, power)
        
        # Actualizar promedio de potencia
        if len(self.power_history) > 0:
            total_power = sum(r["power"] for r in self.power_history)
            self.avg_power = total_power / len(self.power_history)
        
        # Actualizar consumo por componente
        if component not in self.energy_consumption:
            self.energy_consumption[component] = 0.0
        self.energy_consumption[component] += energy
        
        logger.debug(f"Power recorded: {component}={power:.2f}W for {duration:.2f}s")
    
    def estimate_power_consumption(
        self,
        action_type: str,
        parameters: Dict[str, Any]
    ) -> float:
        """
        Estimar consumo de energía para una acción.
        
        Args:
            action_type: Tipo de acción
            parameters: Parámetros de la acción
            
        Returns:
            Potencia estimada en Watts
        """
        if action_type not in self.power_models:
            # Modelo por defecto basado en tipo de acción
            return self._default_power_estimate(action_type, parameters)
        
        model = self.power_models[action_type]
        base_power = model.get("base_power", 10.0)
        
        # Ajustar según parámetros
        power = base_power
        
        # Ajuste por velocidad
        if "speed" in parameters:
            speed = parameters["speed"]
            power *= (1.0 + speed * 0.5)  # Más velocidad = más potencia
        
        # Ajuste por distancia
        if "distance" in parameters:
            distance = parameters.get("distance", 0.0)
            power *= (1.0 + distance * 0.1)
        
        return float(power)
    
    def _default_power_estimate(
        self,
        action_type: str,
        parameters: Dict[str, Any]
    ) -> float:
        """Estimación de potencia por defecto."""
        # Potencias base estimadas (Watts)
        base_powers = {
            "walk": 50.0,
            "stand": 20.0,
            "sit": 10.0,
            "grasp": 30.0,
            "gesture": 15.0,
            "idle": 5.0
        }
        
        return base_powers.get(action_type, 25.0)
    
    def optimize_movement_parameters(
        self,
        action_type: str,
        parameters: Dict[str, Any],
        target_energy: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Optimizar parámetros de movimiento para reducir consumo de energía.
        
        Args:
            action_type: Tipo de acción
            parameters: Parámetros originales
            target_energy: Energía objetivo (opcional)
            
        Returns:
            Parámetros optimizados
        """
        optimized = parameters.copy()
        
        # Reducir velocidad si es posible
        if "speed" in optimized:
            original_speed = optimized["speed"]
            optimized["speed"] = original_speed * 0.8  # Reducir 20%
            logger.debug(f"Speed optimized: {original_speed} -> {optimized['speed']}")
        
        # Reducir aceleración si es posible
        if "acceleration" in optimized:
            original_accel = optimized["acceleration"]
            optimized["acceleration"] = original_accel * 0.7  # Reducir 30%
        
        # Ajustar distancia si hay objetivo de energía
        if target_energy and "distance" in optimized:
            estimated_power = self.estimate_power_consumption(action_type, optimized)
            estimated_duration = optimized.get("distance", 1.0) / optimized.get("speed", 1.0)
            estimated_energy = estimated_power * estimated_duration
            
            if estimated_energy > target_energy:
                # Reducir distancia proporcionalmente
                scale_factor = target_energy / estimated_energy
                optimized["distance"] = optimized["distance"] * scale_factor
        
        return optimized
    
    def check_power_budget(
        self,
        estimated_power: float
    ) -> Tuple[bool, Optional[str]]:
        """
        Verificar si la potencia estimada está dentro del presupuesto.
        
        Args:
            estimated_power: Potencia estimada en Watts
            
        Returns:
            Tupla (is_within_budget, warning_message)
        """
        if not self.enable_power_limiting or self.target_power_budget is None:
            return True, None
        
        if estimated_power > self.target_power_budget:
            warning = (
                f"Estimated power ({estimated_power:.2f}W) exceeds "
                f"budget ({self.target_power_budget:.2f}W)"
            )
            return False, warning
        
        return True, None
    
    def get_energy_statistics(self) -> Dict[str, Any]:
        """
        Obtener estadísticas de energía.
        
        Returns:
            Dict con estadísticas
        """
        # Consumo por componente
        component_breakdown = {
            component: energy
            for component, energy in self.energy_consumption.items()
        }
        
        # Consumo reciente (última hora)
        recent_energy = sum(
            r["energy"] for r in self.power_history
            if (datetime.now() - datetime.fromisoformat(r["timestamp"])).total_seconds() < 3600
        )
        
        return {
            "total_energy_consumed": self.total_energy_consumed,
            "recent_energy_consumed": recent_energy,
            "peak_power": self.peak_power,
            "avg_power": self.avg_power,
            "target_power_budget": self.target_power_budget,
            "component_breakdown": component_breakdown,
            "power_history_size": len(self.power_history)
        }
    
    def get_power_recommendations(self) -> List[str]:
        """
        Obtener recomendaciones para reducir consumo de energía.
        
        Returns:
            Lista de recomendaciones
        """
        recommendations = []
        
        # Verificar si hay componentes con alto consumo
        if self.energy_consumption:
            max_component = max(
                self.energy_consumption.items(),
                key=lambda x: x[1]
            )
            recommendations.append(
                f"Component '{max_component[0]}' has highest energy consumption "
                f"({max_component[1]:.2f}J). Consider optimization."
            )
        
        # Verificar potencia promedio
        if self.target_power_budget and self.avg_power > self.target_power_budget * 0.9:
            recommendations.append(
                f"Average power ({self.avg_power:.2f}W) is close to budget "
                f"({self.target_power_budget:.2f}W). Consider reducing activity."
            )
        
        # Verificar potencia pico
        if self.target_power_budget and self.peak_power > self.target_power_budget:
            recommendations.append(
                f"Peak power ({self.peak_power:.2f}W) exceeded budget "
                f"({self.target_power_budget:.2f}W). Consider power limiting."
            )
        
        return recommendations
    
    def reset_statistics(self) -> None:
        """Resetear estadísticas de energía."""
        self.power_history.clear()
        self.energy_consumption.clear()
        self.total_energy_consumed = 0.0
        self.peak_power = 0.0
        self.avg_power = 0.0
        logger.info("Energy statistics reset")

