"""
Predictive Planning System for Humanoid Devin Robot (Optimizado)
=================================================================

Sistema de planificación predictiva para anticipar y optimizar movimientos futuros.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)


def ErrorCode(description: str):
    """
    Decorador para anotar excepciones con códigos de error y descripciones.
    
    Args:
        description: Descripción del error que se usará en el constructor.
    
    Usage:
        @ErrorCode(description="Invalid input provided")
        class MyException(Exception):
            def __init__(self):
                super().__init__(description)
    """
    def decorator(cls):
        # Almacenar la descripción en la clase
        cls._error_description = description
        return cls
    return decorator


@ErrorCode(description="Error in predictive planning system")
class PredictivePlannerError(Exception):
    """Excepción para errores de planificación predictiva."""
    
    def __init__(self):
        """Initialize exception with description from @ErrorCode decorator."""
        message = getattr(self.__class__, '_error_description', "Error in predictive planning system")
        super().__init__(message)
        self.message = message


class PredictivePlanner:
    """
    Sistema de planificación predictiva para el robot humanoide.
    
    Predice y planifica movimientos futuros basándose en patrones y contexto.
    """
    
    def __init__(
        self,
        prediction_horizon: float = 5.0,
        planning_frequency: float = 10.0
    ):
        """
        Inicializar planificador predictivo.
        
        Args:
            prediction_horizon: Horizonte de predicción en segundos
            planning_frequency: Frecuencia de planificación en Hz
        """
        if not isinstance(prediction_horizon, (int, float)) or prediction_horizon <= 0:
            raise ValueError("prediction_horizon must be a positive number")
        if not isinstance(planning_frequency, (int, float)) or planning_frequency <= 0:
            raise ValueError("planning_frequency must be a positive number")
        
        self.prediction_horizon = prediction_horizon
        self.planning_frequency = planning_frequency
        
        # Historial de movimientos
        self.movement_history: deque = deque(maxlen=1000)
        self.trajectory_history: deque = deque(maxlen=500)
        
        # Modelos predictivos
        self.velocity_predictor = None
        self.position_predictor = None
        
        # Planes predictivos
        self.active_plans: Dict[str, Dict[str, Any]] = {}
        
        # Estadísticas
        self.total_predictions = 0
        self.successful_predictions = 0
        
        logger.info(
            f"Predictive planner initialized: "
            f"horizon={prediction_horizon}s, frequency={planning_frequency}Hz"
        )
    
    def predict_trajectory(
        self,
        current_state: Dict[str, Any],
        target_state: Optional[Dict[str, Any]] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Predecir trayectoria futura.
        
        Args:
            current_state: Estado actual del robot
            target_state: Estado objetivo (opcional)
            constraints: Restricciones (opcional)
            
        Returns:
            Dict con trayectoria predicha
        """
        if not isinstance(current_state, dict):
            raise ValueError("current_state must be a dictionary")
        
        # Obtener estado actual
        current_positions = np.array(current_state.get("joint_positions", []))
        current_velocities = np.array(current_state.get("joint_velocities", []))
        
        if len(current_positions) == 0:
            raise ValueError("current_state must contain joint_positions")
        
        # Calcular número de pasos
        dt = 1.0 / self.planning_frequency
        num_steps = int(self.prediction_horizon * self.planning_frequency)
        
        # Predecir trayectoria
        predicted_positions = []
        predicted_velocities = []
        predicted_times = []
        
        positions = current_positions.copy()
        velocities = current_velocities.copy() if len(current_velocities) > 0 else np.zeros_like(current_positions)
        
        for step in range(num_steps):
            # Predicción simple basada en velocidad actual
            if target_state is not None:
                target_positions = np.array(target_state.get("joint_positions", current_positions))
                # Controlador PD simple
                error = target_positions - positions
                desired_velocity = error * 2.0  # Ganancia proporcional
                velocities = velocities * 0.9 + desired_velocity * 0.1  # Filtro
            else:
                # Decaimiento de velocidad
                velocities = velocities * 0.95
            
            # Integrar
            positions = positions + velocities * dt
            
            # Aplicar límites si están disponibles
            if constraints:
                min_limits = constraints.get("min_joint_limits")
                max_limits = constraints.get("max_joint_limits")
                if min_limits is not None:
                    positions = np.maximum(positions, np.array(min_limits))
                if max_limits is not None:
                    positions = np.minimum(positions, np.array(max_limits))
            
            predicted_positions.append(positions.copy())
            predicted_velocities.append(velocities.copy())
            predicted_times.append(step * dt)
        
        trajectory = {
            "timestamp": datetime.now().isoformat(),
            "prediction_horizon": self.prediction_horizon,
            "num_steps": num_steps,
            "times": predicted_times,
            "positions": [p.tolist() for p in predicted_positions],
            "velocities": [v.tolist() for v in predicted_velocities],
            "current_state": current_state,
            "target_state": target_state
        }
        
        self.trajectory_history.append(trajectory)
        self.total_predictions += 1
        
        return trajectory
    
    def predict_collision(
        self,
        predicted_trajectory: Dict[str, Any],
        obstacles: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Predecir colisiones en trayectoria.
        
        Args:
            predicted_trajectory: Trayectoria predicha
            obstacles: Lista de obstáculos
            
        Returns:
            Dict con información de colisiones predichas
        """
        if not obstacles:
            return {
                "collision_detected": False,
                "collision_time": None,
                "collision_point": None
            }
        
        positions = predicted_trajectory["positions"]
        times = predicted_trajectory["times"]
        
        for i, position in enumerate(positions):
            pos_array = np.array(position)
            
            for obstacle in obstacles:
                obstacle_pos = np.array(obstacle.get("position", [0, 0, 0]))
                obstacle_radius = obstacle.get("radius", 0.1)
                
                # Calcular posición del efector final (simplificado)
                # En un caso real, usaría cinemática directa
                if len(pos_array) >= 3:
                    # Asumir que los primeros 3 valores son posición del efector
                    effector_pos = pos_array[:3]
                else:
                    # Usar posición aproximada
                    effector_pos = np.array([0.5, 0.0, 1.0])
                
                distance = np.linalg.norm(effector_pos - obstacle_pos)
                
                if distance < obstacle_radius:
                    return {
                        "collision_detected": True,
                        "collision_time": times[i],
                        "collision_point": effector_pos.tolist(),
                        "obstacle": obstacle
                    }
        
        return {
            "collision_detected": False,
            "collision_time": None,
            "collision_point": None
        }
    
    def optimize_trajectory(
        self,
        trajectory: Dict[str, Any],
        objectives: List[str],
        weights: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Optimizar trayectoria según objetivos.
        
        Args:
            trajectory: Trayectoria a optimizar
            objectives: Lista de objetivos (energy, smoothness, speed, etc.)
            weights: Pesos para cada objetivo (opcional)
            
        Returns:
            Trayectoria optimizada
        """
        if not objectives:
            return trajectory
        
        if weights is None:
            weights = [1.0] * len(objectives)
        
        if len(weights) != len(objectives):
            raise ValueError("weights must have same length as objectives")
        
        positions = [np.array(p) for p in trajectory["positions"]]
        velocities = [np.array(v) for v in trajectory["velocities"]]
        
        # Optimización simple: suavizar trayectoria
        if "smoothness" in objectives:
            smooth_idx = objectives.index("smoothness")
            weight = weights[smooth_idx]
            
            # Aplicar filtro de suavizado
            for i in range(1, len(positions) - 1):
                positions[i] = (
                    positions[i] * (1 - weight * 0.1) +
                    (positions[i-1] + positions[i+1]) / 2 * (weight * 0.1)
                )
        
        # Optimización de energía: reducir velocidades
        if "energy" in objectives:
            energy_idx = objectives.index("energy")
            weight = weights[energy_idx]
            
            for i in range(len(velocities)):
                velocities[i] = velocities[i] * (1 - weight * 0.2)
        
        optimized = trajectory.copy()
        optimized["positions"] = [p.tolist() for p in positions]
        optimized["velocities"] = [v.tolist() for v in velocities]
        optimized["optimization_applied"] = True
        optimized["objectives"] = objectives
        optimized["weights"] = weights
        
        return optimized
    
    def create_plan(
        self,
        plan_id: str,
        goal: Dict[str, Any],
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Crear plan predictivo.
        
        Args:
            plan_id: ID único del plan
            goal: Objetivo del plan
            constraints: Restricciones (opcional)
            
        Returns:
            Dict con el plan creado
        """
        plan = {
            "plan_id": plan_id,
            "created_at": datetime.now().isoformat(),
            "goal": goal,
            "constraints": constraints or {},
            "status": "active",
            "trajectory": None
        }
        
        self.active_plans[plan_id] = plan
        logger.info(f"Plan created: {plan_id}")
        
        return plan
    
    def update_plan(
        self,
        plan_id: str,
        current_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Actualizar plan con estado actual.
        
        Args:
            plan_id: ID del plan
            current_state: Estado actual
            
        Returns:
            Plan actualizado
        """
        if plan_id not in self.active_plans:
            raise PredictivePlannerError(f"Plan not found: {plan_id}")
        
        plan = self.active_plans[plan_id]
        goal = plan["goal"]
        
        # Predecir nueva trayectoria
        trajectory = self.predict_trajectory(
            current_state=current_state,
            target_state=goal.get("target_state"),
            constraints=plan.get("constraints")
        )
        
        plan["trajectory"] = trajectory
        plan["last_updated"] = datetime.now().isoformat()
        
        return plan
    
    def get_plan(self, plan_id: str) -> Optional[Dict[str, Any]]:
        """
        Obtener plan por ID.
        
        Args:
            plan_id: ID del plan
            
        Returns:
            Plan o None si no existe
        """
        return self.active_plans.get(plan_id)
    
    def cancel_plan(self, plan_id: str) -> None:
        """
        Cancelar plan.
        
        Args:
            plan_id: ID del plan
        """
        if plan_id in self.active_plans:
            self.active_plans[plan_id]["status"] = "cancelled"
            logger.info(f"Plan cancelled: {plan_id}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Obtener estadísticas del planificador.
        
        Returns:
            Dict con estadísticas
        """
        return {
            "total_predictions": self.total_predictions,
            "successful_predictions": self.successful_predictions,
            "success_rate": (
                self.successful_predictions / self.total_predictions
                if self.total_predictions > 0 else 0.0
            ),
            "active_plans": len([p for p in self.active_plans.values() if p["status"] == "active"]),
            "total_plans": len(self.active_plans),
            "trajectory_history_size": len(self.trajectory_history)
        }

