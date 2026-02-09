"""
Universal Robot Controller
==========================

Controlador universal que funciona con cualquier tipo de robot.
"""

import logging
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod

from .robot.robot_types import BaseRobot, RobotType, RobotConfig, RobotState, RobotFactory
from .robot import TrajectoryOptimizer, RobotMovementEngine

logger = logging.getLogger(__name__)


class UniversalRobotController:
    """
    Controlador universal para cualquier tipo de robot.
    
    Proporciona interfaz unificada para controlar diferentes tipos de robots.
    """
    
    def __init__(self, robot: BaseRobot):
        """
        Inicializar controlador.
        
        Args:
            robot: Instancia del robot
        """
        self.robot = robot
        self.trajectory_optimizer = None
        self.movement_engine = None
    
    def connect(self) -> bool:
        """Conectar con el robot."""
        return self.robot.connect()
    
    def disconnect(self):
        """Desconectar del robot."""
        self.robot.disconnect()
    
    def move_to(
        self,
        target_position: List[float],
        target_orientation: Optional[List[float]] = None,
        optimize_trajectory: bool = True
    ) -> bool:
        """
        Mover robot a posición objetivo.
        
        Args:
            target_position: Posición objetivo [x, y, z]
            target_orientation: Orientación objetivo (opcional)
            optimize_trajectory: Optimizar trayectoria (opcional)
            
        Returns:
            True si el movimiento fue exitoso
        """
        if not self.robot.is_connected():
            logger.error("Robot not connected")
            return False
        
        try:
            if optimize_trajectory and self.trajectory_optimizer:
                # Optimizar trayectoria
                current_state = self.robot.get_state()
                current_pos = current_state.end_effector_position or current_state.base_position
                
                trajectory = self.trajectory_optimizer.optimize(
                    start_position=current_pos,
                    target_position=target_position,
                    start_orientation=current_state.end_effector_orientation or current_state.base_orientation,
                    target_orientation=target_orientation
                )
                
                # Ejecutar trayectoria punto por punto
                for point in trajectory.points:
                    success = self.robot.move_to(
                        point.position,
                        point.orientation
                    )
                    if not success:
                        logger.warning(f"Failed to reach intermediate point: {point.position}")
                        return False
                
                return True
            else:
                # Movimiento directo
                return self.robot.move_to(target_position, target_orientation)
                
        except Exception as e:
            logger.error(f"Error in move_to: {e}")
            return False
    
    def move_along_path(
        self,
        waypoints: List[List[float]],
        orientations: Optional[List[List[float]]] = None
    ) -> bool:
        """
        Mover robot a lo largo de una ruta.
        
        Args:
            waypoints: Lista de waypoints [[x, y, z], ...]
            orientations: Lista de orientaciones (opcional)
            
        Returns:
            True si el movimiento fue exitoso
        """
        if not self.robot.is_connected():
            logger.error("Robot not connected")
            return False
        
        try:
            for i, waypoint in enumerate(waypoints):
                orientation = orientations[i] if orientations and i < len(orientations) else None
                
                success = self.move_to(waypoint, orientation)
                if not success:
                    logger.error(f"Failed to reach waypoint {i}: {waypoint}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error in move_along_path: {e}")
            return False
    
    def get_state(self) -> RobotState:
        """Obtener estado actual del robot."""
        return self.robot.get_state()
    
    def set_trajectory_optimizer(self, optimizer: TrajectoryOptimizer):
        """Establecer optimizador de trayectoria."""
        self.trajectory_optimizer = optimizer
    
    def get_robot_type(self) -> RobotType:
        """Obtener tipo de robot."""
        return self.robot.config.robot_type


class UniversalMovementEngine:
    """
    Motor de movimiento universal.
    
    Gestiona múltiples robots de diferentes tipos.
    """
    
    def __init__(self):
        """Inicializar motor."""
        self.robots: Dict[str, UniversalRobotController] = {}
        self.active_robot: Optional[str] = None
    
    def register_robot(
        self,
        robot_id: str,
        robot_config: RobotConfig
    ) -> bool:
        """
        Registrar robot.
        
        Args:
            robot_id: ID único del robot
            robot_config: Configuración del robot
            
        Returns:
            True si se registró exitosamente
        """
        try:
            robot = RobotFactory.create_robot(robot_config)
            controller = UniversalRobotController(robot)
            self.robots[robot_id] = controller
            logger.info(f"Registered robot: {robot_id} ({robot_config.robot_type.value})")
            return True
        except Exception as e:
            logger.error(f"Error registering robot: {e}")
            return False
    
    def connect_robot(self, robot_id: str) -> bool:
        """Conectar robot."""
        if robot_id not in self.robots:
            logger.error(f"Robot not found: {robot_id}")
            return False
        
        success = self.robots[robot_id].connect()
        if success:
            self.active_robot = robot_id
        return success
    
    def move_robot(
        self,
        robot_id: str,
        target_position: List[float],
        target_orientation: Optional[List[float]] = None
    ) -> bool:
        """
        Mover robot específico.
        
        Args:
            robot_id: ID del robot
            target_position: Posición objetivo
            target_orientation: Orientación objetivo (opcional)
            
        Returns:
            True si el movimiento fue exitoso
        """
        if robot_id not in self.robots:
            logger.error(f"Robot not found: {robot_id}")
            return False
        
        return self.robots[robot_id].move_to(target_position, target_orientation)
    
    def move_active_robot(
        self,
        target_position: List[float],
        target_orientation: Optional[List[float]] = None
    ) -> bool:
        """
        Mover robot activo.
        
        Args:
            target_position: Posición objetivo
            target_orientation: Orientación objetivo (opcional)
            
        Returns:
            True si el movimiento fue exitoso
        """
        if self.active_robot is None:
            logger.error("No active robot")
            return False
        
        return self.move_robot(self.active_robot, target_position, target_orientation)
    
    def get_robot_state(self, robot_id: str) -> Optional[RobotState]:
        """Obtener estado de robot."""
        if robot_id not in self.robots:
            return None
        return self.robots[robot_id].get_state()
    
    def list_robots(self) -> List[str]:
        """Listar IDs de robots registrados."""
        return list(self.robots.keys())
    
    def get_robot_type(self, robot_id: str) -> Optional[RobotType]:
        """Obtener tipo de robot."""
        if robot_id not in self.robots:
            return None
        return self.robots[robot_id].get_robot_type()


# Instancia global
_universal_engine: Optional[UniversalMovementEngine] = None


def get_universal_engine() -> UniversalMovementEngine:
    """Obtener instancia global del motor universal."""
    global _universal_engine
    if _universal_engine is None:
        _universal_engine = UniversalMovementEngine()
    return _universal_engine

