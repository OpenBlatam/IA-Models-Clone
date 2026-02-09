"""
Robot Types System
==================

Sistema genérico para diferentes tipos de robots.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class RobotType(Enum):
    """Tipo de robot."""
    MANIPULATOR = "manipulator"  # Brazo robótico
    MOBILE = "mobile"  # Robot móvil
    HUMANOID = "humanoid"  # Robot humanoide
    QUADCOPTER = "quadcopter"  # Drone
    WHEELED = "wheeled"  # Robot con ruedas
    LEGGED = "legged"  # Robot con patas
    CUSTOM = "custom"  # Robot personalizado


@dataclass
class RobotConfig:
    """Configuración base de robot."""
    robot_type: RobotType
    name: str
    dof: int  # Degrees of Freedom
    joint_limits: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    link_lengths: List[float] = field(default_factory=list)
    base_position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RobotState:
    """Estado del robot."""
    joint_positions: List[float] = field(default_factory=list)
    joint_velocities: List[float] = field(default_factory=list)
    joint_accelerations: List[float] = field(default_factory=list)
    end_effector_position: List[float] = field(default_factory=list)
    end_effector_orientation: List[float] = field(default_factory=list)
    base_position: List[float] = field(default_factory=list)
    base_orientation: List[float] = field(default_factory=list)
    timestamp: float = 0.0


class BaseRobot(ABC):
    """
    Clase base para todos los tipos de robots.
    
    Proporciona interfaz común para diferentes tipos de robots.
    """
    
    def __init__(self, config: RobotConfig):
        """
        Inicializar robot.
        
        Args:
            config: Configuración del robot
        """
        self.config = config
        self.state = RobotState()
        self.connected = False
    
    @abstractmethod
    def forward_kinematics(self, joint_positions: List[float]) -> Tuple[List[float], List[float]]:
        """
        Calcular cinemática directa.
        
        Args:
            joint_positions: Posiciones de las articulaciones
            
        Returns:
            Tupla (posición, orientación) del end effector
        """
        pass
    
    @abstractmethod
    def inverse_kinematics(
        self,
        target_position: List[float],
        target_orientation: Optional[List[float]] = None
    ) -> List[float]:
        """
        Calcular cinemática inversa.
        
        Args:
            target_position: Posición objetivo
            target_orientation: Orientación objetivo (opcional)
            
        Returns:
            Posiciones de articulaciones
        """
        pass
    
    @abstractmethod
    def move_to(
        self,
        target_position: List[float],
        target_orientation: Optional[List[float]] = None
    ) -> bool:
        """
        Mover a posición objetivo.
        
        Args:
            target_position: Posición objetivo
            target_orientation: Orientación objetivo (opcional)
            
        Returns:
            True si el movimiento fue exitoso
        """
        pass
    
    @abstractmethod
    def get_state(self) -> RobotState:
        """
        Obtener estado actual del robot.
        
        Returns:
            Estado del robot
        """
        pass
    
    def connect(self) -> bool:
        """Conectar con el robot."""
        self.connected = True
        logger.info(f"Connected to {self.config.name}")
        return True
    
    def disconnect(self):
        """Desconectar del robot."""
        self.connected = False
        logger.info(f"Disconnected from {self.config.name}")
    
    def is_connected(self) -> bool:
        """Verificar si está conectado."""
        return self.connected


class ManipulatorRobot(BaseRobot):
    """
    Robot manipulador (brazo robótico).
    
    Robot con articulaciones rotacionales o prismáticas.
    """
    
    def __init__(self, config: RobotConfig):
        """Inicializar manipulador."""
        if config.robot_type != RobotType.MANIPULATOR:
            raise ValueError("Config must be for MANIPULATOR type")
        super().__init__(config)
    
    def forward_kinematics(self, joint_positions: List[float]) -> Tuple[List[float], List[float]]:
        """Calcular cinemática directa."""
        # Implementación simplificada (en producción usaría matrices de transformación)
        if len(joint_positions) != self.config.dof:
            raise ValueError(f"Expected {self.config.dof} joint positions, got {len(joint_positions)}")
        
        # Calcular posición usando longitudes de eslabones
        x, y, z = self.config.base_position
        for i, (angle, length) in enumerate(zip(joint_positions, self.config.link_lengths)):
            x += length * np.cos(angle)
            y += length * np.sin(angle)
            z += 0.0  # Simplificado
        
        position = [x, y, z]
        orientation = [0.0, 0.0, 0.0, 1.0]  # Quaternion simplificado
        
        return position, orientation
    
    def inverse_kinematics(
        self,
        target_position: List[float],
        target_orientation: Optional[List[float]] = None
    ) -> List[float]:
        """Calcular cinemática inversa."""
        # Implementación simplificada (en producción usaría métodos numéricos/analíticos)
        # Usar método iterativo o analítico según el tipo de manipulador
        joint_positions = [0.0] * self.config.dof
        
        # Simplificado: calcular ángulos básicos
        dx = target_position[0] - self.config.base_position[0]
        dy = target_position[1] - self.config.base_position[1]
        dz = target_position[2] - self.config.base_position[2]
        
        if self.config.dof >= 1:
            joint_positions[0] = np.arctan2(dy, dx)
        if self.config.dof >= 2:
            distance = np.sqrt(dx**2 + dy**2)
            joint_positions[1] = np.arctan2(dz, distance)
        
        return joint_positions
    
    def move_to(
        self,
        target_position: List[float],
        target_orientation: Optional[List[float]] = None
    ) -> bool:
        """Mover a posición objetivo."""
        if not self.connected:
            logger.error("Robot not connected")
            return False
        
        try:
            joint_positions = self.inverse_kinematics(target_position, target_orientation)
            
            # Validar límites de articulaciones
            for i, pos in enumerate(joint_positions):
                joint_name = f"joint_{i}"
                if joint_name in self.config.joint_limits:
                    min_limit, max_limit = self.config.joint_limits[joint_name]
                    if pos < min_limit or pos > max_limit:
                        logger.warning(f"Joint {i} out of limits: {pos} not in [{min_limit}, {max_limit}]")
                        pos = np.clip(pos, min_limit, max_limit)
                        joint_positions[i] = pos
            
            # Actualizar estado
            self.state.joint_positions = joint_positions
            self.state.end_effector_position = target_position
            if target_orientation:
                self.state.end_effector_orientation = target_orientation
            
            logger.info(f"Moved to position: {target_position}")
            return True
            
        except Exception as e:
            logger.error(f"Error moving robot: {e}")
            return False
    
    def get_state(self) -> RobotState:
        """Obtener estado actual."""
        return self.state


class MobileRobot(BaseRobot):
    """
    Robot móvil.
    
    Robot que se mueve en el plano (diferencial, omnidireccional, etc.).
    """
    
    def __init__(self, config: RobotConfig):
        """Inicializar robot móvil."""
        if config.robot_type != RobotType.MOBILE:
            raise ValueError("Config must be for MOBILE type")
        super().__init__(config)
        self.velocity = [0.0, 0.0, 0.0]  # vx, vy, omega
    
    def forward_kinematics(self, joint_positions: List[float]) -> Tuple[List[float], List[float]]:
        """Calcular posición actual."""
        # Para robots móviles, las "articulaciones" son velocidades
        position = self.state.base_position.copy()
        orientation = self.state.base_orientation.copy()
        return position, orientation
    
    def inverse_kinematics(
        self,
        target_position: List[float],
        target_orientation: Optional[List[float]] = None
    ) -> List[float]:
        """Calcular velocidades necesarias."""
        # Calcular velocidades para alcanzar posición objetivo
        dx = target_position[0] - self.state.base_position[0]
        dy = target_position[1] - self.state.base_position[1]
        
        # Velocidades lineales
        vx = dx * 0.5  # Simplificado
        vy = dy * 0.5
        
        # Velocidad angular
        omega = 0.0
        if target_orientation:
            current_yaw = self.state.base_orientation[2] if len(self.state.base_orientation) > 2 else 0.0
            target_yaw = target_orientation[2] if len(target_orientation) > 2 else 0.0
            omega = target_yaw - current_yaw
        
        return [vx, vy, omega]
    
    def move_to(
        self,
        target_position: List[float],
        target_orientation: Optional[List[float]] = None
    ) -> bool:
        """Mover a posición objetivo."""
        if not self.connected:
            logger.error("Robot not connected")
            return False
        
        try:
            velocities = self.inverse_kinematics(target_position, target_orientation)
            
            # Actualizar estado
            self.velocity = velocities
            self.state.base_position = target_position
            if target_orientation:
                self.state.base_orientation = target_orientation
            
            logger.info(f"Moved to position: {target_position}")
            return True
            
        except Exception as e:
            logger.error(f"Error moving robot: {e}")
            return False
    
    def get_state(self) -> RobotState:
        """Obtener estado actual."""
        return self.state


class QuadcopterRobot(BaseRobot):
    """
    Quadcopter (drone).
    
    Robot volador con 4 rotores.
    """
    
    def __init__(self, config: RobotConfig):
        """Inicializar quadcopter."""
        if config.robot_type != RobotType.QUADCOPTER:
            raise ValueError("Config must be for QUADCOPTER type")
        super().__init__(config)
        self.thrust = [0.0, 0.0, 0.0, 0.0]  # Thrust de cada rotor
    
    def forward_kinematics(self, joint_positions: List[float]) -> Tuple[List[float], List[float]]:
        """Calcular posición actual."""
        position = self.state.base_position.copy()
        orientation = self.state.base_orientation.copy()
        return position, orientation
    
    def inverse_kinematics(
        self,
        target_position: List[float],
        target_orientation: Optional[List[float]] = None
    ) -> List[float]:
        """Calcular thrusts necesarios."""
        # Calcular thrusts para alcanzar posición objetivo
        # Simplificado: usar control PID básico
        dx = target_position[0] - self.state.base_position[0]
        dy = target_position[1] - self.state.base_position[1]
        dz = target_position[2] - self.state.base_position[2]
        
        # Thrust base para mantener altura
        base_thrust = 0.5
        
        # Ajustes para movimiento
        thrusts = [
            base_thrust + dz * 0.1,  # Front left
            base_thrust + dz * 0.1,  # Front right
            base_thrust + dz * 0.1,  # Back left
            base_thrust + dz * 0.1   # Back right
        ]
        
        return thrusts
    
    def move_to(
        self,
        target_position: List[float],
        target_orientation: Optional[List[float]] = None
    ) -> bool:
        """Mover a posición objetivo."""
        if not self.connected:
            logger.error("Robot not connected")
            return False
        
        try:
            thrusts = self.inverse_kinematics(target_position, target_orientation)
            
            # Actualizar estado
            self.thrust = thrusts
            self.state.base_position = target_position
            if target_orientation:
                self.state.base_orientation = target_orientation
            
            logger.info(f"Moved to position: {target_position}")
            return True
            
        except Exception as e:
            logger.error(f"Error moving robot: {e}")
            return False
    
    def get_state(self) -> RobotState:
        """Obtener estado actual."""
        return self.state


class RobotFactory:
    """
    Factory para crear diferentes tipos de robots.
    
    Proporciona interfaz unificada para crear cualquier tipo de robot.
    """
    
    _robot_classes = {
        RobotType.MANIPULATOR: ManipulatorRobot,
        RobotType.MOBILE: MobileRobot,
        RobotType.QUADCOPTER: QuadcopterRobot,
    }
    
    @classmethod
    def create_robot(cls, config: RobotConfig) -> BaseRobot:
        """
        Crear robot según configuración.
        
        Args:
            config: Configuración del robot
            
        Returns:
            Instancia del robot
        """
        if config.robot_type not in cls._robot_classes:
            raise ValueError(f"Unsupported robot type: {config.robot_type}")
        
        robot_class = cls._robot_classes[config.robot_type]
        robot = robot_class(config)
        
        logger.info(f"Created {config.robot_type.value} robot: {config.name}")
        return robot
    
    @classmethod
    def register_robot_type(cls, robot_type: RobotType, robot_class: type):
        """
        Registrar nuevo tipo de robot.
        
        Args:
            robot_type: Tipo de robot
            robot_class: Clase del robot
        """
        if not issubclass(robot_class, BaseRobot):
            raise ValueError("Robot class must inherit from BaseRobot")
        
        cls._robot_classes[robot_type] = robot_class
        logger.info(f"Registered robot type: {robot_type.value}")

