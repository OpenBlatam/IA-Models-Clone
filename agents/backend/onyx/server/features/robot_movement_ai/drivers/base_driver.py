"""
Base Robot Driver
=================

Clase base para drivers de diferentes marcas de robots.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)


class BaseRobotDriver(ABC):
    """
    Driver base para robots.
    
    Define la interfaz común para todos los drivers de robots.
    """
    
    def __init__(self, robot_ip: str, robot_port: int = 30001):
        """
        Inicializar driver.
        
        Args:
            robot_ip: IP del robot
            robot_port: Puerto del robot
        """
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        self.connected = False
    
    @abstractmethod
    async def connect(self) -> bool:
        """Conectar con el robot."""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Desconectar del robot."""
        pass
    
    @abstractmethod
    async def get_joint_positions(self) -> List[float]:
        """Obtener posiciones actuales de articulaciones."""
        pass
    
    @abstractmethod
    async def set_joint_positions(
        self,
        positions: List[float],
        velocities: Optional[List[float]] = None
    ) -> bool:
        """Establecer posiciones de articulaciones."""
        pass
    
    @abstractmethod
    async def get_end_effector_pose(self) -> Dict[str, np.ndarray]:
        """Obtener pose del efector final."""
        pass
    
    @abstractmethod
    async def move_to_pose(
        self,
        position: np.ndarray,
        orientation: np.ndarray
    ) -> bool:
        """Mover efector final a pose objetivo."""
        pass
    
    @abstractmethod
    async def stop(self):
        """Detener movimiento."""
        pass
    
    @abstractmethod
    async def emergency_stop(self):
        """Parada de emergencia."""
        pass






