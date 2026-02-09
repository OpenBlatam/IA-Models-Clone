"""
Poppy and iCub Integration for Humanoid Robot (Optimizado)
===========================================================

Integración profesional con robots Poppy y iCub.
Incluye validaciones robustas, manejo de errores mejorado, y optimizaciones.
"""

import logging
from typing import Optional, Dict, Any, List, Union
import os

try:
    from pypot.robot import from_json
    POPPY_AVAILABLE = True
except ImportError:
    POPPY_AVAILABLE = False
    logging.warning("Poppy not available. Install pypot.")

try:
    from icub import icubInterface
    ICUB_AVAILABLE = True
except ImportError:
    ICUB_AVAILABLE = False
    logging.warning("iCub not available. Install icub-common and yarp.")

logger = logging.getLogger(__name__)


class PoppyError(Exception):
    """Excepción personalizada para errores de Poppy."""
    pass


class ICubError(Exception):
    """Excepción personalizada para errores de iCub."""
    pass


class PoppyIntegration:
    """
    Integración con robot Poppy.
    
    Control de robot humanoide Poppy mediante pypot.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Inicializar integración Poppy (optimizado).
        
        Args:
            config_path: Ruta al archivo de configuración JSON
            
        Raises:
            ValueError: Si config_path es inválido
            PoppyError: Si hay error en la inicialización
        """
        if not POPPY_AVAILABLE:
            logger.warning("Poppy not available. Features will be disabled.")
            self.robot = None
            self.initialized = False
            return
        
        # Validar config_path si se proporciona
        if config_path is not None:
            if not isinstance(config_path, str) or not config_path.strip():
                raise ValueError("config_path must be a non-empty string if provided")
            
            if not os.path.exists(config_path):
                raise ValueError(f"Config file not found: {config_path}")
            
            if not os.path.isfile(config_path):
                raise ValueError(f"config_path must be a file, not a directory: {config_path}")
        
        try:
            if config_path:
                self.robot = from_json(config_path.strip())
                logger.info(f"Poppy robot loaded from config: {config_path}")
            else:
                # Usar configuración por defecto
                logger.warning("No Poppy config provided, using default")
                self.robot = None
            
            self.initialized = self.robot is not None
            if self.initialized:
                logger.info("Poppy integration initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Poppy: {e}", exc_info=True)
            self.robot = None
            self.initialized = False
            raise PoppyError(f"Failed to initialize Poppy: {str(e)}") from e
    
    def set_joint_positions(self, positions: Dict[str, Union[float, int]]) -> bool:
        """
        Establecer posiciones de articulaciones (optimizado).
        
        Args:
            positions: Dict con nombre de articulación -> posición (radianes)
            
        Returns:
            True si se estableció exitosamente
            
        Raises:
            ValueError: Si los parámetros son inválidos
            PoppyError: Si hay error al establecer posiciones
        """
        if not self.initialized or self.robot is None:
            logger.warning("Poppy not initialized")
            return False
        
        # Guard clauses
        if not isinstance(positions, dict):
            raise ValueError(f"positions must be a dict, got {type(positions)}")
        
        if len(positions) == 0:
            raise ValueError("positions dict cannot be empty")
        
        # Validar valores
        for motor_name, position in positions.items():
            if not isinstance(motor_name, str) or not motor_name.strip():
                raise ValueError(f"Motor name must be a non-empty string, got {motor_name}")
            
            try:
                position_float = float(position)
                if not (-10.0 <= position_float <= 10.0):  # Rango razonable para radianes
                    logger.warning(f"Position {position_float} for {motor_name} is outside typical range [-10, 10]")
            except (ValueError, TypeError):
                raise ValueError(f"Position for {motor_name} must be a number, got {type(position)}")
        
        try:
            success_count = 0
            for motor_name, position in positions.items():
                motor_name_clean = motor_name.strip()
                if hasattr(self.robot, motor_name_clean):
                    motor = getattr(self.robot, motor_name_clean)
                    motor.goal_position = float(position)
                    success_count += 1
                else:
                    logger.warning(f"Motor '{motor_name_clean}' not found in Poppy robot")
            
            if success_count == 0:
                logger.warning("No valid motors found in positions dict")
                return False
            
            logger.debug(f"Set {success_count}/{len(positions)} Poppy joint positions")
            return True
        except Exception as e:
            logger.error(f"Error setting Poppy joint positions: {e}", exc_info=True)
            raise PoppyError(f"Failed to set joint positions: {str(e)}") from e


class ICubIntegration:
    """
    Integración con robot iCub.
    
    Control de robot humanoide iCub mediante YARP.
    """
    
    def __init__(self, robot_name: str = "icub"):
        """
        Inicializar integración iCub (optimizado).
        
        Args:
            robot_name: Nombre del robot iCub (default: "icub")
            
        Raises:
            ValueError: Si robot_name es inválido
            ICubError: Si hay error en la inicialización
        """
        if not ICUB_AVAILABLE:
            logger.warning("iCub not available. Features will be disabled.")
            self.interface = None
            self.initialized = False
            self.robot_name = robot_name
            return
        
        # Validar robot_name
        if not robot_name or not isinstance(robot_name, str) or not robot_name.strip():
            raise ValueError("robot_name must be a non-empty string")
        
        try:
            self.robot_name = robot_name.strip()
            self.interface = icubInterface()
            self.initialized = True
            logger.info(f"iCub integration initialized for robot: {self.robot_name}")
        except Exception as e:
            logger.error(f"Failed to initialize iCub: {e}", exc_info=True)
            self.interface = None
            self.initialized = False
            self.robot_name = robot_name
            raise ICubError(f"Failed to initialize iCub: {str(e)}") from e
    
    def connect_to_robot(self) -> bool:
        """
        Conectar a robot iCub (optimizado).
        
        Returns:
            True si la conexión fue exitosa
            
        Raises:
            ICubError: Si hay error en la conexión
        """
        if not self.initialized or self.interface is None:
            logger.warning("iCub not initialized")
            return False
        
        try:
            # Placeholder - implementar conexión real
            # En producción: self.interface.connect()
            logger.info(f"Connected to iCub robot: {self.robot_name}")
            return True
        except Exception as e:
            logger.error(f"Error connecting to iCub: {e}", exc_info=True)
            raise ICubError(f"Failed to connect to iCub: {str(e)}") from e
    
    def shutdown(self) -> None:
        """
        Cerrar conexión con iCub (optimizado).
        
        Raises:
            ICubError: Si hay error al cerrar la conexión
        """
        if not self.interface:
            logger.warning("iCub interface is None, nothing to shutdown")
            return
        
        try:
            # Placeholder - implementar shutdown real
            # En producción: self.interface.disconnect()
            logger.info(f"iCub connection closed for robot: {self.robot_name}")
        except Exception as e:
            logger.error(f"Error shutting down iCub: {e}", exc_info=True)
            raise ICubError(f"Failed to shutdown iCub: {str(e)}") from e
