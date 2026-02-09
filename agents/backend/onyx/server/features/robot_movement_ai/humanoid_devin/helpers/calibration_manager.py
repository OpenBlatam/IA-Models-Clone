"""
Calibration Manager for Humanoid Devin Robot (Optimizado)
=========================================================

Gestor de calibración para el robot humanoide.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from pathlib import Path
import json

logger = logging.getLogger(__name__)


def ErrorCode(description: str):
    """Decorador para anotar excepciones con códigos de error y descripciones."""
    def decorator(cls):
        cls._error_description = description
        return cls
    return decorator


@ErrorCode(description="Error in calibration system")
class CalibrationError(Exception):
    """Excepción para errores de calibración."""
    
    def __init__(self):
        """Initialize exception with description from @ErrorCode decorator."""
        message = getattr(self.__class__, '_error_description', "Error in calibration system")
        super().__init__(message)
        self.message = message


class CalibrationManager:
    """
    Gestor de calibración para el robot humanoide.
    
    Maneja calibración de articulaciones, poses y sensores.
    """
    
    def __init__(self, robot_driver, calibration_file: Optional[str] = None):
        """
        Inicializar gestor de calibración.
        
        Args:
            robot_driver: Instancia de HumanoidDevinDriver
            calibration_file: Ruta al archivo de calibración (opcional)
        """
        if robot_driver is None:
            raise ValueError("robot_driver cannot be None")
        
        self.robot = robot_driver
        self.calibration_file = calibration_file
        self.calibration_data: Dict[str, Any] = {
            "joint_offsets": {},
            "joint_scales": {},
            "joint_limits": {},
            "workspace_limits": {},
            "sensor_calibration": {},
            "timestamp": None
        }
        
        if calibration_file:
            self.load_calibration(calibration_file)
        else:
            logger.info("Calibration manager initialized with default values")
    
    def calibrate_joint_zero_position(
        self,
        joint_name: str,
        measured_position: float
    ) -> float:
        """
        Calibrar posición cero de una articulación.
        
        Args:
            joint_name: Nombre de la articulación
            measured_position: Posición medida cuando está en "cero"
            
        Returns:
            Offset calculado
        """
        if not joint_name or not isinstance(joint_name, str):
            raise ValueError("joint_name must be a non-empty string")
        
        if not np.isfinite(measured_position):
            raise ValueError("measured_position must be a finite number")
        
        # El offset es la diferencia entre la posición medida y cero
        offset = -measured_position
        
        self.calibration_data["joint_offsets"][joint_name] = float(offset)
        logger.info(f"Calibrated zero position for {joint_name}: offset={offset:.4f}")
        
        return offset
    
    def calibrate_joint_scale(
        self,
        joint_name: str,
        commanded_position: float,
        measured_position: float
    ) -> float:
        """
        Calibrar escala de una articulación.
        
        Args:
            joint_name: Nombre de la articulación
            commanded_position: Posición comandada
            measured_position: Posición medida
            
        Returns:
            Escala calculada
        """
        if commanded_position == 0:
            raise ValueError("commanded_position cannot be zero for scale calibration")
        
        if not np.isfinite(commanded_position) or not np.isfinite(measured_position):
            raise ValueError("Positions must be finite numbers")
        
        scale = measured_position / commanded_position
        
        self.calibration_data["joint_scales"][joint_name] = float(scale)
        logger.info(f"Calibrated scale for {joint_name}: scale={scale:.4f}")
        
        return scale
    
    def calibrate_joint_limits(
        self,
        joint_name: str,
        min_position: float,
        max_position: float
    ) -> None:
        """
        Calibrar límites de una articulación.
        
        Args:
            joint_name: Nombre de la articulación
            min_position: Posición mínima medida
            max_position: Posición máxima medida
        """
        if min_position >= max_position:
            raise ValueError(f"min_position ({min_position}) must be < max_position ({max_position})")
        
        if not all(np.isfinite([min_position, max_position])):
            raise ValueError("Limits must be finite numbers")
        
        self.calibration_data["joint_limits"][joint_name] = {
            "min": float(min_position),
            "max": float(max_position)
        }
        logger.info(f"Calibrated limits for {joint_name}: [{min_position:.4f}, {max_position:.4f}]")
    
    def calibrate_workspace(
        self,
        measured_positions: List[np.ndarray]
    ) -> Dict[str, Tuple[float, float]]:
        """
        Calibrar límites del espacio de trabajo.
        
        Args:
            measured_positions: Lista de posiciones medidas [x, y, z]
            
        Returns:
            Dict con límites {"x": (min, max), "y": (min, max), "z": (min, max)}
        """
        if not measured_positions or len(measured_positions) == 0:
            raise ValueError("measured_positions cannot be empty")
        
        positions_array = np.array(measured_positions)
        
        if positions_array.shape[1] != 3:
            raise ValueError("Positions must have shape (N, 3)")
        
        if not np.all(np.isfinite(positions_array)):
            raise ValueError("All positions must be finite numbers")
        
        limits = {
            "x": (float(np.min(positions_array[:, 0])), float(np.max(positions_array[:, 0]))),
            "y": (float(np.min(positions_array[:, 1])), float(np.max(positions_array[:, 1]))),
            "z": (float(np.min(positions_array[:, 2])), float(np.max(positions_array[:, 2])))
        }
        
        self.calibration_data["workspace_limits"] = limits
        logger.info(f"Calibrated workspace limits: {limits}")
        
        return limits
    
    def apply_joint_calibration(
        self,
        joint_name: str,
        raw_position: float
    ) -> float:
        """
        Aplicar calibración a una posición de articulación.
        
        Args:
            joint_name: Nombre de la articulación
            raw_position: Posición sin calibrar
            
        Returns:
            Posición calibrada
        """
        calibrated = float(raw_position)
        
        # Aplicar offset
        if joint_name in self.calibration_data["joint_offsets"]:
            calibrated += self.calibration_data["joint_offsets"][joint_name]
        
        # Aplicar escala
        if joint_name in self.calibration_data["joint_scales"]:
            calibrated *= self.calibration_data["joint_scales"][joint_name]
        
        return calibrated
    
    def get_joint_limits(self, joint_name: str) -> Optional[Tuple[float, float]]:
        """
        Obtener límites calibrados de una articulación.
        
        Args:
            joint_name: Nombre de la articulación
            
        Returns:
            Tupla (min, max) o None si no está calibrado
        """
        if joint_name in self.calibration_data["joint_limits"]:
            limits = self.calibration_data["joint_limits"][joint_name]
            return (limits["min"], limits["max"])
        return None
    
    def get_workspace_limits(self) -> Dict[str, Tuple[float, float]]:
        """
        Obtener límites del espacio de trabajo.
        
        Returns:
            Dict con límites {"x": (min, max), "y": (min, max), "z": (min, max)}
        """
        return self.calibration_data.get("workspace_limits", {})
    
    def save_calibration(self, file_path: Optional[str] = None) -> None:
        """
        Guardar calibración en archivo.
        
        Args:
            file_path: Ruta al archivo (usa self.calibration_file si no se proporciona)
            
        Raises:
            CalibrationError: Si hay error al guardar
        """
        from datetime import datetime
        
        save_path = file_path or self.calibration_file
        
        if not save_path:
            raise ValueError("file_path must be provided if calibration_file is not set")
        
        # Actualizar timestamp
        self.calibration_data["timestamp"] = datetime.now().isoformat()
        
        try:
            calib_file = Path(save_path)
            calib_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(calib_file, 'w', encoding='utf-8') as f:
                json.dump(self.calibration_data, f, indent=2)
            
            logger.info(f"Calibration saved to {save_path}")
        except Exception as e:
            logger.error(f"Error saving calibration: {e}", exc_info=True)
            raise CalibrationError(f"Failed to save calibration: {str(e)}") from e
    
    def load_calibration(self, file_path: str) -> None:
        """
        Cargar calibración desde archivo.
        
        Args:
            file_path: Ruta al archivo de calibración
            
        Raises:
            CalibrationError: Si hay error al cargar
        """
        calib_file = Path(file_path)
        
        if not calib_file.exists():
            raise CalibrationError(f"Calibration file not found: {file_path}")
        
        try:
            with open(calib_file, 'r', encoding='utf-8') as f:
                self.calibration_data = json.load(f)
            
            self.calibration_file = file_path
            logger.info(f"Calibration loaded from {file_path}")
        except json.JSONDecodeError as e:
            raise CalibrationError(f"Invalid JSON in calibration file: {e}") from e
        except Exception as e:
            logger.error(f"Error loading calibration: {e}", exc_info=True)
            raise CalibrationError(f"Failed to load calibration: {str(e)}") from e
    
    def get_calibration_status(self) -> Dict[str, Any]:
        """
        Obtener estado de la calibración.
        
        Returns:
            Dict con estado de calibración
        """
        return {
            "joints_calibrated": len(self.calibration_data["joint_offsets"]),
            "scales_calibrated": len(self.calibration_data["joint_scales"]),
            "limits_calibrated": len(self.calibration_data["joint_limits"]),
            "workspace_calibrated": len(self.calibration_data["workspace_limits"]) > 0,
            "calibration_file": self.calibration_file,
            "timestamp": self.calibration_data.get("timestamp")
        }
    
    def reset_calibration(self) -> None:
        """Resetear toda la calibración."""
        self.calibration_data = {
            "joint_offsets": {},
            "joint_scales": {},
            "joint_limits": {},
            "workspace_limits": {},
            "sensor_calibration": {},
            "timestamp": None
        }
        logger.info("Calibration reset")

