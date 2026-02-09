"""
Configuración de Cámara para Control de Calidad
"""

from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class CameraSettings:
    """Configuración de la cámara"""
    camera_index: int = 0
    resolution_width: int = 1920
    resolution_height: int = 1080
    fps: int = 30
    brightness: float = 0.5
    contrast: float = 0.5
    saturation: float = 0.5
    exposure: float = 0.5
    auto_focus: bool = True
    focus_distance: Optional[float] = None
    white_balance: str = "auto"  # "auto", "daylight", "tungsten", "fluorescent"
    iso: Optional[int] = None  # None = auto
    format: str = "RGB"  # "RGB", "BGR", "GRAY"
    flip_horizontal: bool = False
    flip_vertical: bool = False


class CameraConfig:
    """Gestor de configuración de cámara"""
    
    def __init__(self, settings: Optional[CameraSettings] = None):
        """
        Inicializar configuración de cámara
        
        Args:
            settings: Configuración personalizada de la cámara
        """
        self.settings = settings or CameraSettings()
        logger.info(f"Camera config initialized with index {self.settings.camera_index}")
    
    def update_settings(self, **kwargs):
        """
        Actualizar configuración de la cámara
        
        Args:
            **kwargs: Parámetros a actualizar
        """
        for key, value in kwargs.items():
            if hasattr(self.settings, key):
                setattr(self.settings, key, value)
                logger.debug(f"Updated camera setting {key} = {value}")
            else:
                logger.warning(f"Unknown camera setting: {key}")
    
    def get_settings_dict(self) -> dict:
        """Obtener configuración como diccionario"""
        return {
            "camera_index": self.settings.camera_index,
            "resolution": {
                "width": self.settings.resolution_width,
                "height": self.settings.resolution_height
            },
            "fps": self.settings.fps,
            "brightness": self.settings.brightness,
            "contrast": self.settings.contrast,
            "saturation": self.settings.saturation,
            "exposure": self.settings.exposure,
            "auto_focus": self.settings.auto_focus,
            "focus_distance": self.settings.focus_distance,
            "white_balance": self.settings.white_balance,
            "iso": self.settings.iso,
            "format": self.settings.format,
            "flip_horizontal": self.settings.flip_horizontal,
            "flip_vertical": self.settings.flip_vertical,
        }
    
    def validate(self) -> bool:
        """
        Validar configuración
        
        Returns:
            True si la configuración es válida
        """
        if self.settings.camera_index < 0:
            logger.error("Camera index must be >= 0")
            return False
        
        if self.settings.resolution_width <= 0 or self.settings.resolution_height <= 0:
            logger.error("Resolution must be positive")
            return False
        
        if self.settings.fps <= 0:
            logger.error("FPS must be positive")
            return False
        
        if self.settings.white_balance not in ["auto", "daylight", "tungsten", "fluorescent"]:
            logger.error(f"Invalid white balance: {self.settings.white_balance}")
            return False
        
        if self.settings.format not in ["RGB", "BGR", "GRAY"]:
            logger.error(f"Invalid format: {self.settings.format}")
            return False
        
        logger.debug("Camera configuration validated successfully")
        return True






