"""
Configuración de Detección para Control de Calidad
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class DetectionSettings:
    """Configuración de detección"""
    # Detección de objetos
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.4
    max_objects: int = 100
    
    # Detección de anomalías
    anomaly_threshold: float = 0.7
    use_autoencoder: bool = True
    use_statistical: bool = True
    statistical_threshold: float = 3.0  # Desviaciones estándar
    
    # Clasificación de defectos
    defect_categories: List[str] = None
    min_defect_size: int = 10  # píxeles
    max_defect_size: int = 10000  # píxeles
    
    # Procesamiento de imagen
    preprocessing_enabled: bool = True
    noise_reduction: bool = True
    contrast_enhancement: bool = True
    edge_detection: bool = True
    
    # Modelos
    object_detection_model: str = "yolov8"  # "yolov8", "faster_rcnn", "ssd"
    anomaly_detection_model: Optional[str] = None
    defect_classifier_model: Optional[str] = None
    
    def __post_init__(self):
        """Inicializar valores por defecto"""
        if self.defect_categories is None:
            self.defect_categories = [
                "scratch",
                "crack",
                "dent",
                "discoloration",
                "deformation",
                "missing_part",
                "surface_imperfection",
                "contamination",
                "size_variation",
                "other"
            ]


class DetectionConfig:
    """Gestor de configuración de detección"""
    
    def __init__(self, settings: Optional[DetectionSettings] = None):
        """
        Inicializar configuración de detección
        
        Args:
            settings: Configuración personalizada
        """
        self.settings = settings or DetectionSettings()
        logger.info("Detection config initialized")
    
    def update_settings(self, **kwargs):
        """
        Actualizar configuración
        
        Args:
            **kwargs: Parámetros a actualizar
        """
        for key, value in kwargs.items():
            if hasattr(self.settings, key):
                setattr(self.settings, key, value)
                logger.debug(f"Updated detection setting {key} = {value}")
            else:
                logger.warning(f"Unknown detection setting: {key}")
    
    def get_settings_dict(self) -> dict:
        """Obtener configuración como diccionario"""
        return {
            "confidence_threshold": self.settings.confidence_threshold,
            "nms_threshold": self.settings.nms_threshold,
            "max_objects": self.settings.max_objects,
            "anomaly_threshold": self.settings.anomaly_threshold,
            "use_autoencoder": self.settings.use_autoencoder,
            "use_statistical": self.settings.use_statistical,
            "statistical_threshold": self.settings.statistical_threshold,
            "defect_categories": self.settings.defect_categories,
            "min_defect_size": self.settings.min_defect_size,
            "max_defect_size": self.settings.max_defect_size,
            "preprocessing_enabled": self.settings.preprocessing_enabled,
            "noise_reduction": self.settings.noise_reduction,
            "contrast_enhancement": self.settings.contrast_enhancement,
            "edge_detection": self.settings.edge_detection,
            "object_detection_model": self.settings.object_detection_model,
            "anomaly_detection_model": self.settings.anomaly_detection_model,
            "defect_classifier_model": self.settings.defect_classifier_model,
        }
    
    def validate(self) -> bool:
        """
        Validar configuración
        
        Returns:
            True si la configuración es válida
        """
        if not 0.0 <= self.settings.confidence_threshold <= 1.0:
            logger.error("Confidence threshold must be between 0 and 1")
            return False
        
        if not 0.0 <= self.settings.nms_threshold <= 1.0:
            logger.error("NMS threshold must be between 0 and 1")
            return False
        
        if not 0.0 <= self.settings.anomaly_threshold <= 1.0:
            logger.error("Anomaly threshold must be between 0 and 1")
            return False
        
        if self.settings.min_defect_size >= self.settings.max_defect_size:
            logger.error("min_defect_size must be less than max_defect_size")
            return False
        
        if self.settings.object_detection_model not in ["yolov8", "faster_rcnn", "ssd"]:
            logger.error(f"Invalid object detection model: {self.settings.object_detection_model}")
            return False
        
        logger.debug("Detection configuration validated successfully")
        return True






