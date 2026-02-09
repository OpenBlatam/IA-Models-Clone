"""
Model Factory
=============

Factory para crear modelos de deep learning.
"""

import logging
from enum import Enum
from typing import Dict, Any, Optional

from .base_model import BaseRobotModel
from .trajectory_predictor import TrajectoryPredictor
from .motion_controller import MotionController
from .obstacle_detector import ObstacleDetector

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Tipo de modelo."""
    TRAJECTORY_PREDICTOR = "trajectory_predictor"
    MOTION_CONTROLLER = "motion_controller"
    OBSTACLE_DETECTOR = "obstacle_detector"


class ModelFactory:
    """
    Factory para crear modelos de deep learning.
    
    Proporciona una interfaz unificada para crear diferentes tipos de modelos.
    """
    
    _model_registry: Dict[ModelType, type] = {
        ModelType.TRAJECTORY_PREDICTOR: TrajectoryPredictor,
        ModelType.MOTION_CONTROLLER: MotionController,
        ModelType.OBSTACLE_DETECTOR: ObstacleDetector,
    }
    
    @classmethod
    def create_model(
        cls,
        model_type: ModelType,
        config: Dict[str, Any]
    ) -> BaseRobotModel:
        """
        Crear modelo según tipo y configuración.
        
        Args:
            model_type: Tipo de modelo
            config: Configuración del modelo
            
        Returns:
            Modelo instanciado
        """
        if model_type not in cls._model_registry:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model_class = cls._model_registry[model_type]
        
        try:
            if model_type == ModelType.TRAJECTORY_PREDICTOR:
                model = model_class(
                    input_size=config.get("input_size"),
                    output_size=config.get("output_size"),
                    hidden_sizes=config.get("hidden_sizes", [128, 64, 32]),
                    activation=config.get("activation", "relu"),
                    dropout=config.get("dropout", 0.1),
                    use_batch_norm=config.get("use_batch_norm", False)
                )
            elif model_type == ModelType.MOTION_CONTROLLER:
                model = model_class(
                    input_size=config.get("input_size"),
                    output_size=config.get("output_size"),
                    hidden_size=config.get("hidden_size", 128),
                    num_layers=config.get("num_layers", 2),
                    dropout=config.get("dropout", 0.1),
                    bidirectional=config.get("bidirectional", False),
                    use_attention=config.get("use_attention", False)
                )
            elif model_type == ModelType.OBSTACLE_DETECTOR:
                model = model_class(
                    input_channels=config.get("input_channels", 1),
                    num_classes=config.get("num_classes", 2),
                    conv_channels=config.get("conv_channels", [32, 64, 128]),
                    use_residual=config.get("use_residual", False)
                )
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            logger.info(f"Created {model_type.value} model: {model.name}")
            return model
            
        except Exception as e:
            logger.error(f"Error creating model {model_type.value}: {e}")
            raise
    
    @classmethod
    def register_model(cls, model_type: ModelType, model_class: type):
        """
        Registrar nuevo tipo de modelo.
        
        Args:
            model_type: Tipo de modelo
            model_class: Clase del modelo
        """
        if not issubclass(model_class, BaseRobotModel):
            raise ValueError(f"Model class must inherit from BaseRobotModel")
        
        cls._model_registry[model_type] = model_class
        logger.info(f"Registered model type: {model_type.value}")




