"""
Model Factory - Modular Model Creation
======================================

Factory pattern para crear modelos de manera modular y extensible.
"""

import logging
from typing import Dict, Any, Optional, Type
from enum import Enum
import torch
import torch.nn as nn

from ..base_model import BaseRobotModel
from ..transformer_trajectory import TransformerTrajectoryPredictor
from ..diffusion_trajectory import DiffusionTrajectoryGenerator
from ..trajectory_predictor import TrajectoryPredictor

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Tipos de modelos disponibles."""
    TRANSFORMER = "transformer"
    DIFFUSION = "diffusion"
    MLP = "mlp"
    CNN = "cnn"
    LSTM = "lstm"
    GRU = "gru"
    CUSTOM = "custom"


class ModelFactory:
    """
    Factory para crear modelos de manera modular.
    
    Permite registrar y crear modelos de diferentes tipos
    de forma centralizada y extensible.
    """
    
    _registry: Dict[str, Type[BaseRobotModel]] = {}
    _default_configs: Dict[str, Dict[str, Any]] = {}
    
    @classmethod
    def register(
        cls,
        model_type: ModelType,
        model_class: Type[BaseRobotModel],
        default_config: Optional[Dict[str, Any]] = None
    ):
        """
        Registrar un tipo de modelo.
        
        Args:
            model_type: Tipo de modelo
            model_class: Clase del modelo
            default_config: Configuración por defecto
        """
        cls._registry[model_type.value] = model_class
        if default_config:
            cls._default_configs[model_type.value] = default_config
        logger.info(f"Registered model type: {model_type.value}")
    
    @classmethod
    def create(
        cls,
        model_type: ModelType,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> BaseRobotModel:
        """
        Crear un modelo del tipo especificado.
        
        Args:
            model_type: Tipo de modelo
            config: Configuración del modelo
            **kwargs: Argumentos adicionales
            
        Returns:
            Instancia del modelo
        """
        if model_type.value not in cls._registry:
            raise ValueError(f"Unknown model type: {model_type.value}")
        
        model_class = cls._registry[model_type.value]
        
        # Combinar configuración por defecto con la proporcionada
        final_config = cls._default_configs.get(model_type.value, {}).copy()
        if config:
            final_config.update(config)
        final_config.update(kwargs)
        
        try:
            model = model_class(**final_config)
            logger.info(f"Created {model_type.value} model with config: {final_config}")
            return model
        except Exception as e:
            logger.error(f"Error creating model {model_type.value}: {e}")
            raise
    
    @classmethod
    def list_available_models(cls) -> list:
        """Listar modelos disponibles."""
        return list(cls._registry.keys())
    
    @classmethod
    def get_default_config(cls, model_type: ModelType) -> Dict[str, Any]:
        """Obtener configuración por defecto."""
        return cls._default_configs.get(model_type.value, {}).copy()


# Registrar modelos por defecto
ModelFactory.register(
    ModelType.TRANSFORMER,
    TransformerTrajectoryPredictor,
    default_config={
        "input_size": 3,
        "output_size": 3,
        "d_model": 256,
        "num_heads": 8,
        "num_layers": 6
    }
)

ModelFactory.register(
    ModelType.DIFFUSION,
    DiffusionTrajectoryGenerator,
    default_config={
        "trajectory_length": 100,
        "trajectory_dim": 3,
        "hidden_dim": 256,
        "num_timesteps": 1000
    }
)

ModelFactory.register(
    ModelType.MLP,
    TrajectoryPredictor,
    default_config={
        "input_size": 3,
        "output_size": 3,
        "hidden_sizes": [128, 64, 32]
    }
)

