"""
Model Factory
=============

Factory para crear modelos de enrutamiento.
"""

import logging
from typing import Dict, Any, Optional

from .base_model import BaseRouteModel, ModelConfig
from .mlp_model import MLPRoutePredictor

logger = logging.getLogger(__name__)

# Imports condicionales para otros modelos
try:
    from .gnn_model import GCNRoutePredictor, GATRoutePredictor
    GNN_AVAILABLE = True
except ImportError:
    GNN_AVAILABLE = False
    logger.warning("GNN models not available")

try:
    from .transformer_model import TransformerRouteModel
    TRANSFORMER_MODEL_AVAILABLE = True
except ImportError:
    TRANSFORMER_MODEL_AVAILABLE = False
    logger.warning("Transformer model not available")


class ModelFactory:
    """
    Factory para crear modelos de enrutamiento.
    """
    
    @staticmethod
    def create_model(model_type: str, config: ModelConfig) -> BaseRouteModel:
        """
        Crear modelo según tipo.
        
        Args:
            model_type: Tipo de modelo (mlp, gcn, gat, transformer)
            config: Configuración del modelo
            
        Returns:
            Instancia del modelo
        """
        model_type = model_type.lower()
        
        if model_type == "mlp":
            return MLPRoutePredictor(config, use_attention=config.__dict__.get("use_attention", True))
        
        elif model_type == "gcn" and GNN_AVAILABLE:
            return GCNRoutePredictor(config)
        
        elif model_type == "gat" and GNN_AVAILABLE:
            return GATRoutePredictor(config)
        
        elif model_type == "transformer" and TRANSFORMER_MODEL_AVAILABLE:
            return TransformerRouteModel(config)
        
        else:
            raise ValueError(f"Modelo no soportado: {model_type}")
    
    @staticmethod
    def create_from_config(config_dict: Dict[str, Any]) -> BaseRouteModel:
        """
        Crear modelo desde diccionario de configuración.
        
        Args:
            config_dict: Diccionario de configuración
            
        Returns:
            Instancia del modelo
        """
        model_type = config_dict.get("model_type", "mlp")
        model_config_dict = config_dict.get("model", {})
        
        # Crear ModelConfig
        config = ModelConfig(**model_config_dict)
        
        return ModelFactory.create_model(model_type, config)


