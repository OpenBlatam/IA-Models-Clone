"""
Model Utils - Utilidades de Modelos
====================================

Utilidades para gestión y manipulación de modelos.
"""

import logging
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List
import os
import json

logger = logging.getLogger(__name__)


class ModelManager:
    """Gestor de modelos"""
    
    def __init__(self, models_dir: str = "./models"):
        """
        Inicializar gestor
        
        Args:
            models_dir: Directorio de modelos
        """
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        logger.info(f"Model Manager inicializado en {models_dir}")
    
    def save_model(
        self,
        model: nn.Module,
        name: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Guardar modelo
        
        Args:
            model: Modelo a guardar
            name: Nombre del modelo
            metadata: Metadatos adicionales
        """
        model_path = os.path.join(self.models_dir, f"{name}.pt")
        metadata_path = os.path.join(self.models_dir, f"{name}_metadata.json")
        
        # Guardar modelo
        torch.save({
            "model_state_dict": model.state_dict(),
            "model_class": model.__class__.__name__
        }, model_path)
        
        # Guardar metadatos
        if metadata:
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
        
        logger.info(f"Modelo guardado: {model_path}")
    
    def load_model(
        self,
        name: str,
        model_class: Optional[nn.Module] = None
    ) -> Optional[nn.Module]:
        """
        Cargar modelo
        
        Args:
            name: Nombre del modelo
            model_class: Clase del modelo (opcional)
            
        Returns:
            Modelo cargado o None
        """
        model_path = os.path.join(self.models_dir, f"{name}.pt")
        
        if not os.path.exists(model_path):
            logger.error(f"Modelo no encontrado: {model_path}")
            return None
        
        try:
            checkpoint = torch.load(model_path, map_location="cpu")
            
            if model_class:
                model = model_class()
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                logger.warning("model_class no proporcionado, solo se cargó state_dict")
                return checkpoint
            
            logger.info(f"Modelo cargado: {model_path}")
            return model
            
        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
            return None
    
    def list_models(self) -> List[str]:
        """Listar modelos disponibles"""
        models = []
        for file in os.listdir(self.models_dir):
            if file.endswith(".pt"):
                models.append(file[:-3])  # Remover .pt
        return models


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Contar parámetros del modelo
    
    Args:
        model: Modelo
        
    Returns:
        Dict con conteo de parámetros
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": total - trainable
    }


def get_model_size(model: nn.Module, unit: str = "MB") -> float:
    """
    Obtener tamaño del modelo
    
    Args:
        model: Modelo
        unit: Unidad (B, KB, MB, GB)
        
    Returns:
        Tamaño en la unidad especificada
    """
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size = param_size + buffer_size
    
    unit_map = {
        "B": 1,
        "KB": 1024,
        "MB": 1024**2,
        "GB": 1024**3
    }
    
    return total_size / unit_map.get(unit.upper(), 1024**2)


def freeze_layers(model: nn.Module, num_layers: int = 1):
    """
    Congelar primeras N capas
    
    Args:
        model: Modelo
        num_layers: Número de capas a congelar
    """
    layers = list(model.children())
    for i, layer in enumerate(layers[:num_layers]):
        for param in layer.parameters():
            param.requires_grad = False
        logger.info(f"Capa {i} congelada")


def unfreeze_all(model: nn.Module):
    """Descongelar todas las capas"""
    for param in model.parameters():
        param.requires_grad = True




