"""
Model Utils - Utilidades de Modelos
====================================

Utilidades para construcción, carga y gestión de modelos.
"""

import logging
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List
from pathlib import Path
import json

logger = logging.getLogger(__name__)

# Intentar importar transformers
try:
    from transformers import AutoModel, AutoTokenizer, AutoConfig
    _has_transformers = True
except ImportError:
    _has_transformers = False


class ModelBuilder:
    """
    Builder para construir modelos PyTorch.
    """
    
    @staticmethod
    def create_mlp(
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        activation: str = "relu",
        dropout: float = 0.0
    ) -> nn.Module:
        """
        Crear MLP (Multi-Layer Perceptron).
        
        Args:
            input_size: Tamaño de entrada
            hidden_sizes: Tamaños de capas ocultas
            output_size: Tamaño de salida
            activation: Función de activación
            dropout: Dropout rate
            
        Returns:
            Modelo MLP
        """
        layers = []
        prev_size = input_size
        
        # Capas ocultas
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "sigmoid":
                layers.append(nn.Sigmoid())
            elif activation == "gelu":
                layers.append(nn.GELU())
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_size = hidden_size
        
        # Capa de salida
        layers.append(nn.Linear(prev_size, output_size))
        
        return nn.Sequential(*layers)
    
    @staticmethod
    def create_cnn(
        input_channels: int,
        num_classes: int,
        architecture: str = "simple"
    ) -> nn.Module:
        """
        Crear CNN simple.
        
        Args:
            input_channels: Canales de entrada
            num_classes: Número de clases
            architecture: Arquitectura (simple, resnet-like)
            
        Returns:
            Modelo CNN
        """
        if architecture == "simple":
            return nn.Sequential(
                nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(128, num_classes)
            )
        else:
            raise ValueError(f"Unknown architecture: {architecture}")


class ModelCheckpointer:
    """
    Gestor de checkpoints de modelos.
    """
    
    def __init__(self, save_dir: str = "./checkpoints"):
        """
        Inicializar checkpointer.
        
        Args:
            save_dir: Directorio para guardar checkpoints
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def save(
        self,
        model: nn.Module,
        epoch: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False
    ) -> str:
        """
        Guardar checkpoint.
        
        Args:
            model: Modelo
            epoch: Número de epoch
            optimizer: Optimizador (opcional)
            metrics: Métricas (opcional)
            is_best: Si es el mejor modelo
            
        Returns:
            Ruta del checkpoint guardado
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "model_config": self._get_model_config(model)
        }
        
        if optimizer:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        
        if metrics:
            checkpoint["metrics"] = metrics
        
        # Guardar checkpoint regular
        checkpoint_path = self.save_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Guardar mejor modelo
        if is_best:
            best_path = self.save_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved: {best_path}")
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        return str(checkpoint_path)
    
    def load(
        self,
        model: nn.Module,
        path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "cpu"
    ) -> Dict[str, Any]:
        """
        Cargar checkpoint.
        
        Args:
            model: Modelo
            path: Ruta del checkpoint
            optimizer: Optimizador (opcional)
            device: Dispositivo
            
        Returns:
            Información del checkpoint
        """
        checkpoint = torch.load(path, map_location=device)
        
        model.load_state_dict(checkpoint["model_state_dict"])
        
        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        logger.info(f"Checkpoint loaded: {path}")
        return checkpoint
    
    def _get_model_config(self, model: nn.Module) -> Dict[str, Any]:
        """Obtener configuración del modelo"""
        return {
            "type": type(model).__name__,
            "parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)
        }


def load_pretrained_model(
    model_name: str,
    task: str = "classification",
    num_labels: Optional[int] = None
) -> tuple:
    """
    Cargar modelo pre-entrenado.
    
    Args:
        model_name: Nombre del modelo
        task: Tipo de tarea (classification, regression, generation)
        num_labels: Número de labels (para clasificación)
        
    Returns:
        Tupla (modelo, tokenizer)
    """
    if not _has_transformers:
        raise ImportError("transformers library is required")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if task == "classification":
        if num_labels is None:
            num_labels = 2
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
    elif task == "generation":
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_name)
    else:
        model = AutoModel.from_pretrained(model_name)
    
    logger.info(f"Loaded pretrained model: {model_name}")
    return model, tokenizer




