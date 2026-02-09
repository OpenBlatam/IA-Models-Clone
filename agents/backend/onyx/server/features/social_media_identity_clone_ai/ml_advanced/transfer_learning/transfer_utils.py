"""
Utilidades de Transfer Learning
"""

import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class TransferLearningUtils:
    """Utilidades para transfer learning"""
    
    @staticmethod
    def freeze_layers(
        model: nn.Module,
        num_layers: int = 0,
        freeze_embeddings: bool = True
    ) -> nn.Module:
        """
        Congela capas del modelo
        
        Args:
            model: Modelo
            num_layers: Número de capas a congelar (desde el inicio)
            freeze_embeddings: Si congelar embeddings
            
        Returns:
            Modelo con capas congeladas
        """
        # Congelar embeddings
        if freeze_embeddings:
            for param in model.embeddings.parameters():
                param.requires_grad = False
        
        # Congelar primeras N capas
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
            for i, layer in enumerate(model.encoder.layer):
                if i < num_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
        
        logger.info(f"Congeladas {num_layers} capas + embeddings: {freeze_embeddings}")
        return model
    
    @staticmethod
    def add_classification_head(
        base_model: nn.Module,
        num_classes: int,
        dropout: float = 0.1
    ) -> nn.Module:
        """
        Agrega cabeza de clasificación
        
        Args:
            base_model: Modelo base
            num_classes: Número de clases
            dropout: Dropout
            
        Returns:
            Modelo con cabeza de clasificación
        """
        hidden_size = base_model.config.hidden_size if hasattr(base_model, 'config') else 768
        
        classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        # Agregar al modelo
        if hasattr(base_model, 'classifier'):
            base_model.classifier = classifier
        else:
            base_model.add_module('classifier', classifier)
        
        return base_model
    
    @staticmethod
    def get_layer_outputs(
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        layer_indices: Optional[List[int]] = None
    ) -> Dict[int, torch.Tensor]:
        """
        Obtiene outputs de capas específicas
        
        Args:
            model: Modelo
            inputs: Inputs
            layer_indices: Índices de capas (None para todas)
            
        Returns:
            Outputs por capa
        """
        outputs = {}
        
        def hook_fn(layer_idx):
            def hook(module, input, output):
                outputs[layer_idx] = output.detach()
            return hook
        
        handles = []
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
            layers = model.encoder.layer
            for idx, layer in enumerate(layers):
                if layer_indices is None or idx in layer_indices:
                    handle = layer.register_forward_hook(hook_fn(idx))
                    handles.append(handle)
        
        # Forward pass
        with torch.no_grad():
            _ = model(**inputs)
        
        # Remover hooks
        for handle in handles:
            handle.remove()
        
        return outputs
    
    @staticmethod
    def progressive_unfreezing(
        model: nn.Module,
        current_epoch: int,
        total_epochs: int,
        num_layers: int
    ) -> nn.Module:
        """
        Descongelamiento progresivo de capas
        
        Args:
            model: Modelo
            current_epoch: Época actual
            total_epochs: Total de épocas
            num_layers: Número total de capas
            
        Returns:
            Modelo con capas descongeladas progresivamente
        """
        # Calcular cuántas capas descongelar
        progress = current_epoch / total_epochs
        layers_to_unfreeze = int(num_layers * progress)
        
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
            for i, layer in enumerate(model.encoder.layer):
                if i < layers_to_unfreeze:
                    for param in layer.parameters():
                        param.requires_grad = True
        
        return model




