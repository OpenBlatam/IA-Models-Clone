"""
Transfer Learning Utilities - Utilidades de transfer learning
===============================================================
Fine-tuning, feature extraction, y adaptación de modelos
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Callable

logger = logging.getLogger(__name__)


class TransferLearningManager:
    """Gestor de transfer learning"""
    
    def __init__(self):
        self.frozen_layers: List[str] = []
        self.trainable_layers: List[str] = []
    
    def freeze_backbone(
        self,
        model: nn.Module,
        freeze_patterns: Optional[List[str]] = None
    ) -> nn.Module:
        """Congela backbone del modelo"""
        if freeze_patterns is None:
            # Congelar todas las capas excepto las últimas
            freeze_patterns = ["embedding", "encoder"]
        
        for name, param in model.named_parameters():
            should_freeze = any(pattern in name for pattern in freeze_patterns)
            if should_freeze:
                param.requires_grad = False
                self.frozen_layers.append(name)
            else:
                param.requires_grad = True
                self.trainable_layers.append(name)
        
        logger.info(f"Frozen {len(self.frozen_layers)} layers, {len(self.trainable_layers)} trainable")
        return model
    
    def add_classification_head(
        self,
        model: nn.Module,
        num_classes: int,
        hidden_dim: int = 512,
        dropout: float = 0.1
    ) -> nn.Module:
        """Agrega cabeza de clasificación"""
        # Obtener dimensión de salida del modelo base
        if hasattr(model, "config"):
            base_output_dim = model.config.hidden_size
        else:
            # Intentar inferir
            base_output_dim = hidden_dim
        
        classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(base_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Agregar al modelo
        if hasattr(model, "classifier"):
            model.classifier = classifier
        else:
            model.add_module("classifier", classifier)
        
        logger.info(f"Added classification head: {num_classes} classes")
        return model
    
    def add_regression_head(
        self,
        model: nn.Module,
        output_dim: int = 1,
        hidden_dim: int = 512,
        dropout: float = 0.1
    ) -> nn.Module:
        """Agrega cabeza de regresión"""
        if hasattr(model, "config"):
            base_output_dim = model.config.hidden_size
        else:
            base_output_dim = hidden_dim
        
        regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(base_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        if hasattr(model, "regressor"):
            model.regressor = regressor
        else:
            model.add_module("regressor", regressor)
        
        logger.info(f"Added regression head: {output_dim} outputs")
        return model
    
    def progressive_unfreezing(
        self,
        model: nn.Module,
        current_epoch: int,
        unfreeze_schedule: Dict[int, List[str]]
    ) -> nn.Module:
        """Unfreezing progresivo de capas"""
        for epoch, layer_patterns in unfreeze_schedule.items():
            if current_epoch >= epoch:
                for name, param in model.named_parameters():
                    if any(pattern in name for pattern in layer_patterns):
                        param.requires_grad = True
                        if name not in self.trainable_layers:
                            self.trainable_layers.append(name)
                            if name in self.frozen_layers:
                                self.frozen_layers.remove(name)
        
        return model
    
    def get_feature_extractor(
        self,
        model: nn.Module,
        layer_name: Optional[str] = None
    ) -> Callable:
        """Crea extractor de características"""
        def extract_features(x):
            model.eval()
            with torch.no_grad():
                if layer_name:
                    # Extraer de capa específica
                    features = None
                    def hook(module, input, output):
                        nonlocal features
                        features = output
                    
                    target_layer = dict(model.named_modules())[layer_name]
                    handle = target_layer.register_forward_hook(hook)
                    _ = model(x)
                    handle.remove()
                    return features
                else:
                    # Usar salida del modelo base
                    if hasattr(model, "base_model"):
                        return model.base_model(x)
                    else:
                        # Remover última capa
                        base_layers = list(model.children())[:-1]
                        base_model = nn.Sequential(*base_layers)
                        return base_model(x)
        
        return extract_features
    
    def adapt_model(
        self,
        source_model: nn.Module,
        target_num_classes: int,
        freeze_backbone: bool = True
    ) -> nn.Module:
        """Adapta modelo para nueva tarea"""
        # Clonar modelo
        adapted_model = type(source_model)(source_model.config) if hasattr(source_model, 'config') else source_model
        
        # Congelar backbone si es necesario
        if freeze_backbone:
            adapted_model = self.freeze_backbone(adapted_model)
        
        # Agregar nueva cabeza
        adapted_model = self.add_classification_head(adapted_model, target_num_classes)
        
        logger.info(f"Adapted model for {target_num_classes} classes")
        return adapted_model




