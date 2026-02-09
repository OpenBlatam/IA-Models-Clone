"""
Transfer Learning Manager - Gestor de transfer learning
========================================================
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class TransferStrategy(Enum):
    """Estrategias de transfer learning"""
    FINE_TUNING = "fine_tuning"
    FEATURE_EXTRACTION = "feature_extraction"
    PROGRESSIVE_UNFREEZING = "progressive_unfreezing"
    DISCRIMINATIVE_LR = "discriminative_lr"


@dataclass
class TransferConfig:
    """Configuración de transfer learning"""
    strategy: TransferStrategy = TransferStrategy.FINE_TUNING
    freeze_backbone: bool = True
    freeze_layers: List[str] = field(default_factory=list)
    learning_rate_backbone: float = 1e-5
    learning_rate_head: float = 1e-3
    unfreeze_schedule: List[int] = field(default_factory=list)  # Epochs para unfreeze


class TransferLearningManager:
    """Gestor de transfer learning"""
    
    def __init__(self, config: TransferConfig):
        self.config = config
        self.frozen_layers: List[str] = []
    
    def prepare_model(
        self,
        pretrained_model: nn.Module,
        num_classes: int,
        custom_head: Optional[nn.Module] = None
    ) -> nn.Module:
        """Prepara modelo para transfer learning"""
        if self.config.strategy == TransferStrategy.FEATURE_EXTRACTION:
            return self._feature_extraction(pretrained_model, num_classes, custom_head)
        elif self.config.strategy == TransferStrategy.FINE_TUNING:
            return self._fine_tuning(pretrained_model, num_classes, custom_head)
        elif self.config.strategy == TransferStrategy.PROGRESSIVE_UNFREEZING:
            return self._progressive_unfreezing(pretrained_model, num_classes, custom_head)
        else:
            return self._fine_tuning(pretrained_model, num_classes, custom_head)
    
    def _feature_extraction(
        self,
        model: nn.Module,
        num_classes: int,
        custom_head: Optional[nn.Module]
    ) -> nn.Module:
        """Feature extraction: congela backbone"""
        # Congelar todos los parámetros
        for param in model.parameters():
            param.requires_grad = False
        
        # Reemplazar head
        if hasattr(model, 'classifier'):
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)
        elif hasattr(model, 'fc'):
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
        elif custom_head:
            model = nn.Sequential(*list(model.children())[:-1], custom_head)
        else:
            logger.warning("No se pudo encontrar head del modelo")
        
        logger.info("Modelo configurado para feature extraction")
        return model
    
    def _fine_tuning(
        self,
        model: nn.Module,
        num_classes: int,
        custom_head: Optional[nn.Module]
    ) -> nn.Module:
        """Fine-tuning: descongela todo"""
        # Reemplazar head
        if hasattr(model, 'classifier'):
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)
        elif hasattr(model, 'fc'):
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
        elif custom_head:
            model = nn.Sequential(*list(model.children())[:-1], custom_head)
        
        # Todos los parámetros entrenables
        for param in model.parameters():
            param.requires_grad = True
        
        logger.info("Modelo configurado para fine-tuning")
        return model
    
    def _progressive_unfreezing(
        self,
        model: nn.Module,
        num_classes: int,
        custom_head: Optional[nn.Module]
    ) -> nn.Module:
        """Progressive unfreezing"""
        # Inicialmente congelar todo
        for param in model.parameters():
            param.requires_grad = False
        
        # Reemplazar head
        if hasattr(model, 'classifier'):
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)
            model.classifier.requires_grad = True
        elif hasattr(model, 'fc'):
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
            model.fc.requires_grad = True
        elif custom_head:
            model = nn.Sequential(*list(model.children())[:-1], custom_head)
        
        logger.info("Modelo configurado para progressive unfreezing")
        return model
    
    def unfreeze_layers(self, model: nn.Module, layer_names: List[str]):
        """Descongela capas específicas"""
        for name, param in model.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = True
                logger.info(f"Capa {name} descongelada")
    
    def get_parameter_groups(
        self,
        model: nn.Module
    ) -> List[Dict[str, Any]]:
        """Obtiene grupos de parámetros para discriminative LR"""
        backbone_params = []
        head_params = []
        
        for name, param in model.named_parameters():
            if 'classifier' in name or 'fc' in name or 'head' in name:
                head_params.append(param)
            else:
                backbone_params.append(param)
        
        return [
            {"params": backbone_params, "lr": self.config.learning_rate_backbone},
            {"params": head_params, "lr": self.config.learning_rate_head}
        ]




