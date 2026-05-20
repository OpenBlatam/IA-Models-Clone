"""
Transfer Learning Utils - Utilidades de Transfer Learning
==========================================================

Utilidades para transfer learning y fine-tuning.
"""

import logging
import torch
import torch.nn as nn
from typing import Optional, List, Dict, Any, Callable
from pathlib import Path

logger = logging.getLogger(__name__)

# Intentar importar transformers
try:
    from transformers import AutoModel, AutoTokenizer, AutoConfig
    _has_transformers = True
except ImportError:
    _has_transformers = False
    logger.warning("transformers not available, some transfer learning features will be limited")


class FeatureExtractor:
    """
    Extractor de features de modelos pre-entrenados.
    """
    
    def __init__(
        self,
        model: nn.Module,
        layer_name: Optional[str] = None
    ):
        """
        Inicializar extractor.
        
        Args:
            model: Modelo pre-entrenado
            layer_name: Nombre de capa para extraer features
        """
        self.model = model
        self.model.eval()
        self.layer_name = layer_name
        self.features = None
        
        if layer_name:
            self._register_hook()
    
    def _register_hook(self):
        """Registrar hook para extraer features."""
        def hook(module, input, output):
            self.features = output
        
        for name, module in self.model.named_modules():
            if name == self.layer_name:
                module.register_forward_hook(hook)
                break
    
    def extract(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Extraer features.
        
        Args:
            inputs: Inputs
            
        Returns:
            Features extraídas
        """
        with torch.no_grad():
            _ = self.model(inputs)
            return self.features if self.features is not None else inputs


class TransferLearningModel(nn.Module):
    """
    Modelo para transfer learning.
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        freeze_backbone: bool = True,
        custom_head: Optional[nn.Module] = None
    ):
        """
        Inicializar modelo de transfer learning.
        
        Args:
            backbone: Backbone pre-entrenado
            num_classes: Número de clases
            freeze_backbone: Congelar backbone
            custom_head: Head personalizado (opcional)
        """
        super().__init__()
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Obtener tamaño de salida del backbone
        if custom_head is None:
            # Intentar inferir tamaño
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224)
                if hasattr(backbone, 'fc'):
                    # ResNet style
                    backbone_output_size = backbone.fc.in_features
                elif hasattr(backbone, 'classifier'):
                    # VGG style
                    backbone_output_size = backbone.classifier[0].in_features
                else:
                    # Intentar forward pass
                    output = backbone(dummy_input)
                    if isinstance(output, tuple):
                        backbone_output_size = output[0].shape[-1]
                    else:
                        backbone_output_size = output.shape[-1]
            
            self.head = nn.Sequential(
                nn.Linear(backbone_output_size, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
        else:
            self.head = custom_head
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        features = self.backbone(x)
        
        # Manejar diferentes formatos de salida
        if isinstance(features, tuple):
            features = features[0]
        
        # Flatten si es necesario
        if len(features.shape) > 2:
            features = features.view(features.size(0), -1)
        
        return self.head(features)
    
    def unfreeze_backbone(self, num_layers: Optional[int] = None):
        """
        Descongelar backbone.
        
        Args:
            num_layers: Número de capas a descongelar (desde el final)
        """
        self.freeze_backbone = False
        
        if num_layers is None:
            # Descongelar todo
            for param in self.backbone.parameters():
                param.requires_grad = True
        else:
            # Descongelar últimas N capas
            layers = list(self.backbone.children())
            for layer in layers[-num_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True


class ProgressiveUnfreezing:
    """
    Estrategia de unfreezing progresivo.
    """
    
    def __init__(
        self,
        model: TransferLearningModel,
        stages: List[int]
    ):
        """
        Inicializar unfreezing progresivo.
        
        Args:
            model: Modelo de transfer learning
            stages: Lista de épocas para cada etapa
        """
        self.model = model
        self.stages = stages
        self.current_stage = 0
    
    def update(self, epoch: int):
        """
        Actualizar según época.
        
        Args:
            epoch: Época actual
        """
        if self.current_stage < len(self.stages) and epoch >= self.stages[self.current_stage]:
            # Descongelar siguiente grupo de capas
            num_layers = self.current_stage + 1
            self.model.unfreeze_backbone(num_layers=num_layers)
            self.current_stage += 1
            logger.info(f"Unfreezing {num_layers} layers at epoch {epoch}")


class DomainAdaptation:
    """
    Utilidades para domain adaptation.
    """
    
    def __init__(
        self,
        source_model: nn.Module,
        target_domain_size: int
    ):
        """
        Inicializar domain adaptation.
        
        Args:
            source_model: Modelo de dominio fuente
            target_domain_size: Tamaño del dominio objetivo
        """
        self.source_model = source_model
        self.target_domain_size = target_domain_size
    
    def create_domain_classifier(self, feature_size: int) -> nn.Module:
        """
        Crear clasificador de dominio.
        
        Args:
            feature_size: Tamaño de features
            
        Returns:
            Clasificador de dominio
        """
        return nn.Sequential(
            nn.Linear(feature_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)  # Source vs Target
        )
    
    def gradient_reversal_layer(self, x: torch.Tensor, lambda_param: float = 1.0) -> torch.Tensor:
        """
        Gradient Reversal Layer para domain adaptation.
        
        Args:
            x: Input tensor
            lambda_param: Parámetro lambda
            
        Returns:
            Tensor con gradiente revertido
        """
        class GradientReversalFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, lambda_param):
                ctx.lambda_param = lambda_param
                return x.view_as(x)
            
            @staticmethod
            def backward(ctx, grad_output):
                return grad_output.neg() * ctx.lambda_param, None
        
        return GradientReversalFunction.apply(x, lambda_param)


def load_pretrained_backbone(
    model_name: str,
    pretrained: bool = True
) -> nn.Module:
    """
    Cargar backbone pre-entrenado.
    
    Args:
        model_name: Nombre del modelo
        pretrained: Usar pesos pre-entrenados
        
    Returns:
        Modelo pre-entrenado
    """
    try:
        import torchvision.models as models
        
        model_dict = {
            'resnet18': models.resnet18,
            'resnet34': models.resnet34,
            'resnet50': models.resnet50,
            'resnet101': models.resnet101,
            'vgg16': models.vgg16,
            'vgg19': models.vgg19,
            'densenet121': models.densenet121,
            'mobilenet_v2': models.mobilenet_v2,
        }
        
        if model_name.lower() in model_dict:
            model_fn = model_dict[model_name.lower()]
            return model_fn(pretrained=pretrained)
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    except ImportError:
        logger.warning("torchvision not available")
        return None


def create_transfer_model(
    backbone_name: str,
    num_classes: int,
    freeze_backbone: bool = True,
    pretrained: bool = True
) -> TransferLearningModel:
    """
    Crear modelo de transfer learning.
    
    Args:
        backbone_name: Nombre del backbone
        num_classes: Número de clases
        freeze_backbone: Congelar backbone
        pretrained: Usar pesos pre-entrenados
        
    Returns:
        Modelo de transfer learning
    """
    backbone = load_pretrained_backbone(backbone_name, pretrained)
    if backbone is None:
        raise ValueError(f"Could not load backbone: {backbone_name}")
    
    return TransferLearningModel(backbone, num_classes, freeze_backbone)




