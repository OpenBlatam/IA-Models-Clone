"""
Transfer Learning Module
=========================

Sistema profesional de transfer learning.
Incluye fine-tuning, feature extraction, y adaptación de modelos.
"""

import logging
from typing import Dict, Any, Optional, List, Callable
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None

try:
    from transformers import AutoModel, AutoConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers not available. Transfer learning limited.")

logger = logging.getLogger(__name__)


class TransferLearningManager:
    """
    Gestor profesional de transfer learning.
    
    Soporta:
    - Feature extraction (congelar backbone)
    - Fine-tuning completo
    - Fine-tuning discriminativo (diferentes LRs por capas)
    - Progressive unfreezing
    """
    
    def __init__(self, model: nn.Module):
        """
        Inicializar gestor.
        
        Args:
            model: Modelo base
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        
        self.model = model
        self.frozen_layers: List[str] = []
        logger.info("TransferLearningManager initialized")
    
    def freeze_backbone(self, exclude_layers: Optional[List[str]] = None):
        """
        Congelar backbone del modelo.
        
        Args:
            exclude_layers: Capas a excluir del congelado
        """
        exclude_layers = exclude_layers or []
        
        for name, param in self.model.named_parameters():
            if any(excluded in name for excluded in exclude_layers):
                param.requires_grad = True
            else:
                param.requires_grad = False
                if name not in self.frozen_layers:
                    self.frozen_layers.append(name)
        
        logger.info(f"Frozen {len(self.frozen_layers)} layers")
    
    def unfreeze_layers(self, layer_names: List[str]):
        """
        Descongelar capas específicas.
        
        Args:
            layer_names: Nombres de capas a descongelar
        """
        for name, param in self.model.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = True
                if name in self.frozen_layers:
                    self.frozen_layers.remove(name)
        
        logger.info(f"Unfrozen layers: {layer_names}")
    
    def progressive_unfreeze(
        self,
        stage: int,
        total_stages: int = 3
    ):
        """
        Descongelado progresivo de capas.
        
        Args:
            stage: Etapa actual (0 = todo congelado, total_stages = todo descongelado)
            total_stages: Número total de etapas
        """
        all_layers = [name for name, _ in self.model.named_parameters()]
        layers_per_stage = len(all_layers) // total_stages
        
        if stage == 0:
            # Congelar todo
            self.freeze_backbone()
        elif stage >= total_stages:
            # Descongelar todo
            for param in self.model.parameters():
                param.requires_grad = True
        else:
            # Descongelar progresivamente
            layers_to_unfreeze = all_layers[stage * layers_per_stage:(stage + 1) * layers_per_stage]
            self.unfreeze_layers(layers_to_unfreeze)
        
        logger.info(f"Progressive unfreeze: stage {stage}/{total_stages}")
    
    def create_discriminative_optimizer(
        self,
        base_lr: float = 1e-4,
        multiplier: float = 0.1
    ) -> torch.optim.Optimizer:
        """
        Crear optimizador con learning rates discriminativos.
        
        Capas más profundas tienen LR más bajo.
        
        Args:
            base_lr: Learning rate base
            multiplier: Multiplicador para capas congeladas
            
        Returns:
            Optimizador configurado
        """
        param_groups = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # LR más bajo para capas congeladas recientemente
                if name in self.frozen_layers:
                    lr = base_lr * multiplier
                else:
                    lr = base_lr
                
                param_groups.append({
                    'params': param,
                    'lr': lr,
                    'name': name
                })
        
        optimizer = torch.optim.AdamW(param_groups, lr=base_lr)
        logger.info(f"Discriminative optimizer created with {len(param_groups)} parameter groups")
        return optimizer


class PretrainedModelLoader:
    """
    Cargador de modelos pre-entrenados.
    
    Soporta modelos de Hugging Face y PyTorch.
    """
    
    @staticmethod
    def load_transformers_model(
        model_name: str,
        task: str = "base",
        num_labels: Optional[int] = None
    ) -> nn.Module:
        """
        Cargar modelo de Transformers.
        
        Args:
            model_name: Nombre del modelo
            task: Tipo de tarea ("base", "classification", "causal_lm")
            num_labels: Número de clases (para classification)
            
        Returns:
            Modelo cargado
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library is required")
        
        try:
            if task == "base":
                model = AutoModel.from_pretrained(model_name)
            elif task == "classification":
                from transformers import AutoModelForSequenceClassification
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    num_labels=num_labels or 2
                )
            elif task == "causal_lm":
                from transformers import AutoModelForCausalLM
                model = AutoModelForCausalLM.from_pretrained(model_name)
            else:
                raise ValueError(f"Unknown task: {task}")
            
            logger.info(f"Loaded {model_name} for task: {task}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    @staticmethod
    def load_torchvision_model(
        model_name: str,
        pretrained: bool = True,
        num_classes: Optional[int] = None
    ) -> nn.Module:
        """
        Cargar modelo de torchvision.
        
        Args:
            model_name: Nombre del modelo
            pretrained: Usar pesos pre-entrenados
            num_classes: Número de clases (None para mantener original)
            
        Returns:
            Modelo cargado
        """
        try:
            import torchvision.models as models
            
            if hasattr(models, model_name):
                model_fn = getattr(models, model_name)
                model = model_fn(pretrained=pretrained)
                
                if num_classes is not None:
                    # Reemplazar última capa
                    if hasattr(model, 'fc'):
                        model.fc = nn.Linear(model.fc.in_features, num_classes)
                    elif hasattr(model, 'classifier'):
                        if isinstance(model.classifier, nn.Sequential):
                            model.classifier[-1] = nn.Linear(
                                model.classifier[-1].in_features,
                                num_classes
                            )
                
                logger.info(f"Loaded torchvision model: {model_name}")
                return model
            else:
                raise ValueError(f"Unknown torchvision model: {model_name}")
        except ImportError:
            raise ImportError("torchvision is required. Install with: pip install torchvision")


class DomainAdaptation:
    """
    Adaptación de dominio para transfer learning.
    
    Incluye técnicas como adversarial training y domain alignment.
    """
    
    def __init__(self, model: nn.Module):
        """
        Inicializar adaptación de dominio.
        
        Args:
            model: Modelo a adaptar
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        
        self.model = model
        logger.info("DomainAdaptation initialized")
    
    def adversarial_training_step(
        self,
        source_data: torch.Tensor,
        target_data: torch.Tensor,
        source_labels: torch.Tensor,
        domain_classifier: nn.Module,
        optimizer: torch.optim.Optimizer,
        domain_optimizer: torch.optim.Optimizer,
        lambda_domain: float = 0.1
    ) -> Dict[str, float]:
        """
        Paso de entrenamiento adversarial.
        
        Args:
            source_data: Datos del dominio fuente
            target_data: Datos del dominio objetivo
            source_labels: Labels del dominio fuente
            domain_classifier: Clasificador de dominio
            optimizer: Optimizador del modelo principal
            domain_optimizer: Optimizador del clasificador de dominio
            lambda_domain: Peso del loss de dominio
            
        Returns:
            Dict con losses
        """
        # Forward pass
        source_features = self.model(source_data)
        target_features = self.model(target_data)
        
        # Task loss (solo en source)
        task_loss = F.cross_entropy(source_features, source_labels)
        
        # Domain loss (adversarial)
        source_domain_pred = domain_classifier(source_features)
        target_domain_pred = domain_classifier(target_features)
        
        source_domain_labels = torch.zeros(len(source_data), dtype=torch.long)
        target_domain_labels = torch.ones(len(target_data), dtype=torch.long)
        
        domain_loss = (
            F.cross_entropy(source_domain_pred, source_domain_labels) +
            F.cross_entropy(target_domain_pred, target_domain_labels)
        )
        
        # Total loss
        total_loss = task_loss - lambda_domain * domain_loss
        
        # Backward pass
        optimizer.zero_grad()
        domain_optimizer.zero_grad()
        
        total_loss.backward()
        optimizer.step()
        domain_optimizer.step()
        
        return {
            "task_loss": task_loss.item(),
            "domain_loss": domain_loss.item(),
            "total_loss": total_loss.item()
        }

