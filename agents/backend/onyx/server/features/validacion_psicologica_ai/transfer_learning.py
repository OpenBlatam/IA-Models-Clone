"""
Advanced Transfer Learning
==========================
Transfer learning utilities and strategies
"""

from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import structlog

logger = structlog.get_logger()


class TransferLearningManager:
    """
    Manager for transfer learning strategies
    """
    
    def __init__(self):
        """Initialize transfer learning manager"""
        logger.info("TransferLearningManager initialized")
    
    def freeze_backbone(
        self,
        model: nn.Module,
        freeze_layers: Optional[List[str]] = None,
        freeze_all: bool = False
    ) -> nn.Module:
        """
        Freeze model backbone for transfer learning
        
        Args:
            model: Model to freeze
            freeze_layers: Specific layers to freeze (None = freeze all backbone)
            freeze_all: Freeze all parameters
            
        Returns:
            Model with frozen parameters
        """
        if freeze_all:
            for param in model.parameters():
                param.requires_grad = False
            logger.info("All model parameters frozen")
            return model
        
        # Freeze backbone (typically encoder)
        frozen_count = 0
        for name, param in model.named_parameters():
            if freeze_layers:
                if any(layer in name for layer in freeze_layers):
                    param.requires_grad = False
                    frozen_count += 1
            else:
                # Freeze encoder/backbone layers (not classifier head)
                if "classifier" not in name and "head" not in name:
                    param.requires_grad = False
                    frozen_count += 1
        
        logger.info(f"Frozen {frozen_count} parameters for transfer learning")
        return model
    
    def unfreeze_layers(
        self,
        model: nn.Module,
        unfreeze_layers: List[str],
        unfreeze_all: bool = False
    ) -> nn.Module:
        """
        Unfreeze specific layers
        
        Args:
            model: Model
            unfreeze_layers: Layers to unfreeze
            unfreeze_all: Unfreeze all parameters
            
        Returns:
            Model with unfrozen parameters
        """
        if unfreeze_all:
            for param in model.parameters():
                param.requires_grad = True
            logger.info("All model parameters unfrozen")
            return model
        
        unfrozen_count = 0
        for name, param in model.named_parameters():
            if any(layer in name for layer in unfreeze_layers):
                param.requires_grad = True
                unfrozen_count += 1
        
        logger.info(f"Unfrozen {unfrozen_count} parameters")
        return model
    
    def progressive_unfreezing(
        self,
        model: nn.Module,
        epoch: int,
        total_epochs: int,
        num_layers: int
    ) -> nn.Module:
        """
        Progressively unfreeze layers during training
        
        Args:
            model: Model
            epoch: Current epoch
            total_epochs: Total epochs
            num_layers: Number of layers to unfreeze
        
        Returns:
            Model with progressive unfreezing applied
        """
        # Calculate which layers to unfreeze
        unfreeze_ratio = epoch / total_epochs
        layers_to_unfreeze = int(num_layers * unfreeze_ratio)
        
        # Get layer names
        layer_names = [name for name, _ in model.named_parameters()]
        
        # Unfreeze last N layers
        for name, param in model.named_parameters():
            layer_idx = next(
                (i for i, ln in enumerate(layer_names) if ln == name),
                -1
            )
            if layer_idx >= len(layer_names) - layers_to_unfreeze:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        logger.debug(
            f"Progressive unfreezing: {layers_to_unfreeze}/{num_layers} layers unfrozen"
        )
        
        return model
    
    def create_task_head(
        self,
        base_model: nn.Module,
        num_classes: int,
        hidden_dim: Optional[int] = None
    ) -> nn.Module:
        """
        Create task-specific head for transfer learning
        
        Args:
            base_model: Base pre-trained model
            num_classes: Number of output classes
            hidden_dim: Hidden dimension (auto-detect if None)
            
        Returns:
            Model with task head
        """
        # Get output dimension from base model
        if hidden_dim is None:
            # Try to infer from model
            for name, module in base_model.named_modules():
                if hasattr(module, 'out_features'):
                    hidden_dim = module.out_features
                    break
            
            if hidden_dim is None:
                # Default for transformers
                hidden_dim = 768
        
        # Create classifier head
        classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Replace or add classifier
        if hasattr(base_model, 'classifier'):
            base_model.classifier = classifier
        elif hasattr(base_model, 'head'):
            base_model.head = classifier
        else:
            # Add as new attribute
            base_model.classifier = classifier
        
        logger.info("Task head created", num_classes=num_classes, hidden_dim=hidden_dim)
        
        return base_model


class DomainAdaptation:
    """Domain adaptation utilities"""
    
    @staticmethod
    def adversarial_training(
        model: nn.Module,
        domain_classifier: nn.Module,
        source_data: torch.Tensor,
        target_data: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Adversarial training for domain adaptation
        
        Args:
            model: Main model
            domain_classifier: Domain classifier
            source_data: Source domain data
            target_data: Target domain data
            
        Returns:
            Losses
        """
        # Feature extraction
        source_features = model(source_data)
        target_features = model(target_data)
        
        # Domain classification
        source_domain_pred = domain_classifier(source_features)
        target_domain_pred = domain_classifier(target_features)
        
        # Adversarial loss (domain classifier should fail)
        domain_loss = nn.BCELoss()(
            source_domain_pred,
            torch.zeros_like(source_domain_pred)
        ) + nn.BCELoss()(
            target_domain_pred,
            torch.ones_like(target_domain_pred)
        )
        
        return {
            "domain_loss": domain_loss,
            "adversarial_loss": -domain_loss  # Gradient reversal
        }


# Global transfer learning manager
transfer_learning_manager = TransferLearningManager()




