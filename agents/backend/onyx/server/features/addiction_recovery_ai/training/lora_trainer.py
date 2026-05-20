"""
LoRA Fine-tuning for Recovery Models
Efficient fine-tuning using Low-Rank Adaptation
"""

import torch
import torch.nn as nn
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


class LoRALayer(nn.Module):
    """LoRA layer for efficient fine-tuning"""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0
    ):
        """
        Initialize LoRA layer
        
        Args:
            in_features: Input features
            out_features: Output features
            rank: LoRA rank (lower = fewer parameters)
            alpha: Scaling factor
            dropout: Dropout rate
        """
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # LoRA: x @ A^T @ B^T * scaling
        lora_output = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return lora_output * self.scaling


class LoRALinear(nn.Module):
    """LoRA-wrapped linear layer"""
    
    def __init__(
        self,
        linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0
    ):
        """
        Wrap linear layer with LoRA
        
        Args:
            linear: Original linear layer
            rank: LoRA rank
            alpha: Scaling factor
            dropout: Dropout rate
        """
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features,
            linear.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Original + LoRA
        return self.linear(x) + self.lora(x)


def apply_lora_to_model(
    model: nn.Module,
    target_modules: Optional[list] = None,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0
) -> nn.Module:
    """
    Apply LoRA to model
    
    Args:
        model: Model to apply LoRA to
        target_modules: List of module names to apply LoRA (None = all Linear)
        rank: LoRA rank
        alpha: Scaling factor
        dropout: Dropout rate
    
    Returns:
        Model with LoRA applied
    """
    if target_modules is None:
        target_modules = ["Linear"]
    
    for name, module in model.named_modules():
        if any(target in type(module).__name__ for target in target_modules):
            if isinstance(module, nn.Linear):
                # Replace with LoRA-wrapped version
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                
                if parent_name:
                    parent = model
                    for part in parent_name.split("."):
                        parent = getattr(parent, part)
                    setattr(parent, child_name, LoRALinear(module, rank, alpha, dropout))
                else:
                    setattr(model, child_name, LoRALinear(module, rank, alpha, dropout))
    
    logger.info(f"LoRA applied to model with rank={rank}, alpha={alpha}")
    return model


def get_lora_parameters(model: nn.Module) -> list:
    """Get only LoRA parameters for training"""
    lora_params = []
    for name, param in model.named_parameters():
        if "lora" in name.lower():
            lora_params.append(param)
    return lora_params


class LoRATrainer:
    """Trainer for LoRA fine-tuning"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader=None,
        device: Optional[torch.device] = None,
        rank: int = 8,
        alpha: float = 16.0
    ):
        """
        Initialize LoRA trainer
        
        Args:
            model: Model to fine-tune
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to use
            rank: LoRA rank
            alpha: LoRA alpha
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Apply LoRA
        self.model = apply_lora_to_model(model, rank=rank, alpha=alpha)
        self.model = self.model.to(self.device)
        
        # Freeze original parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze LoRA parameters
        for param in get_lora_parameters(self.model):
            param.requires_grad = True
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        logger.info("LoRATrainer initialized")
    
    def get_trainable_parameters(self):
        """Get trainable parameters (only LoRA)"""
        return [p for p in self.model.parameters() if p.requires_grad]
    
    def count_parameters(self) -> Dict[str, int]:
        """Count parameters"""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.get_trainable_parameters())
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
            "trainable_percent": (trainable / total * 100) if total > 0 else 0
        }

