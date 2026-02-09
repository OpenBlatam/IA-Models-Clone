"""
Model Utilities
================
Utility functions for model initialization and management
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import structlog
import math

logger = structlog.get_logger()


def initialize_weights(module: nn.Module, init_type: str = "xavier_uniform") -> None:
    """
    Initialize model weights
    
    Args:
        module: Module to initialize
        init_type: Initialization type (xavier_uniform, xavier_normal, kaiming_uniform, kaiming_normal, normal, zeros)
    """
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        if init_type == "xavier_uniform":
            nn.init.xavier_uniform_(module.weight)
        elif init_type == "xavier_normal":
            nn.init.xavier_normal_(module.weight)
        elif init_type == "kaiming_uniform":
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
        elif init_type == "kaiming_normal":
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
        elif init_type == "normal":
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif init_type == "zeros":
            nn.init.zeros_(module.weight)
        else:
            raise ValueError(f"Unknown initialization type: {init_type}")
        
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    
    elif isinstance(module, (nn.LSTM, nn.GRU)):
        for name, param in module.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)
                # Set forget gate bias to 1
                n = param.size(0)
                param.data[(n // 4):(n // 2)].fill_(1)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count model parameters
    
    Args:
        model: Model
        trainable_only: Count only trainable parameters
        
    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def get_model_summary(model: nn.Module, input_shape: tuple) -> Dict[str, Any]:
    """
    Get model summary
    
    Args:
        model: Model
        input_shape: Input shape
        
    Returns:
        Model summary
    """
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    
    # Estimate model size
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size_mb = (param_size + buffer_size) / (1024 ** 2)
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": total_params - trainable_params,
        "model_size_mb": model_size_mb,
        "input_shape": input_shape
    }


def freeze_bn_layers(model: nn.Module) -> nn.Module:
    """
    Freeze batch normalization layers
    
    Args:
        model: Model
        
    Returns:
        Model with frozen BN layers
    """
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.eval()
            for param in module.parameters():
                param.requires_grad = False
    
    logger.info("Batch normalization layers frozen")
    return model


def apply_dropout(model: nn.Module, dropout_rate: float) -> nn.Module:
    """
    Apply dropout to model
    
    Args:
        model: Model
        dropout_rate: Dropout rate
        
    Returns:
        Model with dropout applied
    """
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = dropout_rate
    
    logger.info(f"Dropout rate set to {dropout_rate}")
    return model


class ModelEMA:
    """
    Exponential Moving Average for model weights
    Improves model stability
    """
    
    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        device: Optional[torch.device] = None
    ):
        """
        Initialize EMA
        
        Args:
            model: Model
            decay: EMA decay factor
            device: Device
        """
        self.model = model
        self.decay = decay
        self.device = device or next(model.parameters()).device
        
        # Create shadow model
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self) -> None:
        """Update EMA weights"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name] +
                    (1 - self.decay) * param.data
                )
    
    def apply_shadow(self) -> None:
        """Apply shadow weights to model"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
    
    def restore(self) -> None:
        """Restore original weights"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}




