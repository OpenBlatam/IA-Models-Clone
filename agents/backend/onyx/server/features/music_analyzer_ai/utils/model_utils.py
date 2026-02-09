"""
Model Utilities
Helper functions for model management and utilities
"""

from typing import Dict, Any, Optional, List
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ModelUtils:
    """
    Utility functions for model management
    """
    
    @staticmethod
    def count_parameters(model: nn.Module, trainable_only: bool = False) -> Dict[str, int]:
        """Count model parameters"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        
        total_params = 0
        trainable_params = 0
        
        for param in model.parameters():
            num_params = param.numel()
            total_params += num_params
            if param.requires_grad:
                trainable_params += num_params
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params if trainable_only else total_params,
            "non_trainable_parameters": total_params - trainable_params
        }
    
    @staticmethod
    def get_model_size_mb(model: nn.Module) -> float:
        """Get model size in MB"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_all_mb = (param_size + buffer_size) / 1024**2
        return size_all_mb
    
    @staticmethod
    def save_model_checkpoint(
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        epoch: int,
        loss: float,
        filepath: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Save model checkpoint with metadata"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "loss": loss,
            "metadata": metadata or {}
        }
        
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, filepath)
        logger.info(f"Saved checkpoint to {filepath}")
    
    @staticmethod
    def load_model_checkpoint(
        model: nn.Module,
        filepath: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "cpu"
    ) -> Dict[str, Any]:
        """Load model checkpoint"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        
        checkpoint = torch.load(filepath, map_location=device)
        
        model.load_state_dict(checkpoint["model_state_dict"])
        
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        logger.info(f"Loaded checkpoint from {filepath}")
        return checkpoint
    
    @staticmethod
    def initialize_weights(model: nn.Module, method: str = "xavier_uniform"):
        """Initialize model weights"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        
        for module in model.modules():
            if isinstance(module, nn.Linear):
                if method == "xavier_uniform":
                    nn.init.xavier_uniform_(module.weight)
                elif method == "xavier_normal":
                    nn.init.xavier_normal_(module.weight)
                elif method == "kaiming_uniform":
                    nn.init.kaiming_uniform_(module.weight)
                elif method == "kaiming_normal":
                    nn.init.kaiming_normal_(module.weight)
                else:
                    raise ValueError(f"Unknown initialization method: {method}")
                
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    @staticmethod
    def get_model_summary(model: nn.Module) -> Dict[str, Any]:
        """Get model summary"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        
        param_info = ModelUtils.count_parameters(model)
        model_size = ModelUtils.get_model_size_mb(model)
        
        # Count layers
        num_layers = sum(1 for _ in model.modules() if isinstance(_, (nn.Linear, nn.Conv1d, nn.Conv2d)))
        
        return {
            "total_parameters": param_info["total_parameters"],
            "trainable_parameters": param_info["trainable_parameters"],
            "model_size_mb": model_size,
            "num_layers": num_layers,
            "architecture": str(type(model).__name__)
        }

