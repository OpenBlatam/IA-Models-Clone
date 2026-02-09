"""
Base Model Class
===============

Base model class with proper initialization, weight management,
and utilities following best practices.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Union
from pathlib import Path
import logging

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)


class BaseModel(nn.Module):
    """
    Base model class with proper initialization and utilities.
    
    Provides:
    - Weight initialization methods
    - Parameter counting
    - Model saving/loading
    - Device management
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize base model.
        
        Args:
            device: Target device (cuda/cpu). If None, auto-detects.
        """
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._initialized = False
    
    def _initialize_weights(self, init_type: str = "xavier_uniform") -> None:
        """
        Initialize model weights using best practices.
        
        Args:
            init_type: Type of initialization
                - xavier_uniform: Xavier uniform initialization
                - xavier_normal: Xavier normal initialization
                - kaiming_uniform: Kaiming uniform (He initialization)
                - kaiming_normal: Kaiming normal (He initialization)
                - orthogonal: Orthogonal initialization
                - normal: Normal distribution initialization
        """
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                if init_type == "xavier_uniform":
                    nn.init.xavier_uniform_(module.weight)
                elif init_type == "xavier_normal":
                    nn.init.xavier_normal_(module.weight)
                elif init_type == "kaiming_uniform":
                    nn.init.kaiming_uniform_(
                        module.weight, 
                        mode='fan_in', 
                        nonlinearity='relu'
                    )
                elif init_type == "kaiming_normal":
                    nn.init.kaiming_normal_(
                        module.weight, 
                        mode='fan_in', 
                        nonlinearity='relu'
                    )
                elif init_type == "orthogonal":
                    nn.init.orthogonal_(module.weight)
                elif init_type == "normal":
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                else:
                    logger.warning(f"Unknown init type: {init_type}, using default")
                
                # Initialize bias
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
            
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
        logger.debug(f"Weights initialized using {init_type}")
    
    def count_parameters(self) -> Dict[str, int]:
        """
        Count trainable and total parameters.
        
        Returns:
            Dictionary with trainable, total, and non_trainable parameter counts
        """
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return {
            "trainable": trainable,
            "total": total,
            "non_trainable": total - trainable,
            "trainable_percentage": 100.0 * trainable / total if total > 0 else 0.0
        }
    
    def save(self, path: Union[str, Path], include_optimizer: bool = False, 
             optimizer: Optional[torch.optim.Optimizer] = None) -> None:
        """
        Save model to disk.
        
        Args:
            path: Path to save model
            include_optimizer: Whether to save optimizer state
            optimizer: Optimizer to save (if include_optimizer is True)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_class': self.__class__.__name__,
        }
        
        if include_optimizer and optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        torch.save(checkpoint, path)
        logger.info(f"✅ Model saved to {path}")
    
    def load(self, path: Union[str, Path], strict: bool = True) -> None:
        """
        Load model from disk.
        
        Args:
            path: Path to model checkpoint
            strict: Whether to strictly enforce that keys match
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {path}")
        
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        logger.info(f"✅ Model loaded from {path}")
    
    def freeze(self) -> None:
        """Freeze all model parameters."""
        for param in self.parameters():
            param.requires_grad = False
        logger.info("Model parameters frozen")
    
    def unfreeze(self) -> None:
        """Unfreeze all model parameters."""
        for param in self.parameters():
            param.requires_grad = True
        logger.info("Model parameters unfrozen")
    
    def get_device(self) -> torch.device:
        """Get model device."""
        return self.device
    
    def to_device(self, device: torch.device) -> None:
        """Move model to specified device."""
        self.device = device
        self.to(device)
        logger.info(f"Model moved to {device}")



