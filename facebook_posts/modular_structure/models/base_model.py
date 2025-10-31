"""
🧠 Base Model Class

Abstract base class for all machine learning models.
Provides common interface and functionality for model management.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


class BaseModel(ABC, nn.Module):
    """
    Abstract base class for all machine learning models.
    
    This class provides a common interface for:
    - Model initialization and configuration
    - Forward pass implementation
    - Model saving and loading
    - Parameter counting and model info
    - Device management
    """
    
    def __init__(self, config: Dict[str, Any], name: str = "base_model"):
        """
        Initialize the base model.
        
        Args:
            config: Model configuration dictionary
            name: Model name for identification
        """
        super().__init__()
        self.config = config
        self.name = name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_info = {}
        
        # Initialize model components
        self._build_model()
        self._setup_device()
        self._log_model_info()
    
    @abstractmethod
    def _build_model(self) -> None:
        """
        Build the model architecture.
        Must be implemented by subclasses.
        """
        pass
    
    def _setup_device(self) -> None:
        """Move model to appropriate device."""
        self.to(self.device)
        logger.info(f"Model {self.name} moved to device: {self.device}")
    
    def _log_model_info(self) -> None:
        """Log model information and statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        self.model_info = {
            "name": self.name,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "device": str(self.device),
            "config": self.config
        }
        
        logger.info(f"Model {self.name} initialized:")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
        logger.info(f"  Device: {self.device}")
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        Must be implemented by subclasses.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        return self.model_info.copy()
    
    def count_parameters(self) -> Tuple[int, int]:
        """
        Count total and trainable parameters.
        
        Returns:
            Tuple of (total_parameters, trainable_parameters)
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
    
    def save_model(self, path: Union[str, Path], save_optimizer: bool = False, 
                   optimizer_state: Optional[Dict] = None) -> None:
        """
        Save model to disk.
        
        Args:
            path: Path to save the model
            save_optimizer: Whether to save optimizer state
            optimizer_state: Optimizer state to save
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            "model_state_dict": self.state_dict(),
            "config": self.config,
            "model_info": self.model_info,
            "name": self.name
        }
        
        if save_optimizer and optimizer_state:
            save_dict["optimizer_state"] = optimizer_state
        
        torch.save(save_dict, path)
        logger.info(f"Model saved to: {path}")
    
    def load_model(self, path: Union[str, Path]) -> None:
        """
        Load model from disk.
        
        Args:
            path: Path to load the model from
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load model state
        self.load_state_dict(checkpoint["model_state_dict"])
        
        # Update config and info if available
        if "config" in checkpoint:
            self.config.update(checkpoint["config"])
        if "model_info" in checkpoint:
            self.model_info.update(checkpoint["model_info"])
        
        logger.info(f"Model loaded from: {path}")
    
    def save_config(self, path: Union[str, Path]) -> None:
        """
        Save model configuration to JSON file.
        
        Args:
            path: Path to save the configuration
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.config, f, indent=2, default=str)
        
        logger.info(f"Model config saved to: {path}")
    
    def load_config(self, path: Union[str, Path]) -> None:
        """
        Load model configuration from JSON file.
        
        Args:
            path: Path to load the configuration from
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path, 'r') as f:
            config = json.load(f)
        
        self.config.update(config)
        logger.info(f"Model config loaded from: {path}")
    
    def freeze_layers(self, layer_names: list) -> None:
        """
        Freeze specific layers by name.
        
        Args:
            layer_names: List of layer names to freeze
        """
        for name, param in self.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = False
                logger.info(f"Frozen layer: {name}")
    
    def unfreeze_layers(self, layer_names: list) -> None:
        """
        Unfreeze specific layers by name.
        
        Args:
            layer_names: List of layer names to unfreeze
        """
        for name, param in self.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = True
                logger.info(f"Unfrozen layer: {name}")
    
    def get_layer_outputs(self, x: torch.Tensor, layer_names: list) -> Dict[str, torch.Tensor]:
        """
        Get intermediate outputs from specific layers.
        
        Args:
            x: Input tensor
            layer_names: List of layer names to get outputs from
            
        Returns:
            Dictionary mapping layer names to outputs
        """
        outputs = {}
        hooks = []
        
        def hook_fn(name):
            def hook(module, input, output):
                outputs[name] = output
            return hook
        
        # Register hooks
        for name, module in self.named_modules():
            if any(layer_name in name for layer_name in layer_names):
                hooks.append(module.register_forward_hook(hook_fn(name)))
        
        # Forward pass
        with torch.no_grad():
            _ = self.forward(x)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return outputs
    
    def summary(self) -> str:
        """
        Generate a summary of the model.
        
        Returns:
            String summary of the model
        """
        total_params, trainable_params = self.count_parameters()
        
        summary = f"""
Model Summary: {self.name}
{'=' * 50}
Total Parameters: {total_params:,}
Trainable Parameters: {trainable_params:,}
Device: {self.device}
Config Keys: {list(self.config.keys())}
        """.strip()
        
        return summary






