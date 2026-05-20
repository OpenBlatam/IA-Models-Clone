"""
Model Serialization
Advanced model serialization and deserialization
"""

import torch
import torch.nn as nn
import pickle
import json
from pathlib import Path
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class ModelSerializer:
    """
    Advanced model serialization
    """
    
    @staticmethod
    def save_model(
        model: nn.Module,
        path: str,
        metadata: Optional[Dict[str, Any]] = None,
        format: str = "pytorch"
    ):
        """
        Save model with metadata
        
        Args:
            model: Model to save
            path: Save path
            metadata: Additional metadata
            format: Save format (pytorch, onnx, torchscript)
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "pytorch":
            # Save PyTorch model
            save_dict = {
                "model_state_dict": model.state_dict(),
                "model_class": model.__class__.__name__,
                "metadata": metadata or {}
            }
            torch.save(save_dict, path)
            logger.info(f"Model saved to {path}")
        
        elif format == "onnx":
            # Save ONNX (requires dummy input)
            logger.warning("ONNX export requires input shape - use export_model instead")
        
        elif format == "torchscript":
            # Save TorchScript (requires dummy input)
            logger.warning("TorchScript export requires input shape - use export_model instead")
    
    @staticmethod
    def load_model(
        model_class: type,
        path: str,
        device: Optional[torch.device] = None
    ) -> tuple[nn.Module, Dict[str, Any]]:
        """
        Load model with metadata
        
        Args:
            model_class: Model class
            path: Model path
            device: Device to load on
            
        Returns:
            (model, metadata)
        """
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        checkpoint = torch.load(path, map_location=device)
        
        # Create model instance
        model = model_class()
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        model.eval()
        
        metadata = checkpoint.get("metadata", {})
        
        logger.info(f"Model loaded from {path}")
        return model, metadata
    
    @staticmethod
    def save_config(config: Dict[str, Any], path: str):
        """
        Save configuration
        
        Args:
            config: Configuration dictionary
            path: Save path
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Config saved to {path}")
    
    @staticmethod
    def load_config(path: str) -> Dict[str, Any]:
        """
        Load configuration
        
        Args:
            path: Config path
            
        Returns:
            Configuration dictionary
        """
        with open(path, 'r') as f:
            config = json.load(f)
        
        logger.info(f"Config loaded from {path}")
        return config


def save_model(model: nn.Module, path: str, **kwargs):
    """Save model"""
    return ModelSerializer.save_model(model, path, **kwargs)


def load_model(model_class: type, path: str, **kwargs) -> tuple[nn.Module, Dict[str, Any]]:
    """Load model"""
    return ModelSerializer.load_model(model_class, path, **kwargs)








