"""
Model Serialization

Utilities for saving and loading models.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ModelSerializer:
    """Serialize and deserialize models."""
    
    def __init__(self, base_dir: str = "./models"):
        """
        Initialize model serializer.
        
        Args:
            base_dir: Base directory for models
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def save_model(
        self,
        model: nn.Module,
        filename: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save complete model.
        
        Args:
            model: Model to save
            filename: Filename
            metadata: Optional metadata
            
        Returns:
            Path to saved model
        """
        model_path = self.base_dir / filename
        
        save_dict = {
            'model': model,
            'metadata': metadata or {}
        }
        
        torch.save(save_dict, model_path)
        logger.info(f"Saved model: {model_path}")
        
        return str(model_path)
    
    def load_model(
        self,
        filename: str,
        device: Optional[torch.device] = None
    ) -> tuple:
        """
        Load complete model.
        
        Args:
            filename: Filename
            device: Device to load on
            
        Returns:
            (model, metadata)
        """
        model_path = self.base_dir / filename
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        save_dict = torch.load(model_path, map_location=device)
        
        model = save_dict['model']
        metadata = save_dict.get('metadata', {})
        
        logger.info(f"Loaded model: {model_path}")
        
        return model, metadata
    
    def save_state_dict(
        self,
        model: nn.Module,
        filename: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save model state dict.
        
        Args:
            model: Model to save
            filename: Filename
            metadata: Optional metadata
            
        Returns:
            Path to saved state dict
        """
        model_path = self.base_dir / filename
        
        save_dict = {
            'state_dict': model.state_dict(),
            'metadata': metadata or {}
        }
        
        torch.save(save_dict, model_path)
        logger.info(f"Saved state dict: {model_path}")
        
        return str(model_path)
    
    def load_state_dict(
        self,
        model: nn.Module,
        filename: str,
        strict: bool = True
    ) -> Dict[str, Any]:
        """
        Load model state dict.
        
        Args:
            model: Model to load into
            filename: Filename
            strict: Strict loading
            
        Returns:
            Metadata dictionary
        """
        model_path = self.base_dir / filename
        
        if not model_path.exists():
            raise FileNotFoundError(f"State dict not found: {model_path}")
        
        save_dict = torch.load(model_path)
        
        model.load_state_dict(save_dict['state_dict'], strict=strict)
        metadata = save_dict.get('metadata', {})
        
        logger.info(f"Loaded state dict: {model_path}")
        
        return metadata


def save_model(
    model: nn.Module,
    filename: str,
    base_dir: str = "./models",
    **kwargs
) -> str:
    """Convenience function to save model."""
    serializer = ModelSerializer(base_dir)
    return serializer.save_model(model, filename, **kwargs)


def load_model(
    filename: str,
    base_dir: str = "./models",
    **kwargs
) -> tuple:
    """Convenience function to load model."""
    serializer = ModelSerializer(base_dir)
    return serializer.load_model(filename, **kwargs)


def save_state_dict(
    model: nn.Module,
    filename: str,
    base_dir: str = "./models",
    **kwargs
) -> str:
    """Convenience function to save state dict."""
    serializer = ModelSerializer(base_dir)
    return serializer.save_state_dict(model, filename, **kwargs)


def load_state_dict(
    model: nn.Module,
    filename: str,
    base_dir: str = "./models",
    **kwargs
) -> Dict[str, Any]:
    """Convenience function to load state dict."""
    serializer = ModelSerializer(base_dir)
    return serializer.load_state_dict(model, filename, **kwargs)



