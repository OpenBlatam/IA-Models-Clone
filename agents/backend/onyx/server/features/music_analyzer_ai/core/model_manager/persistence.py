"""
Model Persistence Module

Model loading and saving functionality.
"""

from typing import Dict, Any, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .manager import ModelManager


class ModelPersistenceMixin:
    """Model persistence mixin."""
    
    def load_model(
        self: ModelManager,
        model_name: str,
        checkpoint_path: str,
        model_type: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> nn.Module:
        """
        Load model from checkpoint.
        
        Args:
            model_name: Name for the model.
            checkpoint_path: Path to checkpoint.
            model_type: Model type (required if not in checkpoint).
            config: Model configuration (required if not in checkpoint).
        
        Returns:
            Loaded model instance.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device_manager.get_device())
        
        # Get model type and config from checkpoint or parameters
        if "model_type" in checkpoint:
            model_type = checkpoint["model_type"]
        if "config" in checkpoint:
            config = checkpoint["config"]
        
        if model_type is None:
            raise ValueError("model_type must be provided or in checkpoint")
        
        # Create model
        model = self.factory.create_model(
            model_type=model_type,
            config=config or {},
            device=self.device_manager.get_device()
        )
        
        # Load weights
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        
        # Register model
        self.models[model_name] = model
        
        logger.info(f"Loaded and registered model: {model_name}")
        return model
    
    def save_model(
        self: ModelManager,
        model_name: str,
        checkpoint_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Save model to checkpoint.
        
        Args:
            model_name: Name of registered model.
            checkpoint_path: Path to save checkpoint.
            metadata: Additional metadata to save.
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        if checkpoint_path is None:
            checkpoint_path = self.model_dir / f"{model_name}.pt"
        else:
            checkpoint_path = Path(checkpoint_path)
        
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "metadata": metadata or {}
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved model {model_name} to {checkpoint_path}")



