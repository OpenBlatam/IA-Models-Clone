"""
Model Serializer
Serialize and deserialize models
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
    logger.warning("PyTorch not available")


class ModelSerializer:
    """Serialize and deserialize models"""
    
    @staticmethod
    def save_model(
        model: nn.Module,
        path: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Save model to file
        
        Args:
            model: Model to save
            path: Path to save
            metadata: Optional metadata
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            "model_state_dict": model.state_dict(),
            "model_class": model.__class__.__name__,
            "metadata": metadata or {}
        }
        
        torch.save(save_dict, path)
        logger.info(f"Model saved to {path}")
    
    @staticmethod
    def load_model(
        path: str,
        model_class: Optional[type] = None,
        map_location: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load model from file
        
        Args:
            path: Path to model file
            model_class: Optional model class
            map_location: Device to load on
        
        Returns:
            Dictionary with model state and metadata
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        checkpoint = torch.load(path, map_location=map_location)
        logger.info(f"Model loaded from {path}")
        
        return checkpoint
    
    @staticmethod
    def save_onnx(
        model: nn.Module,
        path: str,
        input_shape: tuple,
        input_names: Optional[list] = None,
        output_names: Optional[list] = None
    ):
        """
        Export model to ONNX
        
        Args:
            model: Model to export
            path: Output path
            input_shape: Input tensor shape
            input_names: Optional input names
            output_names: Optional output names
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        
        try:
            import torch.onnx
        except ImportError:
            raise ImportError("ONNX export not available")
        
        model.eval()
        dummy_input = torch.randn(input_shape)
        
        torch.onnx.export(
            model,
            dummy_input,
            path,
            input_names=input_names or ["input"],
            output_names=output_names or ["output"],
            dynamic_axes=None
        )
        
        logger.info(f"Model exported to ONNX: {path}")



