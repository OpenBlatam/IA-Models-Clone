"""
Model Manager Module

Main model manager class for lifecycle management.
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
    logger.warning("PyTorch not available")

from ..factories.unified_factory import get_factory
from ...utils.device_manager import get_device_manager
from ...utils.initialization import initialize_weights
from ...inference.pipelines import StandardInferencePipeline


class ModelManager:
    """
    Manages model lifecycle using modular components.
    Handles creation, loading, saving, and inference.
    
    Args:
        device: Device to run models on.
        model_dir: Directory for model storage.
    """
    
    def __init__(
        self,
        device: Optional[str] = None,
        model_dir: str = "./models"
    ):
        self.factory = get_factory()
        self.device_manager = get_device_manager(device)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.models: Dict[str, nn.Module] = {}
        self.inference_pipelines: Dict[str, StandardInferencePipeline] = {}
    
    def list_models(self) -> List[str]:
        """
        List all registered models.
        
        Returns:
            List of model names.
        """
        return list(self.models.keys())
    
    def list_pipelines(self) -> List[str]:
        """
        List all inference pipelines.
        
        Returns:
            List of pipeline names.
        """
        return list(self.inference_pipelines.keys())
    
    def get_model(self, model_name: str) -> Optional[nn.Module]:
        """
        Get registered model.
        
        Args:
            model_name: Name of the model.
        
        Returns:
            Model instance or None.
        """
        return self.models.get(model_name)



