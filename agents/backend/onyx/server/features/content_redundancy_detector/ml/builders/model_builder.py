"""
Model Builder
Builder for creating models
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import logging

from ..models.mobilenet.factory import MobileNetFactory
from ..models.mobilenet.config import MobileNetConfig
from ..helpers import DeviceHelper

logger = logging.getLogger(__name__)


class ModelBuilder:
    """
    Builder for creating models
    """
    
    def __init__(self):
        """Initialize model builder"""
        self.config = None
        self.device = None
        self.pretrained = False
        self.pretrained_path = None
    
    def set_config(self, config: Dict[str, Any]) -> 'ModelBuilder':
        """
        Set model configuration
        
        Args:
            config: Model configuration
            
        Returns:
            Self for chaining
        """
        self.config = config
        return self
    
    def set_device(self, device: Optional[torch.device] = None, use_gpu: bool = True) -> 'ModelBuilder':
        """
        Set device
        
        Args:
            device: Device (None = auto-detect)
            use_gpu: Use GPU if available
            
        Returns:
            Self for chaining
        """
        if device is None:
            self.device = DeviceHelper.get_device(use_gpu=use_gpu)
        else:
            self.device = device
        return self
    
    def set_pretrained(self, path: Optional[str] = None) -> 'ModelBuilder':
        """
        Set pretrained model
        
        Args:
            path: Path to pretrained weights (None = use default)
            
        Returns:
            Self for chaining
        """
        self.pretrained = True
        self.pretrained_path = path
        return self
    
    def build(self) -> nn.Module:
        """
        Build model
        
        Returns:
            Model instance
        """
        if self.config is None:
            raise ValueError("Model configuration not set")
        
        # Create model config
        model_config = MobileNetConfig.from_dict(self.config)
        
        # Create model
        model = MobileNetFactory.create_model(model_config, device=self.device)
        
        # Load pretrained weights if specified
        if self.pretrained:
            if self.pretrained_path:
                checkpoint = torch.load(self.pretrained_path, map_location=self.device)
                model.load_state_dict(checkpoint)
            # Otherwise use default pretrained
        
        logger.info(f"Built model: {model_config.variant}")
        
        return model



