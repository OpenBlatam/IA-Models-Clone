"""
Trainer Builder
Builder for creating trainers
"""

import torch
from typing import Optional, Dict, Any
import logging

from ..training import MobileNetTrainer, TrainingConfig
from ..helpers import DeviceHelper

logger = logging.getLogger(__name__)


class TrainerBuilder:
    """
    Builder for creating trainers
    """
    
    def __init__(self):
        """Initialize trainer builder"""
        self.model = None
        self.device = None
        self.config = None
    
    def set_model(self, model: torch.nn.Module) -> 'TrainerBuilder':
        """
        Set model
        
        Args:
            model: Model instance
            
        Returns:
            Self for chaining
        """
        self.model = model
        return self
    
    def set_device(self, device: Optional[torch.device] = None, use_gpu: bool = True) -> 'TrainerBuilder':
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
    
    def set_config(self, config: Dict[str, Any]) -> 'TrainerBuilder':
        """
        Set training configuration
        
        Args:
            config: Training configuration
            
        Returns:
            Self for chaining
        """
        self.config = config
        return self
    
    def build(self) -> MobileNetTrainer:
        """
        Build trainer
        
        Returns:
            Trainer instance
        """
        if self.model is None:
            raise ValueError("Model not set")
        
        if self.device is None:
            self.device = DeviceHelper.get_device()
        
        if self.config is None:
            training_config = TrainingConfig()
        else:
            training_config = TrainingConfig(**self.config)
        
        trainer = MobileNetTrainer(self.model, self.device, training_config)
        
        logger.info("Built trainer")
        
        return trainer



