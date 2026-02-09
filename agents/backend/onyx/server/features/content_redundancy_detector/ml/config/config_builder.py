"""
Configuration Builder
Build configurations programmatically
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfigBuilder:
    """Builder for model configuration"""
    variant: str = "mobilenet_v2"
    num_classes: int = 1000
    width_mult: float = 1.0
    dropout: float = 0.2
    
    def build(self) -> Dict[str, Any]:
        """Build model configuration"""
        return {
            "variant": self.variant,
            "num_classes": self.num_classes,
            "width_mult": self.width_mult,
            "dropout": self.dropout,
        }


@dataclass
class TrainingConfigBuilder:
    """Builder for training configuration"""
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 50
    optimizer: str = "adam"
    scheduler: str = "cosine"
    weight_decay: float = 0.0001
    
    def build(self) -> Dict[str, Any]:
        """Build training configuration"""
        return {
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "weight_decay": self.weight_decay,
        }


class ConfigBuilder:
    """
    Builder for complete configurations
    """
    
    def __init__(self):
        """Initialize config builder"""
        self.model_builder = ModelConfigBuilder()
        self.training_builder = TrainingConfigBuilder()
        self.data_config = {}
        self.device_config = {}
    
    def set_model(
        self,
        variant: str = "mobilenet_v2",
        num_classes: int = 1000,
        **kwargs
    ) -> 'ConfigBuilder':
        """
        Set model configuration
        
        Args:
            variant: Model variant
            num_classes: Number of classes
            **kwargs: Additional model parameters
            
        Returns:
            Self for chaining
        """
        self.model_builder.variant = variant
        self.model_builder.num_classes = num_classes
        for key, value in kwargs.items():
            if hasattr(self.model_builder, key):
                setattr(self.model_builder, key, value)
        return self
    
    def set_training(
        self,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        num_epochs: int = 50,
        **kwargs
    ) -> 'ConfigBuilder':
        """
        Set training configuration
        
        Args:
            learning_rate: Learning rate
            batch_size: Batch size
            num_epochs: Number of epochs
            **kwargs: Additional training parameters
            
        Returns:
            Self for chaining
        """
        self.training_builder.learning_rate = learning_rate
        self.training_builder.batch_size = batch_size
        self.training_builder.num_epochs = num_epochs
        for key, value in kwargs.items():
            if hasattr(self.training_builder, key):
                setattr(self.training_builder, key, value)
        return self
    
    def set_data(
        self,
        image_size: int = 224,
        num_workers: int = 4,
        **kwargs
    ) -> 'ConfigBuilder':
        """
        Set data configuration
        
        Args:
            image_size: Image size
            num_workers: Number of workers
            **kwargs: Additional data parameters
            
        Returns:
            Self for chaining
        """
        self.data_config = {
            "image_size": image_size,
            "num_workers": num_workers,
            **kwargs
        }
        return self
    
    def set_device(
        self,
        use_gpu: bool = True,
        use_mixed_precision: bool = False,
        **kwargs
    ) -> 'ConfigBuilder':
        """
        Set device configuration
        
        Args:
            use_gpu: Use GPU
            use_mixed_precision: Use mixed precision
            **kwargs: Additional device parameters
            
        Returns:
            Self for chaining
        """
        self.device_config = {
            "use_gpu": use_gpu,
            "use_mixed_precision": use_mixed_precision,
            **kwargs
        }
        return self
    
    def build(self) -> Dict[str, Any]:
        """
        Build complete configuration
        
        Returns:
            Complete configuration dictionary
        """
        return {
            "model": self.model_builder.build(),
            "training": self.training_builder.build(),
            "data": self.data_config,
            "device": self.device_config,
        }



