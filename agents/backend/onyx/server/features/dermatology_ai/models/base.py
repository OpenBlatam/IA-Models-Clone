"""
Base classes and interfaces for ML models
Defines common interfaces and abstract base classes
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all skin analysis models
    Provides common interface and utilities
    Refactored with better organization
    """
    
    def __init__(self, name: str = "BaseModel"):
        super(BaseModel, self).__init__()
        self.name = name
        self._initialized = False
        self._device: Optional[torch.device] = None
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass - must be implemented by subclasses
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary with predictions
        """
        pass
    
    def initialize_weights(self):
        """Initialize model weights - can be overridden"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_size_mb(self) -> float:
        """Get model size in MB"""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / (1024 ** 2)
    
    def to_device(self, device: Union[str, torch.device]):
        """Move model to device and store device"""
        if isinstance(device, str):
            device = torch.device(device)
        self._device = device
        return super().to(device)
    
    def get_device(self) -> Optional[torch.device]:
        """Get current device"""
        if self._device:
            return self._device
        # Try to infer from parameters
        if next(self.parameters(), None) is not None:
            return next(self.parameters()).device
        return None
    
    def export_onnx(
        self,
        output_path: str,
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
        opset_version: int = 11
    ):
        """Export model to ONNX format"""
        try:
            import torch.onnx
            
            self.eval()
            dummy_input = torch.randn(input_shape)
            
            torch.onnx.export(
                self,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            )
            logger.info(f"Model exported to ONNX: {output_path}")
        except ImportError:
            logger.error("ONNX export requires torch.onnx")
            raise
        except Exception as e:
            logger.error(f"Failed to export ONNX: {e}")
            raise


class SkinAnalysisModel(BaseModel):
    """
    Base class for skin analysis models
    Defines standard output format
    """
    
    def __init__(
        self,
        num_conditions: int = 6,
        num_metrics: int = 8,
        name: str = "SkinAnalysisModel"
    ):
        super(SkinAnalysisModel, self).__init__(name=name)
        self.num_conditions = num_conditions
        self.num_metrics = num_metrics
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass returning standardized output:
        {
            'conditions': tensor of shape (batch, num_conditions),
            'metrics': tensor of shape (batch, num_metrics)
        }
        """
        pass
    
    def predict_single(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Predict for a single image (no batch dimension)
        
        Args:
            x: Input tensor of shape (C, H, W)
            
        Returns:
            Dictionary with predictions as numpy arrays
        """
        self.eval()
        with torch.no_grad():
            # Add batch dimension
            if x.dim() == 3:
                x = x.unsqueeze(0)
            
            output = self.forward(x)
            
            # Convert to numpy and remove batch dimension
            result = {}
            for key, value in output.items():
                if isinstance(value, torch.Tensor):
                    value = value.cpu().numpy()
                    if value.ndim > 1 and value.shape[0] == 1:
                        value = value.squeeze(0)
                    result[key] = value
            
            return result


class ModelFactory:
    """
    Factory class for creating model instances
    Implements factory pattern for modular model creation
    """
    
    _model_registry: Dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str, model_class: type):
        """Register a model class"""
        cls._model_registry[name] = model_class
        logger.info(f"Registered model: {name}")
    
    @classmethod
    def create(
        cls,
        model_name: str,
        **kwargs
    ) -> BaseModel:
        """
        Create a model instance
        
        Args:
            model_name: Name of the model to create
            **kwargs: Arguments to pass to model constructor
            
        Returns:
            Model instance
        """
        if model_name not in cls._model_registry:
            raise ValueError(
                f"Model '{model_name}' not found. "
                f"Available models: {list(cls._model_registry.keys())}"
            )
        
        model_class = cls._model_registry[model_name]
        return model_class(**kwargs)
    
    @classmethod
    def list_models(cls) -> list:
        """List all registered models"""
        return list(cls._model_registry.keys())


class ModelConfig:
    """
    Configuration class for models
    Stores hyperparameters and model settings
    """
    
    def __init__(
        self,
        model_name: str,
        model_type: str,
        num_conditions: int = 6,
        num_metrics: int = 8,
        input_size: Tuple[int, int] = (224, 224),
        **kwargs
    ):
        self.model_name = model_name
        self.model_type = model_type
        self.num_conditions = num_conditions
        self.num_metrics = num_metrics
        self.input_size = input_size
        self.extra_params = kwargs
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "num_conditions": self.num_conditions,
            "num_metrics": self.num_metrics,
            "input_size": self.input_size,
            **self.extra_params
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """Create from dictionary"""
        return cls(**config_dict)

