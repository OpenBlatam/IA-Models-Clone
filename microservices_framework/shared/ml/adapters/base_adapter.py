"""
Base Adapters
Adapter pattern for integrating different ML frameworks and libraries.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import torch
import torch.nn as nn


class ModelAdapter(ABC):
    """Base adapter for model integration."""
    
    @abstractmethod
    def adapt(self, model: Any, **kwargs) -> nn.Module:
        """Adapt a model to PyTorch format."""
        pass
    
    @abstractmethod
    def can_adapt(self, model: Any) -> bool:
        """Check if adapter can handle the model."""
        pass


class HuggingFaceAdapter(ModelAdapter):
    """Adapter for HuggingFace models."""
    
    def adapt(self, model: Any, **kwargs) -> nn.Module:
        """Adapt HuggingFace model."""
        # HuggingFace models are already PyTorch modules
        if isinstance(model, nn.Module):
            return model
        raise ValueError("Model is not a PyTorch module")
    
    def can_adapt(self, model: Any) -> bool:
        """Check if model is from HuggingFace."""
        return isinstance(model, nn.Module) and hasattr(model, "config")


class ONNXAdapter(ModelAdapter):
    """Adapter for ONNX models."""
    
    def adapt(self, model_path: str, **kwargs) -> nn.Module:
        """Adapt ONNX model to PyTorch."""
        try:
            import onnxruntime as ort
            from .onnx_wrapper import ONNXWrapper
            
            session = ort.InferenceSession(model_path)
            return ONNXWrapper(session)
        except ImportError:
            raise ImportError("onnxruntime is required for ONNX models")
    
    def can_adapt(self, model: Any) -> bool:
        """Check if model is ONNX."""
        if isinstance(model, str):
            return model.endswith('.onnx')
        return False


class TensorFlowAdapter(ModelAdapter):
    """Adapter for TensorFlow models."""
    
    def adapt(self, model: Any, **kwargs) -> nn.Module:
        """Adapt TensorFlow model to PyTorch."""
        # This would require tensorflow-to-pytorch conversion
        # For now, raise NotImplementedError
        raise NotImplementedError("TensorFlow to PyTorch conversion not implemented")
    
    def can_adapt(self, model: Any) -> bool:
        """Check if model is TensorFlow."""
        try:
            import tensorflow as tf
            return isinstance(model, tf.keras.Model)
        except ImportError:
            return False


class AdapterRegistry:
    """Registry for model adapters."""
    
    def __init__(self):
        self._adapters: list = [
            HuggingFaceAdapter(),
            ONNXAdapter(),
            TensorFlowAdapter(),
        ]
    
    def register(self, adapter: ModelAdapter):
        """Register a new adapter."""
        self._adapters.append(adapter)
    
    def get_adapter(self, model: Any) -> Optional[ModelAdapter]:
        """Get appropriate adapter for model."""
        for adapter in self._adapters:
            if adapter.can_adapt(model):
                return adapter
        return None
    
    def adapt(self, model: Any, **kwargs) -> nn.Module:
        """Adapt model using appropriate adapter."""
        adapter = self.get_adapter(model)
        if adapter is None:
            raise ValueError(f"No adapter found for model type: {type(model)}")
        return adapter.adapt(model, **kwargs)



