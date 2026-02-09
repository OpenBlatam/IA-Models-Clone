"""
Factory Classes
Factory pattern for creating ML components.
"""

from typing import Dict, Any, Optional, Type, List
import torch
import torch.nn as nn
from .interfaces import (
    IModelLoader,
    IInferenceEngine,
    ITrainer,
    IEvaluator,
    IOptimizer,
    IQuantizer,
    IProfiler,
)


class ModelLoaderFactory:
    """Factory for creating model loaders."""
    
    _loaders: Dict[str, Type[IModelLoader]] = {}
    
    @classmethod
    def register(cls, name: str, loader_class: Type[IModelLoader]):
        """Register a model loader class."""
        cls._loaders[name] = loader_class
    
    @classmethod
    def create(cls, name: str, **kwargs) -> IModelLoader:
        """Create a model loader instance."""
        if name not in cls._loaders:
            raise ValueError(f"Unknown model loader: {name}")
        return cls._loaders[name](**kwargs)
    
    @classmethod
    def list_loaders(cls) -> List[str]:
        """List available model loaders."""
        return list(cls._loaders.keys())


class OptimizerFactory:
    """Factory for creating optimizers."""
    
    @staticmethod
    def create(
        optimizer_type: str,
        model: nn.Module,
        learning_rate: float = 1e-3,
        **kwargs
    ) -> torch.optim.Optimizer:
        """
        Create an optimizer.
        
        Args:
            optimizer_type: Type of optimizer (adam, sgd, adamw, etc.)
            model: Model to optimize
            learning_rate: Learning rate
            **kwargs: Additional optimizer parameters
            
        Returns:
            Optimizer instance
        """
        optimizer_type = optimizer_type.lower()
        
        if optimizer_type == "adam":
            return torch.optim.Adam(
                model.parameters(),
                lr=learning_rate,
                **kwargs
            )
        elif optimizer_type == "adamw":
            return torch.optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=kwargs.get("weight_decay", 0.01),
                **{k: v for k, v in kwargs.items() if k != "weight_decay"}
            )
        elif optimizer_type == "sgd":
            return torch.optim.SGD(
                model.parameters(),
                lr=learning_rate,
                momentum=kwargs.get("momentum", 0.9),
                **{k: v for k, v in kwargs.items() if k != "momentum"}
            )
        elif optimizer_type == "rmsprop":
            return torch.optim.RMSprop(
                model.parameters(),
                lr=learning_rate,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")


class LossFunctionFactory:
    """Factory for creating loss functions."""
    
    @staticmethod
    def create(
        loss_type: str,
        **kwargs
    ) -> nn.Module:
        """
        Create a loss function.
        
        Args:
            loss_type: Type of loss (cross_entropy, mse, etc.)
            **kwargs: Additional loss parameters
            
        Returns:
            Loss function
        """
        loss_type = loss_type.lower()
        
        if loss_type == "cross_entropy":
            return nn.CrossEntropyLoss(
                ignore_index=kwargs.get("ignore_index", -100),
                label_smoothing=kwargs.get("label_smoothing", 0.0),
            )
        elif loss_type == "mse":
            return nn.MSELoss(**kwargs)
        elif loss_type == "mae" or loss_type == "l1":
            return nn.L1Loss(**kwargs)
        elif loss_type == "bce":
            return nn.BCELoss(**kwargs)
        elif loss_type == "bce_with_logits":
            return nn.BCEWithLogitsLoss(**kwargs)
        elif loss_type == "focal":
            # Focal loss implementation
            from .losses import FocalLoss
            return FocalLoss(
                alpha=kwargs.get("alpha", 1.0),
                gamma=kwargs.get("gamma", 2.0),
            )
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")


class DeviceFactory:
    """Factory for device management."""
    
    @staticmethod
    def get_device(device: Optional[str] = None) -> torch.device:
        """
        Get appropriate device.
        
        Args:
            device: Device string (cuda, cpu, mps) or None for auto-detect
            
        Returns:
            Device instance
        """
        if device is None:
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        else:
            return torch.device(device)
    
    @staticmethod
    def get_optimal_device() -> torch.device:
        """Get the optimal available device."""
        return DeviceFactory.get_device()


class ComponentFactory:
    """Main factory for creating ML components."""
    
    def __init__(self):
        self.model_loader_factory = ModelLoaderFactory()
        self.optimizer_factory = OptimizerFactory()
        self.loss_factory = LossFunctionFactory()
        self.device_factory = DeviceFactory()
    
    def create_inference_engine(
        self,
        engine_type: str = "default",
        **kwargs
    ) -> IInferenceEngine:
        """Create an inference engine."""
        # Implementation would use registered engines
        from ..inference.inference_engine import InferenceEngine
        return InferenceEngine(**kwargs)
    
    def create_trainer(
        self,
        trainer_type: str = "default",
        **kwargs
    ) -> ITrainer:
        """Create a trainer."""
        from ..training.trainer import Trainer
        return Trainer(**kwargs)
    
    def create_evaluator(
        self,
        evaluator_type: str = "default",
        **kwargs
    ) -> IEvaluator:
        """Create an evaluator."""
        from ..evaluation.evaluator import Evaluator
        return Evaluator(**kwargs)
    
    def create_quantizer(
        self,
        quantizer_type: str = "int8",
        **kwargs
    ) -> IQuantizer:
        """Create a quantizer."""
        from ..quantization.quantization_manager import QuantizationManager
        return QuantizationManager(quantization_type=quantizer_type, **kwargs)
    
    def create_profiler(
        self,
        profiler_type: str = "default",
        **kwargs
    ) -> IProfiler:
        """Create a profiler."""
        from ..monitoring.profiler import ModelProfiler
        return ModelProfiler(**kwargs)

