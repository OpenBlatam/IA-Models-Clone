"""
Strategy Pattern
Strategy pattern for interchangeable algorithms.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch
import torch.nn as nn


class OptimizationStrategy(ABC):
    """Base strategy for optimization."""
    
    @abstractmethod
    def optimize(self, model: nn.Module, **kwargs) -> nn.Module:
        """Optimize the model."""
        pass


class LoRAStrategy(OptimizationStrategy):
    """LoRA optimization strategy."""
    
    def optimize(self, model: nn.Module, r: int = 8, alpha: int = 16, **kwargs) -> nn.Module:
        """Apply LoRA optimization."""
        from ..optimization.lora_manager import LoRAManager
        
        lora_manager = LoRAManager(r=r, alpha=alpha, **kwargs)
        return lora_manager.apply_lora(model)


class QuantizationStrategy(OptimizationStrategy):
    """Quantization optimization strategy."""
    
    def optimize(
        self,
        model: nn.Module,
        quantization_type: str = "int8",
        **kwargs
    ) -> nn.Module:
        """Apply quantization optimization."""
        from ..quantization.quantization_manager import QuantizationManager
        
        quantizer = QuantizationManager(quantization_type=quantization_type)
        return quantizer.quantize_model(model, **kwargs)


class PruningStrategy(OptimizationStrategy):
    """Model pruning strategy."""
    
    def optimize(
        self,
        model: nn.Module,
        pruning_ratio: float = 0.5,
        **kwargs
    ) -> nn.Module:
        """Apply pruning optimization."""
        # Simplified pruning implementation
        # In practice, use torch.nn.utils.prune
        import torch.nn.utils.prune as prune
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=pruning_ratio)
        
        return model


class OptimizationContext:
    """Context for optimization strategies."""
    
    def __init__(self, strategy: Optional[OptimizationStrategy] = None):
        self._strategy = strategy
    
    def set_strategy(self, strategy: OptimizationStrategy):
        """Set the optimization strategy."""
        self._strategy = strategy
    
    def optimize(self, model: nn.Module, **kwargs) -> nn.Module:
        """Optimize model using current strategy."""
        if self._strategy is None:
            raise ValueError("No optimization strategy set")
        return self._strategy.optimize(model, **kwargs)


class TrainingStrategy(ABC):
    """Base strategy for training."""
    
    @abstractmethod
    def train(
        self,
        model: nn.Module,
        train_loader: Any,
        **kwargs
    ) -> Dict[str, Any]:
        """Train the model."""
        pass


class StandardTrainingStrategy(TrainingStrategy):
    """Standard training strategy."""
    
    def train(
        self,
        model: nn.Module,
        train_loader: Any,
        num_epochs: int = 1,
        **kwargs
    ) -> Dict[str, Any]:
        """Standard training."""
        from ..training.trainer import Trainer
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            **kwargs
        )
        trainer.train(num_epochs=num_epochs)
        
        return {"status": "completed"}


class DistributedTrainingStrategy(TrainingStrategy):
    """Distributed training strategy."""
    
    def train(
        self,
        model: nn.Module,
        train_loader: Any,
        num_epochs: int = 1,
        **kwargs
    ) -> Dict[str, Any]:
        """Distributed training."""
        from ..distributed.distributed_trainer import DistributedTrainer
        
        dist_trainer = DistributedTrainer(model=model, **kwargs)
        model = dist_trainer.get_model()
        
        # Use standard trainer with distributed model
        from ..training.trainer import Trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            **kwargs
        )
        trainer.train(num_epochs=num_epochs)
        
        dist_trainer.cleanup()
        return {"status": "completed", "distributed": True}


class TrainingContext:
    """Context for training strategies."""
    
    def __init__(self, strategy: Optional[TrainingStrategy] = None):
        self._strategy = strategy
    
    def set_strategy(self, strategy: TrainingStrategy):
        """Set the training strategy."""
        self._strategy = strategy
    
    def train(self, model: nn.Module, train_loader: Any, **kwargs) -> Dict[str, Any]:
        """Train model using current strategy."""
        if self._strategy is None:
            raise ValueError("No training strategy set")
        return self._strategy.train(model, train_loader, **kwargs)



