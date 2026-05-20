"""
Unified Factory System
Integrates with registry and modular components
"""

from typing import Dict, Any, Optional, Type
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")

from ..core.registry import get_registry
from ..config.model_config import ModelConfig
from ..training.components import (
    create_optimizer,
    create_scheduler,
    ClassificationLoss,
    RegressionLoss,
    MultiTaskLoss
)
from ..training.components.callbacks import (
    EarlyStoppingCallback,
    CheckpointCallback,
    MetricsCallback
)
from ..training.loops import StandardTrainingLoop
from ..inference.pipelines import StandardInferencePipeline


class UnifiedFactory:
    """
    Unified factory that integrates all modular components
    Uses registry for component discovery
    """
    
    def __init__(self):
        self.registry = get_registry()
    
    def create_model(
        self,
        model_type: str,
        config: Optional[Dict[str, Any]] = None,
        device: str = "cuda"
    ) -> nn.Module:
        """
        Create model using registry
        
        Args:
            model_type: Model type name (must be registered)
            config: Model configuration
            device: Target device
        
        Returns:
            Model instance
        """
        config = config or {}
        
        # Get model class from registry
        model_class = self.registry.get_model(model_type)
        if model_class is None:
            raise ValueError(f"Model type '{model_type}' not found in registry. "
                           f"Available: {self.registry.list_models()}")
        
        # Create model
        model = model_class(**config)
        model = model.to(device)
        
        logger.info(f"Created {model_type} model on {device}")
        return model
    
    def create_loss(
        self,
        loss_type: str,
        config: Optional[Dict[str, Any]] = None
    ) -> nn.Module:
        """
        Create loss function
        
        Args:
            loss_type: Loss type name
            config: Loss configuration
        
        Returns:
            Loss function
        """
        config = config or {}
        
        # Try registry first
        loss_class = self.registry.get_loss(loss_type)
        if loss_class:
            return loss_class(**config)
        
        # Fallback to built-in losses
        if loss_type == "classification":
            return ClassificationLoss(**config)
        elif loss_type == "regression":
            return RegressionLoss(**config)
        elif loss_type == "multi_task":
            return MultiTaskLoss(**config)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def create_optimizer(
        self,
        optimizer_type: str,
        parameters,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Create optimizer
        
        Args:
            optimizer_type: Optimizer type
            parameters: Model parameters
            config: Optimizer configuration
        
        Returns:
            Optimizer instance
        """
        config = config or {}
        learning_rate = config.pop("learning_rate", 1e-4)
        
        # Try registry first
        optimizer_factory = self.registry.get_optimizer_factory(optimizer_type)
        if optimizer_factory:
            return optimizer_factory(parameters, learning_rate, **config)
        
        # Fallback to standard factory
        return create_optimizer(optimizer_type, parameters, learning_rate, **config)
    
    def create_scheduler(
        self,
        scheduler_type: str,
        optimizer,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Create learning rate scheduler
        
        Args:
            scheduler_type: Scheduler type
            optimizer: Optimizer instance
            config: Scheduler configuration
        
        Returns:
            Scheduler instance
        """
        config = config or {}
        
        # Try registry first
        scheduler_factory = self.registry.get_scheduler_factory(scheduler_type)
        if scheduler_factory:
            return scheduler_factory(optimizer, **config)
        
        # Fallback to standard factory
        return create_scheduler(scheduler_type, optimizer, **config)
    
    def create_training_loop(
        self,
        model: nn.Module,
        optimizer,
        loss_fn: nn.Module,
        config: Optional[Dict[str, Any]] = None
    ) -> StandardTrainingLoop:
        """
        Create training loop
        
        Args:
            model: Model instance
            optimizer: Optimizer instance
            loss_fn: Loss function
            config: Training loop configuration
        
        Returns:
            Training loop instance
        """
        config = config or {}
        
        return StandardTrainingLoop(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=config.get("device", "cuda"),
            use_mixed_precision=config.get("use_mixed_precision", True),
            gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
            max_grad_norm=config.get("max_grad_norm", 1.0)
        )
    
    def create_inference_pipeline(
        self,
        model: nn.Module,
        config: Optional[Dict[str, Any]] = None
    ) -> StandardInferencePipeline:
        """
        Create inference pipeline
        
        Args:
            model: Model instance
            config: Pipeline configuration
        
        Returns:
            Inference pipeline instance
        """
        config = config or {}
        
        return StandardInferencePipeline(
            model=model,
            preprocess_fn=config.get("preprocess_fn"),
            postprocess_fn=config.get("postprocess_fn"),
            device=config.get("device", "cuda"),
            use_mixed_precision=config.get("use_mixed_precision", True)
        )
    
    def create_callbacks(
        self,
        callback_configs: Dict[str, Dict[str, Any]]
    ) -> list:
        """
        Create training callbacks
        
        Args:
            callback_configs: Dictionary of callback configurations
        
        Returns:
            List of callback instances
        """
        callbacks = []
        
        for callback_type, config in callback_configs.items():
            # Try registry first
            callback_class = self.registry.get_callback(callback_type)
            if callback_class:
                callbacks.append(callback_class(**config))
                continue
            
            # Fallback to built-in callbacks
            if callback_type == "early_stopping":
                callbacks.append(EarlyStoppingCallback(**config))
            elif callback_type == "checkpoint":
                callbacks.append(CheckpointCallback(**config))
            elif callback_type == "metrics":
                callbacks.append(MetricsCallback(**config))
            else:
                logger.warning(f"Unknown callback type: {callback_type}")
        
        return callbacks
    
    def create_from_config(
        self,
        config: ModelConfig
    ) -> Dict[str, Any]:
        """
        Create complete training setup from configuration
        
        Args:
            config: ModelConfig instance
        
        Returns:
            Dictionary with model, optimizer, scheduler, loss, training_loop, callbacks
        """
        # Create model
        model = self.create_model(
            model_type=config.architecture.model_type,
            config={
                "input_dim": config.architecture.input_dim,
                "embed_dim": config.architecture.embed_dim,
                "num_heads": config.architecture.num_heads,
                "num_layers": config.architecture.num_layers,
                "ff_dim": config.architecture.ff_dim,
                "dropout": config.architecture.dropout
            },
            device=config.device
        )
        
        # Create optimizer
        optimizer = self.create_optimizer(
            optimizer_type=config.training.optimizer,
            parameters=model.parameters(),
            config={
                "learning_rate": config.training.learning_rate,
                "weight_decay": config.training.weight_decay
            }
        )
        
        # Create scheduler
        scheduler = self.create_scheduler(
            scheduler_type=config.training.scheduler_type,
            optimizer=optimizer,
            config={
                "T_max": config.training.epochs,
                "warmup_steps": config.training.warmup_steps
            }
        )
        
        # Create loss (simplified - would need task-specific config)
        loss_fn = self.create_loss("classification", {})
        
        # Create training loop
        training_loop = self.create_training_loop(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            config={
                "device": config.device,
                "use_mixed_precision": config.training.use_mixed_precision,
                "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
                "max_grad_norm": config.training.max_grad_norm
            }
        )
        
        # Create callbacks
        callbacks = self.create_callbacks({
            "early_stopping": {
                "patience": config.training.early_stopping_patience
            },
            "checkpoint": {
                "checkpoint_dir": config.training.checkpoint_dir,
                "save_best": config.training.save_best_only
            },
            "metrics": {}
        })
        
        return {
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "loss": loss_fn,
            "training_loop": training_loop,
            "callbacks": callbacks
        }


# Global factory instance
_factory: Optional[UnifiedFactory] = None


def get_factory() -> UnifiedFactory:
    """Get global unified factory"""
    global _factory
    if _factory is None:
        _factory = UnifiedFactory()
    return _factory



