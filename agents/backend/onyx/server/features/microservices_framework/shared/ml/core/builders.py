"""
Builder Classes
Builder pattern for constructing complex ML pipelines.
"""

from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
from .factories import ComponentFactory, OptimizerFactory, LossFunctionFactory


class TrainingPipelineBuilder:
    """Builder for training pipelines."""
    
    def __init__(self):
        self.factory = ComponentFactory()
        self.config: Dict[str, Any] = {}
        self.model: Optional[nn.Module] = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.criterion: Optional[nn.Module] = None
        self.scheduler = None
        self.trainer = None
    
    def with_model(self, model: nn.Module) -> "TrainingPipelineBuilder":
        """Set the model."""
        self.model = model
        return self
    
    def with_data_loaders(
        self,
        train_loader,
        val_loader = None
    ) -> "TrainingPipelineBuilder":
        """Set data loaders."""
        self.train_loader = train_loader
        self.val_loader = val_loader
        return self
    
    def with_optimizer(
        self,
        optimizer_type: str = "adamw",
        learning_rate: float = 5e-5,
        **kwargs
    ) -> "TrainingPipelineBuilder":
        """Set optimizer."""
        if self.model is None:
            raise ValueError("Model must be set before optimizer")
        
        self.optimizer = OptimizerFactory.create(
            optimizer_type,
            self.model,
            learning_rate,
            **kwargs
        )
        return self
    
    def with_loss_function(
        self,
        loss_type: str = "cross_entropy",
        **kwargs
    ) -> "TrainingPipelineBuilder":
        """Set loss function."""
        self.criterion = LossFunctionFactory.create(loss_type, **kwargs)
        return self
    
    def with_scheduler(
        self,
        scheduler_type: str = "cosine",
        num_training_steps: Optional[int] = None,
        **kwargs
    ) -> "TrainingPipelineBuilder":
        """Set learning rate scheduler."""
        if self.optimizer is None:
            raise ValueError("Optimizer must be set before scheduler")
        
        from ..schedulers.learning_rate_scheduler import LearningRateSchedulerFactory
        
        if num_training_steps is None and self.train_loader:
            num_training_steps = len(self.train_loader) * kwargs.get("num_epochs", 1)
        
        self.scheduler = LearningRateSchedulerFactory.create_scheduler(
            self.optimizer,
            scheduler_type,
            num_training_steps=num_training_steps or 1000,
            **kwargs
        )
        return self
    
    def with_trainer_config(
        self,
        use_amp: bool = True,
        max_grad_norm: float = 1.0,
        gradient_accumulation_steps: int = 1,
        **kwargs
    ) -> "TrainingPipelineBuilder":
        """Set trainer configuration."""
        self.config.update({
            "use_amp": use_amp,
            "max_grad_norm": max_grad_norm,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            **kwargs
        })
        return self
    
    def build(self):
        """Build the training pipeline."""
        if self.model is None:
            raise ValueError("Model is required")
        if self.train_loader is None:
            raise ValueError("Train loader is required")
        
        from ..training.trainer import Trainer
        
        self.trainer = Trainer(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            optimizer=self.optimizer,
            criterion=self.criterion,
            **self.config
        )
        
        return self.trainer


class InferencePipelineBuilder:
    """Builder for inference pipelines."""
    
    def __init__(self):
        self.factory = ComponentFactory()
        self.model: Optional[nn.Module] = None
        self.tokenizer = None
        self.engine = None
        self.config: Dict[str, Any] = {}
    
    def with_model(self, model: nn.Module) -> "InferencePipelineBuilder":
        """Set the model."""
        self.model = model
        return self
    
    def with_tokenizer(self, tokenizer) -> "InferencePipelineBuilder":
        """Set the tokenizer."""
        self.tokenizer = tokenizer
        return self
    
    def with_config(
        self,
        use_amp: bool = True,
        max_batch_size: int = 32,
        compile_model: bool = False,
        **kwargs
    ) -> "InferencePipelineBuilder":
        """Set inference configuration."""
        self.config.update({
            "use_amp": use_amp,
            "max_batch_size": max_batch_size,
            "compile_model": compile_model,
            **kwargs
        })
        return self
    
    def build(self):
        """Build the inference pipeline."""
        if self.model is None:
            raise ValueError("Model is required")
        if self.tokenizer is None:
            raise ValueError("Tokenizer is required")
        
        from ..inference.inference_engine import InferenceEngine
        
        self.engine = InferenceEngine(
            model=self.model,
            tokenizer=self.tokenizer,
            **self.config
        )
        
        return self.engine


class ModelOptimizationBuilder:
    """Builder for model optimization pipelines."""
    
    def __init__(self):
        self.model: Optional[nn.Module] = None
        self.optimizations: List[Dict[str, Any]] = []
    
    def with_model(self, model: nn.Module) -> "ModelOptimizationBuilder":
        """Set the model."""
        self.model = model
        return self
    
    def add_quantization(
        self,
        quantization_type: str = "int8",
        **kwargs
    ) -> "ModelOptimizationBuilder":
        """Add quantization optimization."""
        self.optimizations.append({
            "type": "quantization",
            "quantization_type": quantization_type,
            **kwargs
        })
        return self
    
    def add_lora(
        self,
        r: int = 8,
        alpha: int = 16,
        **kwargs
    ) -> "ModelOptimizationBuilder":
        """Add LoRA optimization."""
        self.optimizations.append({
            "type": "lora",
            "r": r,
            "alpha": alpha,
            **kwargs
        })
        return self
    
    def build(self) -> nn.Module:
        """Build optimized model."""
        if self.model is None:
            raise ValueError("Model is required")
        
        optimized_model = self.model
        
        for opt in self.optimizations:
            if opt["type"] == "quantization":
                from ..quantization.quantization_manager import QuantizationManager
                quantizer = QuantizationManager(
                    quantization_type=opt["quantization_type"]
                )
                optimized_model = quantizer.quantize_model(
                    optimized_model,
                    opt.get("quantization_config")
                )
            elif opt["type"] == "lora":
                from ..optimization.lora_manager import LoRAManager
                lora_manager = LoRAManager(
                    r=opt["r"],
                    alpha=opt["alpha"],
                    **{k: v for k, v in opt.items() if k not in ["type", "r", "alpha"]}
                )
                optimized_model = lora_manager.apply_lora(optimized_model)
        
        return optimized_model



