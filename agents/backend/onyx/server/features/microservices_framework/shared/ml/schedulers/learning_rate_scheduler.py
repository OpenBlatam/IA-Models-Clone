"""
Learning Rate Schedulers
Advanced learning rate scheduling strategies.
"""

import torch
from torch.optim.lr_scheduler import (
    LambdaLR,
    StepLR,
    ExponentialLR,
    CosineAnnealingLR,
    ReduceLROnPlateau,
    CosineAnnealingWarmRestarts,
)
from typing import Optional, Callable
import logging

logger = logging.getLogger(__name__)


class LearningRateSchedulerFactory:
    """
    Factory for creating learning rate schedulers.
    """
    
    @staticmethod
    def create_scheduler(
        optimizer: torch.optim.Optimizer,
        scheduler_type: str = "cosine",
        num_training_steps: Optional[int] = None,
        num_warmup_steps: int = 0,
        **kwargs
    ) -> torch.optim.lr_scheduler._LRScheduler:
        """
        Create learning rate scheduler.
        
        Args:
            optimizer: Optimizer
            scheduler_type: Type of scheduler
            num_training_steps: Total training steps
            num_warmup_steps: Number of warmup steps
            **kwargs: Additional scheduler parameters
            
        Returns:
            Learning rate scheduler
        """
        if scheduler_type == "linear":
            return LearningRateSchedulerFactory._create_linear_scheduler(
                optimizer, num_training_steps, num_warmup_steps
            )
        elif scheduler_type == "cosine":
            return LearningRateSchedulerFactory._create_cosine_scheduler(
                optimizer, num_training_steps, num_warmup_steps, **kwargs
            )
        elif scheduler_type == "polynomial":
            return LearningRateSchedulerFactory._create_polynomial_scheduler(
                optimizer, num_training_steps, num_warmup_steps, **kwargs
            )
        elif scheduler_type == "constant":
            return LearningRateSchedulerFactory._create_constant_scheduler(
                optimizer, num_warmup_steps
            )
        elif scheduler_type == "step":
            return StepLR(
                optimizer,
                step_size=kwargs.get("step_size", 30),
                gamma=kwargs.get("gamma", 0.1),
            )
        elif scheduler_type == "exponential":
            return ExponentialLR(
                optimizer,
                gamma=kwargs.get("gamma", 0.95),
            )
        elif scheduler_type == "cosine_restarts":
            return CosineAnnealingWarmRestarts(
                optimizer,
                T_0=kwargs.get("T_0", 10),
                T_mult=kwargs.get("T_mult", 2),
            )
        elif scheduler_type == "reduce_on_plateau":
            return ReduceLROnPlateau(
                optimizer,
                mode=kwargs.get("mode", "min"),
                factor=kwargs.get("factor", 0.1),
                patience=kwargs.get("patience", 10),
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    @staticmethod
    def _create_linear_scheduler(
        optimizer: torch.optim.Optimizer,
        num_training_steps: int,
        num_warmup_steps: int,
    ) -> LambdaLR:
        """Create linear scheduler with warmup."""
        def lr_lambda(current_step: int) -> float:
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0,
                float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
            )
        
        return LambdaLR(optimizer, lr_lambda)
    
    @staticmethod
    def _create_cosine_scheduler(
        optimizer: torch.optim.Optimizer,
        num_training_steps: int,
        num_warmup_steps: int,
        num_cycles: float = 0.5,
    ) -> LambdaLR:
        """Create cosine scheduler with warmup."""
        def lr_lambda(current_step: int) -> float:
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 2.0 * 3.14159 * num_cycles))))
        
        return LambdaLR(optimizer, lr_lambda)
    
    @staticmethod
    def _create_polynomial_scheduler(
        optimizer: torch.optim.Optimizer,
        num_training_steps: int,
        num_warmup_steps: int,
        power: float = 1.0,
    ) -> LambdaLR:
        """Create polynomial scheduler with warmup."""
        def lr_lambda(current_step: int) -> float:
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(0.0, (1.0 - progress) ** power)
        
        return LambdaLR(optimizer, lr_lambda)
    
    @staticmethod
    def _create_constant_scheduler(
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
    ) -> LambdaLR:
        """Create constant scheduler with warmup."""
        def lr_lambda(current_step: int) -> float:
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return 1.0
        
        return LambdaLR(optimizer, lr_lambda)


class EarlyStopping:
    """
    Early stopping callback.
    """
    
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = "min",
        restore_best_weights: bool = True,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
    
    def __call__(self, score: float, model: torch.nn.Module) -> bool:
        """
        Check if training should stop early.
        
        Args:
            score: Current validation score
            model: Model to save weights from
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        elif self._is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights and self.best_weights:
                    model.load_state_dict(self.best_weights)
                    logger.info("Restored best model weights")
        
        return self.early_stop
    
    def _is_better(self, current: float, best: float) -> bool:
        """Check if current score is better than best."""
        if self.mode == "min":
            return current < best - self.min_delta
        else:
            return current > best + self.min_delta



