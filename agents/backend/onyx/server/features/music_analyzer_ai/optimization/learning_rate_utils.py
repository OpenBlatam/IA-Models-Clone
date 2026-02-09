"""
Modular Learning Rate Utilities
Learning rate finding, scheduling, and warmup
"""

from typing import Optional, List, Tuple
import logging
import math

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")


class LearningRateFinder:
    """Find optimal learning rate using learning rate range test"""
    
    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        device: str = "cuda"
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.lr_history: List[float] = []
        self.loss_history: List[float] = []
    
    def find_lr(
        self,
        dataloader,
        start_lr: float = 1e-8,
        end_lr: float = 1.0,
        num_iter: int = 100
    ) -> Tuple[float, List[float], List[float]]:
        """
        Find optimal learning rate
        
        Args:
            dataloader: DataLoader for training
            start_lr: Starting learning rate
            end_lr: Ending learning rate
            num_iter: Number of iterations
        
        Returns:
            Tuple of (optimal_lr, lr_history, loss_history)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        
        self.lr_history.clear()
        self.loss_history.clear()
        
        # Exponential learning rate schedule
        lr_mult = (end_lr / start_lr) ** (1.0 / num_iter)
        
        # Save initial state
        initial_state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        
        best_lr = start_lr
        min_loss = float('inf')
        
        self.model.train()
        for i, batch in enumerate(dataloader):
            if i >= num_iter:
                break
            
            # Calculate learning rate
            lr = start_lr * (lr_mult ** i)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            
            # Forward pass
            batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            outputs = self.model(batch)
            loss = self.loss_fn(outputs, batch)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Record
            self.lr_history.append(lr)
            loss_value = loss.item()
            self.loss_history.append(loss_value)
            
            # Track best LR (lowest loss)
            if loss_value < min_loss:
                min_loss = loss_value
                best_lr = lr
        
        # Restore initial state
        self.model.load_state_dict(initial_state['model'])
        self.optimizer.load_state_dict(initial_state['optimizer'])
        
        logger.info(f"Optimal learning rate: {best_lr:.6f}")
        return best_lr, self.lr_history, self.loss_history


class WarmupScheduler:
    """Warmup learning rate scheduler"""
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_steps: int,
        base_lr: float
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.current_step = 0
    
    def step(self):
        """Update learning rate with warmup"""
        if self.current_step < self.warmup_steps:
            lr = self.base_lr * (self.current_step + 1) / self.warmup_steps
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        self.current_step += 1
    
    def get_lr(self) -> float:
        """Get current learning rate"""
        if self.current_step < self.warmup_steps:
            return self.base_lr * (self.current_step + 1) / self.warmup_steps
        return self.base_lr


class LearningRateScheduler:
    """Wrapper for learning rate schedulers with warmup"""
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler,
        warmup_steps: int = 0
    ):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.warmup_steps = warmup_steps
        self.warmup_scheduler = None
        
        if warmup_steps > 0:
            base_lr = optimizer.param_groups[0]['lr']
            self.warmup_scheduler = WarmupScheduler(optimizer, warmup_steps, base_lr)
    
    def step(self):
        """Step scheduler"""
        if self.warmup_scheduler and self.warmup_scheduler.current_step < self.warmup_steps:
            self.warmup_scheduler.step()
        else:
            self.scheduler.step()
    
    def get_lr(self) -> float:
        """Get current learning rate"""
        return self.optimizer.param_groups[0]['lr']



