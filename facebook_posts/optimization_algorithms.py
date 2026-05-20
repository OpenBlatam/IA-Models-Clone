from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple, Callable
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

#!/usr/bin/env python3
"""
Advanced Optimization Algorithms
Comprehensive implementation of optimization algorithms with PyTorch best practices.
"""

import math
import logging
from dataclasses import dataclass
from enum import Enum

import asyncio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import (
    StepLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ReduceLROnPlateau,
    ExponentialLR,
    OneCycleLR,
)


class OptimizerType(Enum):
    """Types of optimization algorithms."""
    SGD = "sgd"
    ADAM = "adam"
    ADAMW = "adamw"
    ADAGRAD = "adagrad"
    RMSprop = "rmsprop"
    ADADELTA = "adadelta"
    LION = "lion"
    CUSTOM = "custom"


class SchedulerType(Enum):
    """Types of learning rate schedulers."""
    STEP = "step"
    COSINE = "cosine"
    COSINE_WARM_RESTARTS = "cosine_warm_restarts"
    REDUCE_ON_PLATEAU = "reduce_on_plateau"
    EXPONENTIAL = "exponential"
    ONE_CYCLE = "one_cycle"
    CUSTOM = "custom"


@dataclass
class OptimizerConfig:
    """Configuration for optimization algorithms."""
    optimizer_type: OptimizerType = OptimizerType.ADAMW
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    momentum: float = 0.9
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    amsgrad: bool = False
    nesterov: bool = False
    centered: bool = False
    alpha: float = 0.99
    rho: float = 0.9
    lr_decay: float = 0.0


@dataclass
class SchedulerConfig:
    """Configuration for learning rate schedulers."""
    scheduler_type: SchedulerType = SchedulerType.COSINE
    step_size: int = 30
    gamma: float = 0.1
    T_max: int = 100
    T_0: int = 10
    T_mult: int = 2
    eta_min: float = 0.0
    mode: str = "min"
    factor: float = 0.1
    patience: int = 10
    threshold: float = 1e-4
    cooldown: int = 0
    min_lr: float = 0.0
    max_lr: float = 1e-2
    total_steps: int = 1000
    pct_start: float = 0.3
    anneal_strategy: str = "cos"


class AdvancedAdamW(optim.AdamW):
    """Advanced AdamW optimizer with additional features."""
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01, amsgrad=False, maximize=False,
                 capturable=False, differentiable=False, fused=None):
        super().__init__(params, lr, betas, eps, weight_decay, amsgrad,
                        maximize, capturable, differentiable, fused)
        
        # Additional tracking
        self.step_count = 0
        self.lr_history = []
        self.grad_norm_history = []
    
    def step(self, closure=None) -> Any:
        """Step with additional tracking."""
        # Record learning rate
        for group in self.param_groups:
            self.lr_history.append(group['lr'])
        
        # Compute gradient norm before step
        grad_norm = self._compute_grad_norm()
        self.grad_norm_history.append(grad_norm)
        
        # Perform optimization step
        loss = super().step(closure)
        
        # Update step count
        self.step_count += 1
        
        return loss
    
    def _compute_grad_norm(self) -> float:
        """Compute gradient norm across all parameters."""
        total_norm = 0.0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
        return total_norm ** 0.5


class LionOptimizer(optim.Optimizer):
    """Lion optimizer implementation."""
    
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0, eps=1e-8):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    def step(self, closure=None) -> Any:
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Lion does not support sparse gradients')
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                
                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']
                state['step'] += 1
                
                # Weight decay
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])
                
                # Update exp_avg
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Update parameters
                update = exp_avg.sign()
                p.data.add_(update, alpha=-group['lr'])
        
        return loss


class CustomOptimizer(optim.Optimizer):
    """Custom optimizer with advanced features."""
    
    def __init__(self, params, lr=1e-3, momentum=0.9, weight_decay=0.0,
                 dampening=0, nesterov=False, maximize=False) -> Any:
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= dampening < 1.0:
            raise ValueError(f"Invalid dampening value: {dampening}")
        
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                       weight_decay=weight_decay, nesterov=nesterov,
                       maximize=maximize)
        super().__init__(params, defaults)
    
    def step(self, closure=None) -> Any:
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            maximize = group['maximize']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                d_p = p.grad.data
                if maximize:
                    d_p = -d_p
                
                if weight_decay != 0:
                    d_p = d_p.add(p.data, alpha=weight_decay)
                
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf
                
                p.data.add_(d_p, alpha=-group['lr'])
        
        return loss


class AdvancedScheduler:
    """Advanced learning rate scheduler with multiple strategies."""
    
    def __init__(self, optimizer: optim.Optimizer, config: SchedulerConfig):
        self.optimizer = optimizer
        self.config = config
        self.scheduler = self._create_scheduler()
        self.lr_history = []
    
    def _create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Create scheduler based on configuration."""
        if self.config.scheduler_type == SchedulerType.STEP:
            return StepLR(self.optimizer, step_size=self.config.step_size, 
                         gamma=self.config.gamma)
        elif self.config.scheduler_type == SchedulerType.COSINE:
            return CosineAnnealingLR(self.optimizer, T_max=self.config.T_max, 
                                   eta_min=self.config.eta_min)
        elif self.config.scheduler_type == SchedulerType.COSINE_WARM_RESTARTS:
            return CosineAnnealingWarmRestarts(self.optimizer, T_0=self.config.T_0,
                                             T_mult=self.config.T_mult, 
                                             eta_min=self.config.eta_min)
        elif self.config.scheduler_type == SchedulerType.REDUCE_ON_PLATEAU:
            return ReduceLROnPlateau(self.optimizer, mode=self.config.mode,
                                   factor=self.config.factor, 
                                   patience=self.config.patience,
                                   threshold=self.config.threshold,
                                   cooldown=self.config.cooldown,
                                   min_lr=self.config.min_lr)
        elif self.config.scheduler_type == SchedulerType.EXPONENTIAL:
            return ExponentialLR(self.optimizer, gamma=self.config.gamma)
        elif self.config.scheduler_type == SchedulerType.ONE_CYCLE:
            return OneCycleLR(self.optimizer, max_lr=self.config.max_lr,
                            total_steps=self.config.total_steps,
                            pct_start=self.config.pct_start,
                            anneal_strategy=self.config.anneal_strategy)
        else:
            raise ValueError(f"Unknown scheduler type: {self.config.scheduler_type}")
    
    def step(self, metrics: Optional[float] = None):
        """Step the scheduler."""
        if isinstance(self.scheduler, ReduceLROnPlateau):
            if metrics is None:
                raise ValueError("Metrics required for ReduceLROnPlateau scheduler")
            self.scheduler.step(metrics)
        else:
            self.scheduler.step()
        
        # Record learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        self.lr_history.append(current_lr)
    
    def get_last_lr(self) -> List[float]:
        """Get the last learning rate."""
        return self.scheduler.get_last_lr()


class OptimizerFactory:
    """Factory for creating different optimizers."""
    
    @staticmethod
    def create_optimizer(model: nn.Module, config: OptimizerConfig) -> optim.Optimizer:
        """Create optimizer based on configuration."""
        if config.optimizer_type == OptimizerType.SGD:
            return optim.SGD(
                model.parameters(),
                lr=config.learning_rate,
                momentum=config.momentum,
                weight_decay=config.weight_decay,
                nesterov=config.nesterov
            )
        elif config.optimizer_type == OptimizerType.ADAM:
            return optim.Adam(
                model.parameters(),
                lr=config.learning_rate,
                betas=(config.beta1, config.beta2),
                eps=config.eps,
                weight_decay=config.weight_decay,
                amsgrad=config.amsgrad
            )
        elif config.optimizer_type == OptimizerType.ADAMW:
            # Prefer fused AdamW on CUDA; fallback to foreach variant
            fused_supported = torch.cuda.is_available()
            try:
                if fused_supported:
                    return optim.AdamW(
                        model.parameters(),
                        lr=config.learning_rate,
                        betas=(config.beta1, config.beta2),
                        eps=config.eps,
                        weight_decay=config.weight_decay,
                        fused=True  # type: ignore[call-arg]
                    )
            except Exception:
                pass
            return optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                betas=(config.beta1, config.beta2),
                eps=config.eps,
                weight_decay=config.weight_decay,
                foreach=True
            )
        elif config.optimizer_type == OptimizerType.ADAGRAD:
            return optim.Adagrad(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                lr_decay=config.lr_decay
            )
        elif config.optimizer_type == OptimizerType.RMSprop:
            return optim.RMSprop(
                model.parameters(),
                lr=config.learning_rate,
                alpha=config.alpha,
                eps=config.eps,
                weight_decay=config.weight_decay,
                momentum=config.momentum,
                centered=config.centered
            )
        elif config.optimizer_type == OptimizerType.ADADELTA:
            return optim.Adadelta(
                model.parameters(),
                lr=config.learning_rate,
                rho=config.rho,
                eps=config.eps,
                weight_decay=config.weight_decay
            )
        elif config.optimizer_type == OptimizerType.LION:
            return LionOptimizer(
                model.parameters(),
                lr=config.learning_rate,
                betas=(config.beta1, config.beta2),
                weight_decay=config.weight_decay
            )
        elif config.optimizer_type == OptimizerType.CUSTOM:
            return CustomOptimizer(
                model.parameters(),
                lr=config.learning_rate,
                momentum=config.momentum,
                weight_decay=config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer type: {config.optimizer_type}")


class OptimizationManager:
    """Manager for optimization with advanced features."""
    
    def __init__(self, model: nn.Module, optimizer_config: OptimizerConfig,
                 scheduler_config: SchedulerConfig):
        self.model = model
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        
        # Create optimizer
        self.optimizer = OptimizerFactory.create_optimizer(model, optimizer_config)
        
        # Create scheduler
        self.scheduler = AdvancedScheduler(self.optimizer, scheduler_config)
        
        # Training state
        self.step_count = 0
        self.loss_history = []
        self.grad_norm_history = []
    
    def step(self, loss: torch.Tensor, metrics: Optional[float] = None):
        """Perform optimization step."""
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Backward pass
        loss.backward()
        
        # Compute gradient norm
        grad_norm = self._compute_grad_norm()
        self.grad_norm_history.append(grad_norm)
        
        # Optimizer step
        self.optimizer.step()
        
        # Scheduler step
        self.scheduler.step(metrics)
        
        # Update state
        self.step_count += 1
        self.loss_history.append(loss.item())
    
    def _compute_grad_norm(self) -> float:
        """Compute gradient norm across all parameters."""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        return {
            'step_count': self.step_count,
            'current_lr': self.optimizer.param_groups[0]['lr'],
            'loss_history': self.loss_history,
            'grad_norm_history': self.grad_norm_history,
            'lr_history': self.scheduler.lr_history
        }


class OptimizationAnalyzer:
    """Analyze optimization behavior and performance."""
    
    def __init__(self) -> Any:
        self.analysis_results = {}
    
    def analyze_optimizer(self, model: nn.Module, 
                         optimizer_configs: List[OptimizerConfig],
                         num_steps: int = 100) -> Dict[str, Any]:
        """Analyze different optimizers."""
        results = {}
        
        for config in optimizer_configs:
            print(f"Testing {config.optimizer_type.value} optimizer...")
            
            # Create fresh model copy
            model_copy = type(model)()
            
            # Create optimizer
            optimizer = OptimizerFactory.create_optimizer(model_copy, config)
            
            # Test optimization
            test_results = self._test_optimization(model_copy, optimizer, num_steps)
            
            results[config.optimizer_type.value] = {
                'config': config,
                'results': test_results
            }
        
        return results
    
    def _test_optimization(self, model: nn.Module, optimizer: optim.Optimizer,
                          num_steps: int) -> Dict[str, Any]:
        """Test optimization performance."""
        loss_history = []
        grad_norm_history = []
        lr_history = []
        
        for step in range(num_steps):
            # Create dummy loss
            dummy_input = torch.randn(10, 100)
            dummy_target = torch.randn(10, 10)
            output = model(dummy_input)
            loss = F.mse_loss(output, dummy_target)
            
            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            
            # Record gradient norm
            grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5
            
            optimizer.step()
            
            # Record statistics
            loss_history.append(loss.item())
            grad_norm_history.append(grad_norm)
            lr_history.append(optimizer.param_groups[0]['lr'])
        
        return {
            'final_loss': loss_history[-1],
            'loss_history': loss_history,
            'grad_norm_history': grad_norm_history,
            'lr_history': lr_history,
            'convergence_rate': self._compute_convergence_rate(loss_history)
        }
    
    def _compute_convergence_rate(self, loss_history: List[float]) -> float:
        """Compute convergence rate."""
        if len(loss_history) < 2:
            return 0.0
        
        # Compute average rate of loss decrease
        rates = []
        for i in range(1, len(loss_history)):
            if loss_history[i-1] > 0:
                rate = (loss_history[i-1] - loss_history[i]) / loss_history[i-1]
                rates.append(rate)
        
        return np.mean(rates) if rates else 0.0


def demonstrate_optimization_algorithms():
    """Demonstrate different optimization algorithms."""
    print("Optimization Algorithms Demonstration")
    print("=" * 50)
    
    # Create a simple model for testing
    class SimpleModel(nn.Module):
        def __init__(self) -> Any:
            super().__init__()
            self.linear1 = nn.Linear(100, 50)
            self.linear2 = nn.Linear(50, 10)
        
        def forward(self, x) -> Any:
            x = F.relu(self.linear1(x))
            x = self.linear2(x)
            return x
    
    model = SimpleModel()
    
    # Test different optimizers
    optimizer_configs = [
        OptimizerConfig(optimizer_type=OptimizerType.ADAMW, learning_rate=1e-3),
        OptimizerConfig(optimizer_type=OptimizerType.ADAM, learning_rate=1e-3),
        OptimizerConfig(optimizer_type=OptimizerType.SGD, learning_rate=1e-2, momentum=0.9),
        OptimizerConfig(optimizer_type=OptimizerType.RMSprop, learning_rate=1e-3),
        OptimizerConfig(optimizer_type=OptimizerType.LION, learning_rate=1e-4)
    ]
    
    # Test different schedulers
    scheduler_configs = [
        SchedulerConfig(scheduler_type=SchedulerType.COSINE, T_max=100),
        SchedulerConfig(scheduler_type=SchedulerType.STEP, step_size=30, gamma=0.1),
        SchedulerConfig(scheduler_type=SchedulerType.REDUCE_ON_PLATEAU, patience=10),
        SchedulerConfig(scheduler_type=SchedulerType.ONE_CYCLE, max_lr=1e-2, total_steps=100)
    ]
    
    # Analyze optimizers
    analyzer = OptimizationAnalyzer()
    optimizer_results = analyzer.analyze_optimizer(model, optimizer_configs, num_steps=50)
    
    # Print results
    for optimizer_name, result in optimizer_results.items():
        print(f"\n{optimizer_name.upper()}:")
        test_results = result['results']
        print(f"  Final Loss: {test_results['final_loss']:.6f}")
        print(f"  Convergence Rate: {test_results['convergence_rate']:.6f}")
        print(f"  Final LR: {test_results['lr_history'][-1]:.6f}")
    
    return optimizer_results


if __name__ == "__main__":
    # Demonstrate optimization algorithms
    results = demonstrate_optimization_algorithms()
    print("\nOptimization algorithms demonstration completed!") 