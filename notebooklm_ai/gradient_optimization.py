from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import asyncio
import time
import gc
import logging
import json
import os
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import threading
from contextlib import contextmanager
import warnings
from collections import defaultdict, deque
import math
    from prometheus_client import Counter, Histogram, Gauge
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
Advanced Gradient Optimization System
===================================

Comprehensive gradient optimization system with:
- Advanced gradient clipping strategies
- NaN/Inf value detection and handling
- Gradient accumulation and scaling
- Loss scaling and stability
- Training monitoring and recovery
- Adaptive learning rate adjustment

Features: Multiple clipping methods, automatic recovery,
gradient monitoring, and production-ready stability features.
"""



# Performance monitoring
try:
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
if PROMETHEUS_AVAILABLE:
    GRADIENT_NORM = Histogram('gradient_norm', 'Gradient norm values')
    GRADIENT_CLIPPING_EVENTS = Counter('gradient_clipping_events_total', 'Gradient clipping events')
    NAN_INF_EVENTS = Counter('nan_inf_events_total', 'NaN/Inf detection events', ['type'])
    TRAINING_STABILITY = Gauge('training_stability', 'Training stability score')


@dataclass
class GradientConfig:
    """Configuration for gradient optimization."""
    # Gradient clipping
    max_grad_norm: float = 1.0
    clip_grad_norm: bool = True
    clip_grad_value: Optional[float] = None
    clip_grad_norm_type: str = "norm"  # norm, value, adaptive
    
    # NaN/Inf handling
    detect_nan_inf: bool = True
    handle_nan_inf: bool = True
    nan_inf_threshold: float = 1e-6
    max_nan_inf_count: int = 5
    
    # Gradient accumulation
    gradient_accumulation_steps: int = 1
    gradient_accumulation_scale: float = 1.0
    
    # Loss scaling
    use_loss_scaling: bool = True
    loss_scale_factor: float = 1.0
    loss_scale_window: int = 100
    
    # Adaptive features
    adaptive_clipping: bool = False
    adaptive_threshold: float = 0.1
    adaptive_window: int = 50
    
    # Monitoring
    monitor_gradients: bool = True
    gradient_history_size: int = 100
    log_gradient_stats: bool = True
    
    # Recovery
    enable_recovery: bool = True
    recovery_threshold: int = 3
    recovery_strategy: str = "restart"  # restart, skip, reduce_lr
    
    # Mixed precision
    mixed_precision: str = "fp16"  # fp16, fp32, bf16
    use_amp: bool = True


class GradientMonitor:
    """Monitor gradients for stability and optimization."""
    
    def __init__(self, config: GradientConfig):
        
    """__init__ function."""
self.config = config
        self.gradient_history = deque(maxlen=config.gradient_history_size)
        self.nan_inf_count = 0
        self.clipping_count = 0
        self.recovery_count = 0
        self.stability_score = 1.0
        
        # Statistics
        self.grad_norms = []
        self.grad_means = []
        self.grad_stds = []
        
    def update_gradient_stats(self, gradients: List[torch.Tensor]) -> Dict[str, float]:
        """Update gradient statistics."""
        if not gradients:
            return {}
        
        # Compute gradient norm
        total_norm = torch.nn.utils.clip_grad_norm_(gradients, float('inf'))
        
        # Compute gradient statistics
        grad_flat = torch.cat([g.flatten() for g in gradients if g is not None])
        grad_mean = grad_flat.mean().item()
        grad_std = grad_flat.std().item()
        grad_norm = total_norm.item()
        
        # Update history
        self.gradient_history.append({
            'norm': grad_norm,
            'mean': grad_mean,
            'std': grad_std,
            'timestamp': time.time()
        })
        
        # Update statistics
        self.grad_norms.append(grad_norm)
        self.grad_means.append(grad_mean)
        self.grad_stds.append(grad_std)
        
        # Compute stability score
        if len(self.grad_norms) > 10:
            recent_norms = self.grad_norms[-10:]
            stability = 1.0 / (1.0 + np.std(recent_norms))
            self.stability_score = stability
        
        # Update Prometheus metrics
        if PROMETHEUS_AVAILABLE:
            GRADIENT_NORM.observe(grad_norm)
            TRAINING_STABILITY.set(self.stability_score)
        
        return {
            'grad_norm': grad_norm,
            'grad_mean': grad_mean,
            'grad_std': grad_std,
            'stability_score': self.stability_score
        }
    
    def detect_nan_inf(self, gradients: List[torch.Tensor]) -> Tuple[bool, List[str]]:
        """Detect NaN and Inf values in gradients."""
        if not self.config.detect_nan_inf:
            return False, []
        
        issues = []
        
        for i, grad in enumerate(gradients):
            if grad is None:
                continue
            
            # Check for NaN
            if torch.isnan(grad).any():
                issues.append(f"NaN detected in gradient {i}")
                if PROMETHEUS_AVAILABLE:
                    NAN_INF_EVENTS.labels(type="nan").inc()
            
            # Check for Inf
            if torch.isinf(grad).any():
                issues.append(f"Inf detected in gradient {i}")
                if PROMETHEUS_AVAILABLE:
                    NAN_INF_EVENTS.labels(type="inf").inc()
            
            # Check for very large values
            if grad.abs().max() > 1e6:
                issues.append(f"Very large values in gradient {i}")
        
        has_issues = len(issues) > 0
        
        if has_issues:
            self.nan_inf_count += 1
            logger.warning(f"NaN/Inf detected: {issues}")
        
        return has_issues, issues
    
    def get_gradient_stats(self) -> Dict[str, Any]:
        """Get comprehensive gradient statistics."""
        if not self.gradient_history:
            return {}
        
        norms = [h['norm'] for h in self.gradient_history]
        means = [h['mean'] for h in self.gradient_history]
        stds = [h['std'] for h in self.gradient_history]
        
        return {
            'current_norm': norms[-1] if norms else 0.0,
            'mean_norm': np.mean(norms),
            'std_norm': np.std(norms),
            'max_norm': np.max(norms),
            'min_norm': np.min(norms),
            'current_mean': means[-1] if means else 0.0,
            'mean_grad_mean': np.mean(means),
            'mean_grad_std': np.mean(stds),
            'stability_score': self.stability_score,
            'nan_inf_count': self.nan_inf_count,
            'clipping_count': self.clipping_count,
            'recovery_count': self.recovery_count
        }


class AdvancedGradientClipper:
    """Advanced gradient clipping with multiple strategies."""
    
    def __init__(self, config: GradientConfig):
        
    """__init__ function."""
self.config = config
        self.clipping_history = deque(maxlen=100)
        self.adaptive_threshold = config.max_grad_norm
        
    def clip_gradients(self, model: nn.Module, monitor: GradientMonitor) -> Dict[str, Any]:
        """Apply gradient clipping with monitoring."""
        if not self.config.clip_grad_norm:
            return {'clipped': False, 'norm': 0.0}
        
        # Get gradients
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        
        if not gradients:
            return {'clipped': False, 'norm': 0.0}
        
        # Update gradient statistics
        stats = monitor.update_gradient_stats(gradients)
        
        # Detect NaN/Inf
        has_issues, issues = monitor.detect_nan_inf(gradients)
        
        if has_issues and self.config.handle_nan_inf:
            self._handle_nan_inf(model, monitor)
            return {'clipped': False, 'norm': 0.0, 'nan_inf_handled': True}
        
        # Apply clipping based on strategy
        if self.config.clip_grad_norm_type == "norm":
            norm = self._clip_by_norm(gradients)
        elif self.config.clip_grad_norm_type == "value":
            norm = self._clip_by_value(gradients)
        elif self.config.clip_grad_norm_type == "adaptive":
            norm = self._clip_adaptive(gradients, monitor)
        else:
            norm = self._clip_by_norm(gradients)
        
        # Update clipping history
        self.clipping_history.append({
            'norm': norm,
            'timestamp': time.time(),
            'threshold': self.adaptive_threshold
        })
        
        # Update adaptive threshold
        if self.config.adaptive_clipping:
            self._update_adaptive_threshold(monitor)
        
        monitor.clipping_count += 1
        
        if PROMETHEUS_AVAILABLE:
            GRADIENT_CLIPPING_EVENTS.inc()
        
        return {
            'clipped': norm > self.adaptive_threshold,
            'norm': norm,
            'threshold': self.adaptive_threshold,
            'stats': stats
        }
    
    def _clip_by_norm(self, gradients: List[torch.Tensor]) -> float:
        """Clip gradients by norm."""
        total_norm = torch.nn.utils.clip_grad_norm_(
            gradients, 
            self.adaptive_threshold,
            norm_type=2.0
        )
        return total_norm.item()
    
    def _clip_by_value(self, gradients: List[torch.Tensor]) -> float:
        """Clip gradients by value."""
        if self.config.clip_grad_value is None:
            return 0.0
        
        total_norm = 0.0
        for grad in gradients:
            if grad is not None:
                grad.clamp_(-self.config.clip_grad_value, self.config.clip_grad_value)
                total_norm += grad.norm(2.0).item() ** 2
        
        return math.sqrt(total_norm)
    
    def _clip_adaptive(self, gradients: List[torch.Tensor], monitor: GradientMonitor) -> float:
        """Adaptive gradient clipping."""
        # Use stability score to adjust threshold
        stability_factor = monitor.stability_score
        adaptive_threshold = self.adaptive_threshold * stability_factor
        
        total_norm = torch.nn.utils.clip_grad_norm_(
            gradients,
            adaptive_threshold,
            norm_type=2.0
        )
        
        return total_norm.item()
    
    def _update_adaptive_threshold(self, monitor: GradientMonitor):
        """Update adaptive threshold based on gradient statistics."""
        if len(self.clipping_history) < self.config.adaptive_window:
            return
        
        recent_norms = [h['norm'] for h in list(self.clipping_history)[-self.config.adaptive_window:]]
        mean_norm = np.mean(recent_norms)
        
        # Adjust threshold based on recent gradient behavior
        if mean_norm > self.adaptive_threshold * 1.5:
            # Gradients are consistently large, increase threshold
            self.adaptive_threshold *= 1.1
        elif mean_norm < self.adaptive_threshold * 0.5:
            # Gradients are consistently small, decrease threshold
            self.adaptive_threshold *= 0.9
        
        # Keep threshold within reasonable bounds
        self.adaptive_threshold = max(0.1, min(10.0, self.adaptive_threshold))
    
    def _handle_nan_inf(self, model: nn.Module, monitor: GradientMonitor):
        """Handle NaN/Inf values in gradients."""
        logger.warning("Handling NaN/Inf values in gradients")
        
        # Zero out gradients
        for param in model.parameters():
            if param.grad is not None:
                param.grad.zero_()
        
        # Increment recovery count
        monitor.recovery_count += 1
        
        # Check if we need to take recovery action
        if monitor.nan_inf_count >= self.config.max_nan_inf_count:
            if self.config.enable_recovery:
                self._trigger_recovery(model, monitor)
    
    def _trigger_recovery(self, model: nn.Module, monitor: GradientMonitor):
        """Trigger recovery action."""
        if self.config.recovery_strategy == "restart":
            logger.warning("Triggering training restart due to NaN/Inf issues")
            # This would typically be handled by the training loop
            raise RuntimeError("Training restart required due to NaN/Inf issues")
        
        elif self.config.recovery_strategy == "skip":
            logger.warning("Skipping current batch due to NaN/Inf issues")
            # Continue with next batch
        
        elif self.config.recovery_strategy == "reduce_lr":
            logger.warning("Reducing learning rate due to NaN/Inf issues")
            # This would be handled by the optimizer


class LossScaler:
    """Advanced loss scaling for training stability."""
    
    def __init__(self, config: GradientConfig):
        
    """__init__ function."""
self.config = config
        self.scaler = GradScaler() if config.use_amp else None
        self.loss_history = deque(maxlen=config.loss_scale_window)
        self.scale_factor = config.loss_scale_factor
        
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for numerical stability."""
        if not self.config.use_loss_scaling:
            return loss
        
        # Apply scaling
        scaled_loss = loss * self.scale_factor
        
        # Update loss history
        self.loss_history.append(loss.item())
        
        # Adjust scale factor based on loss stability
        if len(self.loss_history) >= 10:
            recent_losses = list(self.loss_history)[-10:]
            loss_std = np.std(recent_losses)
            
            if loss_std > 1.0:
                # Loss is unstable, increase scaling
                self.scale_factor *= 1.1
            elif loss_std < 0.1:
                # Loss is stable, decrease scaling
                self.scale_factor *= 0.9
            
            # Keep scale factor within bounds
            self.scale_factor = max(0.1, min(10.0, self.scale_factor))
        
        return scaled_loss
    
    def unscale_gradients(self, optimizer: Optimizer):
        """Unscale gradients after backward pass."""
        if self.scaler is not None:
            self.scaler.unscale_(optimizer)


class GradientAccumulator:
    """Gradient accumulation for large batch training."""
    
    def __init__(self, config: GradientConfig):
        
    """__init__ function."""
self.config = config
        self.accumulation_steps = config.gradient_accumulation_steps
        self.current_step = 0
        self.accumulated_gradients = {}
        
    def accumulate_gradients(self, model: nn.Module, loss: torch.Tensor, 
                           optimizer: Optimizer, scaler: Optional[GradScaler] = None):
        """Accumulate gradients over multiple steps."""
        # Scale loss for accumulation
        scaled_loss = loss / self.accumulation_steps
        
        # Backward pass
        if scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        
        self.current_step += 1
        
        # Check if we should update parameters
        if self.current_step % self.accumulation_steps == 0:
            # Apply gradient clipping
            if self.config.clip_grad_norm:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    self.config.max_grad_norm
                )
            
            # Optimizer step
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            # Zero gradients
            optimizer.zero_grad()
            
            return True  # Parameters were updated
        
        return False  # Parameters were not updated


class TrainingStabilityManager:
    """Comprehensive training stability manager."""
    
    def __init__(self, config: GradientConfig):
        
    """__init__ function."""
self.config = config
        self.monitor = GradientMonitor(config)
        self.clipper = AdvancedGradientClipper(config)
        self.loss_scaler = LossScaler(config)
        self.accumulator = GradientAccumulator(config)
        
        # Training state
        self.step_count = 0
        self.epoch_count = 0
        self.stability_issues = []
        
    def pre_training_step(self, model: nn.Module) -> Dict[str, Any]:
        """Pre-training step checks."""
        # Check model parameters for NaN/Inf
        param_issues = self._check_model_parameters(model)
        
        if param_issues:
            self.stability_issues.extend(param_issues)
            logger.warning(f"Model parameter issues detected: {param_issues}")
        
        return {
            'param_issues': param_issues,
            'step_count': self.step_count,
            'epoch_count': self.epoch_count
        }
    
    def training_step(self, model: nn.Module, loss: torch.Tensor, 
                     optimizer: Optimizer, scaler: Optional[GradScaler] = None) -> Dict[str, Any]:
        """Execute training step with stability checks."""
        self.step_count += 1
        
        # Scale loss
        scaled_loss = self.loss_scaler.scale_loss(loss)
        
        # Accumulate gradients
        updated = self.accumulator.accumulate_gradients(
            model, scaled_loss, optimizer, scaler
        )
        
        # Apply gradient clipping if parameters were updated
        clipping_info = {}
        if updated:
            clipping_info = self.clipper.clip_gradients(model, self.monitor)
        
        # Get gradient statistics
        grad_stats = self.monitor.get_gradient_stats()
        
        return {
            'loss': loss.item(),
            'scaled_loss': scaled_loss.item(),
            'parameters_updated': updated,
            'clipping_info': clipping_info,
            'gradient_stats': grad_stats,
            'stability_score': self.monitor.stability_score
        }
    
    def post_training_step(self, model: nn.Module) -> Dict[str, Any]:
        """Post-training step checks."""
        # Check for training stability
        stability_info = self._check_training_stability()
        
        # Log statistics if enabled
        if self.config.log_gradient_stats:
            self._log_training_stats()
        
        return stability_info
    
    def _check_model_parameters(self, model: nn.Module) -> List[str]:
        """Check model parameters for issues."""
        issues = []
        
        for name, param in model.named_parameters():
            if param is None:
                continue
            
            # Check for NaN
            if torch.isnan(param).any():
                issues.append(f"NaN in parameter {name}")
            
            # Check for Inf
            if torch.isinf(param).any():
                issues.append(f"Inf in parameter {name}")
            
            # Check for very large values
            if param.abs().max() > 1e6:
                issues.append(f"Very large values in parameter {name}")
        
        return issues
    
    def _check_training_stability(self) -> Dict[str, Any]:
        """Check overall training stability."""
        stability_info = {
            'stable': True,
            'issues': [],
            'recommendations': []
        }
        
        # Check gradient statistics
        grad_stats = self.monitor.get_gradient_stats()
        
        if grad_stats.get('stability_score', 1.0) < 0.5:
            stability_info['stable'] = False
            stability_info['issues'].append("Low gradient stability")
            stability_info['recommendations'].append("Consider reducing learning rate")
        
        if self.monitor.nan_inf_count > 0:
            stability_info['issues'].append(f"NaN/Inf detected {self.monitor.nan_inf_count} times")
            stability_info['recommendations'].append("Check data preprocessing and model initialization")
        
        if self.monitor.clipping_count > 0:
            stability_info['issues'].append(f"Gradient clipping applied {self.monitor.clipping_count} times")
        
        return stability_info
    
    def _log_training_stats(self) -> Any:
        """Log training statistics."""
        grad_stats = self.monitor.get_gradient_stats()
        
        logger.info(f"Training Stats - Step {self.step_count}:")
        logger.info(f"  Gradient Norm: {grad_stats.get('current_norm', 0):.4f}")
        logger.info(f"  Stability Score: {grad_stats.get('stability_score', 0):.4f}")
        logger.info(f"  NaN/Inf Count: {grad_stats.get('nan_inf_count', 0)}")
        logger.info(f"  Clipping Count: {grad_stats.get('clipping_count', 0)}")
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive training statistics."""
        grad_stats = self.monitor.get_gradient_stats()
        
        return {
            'training_info': {
                'step_count': self.step_count,
                'epoch_count': self.epoch_count,
                'stability_issues': self.stability_issues
            },
            'gradient_stats': grad_stats,
            'clipping_info': {
                'adaptive_threshold': self.clipper.adaptive_threshold,
                'clipping_history_size': len(self.clipper.clipping_history)
            },
            'loss_scaling': {
                'scale_factor': self.loss_scaler.scale_factor,
                'loss_history_size': len(self.loss_scaler.loss_history)
            },
            'accumulation': {
                'current_step': self.accumulator.current_step,
                'accumulation_steps': self.accumulator.accumulation_steps
            }
        }


async def main():
    """Example usage of gradient optimization system."""
    # Configuration
    config = GradientConfig(
        max_grad_norm=1.0,
        clip_grad_norm=True,
        detect_nan_inf=True,
        handle_nan_inf=True,
        adaptive_clipping=True,
        use_loss_scaling=True,
        monitor_gradients=True
    )
    
    # Initialize stability manager
    stability_manager = TrainingStabilityManager(config)
    
    # Example model (placeholder)
    model = nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Training loop example
    for epoch in range(10):
        for step in range(100):
            # Pre-training checks
            pre_info = stability_manager.pre_training_step(model)
            
            # Forward pass
            x = torch.randn(32, 10)
            y = torch.randn(32, 1)
            output = model(x)
            loss = F.mse_loss(output, y)
            
            # Training step with stability management
            step_info = stability_manager.training_step(model, loss, optimizer)
            
            # Post-training checks
            post_info = stability_manager.post_training_step(model)
            
            # Log progress
            if step % 10 == 0:
                logger.info(f"Epoch {epoch}, Step {step}: Loss = {loss.item():.4f}, "
                          f"Stability = {step_info['stability_score']:.4f}")
    
    # Get final statistics
    final_stats = stability_manager.get_comprehensive_stats()
    logger.info("Training completed!")
    logger.info(f"Final statistics: {json.dumps(final_stats, indent=2, default=str)}")


match __name__:
    case "__main__":
    asyncio.run(main()) 