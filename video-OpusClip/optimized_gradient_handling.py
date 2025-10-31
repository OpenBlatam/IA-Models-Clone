"""
Optimized Gradient Handling System for Video-OpusClip

Advanced gradient clipping, NaN/Inf detection, and robust training safeguards
for stable and efficient deep learning training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import structlog
from dataclasses import dataclass, field
import time
import warnings
from collections import defaultdict, deque
import math

logger = structlog.get_logger()

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class GradientConfig:
    """Configuration for gradient handling."""
    # Gradient clipping
    clip_gradients: bool = True
    clip_norm: float = 1.0
    clip_value: float = 0.5
    clip_method: str = 'norm'  # 'norm', 'value', 'adaptive'
    
    # NaN/Inf detection
    detect_nan_inf: bool = True
    check_gradients: bool = True
    check_weights: bool = True
    check_loss: bool = True
    
    # Recovery strategies
    skip_batch_on_nan: bool = True
    reset_optimizer_on_nan: bool = False
    reduce_lr_on_nan: bool = True
    lr_reduction_factor: float = 0.5
    
    # Monitoring
    log_gradient_stats: bool = True
    gradient_history_size: int = 100
    alert_on_anomalies: bool = True
    
    # Adaptive clipping
    adaptive_clipping: bool = False
    adaptive_window: int = 100
    adaptive_percentile: float = 95.0

@dataclass
class GradientStats:
    """Container for gradient statistics."""
    # Basic stats
    grad_norm: float = 0.0
    param_norm: float = 0.0
    grad_scale: float = 1.0
    
    # Clipping stats
    clipped_norm: float = 0.0
    clipping_ratio: float = 0.0
    
    # NaN/Inf detection
    has_nan: bool = False
    has_inf: bool = False
    nan_count: int = 0
    inf_count: int = 0
    
    # Adaptive stats
    adaptive_threshold: float = 0.0
    gradient_percentile: float = 0.0
    
    # Performance
    computation_time: float = 0.0

@dataclass
class TrainingState:
    """Training state for gradient handling."""
    step: int = 0
    epoch: int = 0
    nan_occurrences: int = 0
    inf_occurrences: int = 0
    gradient_history: List[float] = field(default_factory=list)
    lr_reductions: int = 0
    optimizer_resets: int = 0
    skipped_batches: int = 0

# =============================================================================
# GRADIENT CLIPPING
# =============================================================================

class OptimizedGradientClipper:
    """Advanced gradient clipping with multiple strategies."""
    
    def __init__(self, config: GradientConfig):
        self.config = config
        self.gradient_history = deque(maxlen=config.gradient_history_size)
        self.clipping_stats = defaultdict(int)
        
        logger.info(f"Initialized gradient clipper: method={config.clip_method}, "
                   f"norm={config.clip_norm}, value={config.clip_value}")
    
    def clip_gradients(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scaler: Optional[torch.cuda.amp.GradScaler] = None
    ) -> GradientStats:
        """Clip gradients using the specified method."""
        stats = GradientStats()
        start_time = time.time()
        
        # Unscale gradients if using mixed precision
        if scaler is not None:
            scaler.unscale_(optimizer)
        
        # Calculate gradient norm
        total_norm = self._compute_gradient_norm(model)
        stats.grad_norm = total_norm
        
        # Store in history
        self.gradient_history.append(total_norm)
        
        # Apply clipping based on method
        if self.config.clip_method == 'norm':
            stats = self._clip_by_norm(model, total_norm, stats)
        elif self.config.clip_method == 'value':
            stats = self._clip_by_value(model, stats)
        elif self.config.clip_method == 'adaptive':
            stats = self._clip_adaptively(model, total_norm, stats)
        
        # Update clipping statistics
        self.clipping_stats['total_clips'] += 1
        if stats.clipping_ratio > 0:
            self.clipping_stats['effective_clips'] += 1
        
        stats.computation_time = time.time() - start_time
        
        return stats
    
    def _compute_gradient_norm(self, model: nn.Module) -> float:
        """Compute total gradient norm."""
        total_norm = 0.0
        param_count = 0
        
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        
        total_norm = total_norm ** (1. / 2)
        return total_norm
    
    def _clip_by_norm(self, model: nn.Module, total_norm: float, stats: GradientStats) -> GradientStats:
        """Clip gradients by norm."""
        if total_norm > self.config.clip_norm:
            clip_coef = self.config.clip_norm / (total_norm + 1e-6)
            stats.grad_scale = clip_coef
            stats.clipping_ratio = 1.0 - clip_coef
            
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.data.mul_(clip_coef)
            
            stats.clipped_norm = self.config.clip_norm
        else:
            stats.clipped_norm = total_norm
            stats.clipping_ratio = 0.0
        
        return stats
    
    def _clip_by_value(self, model: nn.Module, stats: GradientStats) -> GradientStats:
        """Clip gradients by value."""
        clipped_count = 0
        total_count = 0
        
        for p in model.parameters():
            if p.grad is not None:
                total_count += p.grad.numel()
                clipped = torch.clamp(p.grad.data, -self.config.clip_value, self.config.clip_value)
                clipped_count += (clipped != p.grad.data).sum().item()
                p.grad.data = clipped
        
        stats.clipping_ratio = clipped_count / total_count if total_count > 0 else 0.0
        stats.clipped_norm = self._compute_gradient_norm(model)
        
        return stats
    
    def _clip_adaptively(self, model: nn.Module, total_norm: float, stats: GradientStats) -> GradientStats:
        """Adaptive gradient clipping based on history."""
        if len(self.gradient_history) >= self.config.adaptive_window:
            # Calculate adaptive threshold
            history_array = np.array(list(self.gradient_history))
            threshold = np.percentile(history_array, self.config.adaptive_percentile)
            stats.adaptive_threshold = threshold
            stats.gradient_percentile = self.config.adaptive_percentile
            
            # Clip if current norm exceeds threshold
            if total_norm > threshold:
                clip_coef = threshold / (total_norm + 1e-6)
                stats.grad_scale = clip_coef
                stats.clipping_ratio = 1.0 - clip_coef
                
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.data.mul_(clip_coef)
                
                stats.clipped_norm = threshold
            else:
                stats.clipped_norm = total_norm
                stats.clipping_ratio = 0.0
        else:
            # Use standard norm clipping during warmup
            stats = self._clip_by_norm(model, total_norm, stats)
        
        return stats
    
    def get_clipping_stats(self) -> Dict[str, Any]:
        """Get clipping statistics."""
        if len(self.gradient_history) == 0:
            return {}
        
        history_array = np.array(list(self.gradient_history))
        return {
            'total_clips': self.clipping_stats['total_clips'],
            'effective_clips': self.clipping_stats['effective_clips'],
            'clipping_rate': self.clipping_stats['effective_clips'] / max(1, self.clipping_stats['total_clips']),
            'mean_grad_norm': np.mean(history_array),
            'std_grad_norm': np.std(history_array),
            'max_grad_norm': np.max(history_array),
            'min_grad_norm': np.min(history_array),
            'recent_grad_norm': history_array[-1] if len(history_array) > 0 else 0.0
        }

# =============================================================================
# NaN/INF DETECTION AND HANDLING
# =============================================================================

class NanInfDetector:
    """Advanced NaN/Inf detection and handling."""
    
    def __init__(self, config: GradientConfig):
        self.config = config
        self.detection_history = defaultdict(int)
        self.recovery_history = defaultdict(int)
        
        logger.info("Initialized NaN/Inf detector")
    
    def check_tensors(
        self,
        model: nn.Module,
        loss: Optional[torch.Tensor] = None,
        optimizer: Optional[optim.Optimizer] = None
    ) -> GradientStats:
        """Comprehensive tensor checking."""
        stats = GradientStats()
        
        # Check model weights
        if self.config.check_weights:
            stats = self._check_model_weights(model, stats)
        
        # Check gradients
        if self.config.check_gradients:
            stats = self._check_gradients(model, stats)
        
        # Check loss
        if self.config.check_loss and loss is not None:
            stats = self._check_loss(loss, stats)
        
        # Update detection history
        if stats.has_nan:
            self.detection_history['nan_count'] += stats.nan_count
        if stats.has_inf:
            self.detection_history['inf_count'] += stats.inf_count
        
        return stats
    
    def _check_model_weights(self, model: nn.Module, stats: GradientStats) -> GradientStats:
        """Check model weights for NaN/Inf."""
        nan_count = 0
        inf_count = 0
        
        for name, param in model.named_parameters():
            if param.data is not None:
                # Check for NaN
                if torch.isnan(param.data).any():
                    nan_count += torch.isnan(param.data).sum().item()
                    stats.has_nan = True
                    logger.warning(f"NaN detected in model weights: {name}")
                
                # Check for Inf
                if torch.isinf(param.data).any():
                    inf_count += torch.isinf(param.data).sum().item()
                    stats.has_inf = True
                    logger.warning(f"Inf detected in model weights: {name}")
        
        stats.nan_count += nan_count
        stats.inf_count += inf_count
        
        return stats
    
    def _check_gradients(self, model: nn.Module, stats: GradientStats) -> GradientStats:
        """Check gradients for NaN/Inf."""
        nan_count = 0
        inf_count = 0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                # Check for NaN
                if torch.isnan(param.grad).any():
                    nan_count += torch.isnan(param.grad).sum().item()
                    stats.has_nan = True
                    logger.warning(f"NaN detected in gradients: {name}")
                
                # Check for Inf
                if torch.isinf(param.grad).any():
                    inf_count += torch.isinf(param.grad).sum().item()
                    stats.has_inf = True
                    logger.warning(f"Inf detected in gradients: {name}")
        
        stats.nan_count += nan_count
        stats.inf_count += inf_count
        
        return stats
    
    def _check_loss(self, loss: torch.Tensor, stats: GradientStats) -> GradientStats:
        """Check loss for NaN/Inf."""
        if torch.isnan(loss).any():
            stats.has_nan = True
            stats.nan_count += torch.isnan(loss).sum().item()
            logger.warning("NaN detected in loss")
        
        if torch.isinf(loss).any():
            stats.has_inf = True
            stats.inf_count += torch.isinf(loss).sum().item()
            logger.warning("Inf detected in loss")
        
        return stats
    
    def handle_nan_inf(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[Any] = None,
        training_state: Optional[TrainingState] = None
    ) -> Dict[str, Any]:
        """Handle NaN/Inf occurrences."""
        actions_taken = {}
        
        # Zero gradients
        optimizer.zero_grad()
        actions_taken['zeroed_gradients'] = True
        
        # Skip batch if configured
        if self.config.skip_batch_on_nan:
            actions_taken['skip_batch'] = True
            if training_state:
                training_state.skipped_batches += 1
        
        # Reset optimizer if configured
        if self.config.reset_optimizer_on_nan:
            self._reset_optimizer(optimizer)
            actions_taken['reset_optimizer'] = True
            if training_state:
                training_state.optimizer_resets += 1
        
        # Reduce learning rate if configured
        if self.config.reduce_lr_on_nan and scheduler is not None:
            self._reduce_learning_rate(scheduler)
            actions_taken['reduced_lr'] = True
            if training_state:
                training_state.lr_reductions += 1
        
        # Log recovery actions
        logger.warning(f"NaN/Inf recovery actions: {actions_taken}")
        
        return actions_taken
    
    def _reset_optimizer(self, optimizer: optim.Optimizer):
        """Reset optimizer state."""
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.data.zero_()
        
        # Reset optimizer internal state
        optimizer.state.clear()
        logger.info("Optimizer state reset")
    
    def _reduce_learning_rate(self, scheduler: Any):
        """Reduce learning rate."""
        for group in scheduler.optimizer.param_groups:
            group['lr'] *= self.config.lr_reduction_factor
        
        logger.info(f"Learning rate reduced by factor {self.config.lr_reduction_factor}")
    
    def get_detection_stats(self) -> Dict[str, Any]:
        """Get detection statistics."""
        return {
            'total_nan_detections': self.detection_history['nan_count'],
            'total_inf_detections': self.detection_history['inf_count'],
            'total_recoveries': sum(self.recovery_history.values()),
            'recovery_actions': dict(self.recovery_history)
        }

# =============================================================================
# OPTIMIZED GRADIENT HANDLER
# =============================================================================

class OptimizedGradientHandler:
    """Main gradient handling system."""
    
    def __init__(self, config: GradientConfig):
        self.config = config
        self.clipper = OptimizedGradientClipper(config)
        self.detector = NanInfDetector(config)
        self.training_state = TrainingState()
        
        logger.info("Initialized optimized gradient handler")
    
    def handle_gradients(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        loss: Optional[torch.Tensor] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None
    ) -> Tuple[GradientStats, Dict[str, Any]]:
        """Comprehensive gradient handling."""
        # Detect NaN/Inf
        detection_stats = self.detector.check_tensors(model, loss, optimizer)
        
        # Handle NaN/Inf if detected
        recovery_actions = {}
        if detection_stats.has_nan or detection_stats.has_inf:
            recovery_actions = self.detector.handle_nan_inf(
                model, optimizer, scheduler, self.training_state
            )
            
            # Update training state
            if detection_stats.has_nan:
                self.training_state.nan_occurrences += 1
            if detection_stats.has_inf:
                self.training_state.inf_occurrences += 1
            
            # Return early if skipping batch
            if self.config.skip_batch_on_nan:
                return detection_stats, recovery_actions
        
        # Apply gradient clipping
        clipping_stats = self.clipper.clip_gradients(model, optimizer, scaler)
        
        # Combine stats
        combined_stats = GradientStats()
        combined_stats.grad_norm = clipping_stats.grad_norm
        combined_stats.clipped_norm = clipping_stats.clipped_norm
        combined_stats.clipping_ratio = clipping_stats.clipping_ratio
        combined_stats.has_nan = detection_stats.has_nan
        combined_stats.has_inf = detection_stats.has_inf
        combined_stats.nan_count = detection_stats.nan_count
        combined_stats.inf_count = detection_stats.inf_count
        combined_stats.computation_time = clipping_stats.computation_time
        
        # Log gradient statistics
        if self.config.log_gradient_stats:
            self._log_gradient_stats(combined_stats)
        
        # Alert on anomalies
        if self.config.alert_on_anomalies:
            self._check_anomalies(combined_stats)
        
        return combined_stats, recovery_actions
    
    def _log_gradient_stats(self, stats: GradientStats):
        """Log gradient statistics."""
        if self.training_state.step % 100 == 0:  # Log every 100 steps
            logger.info(
                f"Step {self.training_state.step}: "
                f"Grad Norm={stats.grad_norm:.4f}, "
                f"Clipped Norm={stats.clipped_norm:.4f}, "
                f"Clipping Ratio={stats.clipping_ratio:.4f}, "
                f"NaN={stats.has_nan}, Inf={stats.has_inf}"
            )
    
    def _check_anomalies(self, stats: GradientStats):
        """Check for gradient anomalies."""
        # Check for gradient explosion
        if stats.grad_norm > 10.0:
            logger.warning(f"Gradient explosion detected: norm={stats.grad_norm:.4f}")
        
        # Check for gradient vanishing
        if stats.grad_norm < 1e-6:
            logger.warning(f"Gradient vanishing detected: norm={stats.grad_norm:.4f}")
        
        # Check for excessive clipping
        if stats.clipping_ratio > 0.8:
            logger.warning(f"Excessive gradient clipping: ratio={stats.clipping_ratio:.4f}")
    
    def update_training_state(self, step: int, epoch: int):
        """Update training state."""
        self.training_state.step = step
        self.training_state.epoch = epoch
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get training summary."""
        clipping_stats = self.clipper.get_clipping_stats()
        detection_stats = self.detector.get_detection_stats()
        
        return {
            'training_state': {
                'step': self.training_state.step,
                'epoch': self.training_state.epoch,
                'nan_occurrences': self.training_state.nan_occurrences,
                'inf_occurrences': self.training_state.inf_occurrences,
                'skipped_batches': self.training_state.skipped_batches,
                'lr_reductions': self.training_state.lr_reductions,
                'optimizer_resets': self.training_state.optimizer_resets
            },
            'clipping_stats': clipping_stats,
            'detection_stats': detection_stats
        }

# =============================================================================
# TRAINING INTEGRATION
# =============================================================================

class GradientAwareTrainer:
    """Trainer with integrated gradient handling."""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        gradient_config: Optional[GradientConfig] = None,
        device: str = "cuda"
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.gradient_config = gradient_config or GradientConfig()
        self.gradient_handler = OptimizedGradientHandler(self.gradient_config)
        
        logger.info("Initialized gradient-aware trainer")
    
    def training_step(
        self,
        batch: Any,
        loss_fn: nn.Module,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        scheduler: Optional[Any] = None
    ) -> Tuple[torch.Tensor, GradientStats, Dict[str, Any]]:
        """Single training step with gradient handling."""
        # Forward pass
        if isinstance(batch, (list, tuple)):
            inputs = batch[0].to(self.device)
            targets = batch[1].to(self.device)
        else:
            inputs = batch.video_frames.to(self.device)
            targets = batch.labels.to(self.device) if hasattr(batch, 'labels') else None
        
        # Forward pass with mixed precision
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = self.model(inputs)
                if targets is not None:
                    loss = loss_fn(outputs, targets)
                else:
                    loss = outputs.loss if hasattr(outputs, 'loss') else outputs
        else:
            outputs = self.model(inputs)
            if targets is not None:
                loss = loss_fn(outputs, targets)
            else:
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs
        
        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Handle gradients
        gradient_stats, recovery_actions = self.gradient_handler.handle_gradients(
            self.model, self.optimizer, loss, scheduler, scaler
        )
        
        # Skip optimizer step if batch was skipped
        if not recovery_actions.get('skip_batch', False):
            if scaler is not None:
                scaler.step(self.optimizer)
                scaler.update()
            else:
                self.optimizer.step()
            
            self.optimizer.zero_grad()
        
        return loss, gradient_stats, recovery_actions
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get training summary."""
        return self.gradient_handler.get_training_summary()

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_gradient_handler(
    clip_gradients: bool = True,
    clip_norm: float = 1.0,
    detect_nan_inf: bool = True,
    **kwargs
) -> OptimizedGradientHandler:
    """Create gradient handler with default settings."""
    config = GradientConfig(
        clip_gradients=clip_gradients,
        clip_norm=clip_norm,
        detect_nan_inf=detect_nan_inf,
        **kwargs
    )
    return OptimizedGradientHandler(config)

def create_gradient_aware_trainer(
    model: nn.Module,
    optimizer: optim.Optimizer,
    **kwargs
) -> GradientAwareTrainer:
    """Create gradient-aware trainer."""
    return GradientAwareTrainer(model, optimizer, **kwargs)

# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def get_gradient_handler_factory(**default_kwargs):
    """Get gradient handler factory."""
    
    def create_handler(**override_kwargs):
        """Create gradient handler with overridden parameters."""
        params = {**default_kwargs, **override_kwargs}
        return create_gradient_handler(**params)
    
    return create_handler

# Global factory instance
gradient_handler_factory = None

def get_global_gradient_handler_factory(**kwargs):
    """Get global gradient handler factory."""
    global gradient_handler_factory
    if gradient_handler_factory is None:
        gradient_handler_factory = get_gradient_handler_factory(**kwargs)
    return gradient_handler_factory 