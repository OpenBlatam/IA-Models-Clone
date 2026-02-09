"""
Mixed Precision Training System for Video-OpusClip

Comprehensive implementation of mixed precision training using torch.cuda.amp
with automatic mixed precision (AMP), gradient scaling, and performance optimization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import structlog
from dataclasses import dataclass, field
import time
import warnings
from contextlib import contextmanager
import json
from pathlib import Path

logger = structlog.get_logger()

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class MixedPrecisionConfig:
    """Configuration for mixed precision training."""
    # Basic AMP settings
    enabled: bool = True
    dtype: torch.dtype = torch.float16
    device_type: str = 'cuda'
    
    # Gradient scaling
    init_scale: float = 2**16
    growth_factor: float = 2.0
    backoff_factor: float = 0.5
    growth_interval: int = 2000
    
    # Performance optimization
    cache_enabled: bool = True
    autocast_enabled: bool = True
    
    # Memory optimization
    memory_efficient: bool = True
    pin_memory: bool = True
    
    # Monitoring
    log_scaling: bool = True
    log_frequency: int = 100
    save_scaler_state: bool = True
    
    # Error handling
    handle_overflow: bool = True
    overflow_threshold: float = float('inf')
    
    # Multi-GPU settings
    sync_scaler: bool = True
    broadcast_scaler: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        if self.enabled and not torch.cuda.is_available():
            warnings.warn("Mixed precision enabled but CUDA not available")
            self.enabled = False
        
        if self.dtype not in [torch.float16, torch.bfloat16]:
            raise ValueError("dtype must be torch.float16 or torch.bfloat16")
        
        if self.init_scale <= 0:
            raise ValueError("init_scale must be positive")
        
        if self.growth_factor <= 1.0:
            raise ValueError("growth_factor must be greater than 1.0")
        
        if self.backoff_factor <= 0 or self.backoff_factor >= 1.0:
            raise ValueError("backoff_factor must be between 0 and 1")

# =============================================================================
# MIXED PRECISION MANAGER
# =============================================================================

class MixedPrecisionManager:
    """Advanced mixed precision training manager with comprehensive features."""
    
    def __init__(self, config: MixedPrecisionConfig):
        self.config = config
        self.scaler = None
        self.performance_tracker = MixedPrecisionPerformanceTracker()
        self.memory_monitor = MixedPrecisionMemoryMonitor()
        
        if config.enabled:
            self._initialize_scaler()
        
        logger.info(f"Mixed precision manager initialized: enabled={config.enabled}, "
                   f"dtype={config.dtype}")
    
    def _initialize_scaler(self):
        """Initialize gradient scaler."""
        self.scaler = GradScaler(
            init_scale=self.config.init_scale,
            growth_factor=self.config.growth_factor,
            backoff_factor=self.config.backoff_factor,
            growth_interval=self.config.growth_interval
        )
        
        logger.info(f"Gradient scaler initialized: init_scale={self.config.init_scale}")
    
    def autocast_context(self):
        """Get autocast context for mixed precision training."""
        if not self.config.enabled or not self.config.autocast_enabled:
            return contextmanager(lambda: (yield))()
        
        return autocast(
            device_type=self.config.device_type,
            dtype=self.config.dtype,
            cache_enabled=self.config.cache_enabled
        )
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for mixed precision training."""
        if not self.config.enabled or self.scaler is None:
            return loss
        
        return self.scaler.scale(loss)
    
    def unscale_optimizer(self, optimizer: optim.Optimizer):
        """Unscale optimizer gradients."""
        if not self.config.enabled or self.scaler is None:
            return
        
        self.scaler.unscale_(optimizer)
    
    def step_optimizer(self, optimizer: optim.Optimizer) -> bool:
        """Step optimizer with mixed precision support."""
        if not self.config.enabled or self.scaler is None:
            optimizer.step()
            return True
        
        return self.scaler.step(optimizer)
    
    def update_scaler(self):
        """Update gradient scaler."""
        if not self.config.enabled or self.scaler is None:
            return
        
        self.scaler.update()
        
        # Record scaler state
        self.performance_tracker.record_scaler_state(
            self.scaler.get_scale(),
            self.scaler.get_growth_tracker()
        )
    
    def handle_overflow(self, optimizer: optim.Optimizer) -> bool:
        """Handle gradient overflow."""
        if not self.config.enabled or self.scaler is None:
            return False
        
        if self.scaler.is_enabled():
            # Check for overflow
            for param in optimizer.param_groups[0]['params']:
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        logger.warning("Gradient overflow detected")
                        return True
        
        return False
    
    def get_scaler_state(self) -> Dict[str, Any]:
        """Get current scaler state."""
        if not self.config.enabled or self.scaler is None:
            return {}
        
        return {
            'scale': self.scaler.get_scale(),
            'growth_tracker': self.scaler.get_growth_tracker(),
            'enabled': self.scaler.is_enabled()
        }
    
    def save_scaler_state(self, path: str):
        """Save scaler state to file."""
        if not self.config.enabled or self.scaler is None:
            return
        
        state = self.scaler.state_dict()
        torch.save(state, path)
        logger.info(f"Scaler state saved to {path}")
    
    def load_scaler_state(self, path: str):
        """Load scaler state from file."""
        if not self.config.enabled or self.scaler is None:
            return
        
        state = torch.load(path)
        self.scaler.load_state_dict(state)
        logger.info(f"Scaler state loaded from {path}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            'scaler_state': self.get_scaler_state(),
            'performance_metrics': self.performance_tracker.get_metrics(),
            'memory_usage': self.memory_monitor.get_memory_usage(),
            'config': self.config
        }

# =============================================================================
# PERFORMANCE TRACKER
# =============================================================================

class MixedPrecisionPerformanceTracker:
    """Track performance metrics for mixed precision training."""
    
    def __init__(self):
        self.metrics = {
            'scaler_scale': [],
            'scaler_growth_tracker': [],
            'training_time': [],
            'memory_savings': [],
            'overflow_count': 0,
            'total_steps': 0
        }
        self.max_history_size = 1000
    
    def record_scaler_state(self, scale: float, growth_tracker: int):
        """Record scaler state."""
        self.metrics['scaler_scale'].append(scale)
        self.metrics['scaler_growth_tracker'].append(growth_tracker)
        
        if len(self.metrics['scaler_scale']) > self.max_history_size:
            self.metrics['scaler_scale'].pop(0)
            self.metrics['scaler_growth_tracker'].pop(0)
    
    def record_training_time(self, time_taken: float):
        """Record training time."""
        self.metrics['training_time'].append(time_taken)
        
        if len(self.metrics['training_time']) > self.max_history_size:
            self.metrics['training_time'].pop(0)
    
    def record_memory_savings(self, savings_percentage: float):
        """Record memory savings."""
        self.metrics['memory_savings'].append(savings_percentage)
        
        if len(self.metrics['memory_savings']) > self.max_history_size:
            self.metrics['memory_savings'].pop(0)
    
    def record_overflow(self):
        """Record gradient overflow."""
        self.metrics['overflow_count'] += 1
    
    def increment_steps(self):
        """Increment total steps."""
        self.metrics['total_steps'] += 1
    
    def get_metrics(self) -> Dict[str, float]:
        """Get performance metrics."""
        metrics = {}
        
        for key, values in self.metrics.items():
            if isinstance(values, list) and values:
                metrics[f'{key}_mean'] = np.mean(values)
                metrics[f'{key}_std'] = np.std(values)
                metrics[f'{key}_min'] = np.min(values)
                metrics[f'{key}_max'] = np.max(values)
            elif isinstance(values, (int, float)):
                metrics[key] = values
        
        return metrics

# =============================================================================
# MEMORY MONITOR
# =============================================================================

class MixedPrecisionMemoryMonitor:
    """Monitor memory usage for mixed precision training."""
    
    def __init__(self):
        self.memory_history = []
        self.max_history_size = 100
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage."""
        if not torch.cuda.is_available():
            return {
                'gpu_memory_allocated': 0.0,
                'gpu_memory_reserved': 0.0,
                'gpu_memory_free': 0.0,
                'gpu_utilization': 0.0
            }
        
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        total = torch.cuda.get_device_properties(0).total_memory
        
        memory_usage = {
            'gpu_memory_allocated': allocated / 1024**3,  # GB
            'gpu_memory_reserved': reserved / 1024**3,    # GB
            'gpu_memory_free': (total - reserved) / 1024**3,  # GB
            'gpu_utilization': reserved / total
        }
        
        # Update history
        self.memory_history.append(memory_usage)
        if len(self.memory_history) > self.max_history_size:
            self.memory_history.pop(0)
        
        return memory_usage
    
    def estimate_memory_savings(self, fp32_memory: float) -> float:
        """Estimate memory savings from mixed precision."""
        current_memory = self.get_memory_usage()['gpu_memory_allocated']
        
        if fp32_memory > 0:
            savings = (fp32_memory - current_memory) / fp32_memory * 100
            return max(0, savings)
        
        return 0.0

# =============================================================================
# ENHANCED TRAINER WITH MIXED PRECISION
# =============================================================================

class MixedPrecisionTrainer:
    """Enhanced trainer with comprehensive mixed precision support."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        config: Optional[MixedPrecisionConfig] = None,
        loss_fn: Optional[nn.Module] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = "cuda"
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or MixedPrecisionConfig()
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        self.optimizer = optimizer or optim.Adam(model.parameters())
        self.scheduler = scheduler
        self.device = device
        
        # Initialize mixed precision manager
        self.mp_manager = MixedPrecisionManager(self.config)
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_metric = float('inf')
        
        logger.info(f"Mixed precision trainer initialized: "
                   f"enabled={self.config.enabled}, dtype={self.config.dtype}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch with mixed precision."""
        self.model.train()
        self.epoch = epoch
        
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        num_batches = 0
        overflow_count = 0
        
        start_time = time.time()
        
        for batch_idx, batch in enumerate(self.train_loader):
            batch_start_time = time.time()
            
            # Move batch to device
            if isinstance(batch, (list, tuple)):
                inputs = batch[0].to(self.device)
                targets = batch[1].to(self.device)
            else:
                inputs = batch.video_frames.to(self.device)
                targets = batch.labels.to(self.device) if hasattr(batch, 'labels') else None
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with self.mp_manager.autocast_context():
                outputs = self.model(inputs)
                if targets is not None:
                    loss = self.loss_fn(outputs, targets)
                else:
                    loss = outputs.loss if hasattr(outputs, 'loss') else outputs
            
            # Scale loss and backward pass
            scaled_loss = self.mp_manager.scale_loss(loss)
            scaled_loss.backward()
            
            # Handle gradient overflow
            if self.mp_manager.handle_overflow(self.optimizer):
                overflow_count += 1
                self.mp_manager.performance_tracker.record_overflow()
                logger.warning(f"Gradient overflow at batch {batch_idx}")
                continue
            
            # Unscale optimizer and step
            self.mp_manager.unscale_optimizer(self.optimizer)
            
            # Gradient clipping (if needed)
            if hasattr(self.config, 'gradient_clip_norm') and self.config.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_norm
                )
            
            # Step optimizer
            success = self.mp_manager.step_optimizer(self.optimizer)
            if success:
                self.mp_manager.update_scaler()
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Record performance metrics
            batch_time = time.time() - batch_start_time
            self.mp_manager.performance_tracker.record_training_time(batch_time)
            self.mp_manager.performance_tracker.increment_steps()
            
            # Update metrics
            epoch_loss += loss.item()
            if targets is not None:
                if outputs.dim() > 1:
                    accuracy = (outputs.argmax(dim=1) == targets).float().mean().item()
                else:
                    accuracy = ((outputs > 0.5) == targets).float().mean().item()
                epoch_accuracy += accuracy
            
            num_batches += 1
            self.global_step += 1
            
            # Logging
            if batch_idx % self.config.log_frequency == 0:
                self._log_training_progress(batch_idx, loss, epoch_loss / num_batches)
        
        # Calculate epoch metrics
        avg_loss = epoch_loss / num_batches
        avg_accuracy = epoch_accuracy / num_batches if num_batches > 0 else 0.0
        
        # Get performance stats
        stats = self.mp_manager.get_performance_stats()
        
        metrics = {
            'train_loss': avg_loss,
            'train_accuracy': avg_accuracy,
            'epoch': epoch,
            'global_step': self.global_step,
            'time_elapsed': time.time() - start_time,
            'overflow_count': overflow_count,
            'scaler_scale': stats['scaler_state'].get('scale', 1.0),
            'memory_usage': stats['memory_usage'],
            'performance_metrics': stats['performance_metrics']
        }
        
        return metrics
    
    def _log_training_progress(self, batch_idx: int, loss: torch.Tensor, avg_loss: float):
        """Log training progress with mixed precision information."""
        scaler_state = self.mp_manager.get_scaler_state()
        
        logger.info(f"Epoch {self.epoch}, Batch {batch_idx}: "
                   f"Loss={loss.item():.4f}, Avg Loss={avg_loss:.4f}, "
                   f"Scaler Scale={scaler_state.get('scale', 1.0):.2e}")
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate the model with mixed precision."""
        if self.val_loader is None:
            return {'val_loss': float('inf'), 'val_accuracy': 0.0}
        
        self.model.eval()
        
        val_loss = 0.0
        val_accuracy = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move batch to device
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0].to(self.device)
                    targets = batch[1].to(self.device)
                else:
                    inputs = batch.video_frames.to(self.device)
                    targets = batch.labels.to(self.device) if hasattr(batch, 'labels') else None
                
                # Forward pass with mixed precision
                with self.mp_manager.autocast_context():
                    outputs = self.model(inputs)
                    if targets is not None:
                        loss = self.loss_fn(outputs, targets)
                    else:
                        loss = outputs.loss if hasattr(outputs, 'loss') else outputs
                
                # Update metrics
                val_loss += loss.item()
                if targets is not None:
                    if outputs.dim() > 1:
                        accuracy = (outputs.argmax(dim=1) == targets).float().mean().item()
                    else:
                        accuracy = ((outputs > 0.5) == targets).float().mean().item()
                    val_accuracy += accuracy
                
                num_batches += 1
        
        # Calculate average metrics
        avg_loss = val_loss / num_batches
        avg_accuracy = val_accuracy / num_batches if num_batches > 0 else 0.0
        
        return {
            'val_loss': avg_loss,
            'val_accuracy': avg_accuracy
        }
    
    def save_checkpoint(self, epoch: int, path: str):
        """Save checkpoint with mixed precision state."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config,
            'global_step': self.global_step,
            'best_metric': self.best_metric
        }
        
        # Save scaler state if enabled
        if self.config.enabled and self.config.save_scaler_state:
            scaler_path = str(path).replace('.pth', '_scaler.pth')
            self.mp_manager.save_scaler_state(scaler_path)
            checkpoint['scaler_path'] = scaler_path
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: str):
        """Load checkpoint with mixed precision state."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load scaler state if available
        if self.config.enabled and 'scaler_path' in checkpoint:
            scaler_path = checkpoint['scaler_path']
            if Path(scaler_path).exists():
                self.mp_manager.load_scaler_state(scaler_path)
        
        self.epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_metric = checkpoint.get('best_metric', float('inf'))
        
        logger.info(f"Checkpoint loaded: {path}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive training status."""
        stats = self.mp_manager.get_performance_stats()
        stats.update({
            'epoch': self.epoch,
            'global_step': self.global_step,
            'best_metric': self.best_metric,
            'config': self.config
        })
        return stats

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_mixed_precision_config(
    enabled: bool = True,
    dtype: torch.dtype = torch.float16,
    init_scale: float = 2**16,
    memory_efficient: bool = True
) -> MixedPrecisionConfig:
    """Create mixed precision configuration."""
    return MixedPrecisionConfig(
        enabled=enabled,
        dtype=dtype,
        init_scale=init_scale,
        memory_efficient=memory_efficient
    )

def benchmark_mixed_precision(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    config: MixedPrecisionConfig,
    num_steps: int = 100
) -> Dict[str, float]:
    """Benchmark mixed precision vs full precision training."""
    device = next(model.parameters()).device
    
    # Test full precision
    model_fp32 = model.clone().to(torch.float32)
    optimizer_fp32 = optim.Adam(model_fp32.parameters())
    
    start_time = time.time()
    memory_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    
    for i, batch in enumerate(train_loader):
        if i >= num_steps:
            break
        
        if isinstance(batch, (list, tuple)):
            inputs = batch[0].to(device)
            targets = batch[1].to(device)
        else:
            inputs = batch.video_frames.to(device)
            targets = batch.labels.to(device) if hasattr(batch, 'labels') else None
        
        optimizer_fp32.zero_grad()
        outputs = model_fp32(inputs)
        loss = nn.CrossEntropyLoss()(outputs, targets)
        loss.backward()
        optimizer_fp32.step()
    
    fp32_time = time.time() - start_time
    fp32_memory = torch.cuda.memory_allocated() - memory_before if torch.cuda.is_available() else 0
    
    # Test mixed precision
    model_mp = model.clone()
    mp_config = config
    mp_trainer = MixedPrecisionTrainer(
        model=model_mp,
        train_loader=train_loader,
        config=mp_config,
        optimizer=optim.Adam(model_mp.parameters())
    )
    
    start_time = time.time()
    memory_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    
    for i, batch in enumerate(train_loader):
        if i >= num_steps:
            break
        
        if isinstance(batch, (list, tuple)):
            inputs = batch[0].to(device)
            targets = batch[1].to(device)
        else:
            inputs = batch.video_frames.to(device)
            targets = batch.labels.to(device) if hasattr(batch, 'labels') else None
        
        mp_trainer.optimizer.zero_grad()
        
        with mp_trainer.mp_manager.autocast_context():
            outputs = mp_trainer.model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, targets)
        
        scaled_loss = mp_trainer.mp_manager.scale_loss(loss)
        scaled_loss.backward()
        mp_trainer.mp_manager.step_optimizer(mp_trainer.optimizer)
        mp_trainer.mp_manager.update_scaler()
    
    mp_time = time.time() - start_time
    mp_memory = torch.cuda.memory_allocated() - memory_before if torch.cuda.is_available() else 0
    
    # Calculate metrics
    speedup = fp32_time / mp_time if mp_time > 0 else 1.0
    memory_savings = (fp32_memory - mp_memory) / fp32_memory * 100 if fp32_memory > 0 else 0.0
    
    return {
        'fp32_time': fp32_time,
        'mp_time': mp_time,
        'speedup': speedup,
        'fp32_memory': fp32_memory / 1024**3,  # GB
        'mp_memory': mp_memory / 1024**3,      # GB
        'memory_savings': memory_savings
    }

# =============================================================================
# CONTEXT MANAGERS
# =============================================================================

@contextmanager
def mixed_precision_context(config: MixedPrecisionConfig):
    """Context manager for mixed precision training."""
    manager = MixedPrecisionManager(config)
    try:
        yield manager
    finally:
        # Cleanup if needed
        pass

# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def example_mixed_precision_training():
    """Example of mixed precision training setup."""
    # Create configuration
    config = create_mixed_precision_config(
        enabled=True,
        dtype=torch.float16,
        init_scale=2**16,
        memory_efficient=True
    )
    
    # Create model and dataset
    model = nn.Sequential(
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 10)
    )
    
    # Create synthetic dataset
    train_data = torch.randn(1000, 784)
    train_labels = torch.randint(0, 10, (1000,))
    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )
    
    # Create trainer
    trainer = MixedPrecisionTrainer(
        model=model,
        train_loader=train_loader,
        config=config,
        optimizer=optim.Adam(model.parameters(), lr=1e-3)
    )
    
    # Training loop
    for epoch in range(5):
        metrics = trainer.train_epoch(epoch)
        print(f"Epoch {epoch}: Loss={metrics['train_loss']:.4f}, "
              f"Scaler Scale={metrics['scaler_scale']:.2e}")

if __name__ == "__main__":
    example_mixed_precision_training() 