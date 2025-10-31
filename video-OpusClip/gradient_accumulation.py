"""
Enhanced Gradient Accumulation System for Video-OpusClip

Advanced gradient accumulation implementation for training with large effective
batch sizes while managing memory constraints. Supports multi-GPU training,
mixed precision, and various accumulation strategies.
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

logger = structlog.get_logger()

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class GradientAccumulationConfig:
    """Configuration for gradient accumulation."""
    # Basic accumulation settings
    accumulation_steps: int = 4
    effective_batch_size: Optional[int] = None  # Target effective batch size
    max_batch_size: int = 32  # Maximum physical batch size
    
    # Accumulation strategy
    strategy: str = 'standard'  # 'standard', 'dynamic', 'adaptive'
    
    # Memory management
    memory_threshold: float = 0.8  # Memory usage threshold (0.8 = 80%)
    auto_adjust: bool = True  # Automatically adjust accumulation steps
    
    # Mixed precision
    use_amp: bool = True
    amp_dtype: torch.dtype = torch.float16
    
    # Gradient handling
    gradient_clip_norm: float = 1.0
    gradient_clip_value: Optional[float] = None
    accumulate_gradients: bool = True
    
    # Monitoring
    log_accumulation: bool = True
    log_frequency: int = 10
    
    # Multi-GPU settings
    sync_across_devices: bool = True
    reduce_gradients: bool = True
    
    def __post_init__(self):
        """Validate and set default values."""
        if self.effective_batch_size is not None and self.accumulation_steps == 1:
            # Calculate accumulation steps based on effective batch size
            self.accumulation_steps = max(1, self.effective_batch_size // self.max_batch_size)
        
        if self.accumulation_steps < 1:
            raise ValueError("accumulation_steps must be >= 1")
        
        if self.memory_threshold <= 0 or self.memory_threshold >= 1:
            raise ValueError("memory_threshold must be between 0 and 1")

# =============================================================================
# GRADIENT ACCUMULATION MANAGER
# =============================================================================

class GradientAccumulationManager:
    """Advanced gradient accumulation manager with memory monitoring and optimization."""
    
    def __init__(self, config: GradientAccumulationConfig):
        self.config = config
        self.current_step = 0
        self.accumulated_gradients = 0
        self.memory_monitor = MemoryMonitor()
        self.performance_tracker = PerformanceTracker()
        
        # Initialize scaler for mixed precision
        self.scaler = GradScaler() if config.use_amp else None
        
        logger.info(f"Gradient accumulation manager initialized: "
                   f"steps={config.accumulation_steps}, "
                   f"strategy={config.strategy}")
    
    def should_accumulate(self, batch_idx: int) -> bool:
        """Determine if gradients should be accumulated."""
        return (batch_idx + 1) % self.config.accumulation_steps != 0
    
    def should_update(self, batch_idx: int) -> bool:
        """Determine if optimizer should be updated."""
        return (batch_idx + 1) % self.config.accumulation_steps == 0
    
    def get_effective_batch_size(self, physical_batch_size: int) -> int:
        """Calculate effective batch size."""
        return physical_batch_size * self.config.accumulation_steps
    
    def get_learning_rate_scale(self) -> float:
        """Get learning rate scaling factor for gradient accumulation."""
        return 1.0  # No scaling by default, can be overridden
    
    def accumulate_gradients(self, loss: torch.Tensor, model: nn.Module) -> torch.Tensor:
        """Accumulate gradients from loss."""
        if self.config.use_amp:
            scaled_loss = self.scaler.scale(loss / self.config.accumulation_steps)
            scaled_loss.backward()
        else:
            (loss / self.config.accumulation_steps).backward()
        
        self.accumulated_gradients += 1
        return loss
    
    def update_optimizer(self, optimizer: optim.Optimizer, model: nn.Module) -> Dict[str, Any]:
        """Update optimizer with accumulated gradients."""
        update_info = {
            'gradients_accumulated': self.accumulated_gradients,
            'memory_usage': self.memory_monitor.get_memory_usage(),
            'performance_metrics': self.performance_tracker.get_metrics()
        }
        
        # Gradient clipping
        if self.config.gradient_clip_norm > 0:
            if self.config.use_amp:
                self.scaler.unscale_(optimizer)
            
            if self.config.gradient_clip_value is not None:
                torch.nn.utils.clip_grad_value_(
                    model.parameters(),
                    self.config.gradient_clip_value
                )
            else:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    self.config.gradient_clip_norm
                )
        
        # Optimizer step
        if self.config.use_amp:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Reset accumulation counter
        self.accumulated_gradients = 0
        
        # Log update
        if self.config.log_accumulation:
            logger.info(f"Optimizer updated: {update_info}")
        
        return update_info
    
    def check_memory_and_adjust(self, model: nn.Module) -> bool:
        """Check memory usage and adjust accumulation if needed."""
        if not self.config.auto_adjust:
            return False
        
        memory_usage = self.memory_monitor.get_memory_usage()
        current_threshold = memory_usage['gpu_utilization']
        
        if current_threshold > self.config.memory_threshold:
            # Increase accumulation steps to reduce memory usage
            old_steps = self.config.accumulation_steps
            self.config.accumulation_steps = min(
                self.config.accumulation_steps * 2,
                self.config.effective_batch_size // 1  # Minimum batch size of 1
            )
            
            if self.config.accumulation_steps != old_steps:
                logger.warning(f"Memory threshold exceeded ({current_threshold:.2f} > "
                             f"{self.config.memory_threshold}). "
                             f"Increased accumulation steps from {old_steps} to "
                             f"{self.config.accumulation_steps}")
                return True
        
        return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of gradient accumulation."""
        return {
            'accumulation_steps': self.config.accumulation_steps,
            'current_step': self.current_step,
            'accumulated_gradients': self.accumulated_gradients,
            'effective_batch_size': self.get_effective_batch_size(self.config.max_batch_size),
            'memory_usage': self.memory_monitor.get_memory_usage(),
            'performance_metrics': self.performance_tracker.get_metrics()
        }

# =============================================================================
# MEMORY MONITOR
# =============================================================================

class MemoryMonitor:
    """Monitor GPU memory usage and provide recommendations."""
    
    def __init__(self):
        self.memory_history = []
        self.max_history_size = 100
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        if not torch.cuda.is_available():
            return {
                'gpu_utilization': 0.0,
                'memory_allocated': 0.0,
                'memory_reserved': 0.0,
                'memory_free': 0.0
            }
        
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        total = torch.cuda.get_device_properties(0).total_memory
        
        memory_usage = {
            'gpu_utilization': reserved / total,
            'memory_allocated': allocated / 1024**3,  # GB
            'memory_reserved': reserved / 1024**3,    # GB
            'memory_free': (total - reserved) / 1024**3  # GB
        }
        
        # Update history
        self.memory_history.append(memory_usage)
        if len(self.memory_history) > self.max_history_size:
            self.memory_history.pop(0)
        
        return memory_usage
    
    def get_memory_trend(self) -> Dict[str, float]:
        """Get memory usage trend over time."""
        if len(self.memory_history) < 2:
            return {'trend': 'stable', 'change_rate': 0.0}
        
        recent = self.memory_history[-10:]  # Last 10 measurements
        if len(recent) < 2:
            return {'trend': 'stable', 'change_rate': 0.0}
        
        utilization_values = [m['gpu_utilization'] for m in recent]
        change_rate = (utilization_values[-1] - utilization_values[0]) / len(utilization_values)
        
        if change_rate > 0.01:
            trend = 'increasing'
        elif change_rate < -0.01:
            trend = 'decreasing'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'change_rate': change_rate,
            'average_utilization': np.mean(utilization_values)
        }
    
    def get_memory_recommendations(self) -> List[str]:
        """Get memory optimization recommendations."""
        recommendations = []
        memory_usage = self.get_memory_usage()
        memory_trend = self.get_memory_trend()
        
        if memory_usage['gpu_utilization'] > 0.9:
            recommendations.append("High GPU memory usage detected. Consider reducing batch size or increasing gradient accumulation steps.")
        
        if memory_trend['trend'] == 'increasing':
            recommendations.append("Memory usage is increasing. Monitor for potential memory leaks.")
        
        if memory_usage['memory_free'] < 1.0:  # Less than 1GB free
            recommendations.append("Low free memory. Consider clearing cache or reducing model size.")
        
        return recommendations

# =============================================================================
# PERFORMANCE TRACKER
# =============================================================================

class PerformanceTracker:
    """Track performance metrics during gradient accumulation."""
    
    def __init__(self):
        self.metrics = {
            'accumulation_time': [],
            'update_time': [],
            'memory_peak': [],
            'gradient_norm': []
        }
        self.max_history_size = 100
    
    def record_accumulation_time(self, time_taken: float):
        """Record time taken for gradient accumulation."""
        self.metrics['accumulation_time'].append(time_taken)
        if len(self.metrics['accumulation_time']) > self.max_history_size:
            self.metrics['accumulation_time'].pop(0)
    
    def record_update_time(self, time_taken: float):
        """Record time taken for optimizer update."""
        self.metrics['update_time'].append(time_taken)
        if len(self.metrics['update_time']) > self.max_history_size:
            self.metrics['update_time'].pop(0)
    
    def record_memory_peak(self, peak_memory: float):
        """Record peak memory usage."""
        self.metrics['memory_peak'].append(peak_memory)
        if len(self.metrics['memory_peak']) > self.max_history_size:
            self.metrics['memory_peak'].pop(0)
    
    def record_gradient_norm(self, norm: float):
        """Record gradient norm."""
        self.metrics['gradient_norm'].append(norm)
        if len(self.metrics['gradient_norm']) > self.max_history_size:
            self.metrics['gradient_norm'].pop(0)
    
    def get_metrics(self) -> Dict[str, float]:
        """Get performance metrics."""
        metrics = {}
        
        for key, values in self.metrics.items():
            if values:
                metrics[f'{key}_mean'] = np.mean(values)
                metrics[f'{key}_std'] = np.std(values)
                metrics[f'{key}_min'] = np.min(values)
                metrics[f'{key}_max'] = np.max(values)
            else:
                metrics[f'{key}_mean'] = 0.0
                metrics[f'{key}_std'] = 0.0
                metrics[f'{key}_min'] = 0.0
                metrics[f'{key}_max'] = 0.0
        
        return metrics

# =============================================================================
# ENHANCED TRAINER WITH GRADIENT ACCUMULATION
# =============================================================================

class GradientAccumulationTrainer:
    """Enhanced trainer with advanced gradient accumulation capabilities."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        config: Optional[GradientAccumulationConfig] = None,
        loss_fn: Optional[nn.Module] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = "cuda"
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or GradientAccumulationConfig()
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        self.optimizer = optimizer or optim.Adam(model.parameters())
        self.scheduler = scheduler
        self.device = device
        
        # Initialize gradient accumulation manager
        self.accumulation_manager = GradientAccumulationManager(self.config)
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_metric = float('inf')
        
        logger.info(f"Gradient accumulation trainer initialized: "
                   f"effective_batch_size={self.accumulation_manager.get_effective_batch_size(self.config.max_batch_size)}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch with gradient accumulation."""
        self.model.train()
        self.epoch = epoch
        
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        num_batches = 0
        accumulation_count = 0
        
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
            
            # Forward pass
            if self.config.use_amp:
                with autocast(dtype=self.config.amp_dtype):
                    outputs = self.model(inputs)
                    if targets is not None:
                        loss = self.loss_fn(outputs, targets)
                    else:
                        loss = outputs.loss if hasattr(outputs, 'loss') else outputs
            else:
                outputs = self.model(inputs)
                if targets is not None:
                    loss = self.loss_fn(outputs, targets)
                else:
                    loss = outputs.loss if hasattr(outputs, 'loss') else outputs
            
            # Accumulate gradients
            self.accumulation_manager.accumulate_gradients(loss, self.model)
            
            # Record performance metrics
            accumulation_time = time.time() - batch_start_time
            self.accumulation_manager.performance_tracker.record_accumulation_time(accumulation_time)
            
            # Check if we should update the optimizer
            if self.accumulation_manager.should_update(batch_idx):
                update_start_time = time.time()
                
                # Check memory and adjust if needed
                self.accumulation_manager.check_memory_and_adjust(self.model)
                
                # Update optimizer
                update_info = self.accumulation_manager.update_optimizer(self.optimizer, self.model)
                
                # Update scheduler if available
                if self.scheduler is not None:
                    self.scheduler.step()
                
                # Record update time
                update_time = time.time() - update_start_time
                self.accumulation_manager.performance_tracker.record_update_time(update_time)
                
                # Record memory peak
                memory_usage = self.accumulation_manager.memory_monitor.get_memory_usage()
                self.accumulation_manager.performance_tracker.record_memory_peak(
                    memory_usage['memory_allocated']
                )
                
                accumulation_count += 1
                self.global_step += 1
            
            # Update metrics
            epoch_loss += loss.item()
            if targets is not None:
                if outputs.dim() > 1:
                    accuracy = (outputs.argmax(dim=1) == targets).float().mean().item()
                else:
                    accuracy = ((outputs > 0.5) == targets).float().mean().item()
                epoch_accuracy += accuracy
            
            num_batches += 1
            
            # Logging
            if batch_idx % self.config.log_frequency == 0:
                self._log_training_progress(batch_idx, loss, epoch_loss / num_batches)
        
        # Calculate epoch metrics
        avg_loss = epoch_loss / num_batches
        avg_accuracy = epoch_accuracy / num_batches if num_batches > 0 else 0.0
        
        # Get final status
        status = self.accumulation_manager.get_status()
        
        metrics = {
            'train_loss': avg_loss,
            'train_accuracy': avg_accuracy,
            'epoch': epoch,
            'global_step': self.global_step,
            'accumulation_count': accumulation_count,
            'effective_batch_size': status['effective_batch_size'],
            'time_elapsed': time.time() - start_time,
            'memory_usage': status['memory_usage'],
            'performance_metrics': status['performance_metrics']
        }
        
        return metrics
    
    def _log_training_progress(self, batch_idx: int, loss: torch.Tensor, avg_loss: float):
        """Log training progress with accumulation information."""
        status = self.accumulation_manager.get_status()
        
        logger.info(f"Epoch {self.epoch}, Batch {batch_idx}: "
                   f"Loss={loss.item():.4f}, Avg Loss={avg_loss:.4f}, "
                   f"Accumulated={status['accumulated_gradients']}/{self.config.accumulation_steps}, "
                   f"Memory={status['memory_usage']['gpu_utilization']:.2f}")
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate the model."""
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
                
                # Forward pass
                if self.config.use_amp:
                    with autocast(dtype=self.config.amp_dtype):
                        outputs = self.model(inputs)
                        if targets is not None:
                            loss = self.loss_fn(outputs, targets)
                        else:
                            loss = outputs.loss if hasattr(outputs, 'loss') else outputs
                else:
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
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive training status."""
        status = self.accumulation_manager.get_status()
        status.update({
            'epoch': self.epoch,
            'global_step': self.global_step,
            'best_metric': self.best_metric,
            'config': self.config
        })
        return status

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def calculate_optimal_accumulation_steps(
    target_batch_size: int,
    max_memory_batch_size: int,
    available_memory_gb: float,
    model_memory_gb: float
) -> int:
    """Calculate optimal accumulation steps based on memory constraints."""
    if max_memory_batch_size >= target_batch_size:
        return 1
    
    # Calculate how many accumulation steps needed
    accumulation_steps = target_batch_size // max_memory_batch_size
    
    # Check if we have enough memory for the model
    required_memory = model_memory_gb + (max_memory_batch_size * 0.1)  # Estimate per sample memory
    if required_memory > available_memory_gb * 0.8:  # Leave 20% buffer
        # Reduce batch size to fit in memory
        max_memory_batch_size = int((available_memory_gb * 0.8 - model_memory_gb) / 0.1)
        accumulation_steps = target_batch_size // max_memory_batch_size
    
    return max(1, accumulation_steps)

def create_accumulation_config(
    target_batch_size: int = 128,
    max_batch_size: int = 32,
    use_amp: bool = True,
    strategy: str = 'standard'
) -> GradientAccumulationConfig:
    """Create gradient accumulation configuration."""
    accumulation_steps = max(1, target_batch_size // max_batch_size)
    
    return GradientAccumulationConfig(
        accumulation_steps=accumulation_steps,
        effective_batch_size=target_batch_size,
        max_batch_size=max_batch_size,
        strategy=strategy,
        use_amp=use_amp,
        auto_adjust=True
    )

# =============================================================================
# CONTEXT MANAGERS
# =============================================================================

@contextmanager
def gradient_accumulation_context(manager: GradientAccumulationManager):
    """Context manager for gradient accumulation."""
    try:
        yield manager
    finally:
        # Ensure any remaining gradients are processed
        if manager.accumulated_gradients > 0:
            logger.warning(f"Unprocessed gradients: {manager.accumulated_gradients}")

# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def example_gradient_accumulation():
    """Example of using gradient accumulation for large batch training."""
    # Create configuration for large batch training
    config = create_accumulation_config(
        target_batch_size=256,  # Target effective batch size
        max_batch_size=32,      # Maximum physical batch size
        use_amp=True,           # Use mixed precision
        strategy='standard'
    )
    
    # Create model and data loaders
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
        batch_size=config.max_batch_size,
        shuffle=True
    )
    
    # Create trainer
    trainer = GradientAccumulationTrainer(
        model=model,
        train_loader=train_loader,
        config=config,
        optimizer=optim.Adam(model.parameters(), lr=1e-3)
    )
    
    # Training loop
    for epoch in range(5):
        metrics = trainer.train_epoch(epoch)
        print(f"Epoch {epoch}: Loss={metrics['train_loss']:.4f}, "
              f"Effective Batch Size={metrics['effective_batch_size']}")

if __name__ == "__main__":
    example_gradient_accumulation() 