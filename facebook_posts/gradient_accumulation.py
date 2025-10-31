from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import torch
import torch.nn as nn
import torch.utils.data as data
from data_loader_utils import make_loader
import numpy as np
import logging
import time
import os
import gc
from typing import Dict, Any, List, Tuple, Optional, Union, Callable, Iterator
from dataclasses import dataclass, field
from enum import Enum
import warnings
from pathlib import Path
import threading
import queue
from contextlib import contextmanager
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, deque
    from deep_learning_framework import DeepLearningFramework, FrameworkConfig, TaskType
    from evaluation_metrics import EvaluationManager, MetricConfig, MetricType
    from gradient_clipping_nan_handling import NumericalStabilityManager
    from early_stopping_scheduling import TrainingManager
    from efficient_data_loading import EfficientDataLoader
    from data_splitting_validation import DataSplitter
    from training_evaluation import TrainingManager as TrainingEvalManager
    from diffusion_models import DiffusionModel, DiffusionConfig
    from advanced_transformers import AdvancedTransformerModel
    from llm_training import AdvancedLLMTrainer
    from model_finetuning import ModelFineTuner
    from custom_modules import AdvancedNeuralNetwork
    from weight_initialization import AdvancedWeightInitializer
    from normalization_techniques import AdvancedLayerNorm
    from loss_functions import AdvancedCrossEntropyLoss
    from optimization_algorithms import AdvancedAdamW
    from attention_mechanisms import MultiHeadAttention
    from tokenization_sequence import AdvancedTokenizer
    from framework_utils import MetricsTracker, ModelAnalyzer, PerformanceMonitor
    from deep_learning_integration import DeepLearningIntegration, IntegrationConfig, IntegrationType, ComponentType
    from robust_error_handling import RobustErrorHandler, RobustDataLoader, RobustModelHandler
    from training_logging_system import TrainingLogger, TrainingProgressTracker, TrainingLoggingManager
    from pytorch_debugging_tools import PyTorchDebugger, PyTorchDebugManager, DebugConfig
    from multi_gpu_training import MultiGPUTrainer, MultiGPUConfig, MultiGPUMode, ParallelStrategy
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Gradient Accumulation System
Comprehensive gradient accumulation for large batch sizes and memory-efficient training.
"""

warnings.filterwarnings('ignore')

# Import our custom modules
try:
except ImportError as e:
    print(f"Warning: Some modules not available: {e}")


class AccumulationMode(Enum):
    """Gradient accumulation modes."""
    STANDARD = "standard"           # Standard gradient accumulation
    EFFICIENT = "efficient"         # Memory-efficient accumulation
    ADAPTIVE = "adaptive"           # Adaptive accumulation based on memory
    PROGRESSIVE = "progressive"     # Progressive accumulation with scaling
    SELECTIVE = "selective"         # Selective parameter accumulation


class AccumulationStrategy(Enum):
    """Gradient accumulation strategies."""
    UNIFORM = "uniform"             # Uniform accumulation across all parameters
    LAYER_WISE = "layer_wise"       # Layer-wise accumulation
    PARAMETER_GROUPS = "parameter_groups"  # Parameter group-based accumulation
    ADAPTIVE_LAYERS = "adaptive_layers"    # Adaptive layer accumulation
    MEMORY_AWARE = "memory_aware"   # Memory-aware accumulation


@dataclass
class GradientAccumulationConfig:
    """Configuration for gradient accumulation."""
    accumulation_steps: int = 4
    mode: AccumulationMode = AccumulationMode.STANDARD
    strategy: AccumulationStrategy = AccumulationStrategy.UNIFORM
    effective_batch_size: int = 64
    target_batch_size: int = 256
    memory_threshold: float = 0.8  # 80% of available memory
    gradient_scaling: bool = True
    loss_scaling: bool = True
    adaptive_scaling: bool = False
    progressive_scaling: bool = False
    selective_accumulation: bool = False
    accumulation_schedule: List[int] = field(default_factory=list)
    layer_accumulation_weights: Dict[str, float] = field(default_factory=dict)
    parameter_group_weights: Dict[str, float] = field(default_factory=dict)
    memory_monitoring: bool = True
    gradient_clipping: bool = True
    clipping_norm: float = 1.0
    nan_handling: bool = True
    checkpoint_frequency: int = 1
    logging_frequency: int = 10
    validation_frequency: int = 1
    save_accumulation_state: bool = True
    restore_accumulation_state: bool = True


@dataclass
class AccumulationState:
    """State for gradient accumulation."""
    current_step: int = 0
    accumulation_step: int = 0
    total_steps: int = 0
    effective_batch_size: int = 0
    accumulated_loss: float = 0.0
    accumulated_gradients: Dict[str, torch.Tensor] = field(default_factory=dict)
    gradient_norms: List[float] = field(default_factory=list)
    memory_usage: List[float] = field(default_factory=list)
    loss_history: List[float] = field(default_factory=list)
    scaling_factors: List[float] = field(default_factory=list)
    nan_detected: bool = False
    overflow_detected: bool = False
    last_optimizer_step: int = 0
    last_checkpoint_step: int = 0


class GradientAccumulator:
    """Comprehensive gradient accumulation system."""
    
    def __init__(self, config: GradientAccumulationConfig):
        self.config = config
        self.logger = self._setup_logger()
        self.state = AccumulationState()
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.scaler = None
        self.device = None
        
        # Performance tracking
        self.performance_metrics = {
            'total_accumulation_steps': 0,
            'total_optimizer_steps': 0,
            'average_gradient_norm': 0.0,
            'memory_efficiency': 0.0,
            'training_time': 0.0
        }
        
        # Initialize accumulation schedule
        self._initialize_accumulation_schedule()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for gradient accumulation."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger("gradient_accumulator")
    
    def _initialize_accumulation_schedule(self) -> Any:
        """Initialize accumulation schedule."""
        if not self.config.accumulation_schedule:
            # Default schedule: uniform accumulation
            self.config.accumulation_schedule = [self.config.accumulation_steps] * 1000
        
        self.logger.info(f"Accumulation schedule initialized: {self.config.accumulation_steps} steps")
    
    def setup_training(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                      criterion: Callable, scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                      device: Optional[torch.device] = None):
        """Setup training components for gradient accumulation."""
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device or next(model.parameters()).device
        
        # Setup mixed precision scaler
        if hasattr(torch.cuda, 'amp') and torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Initialize accumulation state
        self._initialize_accumulation_state()
        
        self.logger.info(f"Gradient accumulation setup completed on device: {self.device}")
    
    def _initialize_accumulation_state(self) -> Any:
        """Initialize accumulation state."""
        self.state = AccumulationState()
        self.state.effective_batch_size = self.config.effective_batch_size
        
        # Initialize accumulated gradients
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.state.accumulated_gradients[name] = torch.zeros_like(param.grad) if param.grad is not None else torch.zeros_like(param.data)
    
    def accumulate_gradients(self, loss: torch.Tensor, backward: bool = True) -> Dict[str, Any]:
        """Accumulate gradients for the current step."""
        # Scale loss if needed
        if self.config.loss_scaling:
            scaled_loss = loss / self.config.accumulation_steps
        else:
            scaled_loss = loss
        
        # Backward pass
        if backward:
            if self.scaler is not None:
                self.scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
        
        # Accumulate gradients
        self._accumulate_gradients()
        
        # Update state
        self.state.accumulation_step += 1
        self.state.total_steps += 1
        self.state.accumulated_loss += loss.item()
        
        # Check if optimization step should be performed
        should_optimize = self._should_perform_optimization()
        
        # Perform optimization if needed
        if should_optimize:
            self._perform_optimization_step()
        
        # Log progress
        if self.state.total_steps % self.config.logging_frequency == 0:
            self._log_accumulation_progress()
        
        return {
            'loss': loss.item(),
            'accumulated_loss': self.state.accumulated_loss,
            'accumulation_step': self.state.accumulation_step,
            'should_optimize': should_optimize,
            'effective_batch_size': self.state.effective_batch_size
        }
    
    def _accumulate_gradients(self) -> Any:
        """Accumulate gradients based on strategy."""
        if self.config.strategy == AccumulationStrategy.UNIFORM:
            self._uniform_accumulation()
        elif self.config.strategy == AccumulationStrategy.LAYER_WISE:
            self._layer_wise_accumulation()
        elif self.config.strategy == AccumulationStrategy.PARAMETER_GROUPS:
            self._parameter_group_accumulation()
        elif self.config.strategy == AccumulationStrategy.ADAPTIVE_LAYERS:
            self._adaptive_layer_accumulation()
        elif self.config.strategy == AccumulationStrategy.MEMORY_AWARE:
            self._memory_aware_accumulation()
        else:
            self._uniform_accumulation()
    
    def _uniform_accumulation(self) -> Any:
        """Uniform gradient accumulation across all parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if name not in self.state.accumulated_gradients:
                    self.state.accumulated_gradients[name] = torch.zeros_like(param.grad)
                
                # Accumulate gradients
                self.state.accumulated_gradients[name] += param.grad.clone()
                
                # Clear current gradients
                param.grad.zero_()
    
    def _layer_wise_accumulation(self) -> Any:
        """Layer-wise gradient accumulation."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if name not in self.state.accumulated_gradients:
                    self.state.accumulated_gradients[name] = torch.zeros_like(param.grad)
                
                # Get layer weight
                layer_weight = self.config.layer_accumulation_weights.get(name, 1.0)
                
                # Accumulate gradients with layer weight
                self.state.accumulated_gradients[name] += param.grad.clone() * layer_weight
                
                # Clear current gradients
                param.grad.zero_()
    
    def _parameter_group_accumulation(self) -> Any:
        """Parameter group-based gradient accumulation."""
        for group_idx, group in enumerate(self.optimizer.param_groups):
            group_weight = self.config.parameter_group_weights.get(f"group_{group_idx}", 1.0)
            
            for param in group['params']:
                if param.requires_grad and param.grad is not None:
                    param_name = None
                    for name, p in self.model.named_parameters():
                        if p is param:
                            param_name = name
                            break
                    
                    if param_name and param_name not in self.state.accumulated_gradients:
                        self.state.accumulated_gradients[param_name] = torch.zeros_like(param.grad)
                    
                    if param_name:
                        # Accumulate gradients with group weight
                        self.state.accumulated_gradients[param_name] += param.grad.clone() * group_weight
                        
                        # Clear current gradients
                        param.grad.zero_()
    
    def _adaptive_layer_accumulation(self) -> Any:
        """Adaptive layer accumulation based on gradient norms."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if name not in self.state.accumulated_gradients:
                    self.state.accumulated_gradients[name] = torch.zeros_like(param.grad)
                
                # Calculate gradient norm
                grad_norm = param.grad.norm().item()
                
                # Adaptive scaling based on gradient norm
                if grad_norm > 1.0:
                    scaling_factor = 1.0 / grad_norm
                else:
                    scaling_factor = 1.0
                
                # Accumulate gradients with adaptive scaling
                self.state.accumulated_gradients[name] += param.grad.clone() * scaling_factor
                
                # Clear current gradients
                param.grad.zero_()
    
    def _memory_aware_accumulation(self) -> Any:
        """Memory-aware gradient accumulation."""
        # Check memory usage
        if torch.cuda.is_available():
            memory_usage = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
        else:
            memory_usage = 0.0
        
        self.state.memory_usage.append(memory_usage)
        
        # Adjust accumulation based on memory usage
        if memory_usage > self.config.memory_threshold:
            # Reduce accumulation to prevent OOM
            scaling_factor = 0.5
        else:
            scaling_factor = 1.0
        
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if name not in self.state.accumulated_gradients:
                    self.state.accumulated_gradients[name] = torch.zeros_like(param.grad)
                
                # Accumulate gradients with memory-aware scaling
                self.state.accumulated_gradients[name] += param.grad.clone() * scaling_factor
                
                # Clear current gradients
                param.grad.zero_()
    
    def _should_perform_optimization(self) -> bool:
        """Check if optimization step should be performed."""
        # Check if accumulation steps reached
        if self.state.accumulation_step >= self.config.accumulation_steps:
            return True
        
        # Check adaptive accumulation schedule
        if self.config.accumulation_schedule and self.state.total_steps < len(self.config.accumulation_schedule):
            target_steps = self.config.accumulation_schedule[self.state.total_steps]
            if self.state.accumulation_step >= target_steps:
                return True
        
        return False
    
    def _perform_optimization_step(self) -> Any:
        """Perform optimization step with accumulated gradients."""
        # Apply accumulated gradients
        self._apply_accumulated_gradients()
        
        # Gradient clipping
        if self.config.gradient_clipping:
            self._clip_gradients()
        
        # NaN handling
        if self.config.nan_handling:
            self._handle_nan_gradients()
        
        # Perform optimization step
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        # Update scheduler
        if self.scheduler is not None:
            self.scheduler.step()
        
        # Reset accumulation state
        self._reset_accumulation_state()
        
        # Update performance metrics
        self.performance_metrics['total_optimizer_steps'] += 1
        self.state.last_optimizer_step = self.state.total_steps
    
    def _apply_accumulated_gradients(self) -> Any:
        """Apply accumulated gradients to model parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.state.accumulated_gradients:
                # Apply accumulated gradients
                param.grad = self.state.accumulated_gradients[name].clone()
                
                # Calculate gradient norm
                grad_norm = param.grad.norm().item()
                self.state.gradient_norms.append(grad_norm)
    
    def _clip_gradients(self) -> Any:
        """Clip gradients to prevent exploding gradients."""
        if self.config.gradient_clipping:
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clipping_norm)
    
    def _handle_nan_gradients(self) -> Any:
        """Handle NaN gradients."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if torch.isnan(param.grad).any():
                    self.state.nan_detected = True
                    self.logger.warning(f"NaN detected in gradients of {name}")
                    param.grad.data.zero_()
    
    def _reset_accumulation_state(self) -> Any:
        """Reset accumulation state after optimization step."""
        self.state.accumulation_step = 0
        self.state.accumulated_loss = 0.0
        
        # Clear accumulated gradients
        for name in self.state.accumulated_gradients:
            self.state.accumulated_gradients[name].zero_()
        
        # Update effective batch size
        self.state.effective_batch_size = self.config.effective_batch_size * self.config.accumulation_steps
    
    def _log_accumulation_progress(self) -> Any:
        """Log accumulation progress."""
        avg_gradient_norm = np.mean(self.state.gradient_norms[-self.config.accumulation_steps:]) if self.state.gradient_norms else 0.0
        
        self.logger.info(
            f"Step {self.state.total_steps}, "
            f"Accumulation: {self.state.accumulation_step}/{self.config.accumulation_steps}, "
            f"Loss: {self.state.accumulated_loss:.4f}, "
            f"Gradient Norm: {avg_gradient_norm:.4f}, "
            f"Effective Batch Size: {self.state.effective_batch_size}"
        )
    
    def get_accumulation_stats(self) -> Dict[str, Any]:
        """Get accumulation statistics."""
        return {
            'current_step': self.state.total_steps,
            'accumulation_step': self.state.accumulation_step,
            'accumulation_steps': self.config.accumulation_steps,
            'effective_batch_size': self.state.effective_batch_size,
            'accumulated_loss': self.state.accumulated_loss,
            'average_gradient_norm': np.mean(self.state.gradient_norms) if self.state.gradient_norms else 0.0,
            'total_optimizer_steps': self.performance_metrics['total_optimizer_steps'],
            'nan_detected': self.state.nan_detected,
            'overflow_detected': self.state.overflow_detected,
            'memory_usage': self.state.memory_usage[-1] if self.state.memory_usage else 0.0
        }
    
    def save_accumulation_state(self, filepath: str):
        """Save accumulation state."""
        state_dict = {
            'config': self.config,
            'state': self.state,
            'performance_metrics': self.performance_metrics,
            'accumulated_gradients': {name: grad.cpu() for name, grad in self.state.accumulated_gradients.items()}
        }
        
        torch.save(state_dict, filepath)
        self.logger.info(f"Accumulation state saved to {filepath}")
    
    def load_accumulation_state(self, filepath: str):
        """Load accumulation state."""
        state_dict = torch.load(filepath, map_location=self.device)
        
        self.config = state_dict['config']
        self.state = state_dict['state']
        self.performance_metrics = state_dict['performance_metrics']
        
        # Load accumulated gradients
        for name, grad in state_dict['accumulated_gradients'].items():
            self.state.accumulated_gradients[name] = grad.to(self.device)
        
        self.logger.info(f"Accumulation state loaded from {filepath}")


class GradientAccumulationTrainer:
    """High-level trainer with gradient accumulation."""
    
    def __init__(self, config: GradientAccumulationConfig):
        self.config = config
        self.accumulator = GradientAccumulator(config)
        self.logger = self.accumulator.logger
        
        # Training components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.device = None
        
        # Training state
        self.current_epoch = 0
        self.training_history = []
    
    def setup_training(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                      criterion: Callable, scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                      device: Optional[torch.device] = None):
        """Setup training components."""
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device or next(model.parameters()).device
        
        # Setup accumulator
        self.accumulator.setup_training(model, optimizer, criterion, scheduler, device)
        
        self.logger.info("Gradient accumulation training setup completed")
    
    def train_step(self, data_batch: torch.Tensor, target_batch: torch.Tensor) -> Dict[str, Any]:
        """Perform training step with gradient accumulation."""
        # Move data to device
        data_batch = data_batch.to(self.device, non_blocking=True)
        target_batch = target_batch.to(self.device, non_blocking=True)
        
        # Forward pass
        output = self.model(data_batch)
        loss = self.criterion(output, target_batch)
        
        # Accumulate gradients
        accumulation_result = self.accumulator.accumulate_gradients(loss)
        
        # Calculate metrics
        metrics = {
            'loss': loss.item(),
            'output_shape': output.shape,
            'target_shape': target_batch.shape,
            'accumulation_step': accumulation_result['accumulation_step'],
            'should_optimize': accumulation_result['should_optimize'],
            'effective_batch_size': accumulation_result['effective_batch_size']
        }
        
        return metrics
    
    def train_epoch(self, train_loader: data.DataLoader) -> Dict[str, Any]:
        """Train for one epoch with gradient accumulation."""
        self.model.train()
        
        epoch_loss = 0.0
        num_batches = len(train_loader)
        optimization_steps = 0
        
        for batch_idx, (data_batch, target_batch) in enumerate(train_loader):
            # Training step
            metrics = self.train_step(data_batch, target_batch)
            
            epoch_loss += metrics['loss']
            
            if metrics['should_optimize']:
                optimization_steps += 1
            
            # Log progress
            if batch_idx % self.config.logging_frequency == 0:
                self.logger.info(
                    f"Epoch {self.current_epoch}, "
                    f"Batch {batch_idx}/{num_batches}, "
                    f"Loss: {metrics['loss']:.4f}, "
                    f"Accumulation: {metrics['accumulation_step']}/{self.config.accumulation_steps}"
                )
        
        avg_loss = epoch_loss / num_batches
        
        return {
            'epoch_loss': avg_loss,
            'num_batches': num_batches,
            'optimization_steps': optimization_steps,
            'effective_batch_size': self.config.effective_batch_size * self.config.accumulation_steps
        }
    
    def validate_epoch(self, val_loader: data.DataLoader) -> Dict[str, Any]:
        """Validate for one epoch."""
        self.model.eval()
        
        val_loss = 0.0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for batch_idx, (data_batch, target_batch) in enumerate(val_loader):
                # Move data to device
                data_batch = data_batch.to(self.device, non_blocking=True)
                target_batch = target_batch.to(self.device, non_blocking=True)
                
                # Forward pass
                output = self.model(data_batch)
                loss = self.criterion(output, target_batch)
                
                val_loss += loss.item()
        
        avg_loss = val_loss / num_batches
        
        return {'val_loss': avg_loss, 'num_batches': num_batches}
    
    def train(self, train_loader: data.DataLoader, val_loader: Optional[data.DataLoader] = None,
              num_epochs: int = 10, save_dir: str = "checkpoints"):
        """Complete training loop with gradient accumulation."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            
            # Validation
            val_metrics = None
            if val_loader is not None and epoch % self.config.validation_frequency == 0:
                val_metrics = self.validate_epoch(val_loader)
            
            # Log results
            self.logger.info(f"Epoch {epoch} - Train Loss: {train_metrics['epoch_loss']:.4f}")
            if val_metrics:
                self.logger.info(f"Epoch {epoch} - Val Loss: {val_metrics['val_loss']:.4f}")
            
            # Save checkpoint
            if epoch % self.config.checkpoint_frequency == 0:
                self.save_checkpoint(save_path / f"checkpoint_epoch_{epoch}.pth", epoch)
            
            # Save accumulation state
            if self.config.save_accumulation_state:
                self.accumulator.save_accumulation_state(
                    save_path / f"accumulation_state_epoch_{epoch}.pth"
                )
        
        # Save final checkpoint
        self.save_checkpoint(save_path / "final_model.pth", num_epochs - 1)
        
        self.logger.info("Training completed")
    
    def save_checkpoint(self, filepath: str, epoch: int):
        """Save training checkpoint."""
        cpu_model_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': cpu_model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'training_history': self.training_history
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, filepath)
        self.logger.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load training checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.training_history = checkpoint.get('training_history', [])
        
        self.logger.info(f"Checkpoint loaded from {filepath}")
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        accumulator_stats = self.accumulator.get_accumulation_stats()
        
        return {
            'current_epoch': self.current_epoch,
            'accumulation_stats': accumulator_stats,
            'training_history': self.training_history
        }


def demonstrate_gradient_accumulation():
    """Demonstrate gradient accumulation."""
    print("Gradient Accumulation Demonstration")
    print("=" * 60)
    
    # Create configuration
    config = GradientAccumulationConfig(
        accumulation_steps=4,
        mode=AccumulationMode.STANDARD,
        strategy=AccumulationStrategy.UNIFORM,
        effective_batch_size=16,
        target_batch_size=64,
        gradient_scaling=True,
        loss_scaling=True,
        gradient_clipping=True,
        clipping_norm=1.0,
        nan_handling=True,
        memory_monitoring=True,
        logging_frequency=5
    )
    
    # Create trainer
    trainer = GradientAccumulationTrainer(config)
    
    # Create simple model
    model = nn.Sequential(
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 10)
    )
    
    # Create optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    # Setup training
    trainer.setup_training(model, optimizer, criterion, scheduler)
    
    # Create dummy dataset
    class DummyDataset(data.Dataset):
        def __init__(self, num_samples=1000) -> Any:
            self.data = torch.randn(num_samples, 784)
            self.targets = torch.randint(0, 10, (num_samples,))
        
        def __len__(self) -> Any:
            return len(self.data)
        
        def __getitem__(self, idx) -> Optional[Dict[str, Any]]:
            return self.data[idx], self.targets[idx]
    
    # Create dataloader
    train_dataset = DummyDataset(1000)
    train_loader = make_loader(train_dataset, batch_size=16, shuffle=True, generator_seed=42)
    
    # Train for a few epochs
    print("\nStarting gradient accumulation training...")
    trainer.train(train_loader, num_epochs=2)
    
    # Get training stats
    stats = trainer.get_training_stats()
    print(f"\nTraining completed!")
    print(f"Effective batch size: {stats['accumulation_stats']['effective_batch_size']}")
    print(f"Total optimizer steps: {stats['accumulation_stats']['total_optimizer_steps']}")
    print(f"Average gradient norm: {stats['accumulation_stats']['average_gradient_norm']:.4f}")


if __name__ == "__main__":
    # Demonstrate gradient accumulation
    demonstrate_gradient_accumulation() 