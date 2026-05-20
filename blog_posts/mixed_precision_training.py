from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import logging
import time
import json
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
import threading
from contextlib import contextmanager
from typing import Any, List, Dict, Optional
import asyncio
"""
🚀 Mixed Precision Training System with torch.cuda.amp
=====================================================

Comprehensive mixed precision training implementation using PyTorch's Automatic Mixed Precision (AMP)
with advanced features for optimal performance and memory efficiency.
"""


# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class MixedPrecisionConfig:
    """Configuration for mixed precision training."""
    
    # Basic settings
    enabled: bool = True
    dtype: torch.dtype = torch.float16
    init_scale: float = 2**16
    growth_factor: float = 2.0
    backoff_factor: float = 0.5
    growth_interval: int = 2000
    
    # Performance settings
    memory_efficient: bool = True
    use_dynamic_loss_scaling: bool = True
    use_grad_scaling: bool = True
    
    # Monitoring
    log_scaling: bool = True
    track_performance: bool = True
    save_scaler_state: bool = True
    
    # Advanced settings
    min_scale: float = 1.0
    max_scale: float = 2**24
    scale_window: int = 2000
    
    # Model-specific settings
    cast_model_outputs: bool = True
    cast_batchnorm: bool = True
    cast_linear: bool = True
    cast_conv: bool = True


class MixedPrecisionTrainer:
    """Advanced mixed precision trainer with comprehensive features."""
    
    def __init__(self, config: MixedPrecisionConfig = None):
        
    """__init__ function."""
self.config = config or MixedPrecisionConfig()
        self.scaler = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Performance tracking
        self.performance_metrics = {
            'training_steps': 0,
            'successful_steps': 0,
            'scaler_updates': 0,
            'scaler_skips': 0,
            'memory_savings': [],
            'speed_improvements': [],
            'loss_scales': []
        }
        
        # Initialize scaler if enabled
        if self.config.enabled and torch.cuda.is_available():
            self._initialize_scaler()
        
        logger.info(f"MixedPrecisionTrainer initialized: enabled={self.config.enabled}")
    
    def _initialize_scaler(self) -> Any:
        """Initialize GradScaler with configuration."""
        try:
            self.scaler = GradScaler(
                init_scale=self.config.init_scale,
                growth_factor=self.config.growth_factor,
                backoff_factor=self.config.backoff_factor,
                growth_interval=self.config.growth_interval
            )
            logger.info(f"✅ GradScaler initialized with scale={self.config.init_scale}")
        except Exception as e:
            logger.error(f"Failed to initialize GradScaler: {e}")
            self.scaler = None
    
    def prepare_model(self, model: nn.Module) -> nn.Module:
        """Prepare model for mixed precision training."""
        if not self.config.enabled:
            return model
        
        try:
            # Move model to device
            model = model.to(self.device)
            
            # Convert model to mixed precision if needed
            if self.config.cast_model_outputs:
                model = self._cast_model_outputs(model)
            
            logger.info("✅ Model prepared for mixed precision training")
            return model
            
        except Exception as e:
            logger.error(f"Failed to prepare model for mixed precision: {e}")
            return model
    
    def _cast_model_outputs(self, model: nn.Module) -> nn.Module:
        """Cast model outputs to appropriate precision."""
        try:
            # Create a wrapper that casts outputs
            class MixedPrecisionWrapper(nn.Module):
                def __init__(self, module) -> Any:
                    super().__init__()
                    self.module = module
                
                def forward(self, *args, **kwargs) -> Any:
                    with autocast():
                        output = self.module(*args, **kwargs)
                    return output
            
            return MixedPrecisionWrapper(model)
            
        except Exception as e:
            logger.warning(f"Failed to cast model outputs: {e}")
            return model
    
    def train_step(self, model: nn.Module, optimizer: optim.Optimizer, 
                  criterion: nn.Module, data, target, 
                  backward_kwargs: Dict[str, Any] = None) -> Dict[str, Any]:
        """Perform a single training step with mixed precision."""
        
        if not self.config.enabled or self.scaler is None:
            # Fallback to regular training
            return self._regular_train_step(model, optimizer, criterion, data, target)
        
        try:
            # Move data to device
            if isinstance(data, torch.Tensor):
                data = data.to(self.device)
            if isinstance(target, torch.Tensor):
                target = target.to(self.device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass with autocast
            with autocast():
                output = model(data)
                loss = criterion(output, target)
            
            # Backward pass with gradient scaling
            if backward_kwargs is None:
                backward_kwargs = {}
            
            self.scaler.scale(loss).backward(**backward_kwargs)
            
            # Optimizer step with gradient scaling
            self.scaler.step(optimizer)
            self.scaler.update()
            
            # Update performance metrics
            self._update_performance_metrics(loss.item())
            
            return {
                'loss': loss.item(),
                'scaler_scale': self.scaler.get_scale(),
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Mixed precision training step failed: {e}")
            return {
                'loss': float('inf'),
                'error': str(e),
                'success': False
            }
    
    def _regular_train_step(self, model: nn.Module, optimizer: optim.Optimizer,
                           criterion: nn.Module, data, target) -> Dict[str, Any]:
        """Regular training step without mixed precision."""
        try:
            # Move data to device
            if isinstance(data, torch.Tensor):
                data = data.to(self.device)
            if isinstance(target, torch.Tensor):
                target = target.to(self.device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            return {
                'loss': loss.item(),
                'scaler_scale': 1.0,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Regular training step failed: {e}")
            return {
                'loss': float('inf'),
                'error': str(e),
                'success': False
            }
    
    def _update_performance_metrics(self, loss: float):
        """Update performance tracking metrics."""
        self.performance_metrics['training_steps'] += 1
        
        if self.scaler:
            current_scale = self.scaler.get_scale()
            self.performance_metrics['loss_scales'].append(current_scale)
            
            if self.config.log_scaling:
                logger.debug(f"Training step {self.performance_metrics['training_steps']}: "
                           f"loss={loss:.4f}, scale={current_scale}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        summary = {
            'config': {
                'enabled': self.config.enabled,
                'dtype': str(self.config.dtype),
                'init_scale': self.config.init_scale,
                'memory_efficient': self.config.memory_efficient
            },
            'performance': self.performance_metrics.copy(),
            'scaler_info': {}
        }
        
        if self.scaler:
            summary['scaler_info'] = {
                'current_scale': self.scaler.get_scale(),
                'growth_factor': self.config.growth_factor,
                'backoff_factor': self.config.backoff_factor
            }
        
        # Calculate memory savings
        if self.performance_metrics['training_steps'] > 0:
            # Estimate memory savings (typically 50% for FP16 vs FP32)
            estimated_savings = 0.5 if self.config.enabled else 0.0
            summary['estimated_memory_savings'] = f"{estimated_savings * 100:.1f}%"
        
        return summary
    
    def save_scaler_state(self, path: str):
        """Save scaler state for checkpointing."""
        if self.scaler and self.config.save_scaler_state:
            try:
                torch.save(self.scaler.state_dict(), path)
                logger.info(f"✅ Scaler state saved to {path}")
            except Exception as e:
                logger.error(f"Failed to save scaler state: {e}")
    
    def load_scaler_state(self, path: str):
        """Load scaler state from checkpoint."""
        if self.scaler and self.config.save_scaler_state:
            try:
                self.scaler.load_state_dict(torch.load(path))
                logger.info(f"✅ Scaler state loaded from {path}")
            except Exception as e:
                logger.error(f"Failed to load scaler state: {e}")


class AdaptiveMixedPrecisionTrainer(MixedPrecisionTrainer):
    """Adaptive mixed precision trainer that adjusts settings based on performance."""
    
    def __init__(self, config: MixedPrecisionConfig = None):
        
    """__init__ function."""
super().__init__(config)
        self.adaptation_history = []
        self.performance_threshold = 0.1  # 10% performance improvement threshold
        
    def adapt_config(self, current_performance: float, baseline_performance: float):
        """Adapt mixed precision configuration based on performance."""
        try:
            performance_improvement = (current_performance - baseline_performance) / baseline_performance
            
            if performance_improvement < -self.performance_threshold:
                # Performance degraded, reduce mixed precision aggressiveness
                self._reduce_mixed_precision_aggressiveness()
            elif performance_improvement > self.performance_threshold:
                # Performance improved, increase mixed precision aggressiveness
                self._increase_mixed_precision_aggressiveness()
            
            self.adaptation_history.append({
                'timestamp': datetime.now().isoformat(),
                'performance_improvement': performance_improvement,
                'current_config': {
                    'init_scale': self.config.init_scale,
                    'growth_factor': self.config.growth_factor,
                    'memory_efficient': self.config.memory_efficient
                }
            })
            
        except Exception as e:
            logger.warning(f"Failed to adapt mixed precision config: {e}")
    
    def _reduce_mixed_precision_aggressiveness(self) -> Any:
        """Reduce mixed precision aggressiveness."""
        try:
            # Increase init scale for more conservative scaling
            self.config.init_scale = min(self.config.max_scale, self.config.init_scale * 2)
            
            # Reduce growth factor
            self.config.growth_factor = max(1.1, self.config.growth_factor * 0.9)
            
            # Reinitialize scaler with new settings
            if self.scaler:
                self._initialize_scaler()
            
            logger.info(f"Reduced mixed precision aggressiveness: "
                       f"init_scale={self.config.init_scale}, growth_factor={self.config.growth_factor}")
            
        except Exception as e:
            logger.warning(f"Failed to reduce mixed precision aggressiveness: {e}")
    
    def _increase_mixed_precision_aggressiveness(self) -> Any:
        """Increase mixed precision aggressiveness."""
        try:
            # Decrease init scale for more aggressive scaling
            self.config.init_scale = max(self.config.min_scale, self.config.init_scale * 0.5)
            
            # Increase growth factor
            self.config.growth_factor = min(4.0, self.config.growth_factor * 1.1)
            
            # Reinitialize scaler with new settings
            if self.scaler:
                self._initialize_scaler()
            
            logger.info(f"Increased mixed precision aggressiveness: "
                       f"init_scale={self.config.init_scale}, growth_factor={self.config.growth_factor}")
            
        except Exception as e:
            logger.warning(f"Failed to increase mixed precision aggressiveness: {e}")
    
    def get_adaptation_summary(self) -> Dict[str, Any]:
        """Get adaptation history summary."""
        return {
            'adaptation_history': self.adaptation_history,
            'total_adaptations': len(self.adaptation_history),
            'current_config': {
                'init_scale': self.config.init_scale,
                'growth_factor': self.config.growth_factor,
                'memory_efficient': self.config.memory_efficient
            }
        }


# Utility functions for mixed precision training
def create_mixed_precision_config(enabled: bool = True, 
                                 dtype: torch.dtype = torch.float16,
                                 init_scale: float = 2**16,
                                 memory_efficient: bool = True) -> MixedPrecisionConfig:
    """Create mixed precision configuration."""
    return MixedPrecisionConfig(
        enabled=enabled,
        dtype=dtype,
        init_scale=init_scale,
        memory_efficient=memory_efficient
    )


def should_use_mixed_precision(model: nn.Module, batch_size: int, 
                              available_memory: float) -> bool:
    """Determine if mixed precision should be used."""
    try:
        # Check if CUDA is available
        if not torch.cuda.is_available():
            return False
        
        # Check if model is large enough to benefit
        total_params = sum(p.numel() for p in model.parameters())
        if total_params < 1e6:  # Less than 1M parameters
            return False
        
        # Check if batch size is large enough
        if batch_size < 8:
            return False
        
        # Check if memory is constrained
        if available_memory < 4.0:  # Less than 4GB
            return True
        
        # Check if model has operations that benefit from mixed precision
        has_conv = any(isinstance(m, nn.Conv2d) for m in model.modules())
        has_linear = any(isinstance(m, nn.Linear) for m in model.modules())
        
        return has_conv or has_linear
        
    except Exception as e:
        logger.warning(f"Failed to determine mixed precision usage: {e}")
        return True  # Default to True for safety


def optimize_mixed_precision_settings(model: nn.Module, 
                                    batch_size: int,
                                    available_memory: float) -> MixedPrecisionConfig:
    """Optimize mixed precision settings for the given model and hardware."""
    try:
        # Determine if mixed precision should be used
        use_mixed_precision = should_use_mixed_precision(model, batch_size, available_memory)
        
        if not use_mixed_precision:
            return MixedPrecisionConfig(enabled=False)
        
        # Calculate optimal settings based on model and memory
        total_params = sum(p.numel() for p in model.parameters())
        
        # Adjust init scale based on model size
        if total_params > 1e8:  # Large model (>100M parameters)
            init_scale = 2**20
        elif total_params > 1e7:  # Medium model (>10M parameters)
            init_scale = 2**18
        else:
            init_scale = 2**16
        
        # Adjust growth factor based on memory
        if available_memory < 8.0:
            growth_factor = 1.5  # More conservative
        else:
            growth_factor = 2.0  # More aggressive
        
        # Enable memory efficient mode for large models
        memory_efficient = total_params > 1e7
        
        config = MixedPrecisionConfig(
            enabled=True,
            init_scale=init_scale,
            growth_factor=growth_factor,
            memory_efficient=memory_efficient
        )
        
        logger.info(f"Optimized mixed precision settings: "
                   f"init_scale={init_scale}, growth_factor={growth_factor}, "
                   f"memory_efficient={memory_efficient}")
        
        return config
        
    except Exception as e:
        logger.error(f"Failed to optimize mixed precision settings: {e}")
        return MixedPrecisionConfig(enabled=True)  # Default fallback


@contextmanager
def mixed_precision_context(enabled: bool = True, dtype: torch.dtype = torch.float16):
    """Context manager for mixed precision operations."""
    if enabled and torch.cuda.is_available():
        with autocast():
            yield
    else:
        yield


def benchmark_mixed_precision(model: nn.Module, 
                            data: torch.Tensor,
                            num_iterations: int = 100) -> Dict[str, float]:
    """Benchmark mixed precision vs regular precision."""
    try:
        device = next(model.parameters()).device
        model.eval()
        
        # Warm up
        with torch.no_grad():
            for _ in range(10):
                _ = model(data)
        
        torch.cuda.synchronize()
        
        # Benchmark regular precision
        start_time = time.time()
        memory_before = torch.cuda.memory_allocated()
        
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(data)
        
        torch.cuda.synchronize()
        regular_time = time.time() - start_time
        regular_memory = torch.cuda.memory_allocated() - memory_before
        
        # Benchmark mixed precision
        start_time = time.time()
        memory_before = torch.cuda.memory_allocated()
        
        with torch.no_grad():
            with autocast():
                for _ in range(num_iterations):
                    _ = model(data)
        
        torch.cuda.synchronize()
        mixed_time = time.time() - start_time
        mixed_memory = torch.cuda.memory_allocated() - memory_before
        
        # Calculate improvements
        speed_improvement = (regular_time - mixed_time) / regular_time * 100
        memory_improvement = (regular_memory - mixed_memory) / regular_memory * 100
        
        results = {
            'regular_time': regular_time,
            'mixed_time': mixed_time,
            'regular_memory': regular_memory / (1024**3),  # GB
            'mixed_memory': mixed_memory / (1024**3),  # GB
            'speed_improvement_percent': speed_improvement,
            'memory_improvement_percent': memory_improvement,
            'iterations': num_iterations
        }
        
        logger.info(f"Mixed precision benchmark: "
                   f"Speed improvement: {speed_improvement:.1f}%, "
                   f"Memory improvement: {memory_improvement:.1f}%")
        
        return results
        
    except Exception as e:
        logger.error(f"Failed to benchmark mixed precision: {e}")
        return {}


# Enhanced training function with mixed precision
def train_with_mixed_precision(model: nn.Module,
                              train_loader: torch.utils.data.DataLoader,
                              optimizer: torch.optim.Optimizer,
                              criterion: nn.Module,
                              num_epochs: int = 10,
                              config: MixedPrecisionConfig = None,
                              adaptive: bool = True) -> Dict[str, Any]:
    """Train model with comprehensive mixed precision support."""
    
    try:
        # Setup mixed precision trainer
        if adaptive:
            trainer = AdaptiveMixedPrecisionTrainer(config)
        else:
            trainer = MixedPrecisionTrainer(config)
        
        # Prepare model
        model = trainer.prepare_model(model)
        
        # Training metrics
        training_metrics = {
            'epochs': [],
            'train_losses': [],
            'scaler_scales': [],
            'memory_usage': [],
            'training_time': [],
            'mixed_precision_summary': {}
        }
        
        total_start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            epoch_loss = 0.0
            num_batches = 0
            
            model.train()
            
            for batch_idx, (data, target) in enumerate(train_loader):
                try:
                    # Training step with mixed precision
                    step_result = trainer.train_step(model, optimizer, criterion, data, target)
                    
                    if step_result['success']:
                        epoch_loss += step_result['loss']
                        num_batches += 1
                        
                        # Log progress
                        if batch_idx % 10 == 0:
                            logger.info(f"Epoch {epoch + 1}, Batch {batch_idx}: "
                                       f"Loss = {step_result['loss']:.4f}, "
                                       f"Scale = {step_result['scaler_scale']:.0f}")
                    else:
                        logger.warning(f"Training step failed at batch {batch_idx}: {step_result.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    logger.error(f"Training step failed at batch {batch_idx}: {e}")
                    continue
            
            # Calculate epoch metrics
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            epoch_time = time.time() - epoch_start_time
            
            # Record metrics
            training_metrics['epochs'].append(epoch + 1)
            training_metrics['train_losses'].append(avg_loss)
            training_metrics['training_time'].append(epoch_time)
            
            # Get scaler scale
            if trainer.scaler:
                training_metrics['scaler_scales'].append(trainer.scaler.get_scale())
            
            # Memory usage
            if torch.cuda.is_available():
                memory_usage = torch.cuda.memory_allocated() / (1024**3)
                training_metrics['memory_usage'].append(memory_usage)
            
            logger.info(f"Epoch {epoch + 1}/{num_epochs}: "
                       f"Loss = {avg_loss:.4f}, Time = {epoch_time:.2f}s")
        
        total_time = time.time() - total_start_time
        training_metrics['total_training_time'] = total_time
        training_metrics['final_loss'] = training_metrics['train_losses'][-1] if training_metrics['train_losses'] else 0.0
        training_metrics['mixed_precision_summary'] = trainer.get_performance_summary()
        
        if adaptive:
            training_metrics['adaptation_summary'] = trainer.get_adaptation_summary()
        
        logger.info(f"✅ Mixed precision training completed in {total_time:.2f}s")
        return training_metrics
        
    except Exception as e:
        logger.error(f"Mixed precision training failed: {e}")
        return {'error': str(e)}


# Gradio interface functions
def train_model_with_mixed_precision_interface(model_type: str, num_epochs: int, 
                                             batch_size: int, learning_rate: float,
                                             use_mixed_precision: bool, adaptive: bool) -> str:
    """Train model with mixed precision for the Gradio interface."""
    try:
        # Create model based on type
        if model_type == "linear":
            model = torch.nn.Linear(10, 2)
        elif model_type == "mlp":
            model = torch.nn.Sequential(
                torch.nn.Linear(10, 64), torch.nn.ReLU(),
                torch.nn.Linear(64, 32), torch.nn.ReLU(),
                torch.nn.Linear(32, 2)
            )
        else:
            model = torch.nn.Linear(10, 2)
        
        # Create dataset and data loader
        X = torch.randn(1000, 10)
        y = torch.randint(0, 2, (1000,))
        dataset = torch.utils.data.TensorDataset(X, y)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Setup optimizer and loss
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Create mixed precision config
        if use_mixed_precision:
            available_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 8.0
            config = optimize_mixed_precision_settings(model, batch_size, available_memory)
        else:
            config = MixedPrecisionConfig(enabled=False)
        
        # Train with mixed precision
        training_metrics = train_with_mixed_precision(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            num_epochs=num_epochs,
            config=config,
            adaptive=adaptive
        )
        
        return json.dumps(training_metrics, indent=2, default=str)
        
    except Exception as e:
        logger.error(f"Mixed precision training error: {e}")
        return json.dumps({"error": str(e)}, indent=2)


def benchmark_mixed_precision_interface(model_type: str, batch_size: int) -> str:
    """Benchmark mixed precision performance for the Gradio interface."""
    try:
        # Create model based on type
        if model_type == "linear":
            model = torch.nn.Linear(10, 2)
        elif model_type == "mlp":
            model = torch.nn.Sequential(
                torch.nn.Linear(10, 64), torch.nn.ReLU(),
                torch.nn.Linear(64, 32), torch.nn.ReLU(),
                torch.nn.Linear(32, 2)
            )
        else:
            model = torch.nn.Linear(10, 2)
        
        # Move model to GPU
        if torch.cuda.is_available():
            model = model.cuda()
        
        # Create sample data
        data = torch.randn(batch_size, 10)
        if torch.cuda.is_available():
            data = data.cuda()
        
        # Run benchmark
        benchmark_results = benchmark_mixed_precision(model, data, num_iterations=50)
        
        return json.dumps(benchmark_results, indent=2, default=str)
        
    except Exception as e:
        logger.error(f"Benchmark error: {e}")
        return json.dumps({"error": str(e)}, indent=2)


def get_mixed_precision_recommendations() -> str:
    """Get mixed precision recommendations based on system."""
    try:
        recommendations = {
            'system_info': {},
            'recommendations': {}
        }
        
        if torch.cuda.is_available():
            # Get GPU information
            gpu_props = torch.cuda.get_device_properties(0)
            total_memory = gpu_props.total_memory / (1024**3)
            
            recommendations['system_info'] = {
                'gpu_name': gpu_props.name,
                'total_memory_gb': total_memory,
                'compute_capability': f"{gpu_props.major}.{gpu_props.minor}",
                'cuda_available': True
            }
            
            # Generate recommendations
            if total_memory >= 16:
                recommendations['recommendations']['use_mixed_precision'] = True
                recommendations['recommendations']['init_scale'] = 2**16
                recommendations['recommendations']['growth_factor'] = 2.0
                recommendations['recommendations']['memory_efficient'] = False
            elif total_memory >= 8:
                recommendations['recommendations']['use_mixed_precision'] = True
                recommendations['recommendations']['init_scale'] = 2**18
                recommendations['recommendations']['growth_factor'] = 1.5
                recommendations['recommendations']['memory_efficient'] = True
            else:
                recommendations['recommendations']['use_mixed_precision'] = True
                recommendations['recommendations']['init_scale'] = 2**20
                recommendations['recommendations']['growth_factor'] = 1.2
                recommendations['recommendations']['memory_efficient'] = True
            
            recommendations['recommendations']['adaptive'] = True
            recommendations['recommendations']['batch_size'] = min(64, int(total_memory * 4))
        else:
            recommendations['system_info']['cuda_available'] = False
            recommendations['recommendations']['use_mixed_precision'] = False
        
        return json.dumps(recommendations, indent=2, default=str)
        
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)


if __name__ == "__main__":
    # Example usage
    print("🚀 Mixed Precision Training System")
    print("=" * 50)
    
    # Create a simple model
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 2)
    )
    
    if torch.cuda.is_available():
        model = model.cuda()
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        
        # Benchmark mixed precision
        data = torch.randn(32, 10).cuda()
        results = benchmark_mixed_precision(model, data)
        print(f"📊 Benchmark results: {results}")
        
        # Get recommendations
        recommendations = get_mixed_precision_recommendations()
        print(f"💡 Recommendations: {recommendations}")
    else:
        print("❌ CUDA not available") 