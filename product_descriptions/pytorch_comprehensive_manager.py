from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DataParallel, DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.checkpoint import checkpoint
from torch.func import functional_call, vmap, grad
from torch.export import export
from torch._dynamo import optimize
import torch._dynamo as dynamo
import numpy as np
import logging
import os
import time
import gc
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import structlog
from contextlib import contextmanager
import psutil
import GPUtil
        import threading
        import time
from typing import Any, List, Dict, Optional
import asyncio
"""
Comprehensive PyTorch Management System

This module provides a unified PyTorch management system that consolidates
all PyTorch functionality across the codebase with:

- Advanced PyTorch 2.0+ optimizations (torch.compile, flash attention)
- Comprehensive device management (CUDA, MPS, CPU)
- Memory optimization and monitoring
- Mixed precision training with automatic scaling
- Distributed training support
- Model compilation and optimization
- Performance profiling and debugging
- Security and validation features
- Integration with existing modules
"""


# Modern PyTorch 2.0+ features

# Additional imports

logger = structlog.get_logger(__name__)


class DeviceType(Enum):
    """Device type enumeration."""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
    AUTO = "auto"


class OptimizationLevel(Enum):
    """Optimization level enumeration."""
    NONE = "none"
    BASIC = "basic"
    ADVANCED = "advanced"
    MAXIMUM = "maximum"


@dataclass
class PyTorchConfig:
    """Comprehensive PyTorch configuration."""
    
    # Device configuration
    device: DeviceType = DeviceType.AUTO
    num_gpus: int = 1
    distributed_training: bool = False
    backend: str = "nccl"  # "nccl" for GPU, "gloo" for CPU
    
    # Memory and optimization
    memory_fraction: float = 0.9
    enable_cudnn_benchmark: bool = True
    enable_cudnn_deterministic: bool = False
    enable_tf32: bool = True  # For Ampere+ GPUs
    enable_amp: bool = True  # Automatic Mixed Precision
    enable_flash_attention: bool = True
    enable_compile: bool = True  # PyTorch 2.0+ compile
    
    # Training optimizations
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    use_gradient_checkpointing: bool = False
    use_checkpoint: bool = False
    
    # Data loading
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    
    # Performance monitoring
    enable_profiling: bool = False
    enable_memory_tracking: bool = True
    log_memory_usage: bool = True
    profile_memory: bool = False
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = False
    
    # Security
    enable_security_checks: bool = True
    validate_inputs: bool = True
    sanitize_outputs: bool = True
    
    # Debugging
    enable_debugging: bool = False
    enable_gradient_checking: bool = False
    enable_anomaly_detection: bool = False


class PyTorchDeviceManager:
    """Manages PyTorch device configuration and optimization."""
    
    def __init__(self, config: PyTorchConfig):
        
    """__init__ function."""
self.config = config
        self.device = self._setup_device()
        self._setup_device_optimizations()
        
    def _setup_device(self) -> torch.device:
        """Setup and validate device."""
        if self.config.device == DeviceType.AUTO:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(self.config.device.value)
        
        # Validate device
        if device.type == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available")
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"PyTorch version: {torch.__version__}")
            logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        elif device.type == 'mps':
            logger.info("Using MPS (Apple Silicon) device")
        else:
            logger.info("Using CPU device")
            
        return device
    
    def _setup_device_optimizations(self) -> None:
        """Setup device-specific optimizations."""
        if self.device.type == 'cuda':
            self._setup_cuda_optimizations()
        elif self.device.type == 'mps':
            self._setup_mps_optimizations()
        
        # Set random seed for reproducibility
        self._set_seed()
    
    def _setup_cuda_optimizations(self) -> None:
        """Setup CUDA-specific optimizations."""
        # Enable cuDNN benchmark for better performance
        if self.config.enable_cudnn_benchmark:
            torch.backends.cudnn.benchmark = True
            logger.info("Enabled cuDNN benchmark")
        
        # Enable deterministic mode if requested
        if self.config.enable_cudnn_deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            logger.info("Enabled deterministic mode")
        
        # Enable TF32 for Ampere+ GPUs
        if self.config.enable_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("Enabled TF32 for Ampere+ GPUs")
        
        # Enable flash attention
        if self.config.enable_flash_attention:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
            logger.info("Enabled flash attention optimizations")
        
        # Set memory fraction
        if self.config.memory_fraction < 1.0:
            torch.cuda.set_per_process_memory_fraction(self.config.memory_fraction)
            logger.info(f"Set GPU memory fraction to {self.config.memory_fraction}")
    
    def _setup_mps_optimizations(self) -> None:
        """Setup MPS-specific optimizations."""
        # MPS optimizations for Apple Silicon
        logger.info("MPS optimizations applied")
    
    def _set_seed(self) -> None:
        """Set random seed for reproducibility."""
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.seed)
            torch.cuda.manual_seed_all(self.config.seed)
        np.random.seed(self.config.seed)
        logger.info(f"Set random seed to {self.config.seed}")
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get comprehensive device information."""
        info = {
            'device_type': self.device.type,
            'device_index': self.device.index,
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'mps_available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        }
        
        if self.device.type == 'cuda':
            info.update({
                'gpu_name': torch.cuda.get_device_name(),
                'gpu_memory_total': torch.cuda.get_device_properties(0).total_memory,
                'gpu_memory_allocated': torch.cuda.memory_allocated(),
                'gpu_memory_reserved': torch.cuda.memory_reserved(),
                'gpu_count': torch.cuda.device_count()
            })
        
        return info


class PyTorchMemoryManager:
    """Manages PyTorch memory optimization and monitoring."""
    
    def __init__(self, device_manager: PyTorchDeviceManager):
        
    """__init__ function."""
self.device_manager = device_manager
        self.device = device_manager.device
        self.memory_stats = {}
        
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics."""
        stats = {
            'system_ram': {
                'total': psutil.virtual_memory().total,
                'available': psutil.virtual_memory().available,
                'used': psutil.virtual_memory().used,
                'percent': psutil.virtual_memory().percent
            }
        }
        
        if self.device.type == 'cuda':
            stats['gpu_memory'] = {
                'allocated': torch.cuda.memory_allocated(),
                'reserved': torch.cuda.memory_reserved(),
                'max_allocated': torch.cuda.max_memory_allocated(),
                'max_reserved': torch.cuda.max_memory_reserved(),
                'total': torch.cuda.get_device_properties(0).total_memory
            }
            
            # Get GPU utilization
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    stats['gpu_utilization'] = {
                        'load': gpus[0].load * 100,
                        'memory_used': gpus[0].memoryUsed,
                        'memory_total': gpus[0].memoryTotal,
                        'temperature': gpus[0].temperature
                    }
            except Exception as e:
                logger.warning(f"Could not get GPU utilization: {e}")
        
        return stats
    
    def clear_cache(self) -> None:
        """Clear PyTorch cache."""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("Cleared PyTorch cache")
    
    def monitor_memory(self, interval: float = 1.0) -> None:
        """Monitor memory usage continuously."""
        
        def monitor():
            
    """monitor function."""
while True:
                stats = self.get_memory_stats()
                logger.info(f"Memory stats: {stats}")
                time.sleep(interval)
        
        thread = threading.Thread(target=monitor, daemon=True)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        thread.start()
        logger.info(f"Started memory monitoring with {interval}s interval")
    
    @contextmanager
    def memory_tracking(self, operation_name: str):
        """Context manager for memory tracking."""
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        
        start_stats = self.get_memory_stats()
        start_time = time.time()
        
        try:
            yield
        finally:
            end_time = time.time()
            end_stats = self.get_memory_stats()
            
            duration = end_time - start_time
            
            if self.device.type == 'cuda':
                peak_memory = torch.cuda.max_memory_allocated()
                logger.info(f"{operation_name}: Duration={duration:.3f}s, Peak GPU Memory={peak_memory/1e9:.2f}GB")
            else:
                logger.info(f"{operation_name}: Duration={duration:.3f}s")


class PyTorchOptimizer:
    """Advanced PyTorch optimization utilities."""
    
    def __init__(self, device_manager: PyTorchDeviceManager):
        
    """__init__ function."""
self.device_manager = device_manager
        self.device = device_manager.device
        self.config = device_manager.config
    
    def compile_model(self, model: nn.Module, mode: str = "reduce-overhead") -> nn.Module:
        """Compile model using torch.compile."""
        if not self.config.enable_compile:
            return model
        
        try:
            compiled_model = torch.compile(
                model,
                mode=mode,
                fullgraph=True,
                dynamic=True
            )
            logger.info(f"Model compiled successfully with mode: {mode}")
            return compiled_model
        except Exception as e:
            logger.warning(f"Model compilation failed: {e}, using original model")
            return model
    
    def optimize_model(self, model: nn.Module, optimization_level: OptimizationLevel) -> nn.Module:
        """Apply comprehensive model optimizations."""
        if optimization_level == OptimizationLevel.NONE:
            return model
        
        # Basic optimizations
        if optimization_level.value >= OptimizationLevel.BASIC.value:
            model.eval()  # Set to evaluation mode for inference
        
        # Advanced optimizations
        if optimization_level.value >= OptimizationLevel.ADVANCED.value:
            # Use channels_last memory format for better performance
            if self.device.type == 'cuda':
                model = model.to(memory_format=torch.channels_last)
            
            # Enable gradient checkpointing if requested
            if self.config.use_gradient_checkpointing:
                model.gradient_checkpointing_enable()
        
        # Maximum optimizations
        if optimization_level.value >= OptimizationLevel.MAXIMUM.value:
            # Compile model
            model = self.compile_model(model, mode="max-autotune")
            
            # Quantization for CPU
            if self.device.type == 'cpu':
                model = torch.quantization.quantize_dynamic(
                    model, {nn.Linear}, dtype=torch.qint8
                )
        
        return model
    
    def create_optimizer(self, model: nn.Module, lr: float = 1e-3, 
                        optimizer_type: str = "adamw") -> optim.Optimizer:
        """Create optimized optimizer."""
        if optimizer_type.lower() == "adamw":
            return optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        elif optimizer_type.lower() == "adam":
            return optim.Adam(model.parameters(), lr=lr)
        elif optimizer_type.lower() == "sgd":
            return optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    def create_scheduler(self, optimizer: optim.Optimizer, 
                        scheduler_type: str = "cosine", 
                        num_training_steps: int = 1000) -> Any:
        """Create learning rate scheduler."""
        if scheduler_type.lower() == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_training_steps)
        elif scheduler_type.lower() == "linear":
            return optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=num_training_steps)
        elif scheduler_type.lower() == "step":
            return optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")


class PyTorchTrainer:
    """Advanced PyTorch training utilities."""
    
    def __init__(self, device_manager: PyTorchDeviceManager, 
                 memory_manager: PyTorchMemoryManager,
                 optimizer: PyTorchOptimizer):
        
    """__init__ function."""
self.device_manager = device_manager
        self.memory_manager = memory_manager
        self.optimizer = optimizer
        self.device = device_manager.device
        self.config = device_manager.config
        
        # Setup mixed precision
        self.scaler = None
        if self.config.enable_amp and self.device.type in ['cuda', 'mps']:
            self.scaler = GradScaler()
            logger.info("Initialized Automatic Mixed Precision (AMP)")
        
        # Setup TensorBoard
        self.writer = None
        if self.config.enable_profiling:
            self.writer = SummaryWriter(log_dir='./logs/pytorch')
            logger.info("Initialized TensorBoard writer")
    
    def train_step(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        batch: Dict[str, torch.Tensor],
        loss_fn: Callable,
        gradient_accumulation_steps: int = 1
    ) -> Dict[str, float]:
        """Optimized training step with mixed precision."""
        
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Forward pass with mixed precision
        if self.scaler is not None:
            with autocast():
                outputs = model(**batch)
                loss = loss_fn(outputs, batch['labels']) if 'labels' in batch else outputs
                scaled_loss = loss / gradient_accumulation_steps
            
            # Backward pass
            self.scaler.scale(scaled_loss).backward()
            
            # Gradient accumulation
            if (optimizer.param_groups[0]['step'] + 1) % gradient_accumulation_steps == 0:
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad()
        else:
            # Standard precision training
            outputs = model(**batch)
            loss = loss_fn(outputs, batch['labels']) if 'labels' in batch else outputs
            loss.backward()
            
            if (optimizer.param_groups[0]['step'] + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
        
        return {'loss': loss.item()}
    
    def validate_step(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        loss_fn: Callable
    ) -> Dict[str, float]:
        """Validation step."""
        model.eval()
        with torch.no_grad():
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            if self.scaler is not None:
                with autocast():
                    outputs = model(**batch)
                    loss = loss_fn(outputs, batch['labels']) if 'labels' in batch else outputs
            else:
                outputs = model(**batch)
                loss = loss_fn(outputs, batch['labels']) if 'labels' in batch else outputs
        
        return {'loss': loss.item()}
    
    @contextmanager
    def profiling_context(self, operation_name: str):
        """Context manager for profiling."""
        if not self.config.enable_profiling:
            yield
            return
        
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True
        ) as prof:
            with record_function(operation_name):
                yield
        
        # Save profiling results
        prof.export_chrome_trace(f"trace_{operation_name}.json")
        logger.info(f"Profiling results saved for {operation_name}")


class PyTorchSecurityManager:
    """Manages PyTorch security and validation."""
    
    def __init__(self, config: PyTorchConfig):
        
    """__init__ function."""
self.config = config
    
    def validate_inputs(self, inputs: Dict[str, torch.Tensor]) -> bool:
        """Validate input tensors for security."""
        if not self.config.validate_inputs:
            return True
        
        for key, tensor in inputs.items():
            if not isinstance(tensor, torch.Tensor):
                logger.warning(f"Input {key} is not a tensor")
                return False
            
            # Check for NaN or Inf values
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                logger.warning(f"Input {key} contains NaN or Inf values")
                return False
            
            # Check for reasonable value ranges
            if tensor.numel() > 0:
                if torch.abs(tensor).max() > 1e6:
                    logger.warning(f"Input {key} has very large values")
                    return False
        
        return True
    
    def sanitize_outputs(self, outputs: torch.Tensor) -> torch.Tensor:
        """Sanitize output tensors."""
        if not self.config.sanitize_outputs:
            return outputs
        
        # Replace NaN and Inf values
        outputs = torch.nan_to_num(outputs, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return outputs
    
    def check_model_security(self, model: nn.Module) -> Dict[str, bool]:
        """Check model for security issues."""
        security_checks = {
            'has_nan_weights': False,
            'has_inf_weights': False,
            'has_large_weights': False,
            'is_valid': True
        }
        
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                security_checks['has_nan_weights'] = True
                security_checks['is_valid'] = False
                logger.warning(f"Parameter {name} contains NaN values")
            
            if torch.isinf(param).any():
                security_checks['has_inf_weights'] = True
                security_checks['is_valid'] = False
                logger.warning(f"Parameter {name} contains Inf values")
            
            if torch.abs(param).max() > 1e6:
                security_checks['has_large_weights'] = True
                logger.warning(f"Parameter {name} has very large values")
        
        return security_checks


class PyTorchDebugger:
    """Advanced PyTorch debugging utilities."""
    
    def __init__(self, config: PyTorchConfig):
        
    """__init__ function."""
self.config = config
    
    def enable_debugging(self) -> None:
        """Enable PyTorch debugging features."""
        if not self.config.enable_debugging:
            return
        
        # Enable anomaly detection
        if self.config.enable_anomaly_detection:
            torch.autograd.set_detect_anomaly(True)
            logger.info("Enabled autograd anomaly detection")
        
        # Enable gradient checking
        if self.config.enable_gradient_checking:
            torch.autograd.set_detect_anomaly(True)
            logger.info("Enabled gradient checking")
    
    def check_gradients(self, model: nn.Module) -> Dict[str, Any]:
        """Check model gradients."""
        gradient_stats = {
            'total_params': 0,
            'params_with_grad': 0,
            'grad_norm': 0.0,
            'max_grad': 0.0,
            'min_grad': 0.0,
            'has_nan_grad': False,
            'has_inf_grad': False
        }
        
        total_norm = 0.0
        max_grad = float('-inf')
        min_grad = float('inf')
        
        for param in model.parameters():
            gradient_stats['total_params'] += param.numel()
            
            if param.grad is not None:
                gradient_stats['params_with_grad'] += param.numel()
                
                # Check for NaN and Inf
                if torch.isnan(param.grad).any():
                    gradient_stats['has_nan_grad'] = True
                
                if torch.isinf(param.grad).any():
                    gradient_stats['has_inf_grad'] = True
                
                # Calculate gradient norm
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                
                # Track max/min gradients
                max_grad = max(max_grad, param.grad.data.max().item())
                min_grad = min(min_grad, param.grad.data.min().item())
        
        gradient_stats['grad_norm'] = total_norm ** 0.5
        gradient_stats['max_grad'] = max_grad
        gradient_stats['min_grad'] = min_grad
        
        return gradient_stats


class ComprehensivePyTorchManager:
    """Comprehensive PyTorch management system."""
    
    def __init__(self, config: PyTorchConfig):
        
    """__init__ function."""
self.config = config
        
        # Initialize components
        self.device_manager = PyTorchDeviceManager(config)
        self.memory_manager = PyTorchMemoryManager(self.device_manager)
        self.optimizer = PyTorchOptimizer(self.device_manager)
        self.trainer = PyTorchTrainer(self.device_manager, self.memory_manager, self.optimizer)
        self.security_manager = PyTorchSecurityManager(config)
        self.debugger = PyTorchDebugger(config)
        
        # Enable debugging if requested
        self.debugger.enable_debugging()
        
        logger.info("Comprehensive PyTorch Manager initialized")
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information."""
        return {
            'device_info': self.device_manager.get_device_info(),
            'memory_stats': self.memory_manager.get_memory_stats(),
            'config': {
                'device': self.config.device.value,
                'enable_amp': self.config.enable_amp,
                'enable_compile': self.config.enable_compile,
                'enable_flash_attention': self.config.enable_flash_attention,
                'distributed_training': self.config.distributed_training
            }
        }
    
    def optimize_model(self, model: nn.Module, 
                      optimization_level: OptimizationLevel = OptimizationLevel.ADVANCED) -> nn.Module:
        """Apply comprehensive model optimization."""
        # Security check
        security_checks = self.security_manager.check_model_security(model)
        if not security_checks['is_valid']:
            logger.warning("Model security checks failed")
        
        # Move to device
        model = model.to(self.device_manager.device)
        
        # Apply optimizations
        model = self.optimizer.optimize_model(model, optimization_level)
        
        return model
    
    def create_training_pipeline(self, model: nn.Module, 
                               lr: float = 1e-3,
                               optimizer_type: str = "adamw",
                               scheduler_type: str = "cosine") -> Dict[str, Any]:
        """Create complete training pipeline."""
        # Optimize model
        model = self.optimize_model(model)
        
        # Create optimizer and scheduler
        optimizer = self.optimizer.create_optimizer(model, lr, optimizer_type)
        scheduler = self.optimizer.create_scheduler(optimizer, scheduler_type)
        
        return {
            'model': model,
            'optimizer': optimizer,
            'scheduler': scheduler,
            'trainer': self.trainer,
            'memory_manager': self.memory_manager
        }
    
    def profile_model(self, model: nn.Module, input_shape: Tuple[int, ...]) -> Dict[str, Any]:
        """Profile model performance."""
        model = model.to(self.device_manager.device)
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(input_shape).to(self.device_manager.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)
        
        # Profile
        with self.trainer.profiling_context("model_inference"):
            with torch.no_grad():
                start_time = time.time()
                output = model(dummy_input)
                end_time = time.time()
        
        return {
            'inference_time': end_time - start_time,
            'output_shape': output.shape,
            'memory_stats': self.memory_manager.get_memory_stats()
        }


# Utility functions
def setup_pytorch_environment(config: PyTorchConfig) -> ComprehensivePyTorchManager:
    """Setup complete PyTorch environment."""
    return ComprehensivePyTorchManager(config)


def get_optimal_config(device_type: DeviceType = DeviceType.AUTO) -> PyTorchConfig:
    """Get optimal configuration for given device type."""
    config = PyTorchConfig(device=device_type)
    
    if device_type == DeviceType.CUDA:
        config.enable_amp = True
        config.enable_compile = True
        config.enable_flash_attention = True
        config.enable_tf32 = True
    elif device_type == DeviceType.MPS:
        config.enable_amp = True
        config.enable_compile = True
    else:  # CPU
        config.enable_compile = True
        config.num_workers = min(8, os.cpu_count() or 4)
    
    return config


if __name__ == "__main__":
    # Example usage
    config = get_optimal_config()
    manager = setup_pytorch_environment(config)
    
    # Print system info
    system_info = manager.get_system_info()
    print(json.dumps(system_info, indent=2, default=str)) 