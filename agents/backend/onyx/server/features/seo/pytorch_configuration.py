from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import logging
import os
import gc
import time
import warnings
from contextlib import contextmanager
import numpy as np
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
PyTorch Configuration and Optimization for SEO Service
Primary deep learning framework configuration with advanced optimizations
"""


logger = logging.getLogger(__name__)

@dataclass
class PyTorchConfig:
    """Comprehensive PyTorch configuration for optimal performance"""
    
    # Device configuration
    device: str = "auto"  # "auto", "cuda", "cpu", "mps"
    num_gpus: int = 1
    distributed_training: bool = False
    backend: str = "nccl"  # "nccl" for GPU, "gloo" for CPU
    
    # Memory and optimization
    memory_fraction: float = 0.9
    enable_cudnn_benchmark: bool = True
    enable_cudnn_deterministic: bool = False
    enable_tf32: bool = True  # For Ampere+ GPUs
    enable_amp: bool = True  # Automatic Mixed Precision
    
    # Training optimizations
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    use_gradient_checkpointing: bool = False
    use_compile: bool = True  # PyTorch 2.0+ compile
    
    # Data loading
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    
    # Performance monitoring
    enable_profiling: bool = False
    enable_memory_tracking: bool = True
    log_memory_usage: bool = True
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = False
    
    def __post_init__(self) -> Any:
        """Validate and set default values"""
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

class PyTorchManager:
    """Manages PyTorch configuration and optimizations"""
    
    def __init__(self, config: PyTorchConfig):
        
    """__init__ function."""
self.config = config
        self.device = self._setup_device()
        self.scaler = None
        self.writer = None
        self._setup_pytorch_environment()
        
    def _setup_device(self) -> torch.device:
        """Setup and validate device"""
        device = torch.device(self.config.device)
        
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
    
    def _setup_pytorch_environment(self) -> None:
        """Setup PyTorch environment with optimizations"""
        # Set random seed for reproducibility
        self._set_seed()
        
        # Setup device-specific optimizations
        if self.device.type == 'cuda':
            self._setup_cuda_optimizations()
        elif self.device.type == 'mps':
            self._setup_mps_optimizations()
        
        # Setup mixed precision
        if self.config.enable_amp and self.device.type in ['cuda', 'mps']:
            self.scaler = GradScaler()
            logger.info("Initialized Automatic Mixed Precision (AMP)")
        
        # Setup tensorboard writer
        if self.config.enable_profiling:
            self.writer = SummaryWriter(log_dir='./logs/pytorch')
            logger.info("Initialized TensorBoard writer")
    
    def _set_seed(self) -> None:
        """Set random seed for reproducibility"""
        torch.manual_seed(self.config.seed)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed(self.config.seed)
            torch.cuda.manual_seed_all(self.config.seed)
        np.random.seed(self.config.seed)
        
        if self.config.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            logger.info("Enabled deterministic mode")
    
    def _setup_cuda_optimizations(self) -> None:
        """Setup CUDA-specific optimizations"""
        # Memory management
        torch.cuda.set_per_process_memory_fraction(self.config.memory_fraction)
        
        # cuDNN optimizations
        if self.config.enable_cudnn_benchmark:
            torch.backends.cudnn.benchmark = True
            logger.info("Enabled cuDNN benchmark mode")
        
        if self.config.enable_cudnn_deterministic:
            torch.backends.cudnn.deterministic = True
            logger.info("Enabled cuDNN deterministic mode")
        
        # TF32 for Ampere+ GPUs
        if self.config.enable_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("Enabled TF32 for improved performance")
        
        # Memory pool optimization
        torch.cuda.empty_cache()
        gc.collect()
    
    def _setup_mps_optimizations(self) -> None:
        """Setup MPS-specific optimizations"""
        # MPS optimizations for Apple Silicon
        if hasattr(torch.backends, 'mps'):
            logger.info("MPS optimizations applied")
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get comprehensive device information"""
        info = {
            'device_type': self.device.type,
            'device_index': self.device.index if self.device.index else 0,
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'amp_enabled': self.config.enable_amp,
            'deterministic': self.config.deterministic
        }
        
        if self.device.type == 'cuda':
            info.update({
                'gpu_name': torch.cuda.get_device_name(),
                'gpu_memory_total_gb': torch.cuda.get_device_properties(0).total_memory / 1e9,
                'gpu_memory_allocated_gb': torch.cuda.memory_allocated() / 1e9,
                'gpu_memory_cached_gb': torch.cuda.memory_reserved() / 1e9,
                'gpu_count': torch.cuda.device_count()
            })
        
        return info
    
    def optimize_model(self, model: nn.Module) -> nn.Module:
        """Apply PyTorch optimizations to model"""
        # Move to device
        model = model.to(self.device)
        
        # Enable gradient checkpointing if requested
        if self.config.use_gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing")
        
        # Use PyTorch 2.0 compile if available
        if self.config.use_compile and hasattr(torch, 'compile'):
            try:
                model = torch.compile(model, mode='default')
                logger.info("Applied PyTorch 2.0 compile optimization")
            except Exception as e:
                logger.warning(f"PyTorch compile failed: {e}")
        
        return model
    
    def create_optimized_dataloader(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = True,
        **kwargs
    ) -> DataLoader:
        """Create optimized DataLoader with PyTorch best practices"""
        dataloader_kwargs = {
            'batch_size': batch_size,
            'shuffle': shuffle,
            'num_workers': self.config.num_workers,
            'pin_memory': self.config.pin_memory and self.device.type == 'cuda',
            'persistent_workers': self.config.persistent_workers,
            'prefetch_factor': self.config.prefetch_factor,
            **kwargs
        }
        
        return DataLoader(dataset, **dataloader_kwargs)
    
    def clear_memory(self) -> None:
        """Clear GPU memory and garbage collect"""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("Cleared device memory")
    
    @contextmanager
    def memory_monitor(self, operation_name: str):
        """Context manager for monitoring memory usage"""
        if self.device.type != 'cuda':
            yield
            return
        
        initial_memory = torch.cuda.memory_allocated()
        start_time = time.time()
        
        try:
            yield
        finally:
            final_memory = torch.cuda.memory_allocated()
            duration = time.time() - start_time
            
            memory_used = (final_memory - initial_memory) / 1e9
            if self.config.log_memory_usage:
                logger.info(f"{operation_name}: Memory used: {memory_used:.2f} GB, Duration: {duration:.2f}s")

class PyTorchTrainer:
    """Advanced PyTorch training utilities"""
    
    def __init__(self, pytorch_manager: PyTorchManager):
        
    """__init__ function."""
self.manager = pytorch_manager
        self.device = pytorch_manager.device
        self.scaler = pytorch_manager.scaler
    
    def train_step(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        batch: Dict[str, torch.Tensor],
        loss_fn: Callable,
        gradient_accumulation_steps: int = 1
    ) -> Dict[str, float]:
        """Optimized training step with mixed precision"""
        
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.manager.config.max_grad_norm)
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad()
        else:
            # Standard precision training
            outputs = model(**batch)
            loss = loss_fn(outputs, batch['labels']) if 'labels' in batch else outputs
            loss.backward()
            
            if (optimizer.param_groups[0]['step'] + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.manager.config.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
        
        return {'loss': loss.item()}
    
    def validation_step(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        loss_fn: Callable
    ) -> Dict[str, float]:
        """Validation step with no gradient computation"""
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

class PyTorchProfiler:
    """PyTorch performance profiling utilities"""
    
    def __init__(self, pytorch_manager: PyTorchManager):
        
    """__init__ function."""
self.manager = pytorch_manager
        self.writer = pytorch_manager.writer
    
    @contextmanager
    def profile_operation(self, operation_name: str):
        """Profile a specific operation"""
        if not self.manager.config.enable_profiling:
            yield
            return
        
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True
        ) as prof:
            yield
        
        # Save profiling results
        prof.export_chrome_trace(f"./logs/profiler_{operation_name}_{int(time.time())}.json")
        logger.info(f"Profiling results saved for {operation_name}")

def setup_pytorch_environment(config: PyTorchConfig) -> PyTorchManager:
    """Setup PyTorch environment with given configuration"""
    return PyTorchManager(config)

def create_pytorch_trainer(pytorch_manager: PyTorchManager) -> PyTorchTrainer:
    """Create PyTorch trainer with optimizations"""
    return PyTorchTrainer(pytorch_manager)

def get_optimal_batch_size(model: nn.Module, pytorch_manager: PyTorchManager, max_memory_gb: float = 8.0) -> int:
    """Find optimal batch size for given model and memory constraints"""
    if pytorch_manager.device.type != 'cuda':
        return 32  # Default for CPU
    
    # Simple heuristic for batch size optimization
    total_params = sum(p.numel() for p in model.parameters())
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    # Conservative estimate: 4 bytes per parameter for gradients + activations
    estimated_memory_per_sample = total_params * 4 / 1e9  # GB per sample
    
    optimal_batch_size = int(min(max_memory_gb / estimated_memory_per_sample, 64))
    return max(1, optimal_batch_size)

# Default configurations
DEFAULT_PYTORCH_CONFIG = PyTorchConfig()
HIGH_PERFORMANCE_CONFIG = PyTorchConfig(
    enable_cudnn_benchmark=True,
    enable_tf32=True,
    enable_amp=True,
    use_compile=True,
    num_workers=8,
    prefetch_factor=4
)
MEMORY_EFFICIENT_CONFIG = PyTorchConfig(
    memory_fraction=0.7,
    use_gradient_checkpointing=True,
    enable_amp=True,
    num_workers=2,
    prefetch_factor=1
) 