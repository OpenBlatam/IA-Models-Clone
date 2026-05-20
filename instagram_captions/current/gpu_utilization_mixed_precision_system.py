"""
GPU Utilization and Mixed Precision Training System
Implements comprehensive GPU optimization and Automatic Mixed Precision (AMP) training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DataParallel, DistributedDataParallel
import torch.distributed as dist
from torch.utils.data import DataLoader
import time
import psutil
import gc
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import logging
import warnings
import numpy as np
from contextlib import contextmanager
import threading
import queue

# =============================================================================
# GPU UTILIZATION SYSTEM
# =============================================================================

@dataclass
class GPUConfig:
    """Configuration for GPU utilization"""
    device_ids: List[int] = field(default_factory=list)
    memory_fraction: float = 0.9
    enable_mixed_precision: bool = True
    enable_gradient_checkpointing: bool = True
    enable_data_parallel: bool = False
    enable_distributed: bool = False
    pin_memory: bool = True
    num_workers: int = 4
    prefetch_factor: int = 2
    persistent_workers: bool = True

class GPUManager:
    """Comprehensive GPU management and optimization"""
    
    def __init__(self, config: GPUConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.device = self._get_optimal_device()
        self.gpu_info = self._get_gpu_info()
        
        # Setup GPU optimization
        self._setup_gpu_optimization()
        
        self.logger.info(f"GPU Manager initialized on device: {self.device}")
        self.logger.info(f"GPU Info: {self.gpu_info}")
    
    def _get_optimal_device(self) -> str:
        """Get optimal device for training"""
        if not torch.cuda.is_available():
            self.logger.warning("CUDA not available, using CPU")
            return "cpu"
        
        # If specific device IDs are specified
        if self.config.device_ids:
            device_id = self.config.device_ids[0]
            if device_id < torch.cuda.device_count():
                torch.cuda.set_device(device_id)
                return f"cuda:{device_id}"
            else:
                self.logger.warning(f"Device {device_id} not available, using default")
        
        # Use default GPU
        return "cuda"
    
    def _get_gpu_info(self) -> Dict[str, Any]:
        """Get comprehensive GPU information"""
        if not torch.cuda.is_available():
            return {"available": False}
        
        device = torch.cuda.current_device()
        info = {
            "available": True,
            "device_count": torch.cuda.device_count(),
            "current_device": device,
            "device_name": torch.cuda.get_device_name(device),
            "compute_capability": torch.cuda.get_device_capability(device),
            "total_memory": torch.cuda.get_device_properties(device).total_memory,
            "memory_allocated": torch.cuda.memory_allocated(device),
            "memory_reserved": torch.cuda.memory_reserved(device),
            "memory_cached": torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
        }
        
        return info
    
    def _setup_gpu_optimization(self):
        """Setup GPU optimization settings"""
        if not torch.cuda.is_available():
            return
        
        # Set memory fraction
        if self.config.memory_fraction < 1.0:
            torch.cuda.set_per_process_memory_fraction(self.config.memory_fraction)
            self.logger.info(f"GPU memory fraction set to {self.config.memory_fraction}")
        
        # Enable memory efficient algorithms
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Enable TensorFloat-32 for Ampere GPUs
        if torch.cuda.get_device_capability()[0] >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            self.logger.info("TensorFloat-32 enabled for Ampere GPU")
        
        # Set memory growth
        torch.cuda.empty_cache()
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current GPU memory usage"""
        if not torch.cuda.is_available():
            return {"available": False}
        
        device = torch.cuda.current_device()
        memory_info = {
            "device": device,
            "allocated": torch.cuda.memory_allocated(device),
            "reserved": torch.cuda.memory_reserved(device),
            "cached": torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device),
            "max_allocated": torch.cuda.max_memory_allocated(device),
            "max_reserved": torch.cuda.max_memory_reserved(device)
        }
        
        # Convert to GB for readability
        for key in ["allocated", "reserved", "cached", "max_allocated", "max_reserved"]:
            memory_info[f"{key}_gb"] = memory_info[key] / (1024**3)
        
        return memory_info
    
    def clear_memory(self):
        """Clear GPU memory cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            self.logger.info("GPU memory cache cleared")
    
    def optimize_model_for_gpu(self, model: nn.Module) -> nn.Module:
        """Optimize model for GPU usage"""
        if not torch.cuda.is_available():
            return model
        
        # Move model to GPU
        model = model.to(self.device)
        
        # Enable gradient checkpointing if specified
        if self.config.enable_gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            self.logger.info("Gradient checkpointing enabled")
        
        # Apply DataParallel if specified and multiple GPUs available
        if self.config.enable_data_parallel and torch.cuda.device_count() > 1:
            if self.config.device_ids:
                model = DataParallel(model, device_ids=self.config.device_ids)
            else:
                model = DataParallel(model)
            self.logger.info(f"DataParallel enabled on {torch.cuda.device_count()} GPUs")
        
        # Apply DistributedDataParallel if specified
        if self.config.enable_distributed:
            if not dist.is_initialized():
                self.logger.warning("Distributed training not initialized")
            else:
                model = DistributedDataParallel(model)
                self.logger.info("DistributedDataParallel enabled")
        
        return model
    
    def optimize_dataloader_for_gpu(self, dataloader: DataLoader) -> DataLoader:
        """Optimize DataLoader for GPU usage"""
        if not torch.cuda.is_available():
            return dataloader
        
        # Set pin_memory
        if self.config.pin_memory:
            dataloader.pin_memory = True
        
        # Set num_workers
        if self.config.num_workers > 0:
            dataloader.num_workers = self.config.num_workers
        
        # Set prefetch_factor
        if hasattr(dataloader, 'prefetch_factor'):
            dataloader.prefetch_factor = self.config.prefetch_factor
        
        # Set persistent_workers
        if hasattr(dataloader, 'persistent_workers'):
            dataloader.persistent_workers = self.config.persistent_workers
        
        return dataloader
    
    def monitor_gpu_usage(self, interval: float = 1.0, duration: Optional[float] = None):
        """Monitor GPU usage in real-time"""
        if not torch.cuda.is_available():
            self.logger.warning("Cannot monitor GPU usage on CPU")
            return
        
        def monitor_loop():
            start_time = time.time()
            while True:
                memory_info = self.get_memory_usage()
                
                self.logger.info(
                    f"GPU Memory - Allocated: {memory_info['allocated_gb']:.2f}GB, "
                    f"Reserved: {memory_info['reserved_gb']:.2f}GB, "
                    f"Cached: {memory_info['cached_gb']:.2f}GB"
                )
                
                # Check if duration exceeded
                if duration and (time.time() - start_time) > duration:
                    break
                
                time.sleep(interval)
        
        # Start monitoring in background thread
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        
        return monitor_thread

# =============================================================================
# MIXED PRECISION TRAINING SYSTEM
# =============================================================================

@dataclass
class MixedPrecisionConfig:
    """Configuration for mixed precision training"""
    enabled: bool = True
    dtype: str = "float16"  # "float16", "bfloat16"
    loss_scaling: bool = True
    initial_scale: float = 2**16
    growth_factor: float = 2.0
    backoff_factor: float = 0.5
    growth_interval: int = 2000
    hysteresis: int = 2000
    min_scale: float = 1.0
    max_scale: float = 2**24

class MixedPrecisionTrainer:
    """Advanced mixed precision training with automatic scaling"""
    
    def __init__(self, config: MixedPrecisionConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize mixed precision components
        self.scaler = None
        self.autocast_context = None
        
        if self.config.enabled:
            self._setup_mixed_precision()
    
    def _setup_mixed_precision(self):
        """Setup mixed precision training components"""
        if not torch.cuda.is_available():
            self.logger.warning("Mixed precision requested but CUDA not available")
            self.config.enabled = False
            return
        
        # Setup autocast context
        if self.config.dtype == "float16":
            self.autocast_context = autocast
        elif self.config.dtype == "bfloat16":
            self.autocast_context = autocast(dtype=torch.bfloat16)
        else:
            raise ValueError(f"Unsupported dtype: {self.config.dtype}")
        
        # Setup gradient scaler
        if self.config.loss_scaling:
            self.scaler = GradScaler(
                init_scale=self.config.initial_scale,
                growth_factor=self.config.growth_factor,
                backoff_factor=self.config.backoff_factor,
                growth_interval=self.config.growth_interval,
                hysteresis=self.config.hysteresis
            )
        
        self.logger.info(f"Mixed precision training enabled with {self.config.dtype}")
        if self.config.loss_scaling:
            self.logger.info("Gradient scaling enabled")
    
    def forward_pass(self, model: nn.Module, *args, **kwargs) -> torch.Tensor:
        """Forward pass with mixed precision"""
        if not self.config.enabled or self.autocast_context is None:
            return model(*args, **kwargs)
        
        with self.autocast_context():
            return model(*args, **kwargs)
    
    def compute_loss(self, criterion: nn.Module, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss with mixed precision"""
        if not self.config.enabled or self.autocast_context is None:
            return criterion(predictions, targets)
        
        with self.autocast_context():
            return criterion(predictions, targets)
    
    def backward_pass(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer):
        """Backward pass with mixed precision and gradient scaling"""
        if not self.config.enabled or self.scaler is None:
            loss.backward()
            return
        
        # Scale loss and backward pass
        self.scaler.scale(loss).backward()
    
    def optimizer_step(self, optimizer: torch.optim.Optimizer):
        """Optimizer step with mixed precision"""
        if not self.config.enabled or self.scaler is None:
            optimizer.step()
            return
        
        # Unscale gradients and optimizer step
        self.scaler.unscale_(optimizer)
        
        # Gradient clipping (optional)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Optimizer step
        self.scaler.step(optimizer)
        self.scaler.update()
    
    def get_scaler_state(self) -> Dict[str, Any]:
        """Get current scaler state"""
        if self.scaler is None:
            return {}
        
        return {
            "scale": self.scaler.get_scale(),
            "growth_tracker": self.scaler._growth_tracker
        }

class AdvancedMixedPrecisionTrainer:
    """Advanced mixed precision trainer with dynamic precision adjustment"""
    
    def __init__(self, config: MixedPrecisionConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.mixed_precision_trainer = MixedPrecisionTrainer(config)
        
        # Dynamic precision adjustment
        self.precision_history = []
        self.performance_metrics = []
        
        # Setup dynamic precision
        self._setup_dynamic_precision()
    
    def _setup_dynamic_precision(self):
        """Setup dynamic precision adjustment"""
        if not self.config.enabled:
            return
        
        self.logger.info("Dynamic precision adjustment enabled")
    
    def train_step(self, model: nn.Module, data: torch.Tensor, target: torch.Tensor,
                   criterion: nn.Module, optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
        """Complete training step with mixed precision"""
        start_time = time.time()
        
        # Forward pass with mixed precision
        predictions = self.mixed_precision_trainer.forward_pass(model, data)
        
        # Compute loss with mixed precision
        loss = self.mixed_precision_trainer.compute_loss(criterion, predictions, target)
        
        # Backward pass with mixed precision
        self.mixed_precision_trainer.backward_pass(loss, optimizer)
        
        # Optimizer step with mixed precision
        self.mixed_precision_trainer.optimizer_step(optimizer)
        
        # Calculate step time
        step_time = time.time() - start_time
        
        # Get scaler state
        scaler_state = self.mixed_precision_trainer.get_scaler_state()
        
        return {
            "loss": loss.item(),
            "step_time": step_time,
            "scaler_scale": scaler_state.get("scale", 1.0),
            "precision": self.config.dtype if self.config.enabled else "float32"
        }
    
    def validate_step(self, model: nn.Module, data: torch.Tensor, target: torch.Tensor,
                     criterion: nn.Module) -> Dict[str, Any]:
        """Validation step with mixed precision"""
        start_time = time.time()
        
        model.eval()
        with torch.no_grad():
            # Forward pass with mixed precision
            predictions = self.mixed_precision_trainer.forward_pass(model, data)
            
            # Compute loss
            loss = self.mixed_precision_trainer.compute_loss(criterion, predictions, target)
        
        step_time = time.time() - start_time
        
        return {
            "loss": loss.item(),
            "step_time": step_time,
            "precision": self.config.dtype if self.config.enabled else "float32"
        }

# =============================================================================
# GPU MEMORY OPTIMIZATION
# =============================================================================

class GPUMemoryOptimizer:
    """Advanced GPU memory optimization"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.memory_tracker = {}
    
    def optimize_model_memory(self, model: nn.Module, input_shape: Tuple[int, ...]) -> nn.Module:
        """Optimize model for memory efficiency"""
        if not torch.cuda.is_available():
            return model
        
        # Enable gradient checkpointing
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            self.logger.info("Gradient checkpointing enabled for memory optimization")
        
        # Use memory efficient attention if available
        if hasattr(model, 'config') and hasattr(model.config, 'attention_mode'):
            if model.config.attention_mode != "flash_attention_2":
                try:
                    model.config.attention_mode = "flash_attention_2"
                    self.logger.info("Memory efficient attention enabled")
                except Exception as e:
                    self.logger.warning(f"Failed to enable memory efficient attention: {e}")
        
        # Enable memory efficient forward pass
        if hasattr(model, 'config') and hasattr(model.config, 'use_memory_efficient_forward'):
            model.config.use_memory_efficient_forward = True
            self.logger.info("Memory efficient forward pass enabled")
        
        return model
    
    def optimize_batch_size(self, model: nn.Module, input_shape: Tuple[int, ...],
                           target_memory_gb: float = 8.0) -> int:
        """Find optimal batch size for given memory constraint"""
        if not torch.cuda.is_available():
            return 32  # Default for CPU
        
        device = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device).total_memory
        target_memory_bytes = target_memory_gb * (1024**3)
        
        # Start with batch size 1
        batch_size = 1
        max_batch_size = 1024  # Safety limit
        
        while batch_size <= max_batch_size:
            try:
                # Create dummy input
                dummy_input = torch.randn(batch_size, *input_shape[1:], device=device)
                
                # Forward pass to measure memory
                with torch.no_grad():
                    _ = model(dummy_input)
                
                # Check memory usage
                memory_used = torch.cuda.memory_allocated(device)
                
                if memory_used > target_memory_bytes:
                    # Reduce batch size and break
                    batch_size = max(1, batch_size - 1)
                    break
                
                # Clear memory
                del dummy_input
                torch.cuda.empty_cache()
                
                batch_size *= 2
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    # Reduce batch size and break
                    batch_size = max(1, batch_size // 2)
                    break
                else:
                    raise e
        
        self.logger.info(f"Optimal batch size: {batch_size}")
        return batch_size
    
    def profile_memory_usage(self, model: nn.Module, input_shape: Tuple[int, ...],
                            batch_sizes: List[int] = None) -> Dict[str, Any]:
        """Profile memory usage for different batch sizes"""
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
        
        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8, 16, 32, 64]
        
        device = torch.cuda.current_device()
        memory_profiles = {}
        
        for batch_size in batch_sizes:
            try:
                # Clear memory
                torch.cuda.empty_cache()
                
                # Create dummy input
                dummy_input = torch.randn(batch_size, *input_shape[1:], device=device)
                
                # Measure initial memory
                initial_memory = torch.cuda.memory_allocated(device)
                
                # Forward pass
                with torch.no_grad():
                    _ = model(dummy_input)
                
                # Measure peak memory
                peak_memory = torch.cuda.max_memory_allocated(device)
                
                # Calculate memory usage
                memory_used = peak_memory - initial_memory
                
                memory_profiles[batch_size] = {
                    "memory_used_bytes": memory_used,
                    "memory_used_gb": memory_used / (1024**3),
                    "peak_memory_gb": peak_memory / (1024**3)
                }
                
                # Cleanup
                del dummy_input
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                memory_profiles[batch_size] = {"error": str(e)}
        
        return memory_profiles

# =============================================================================
# GPU PERFORMANCE MONITORING
# =============================================================================

class GPUPerformanceMonitor:
    """Real-time GPU performance monitoring"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.monitoring = False
        self.monitor_thread = None
        self.metrics_queue = queue.Queue()
    
    def start_monitoring(self, interval: float = 1.0):
        """Start GPU performance monitoring"""
        if not torch.cuda.is_available():
            self.logger.warning("Cannot monitor GPU performance on CPU")
            return
        
        if self.monitoring:
            self.logger.warning("Monitoring already active")
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval,), daemon=True)
        self.monitor_thread.start()
        
        self.logger.info("GPU performance monitoring started")
    
    def stop_monitoring(self):
        """Stop GPU performance monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        self.logger.info("GPU performance monitoring stopped")
    
    def _monitor_loop(self, interval: float):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                metrics = self._collect_metrics()
                self.metrics_queue.put(metrics)
                time.sleep(interval)
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval)
    
    def _collect_metrics(self) -> Dict[str, Any]:
        """Collect GPU performance metrics"""
        if not torch.cuda.is_available():
            return {}
        
        device = torch.cuda.current_device()
        
        # Memory metrics
        memory_allocated = torch.cuda.memory_allocated(device)
        memory_reserved = torch.cuda.memory_reserved(device)
        memory_cached = memory_reserved - memory_allocated
        
        # Performance metrics
        try:
            # Get GPU utilization (requires nvidia-ml-py3)
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(device)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_utilization = utilization.gpu
            memory_utilization = utilization.memory
        except ImportError:
            gpu_utilization = None
            memory_utilization = None
        
        metrics = {
            "timestamp": time.time(),
            "device": device,
            "memory_allocated_gb": memory_allocated / (1024**3),
            "memory_reserved_gb": memory_reserved / (1024**3),
            "memory_cached_gb": memory_cached / (1024**3),
            "gpu_utilization": gpu_utilization,
            "memory_utilization": memory_utilization
        }
        
        return metrics
    
    def get_latest_metrics(self) -> Optional[Dict[str, Any]]:
        """Get latest performance metrics"""
        try:
            return self.metrics_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_metrics_history(self) -> List[Dict[str, Any]]:
        """Get all collected metrics"""
        metrics = []
        while not self.metrics_queue.empty():
            try:
                metrics.append(self.metrics_queue.get_nowait())
            except queue.Empty:
                break
        
        return metrics

# =============================================================================
# INTEGRATED GPU TRAINING SYSTEM
# =============================================================================

class IntegratedGPUTrainingSystem:
    """Integrated system for GPU-optimized training"""
    
    def __init__(self, gpu_config: GPUConfig, mixed_precision_config: MixedPrecisionConfig):
        self.gpu_config = gpu_config
        self.mixed_precision_config = mixed_precision_config
        
        # Initialize components
        self.gpu_manager = GPUManager(gpu_config)
        self.mixed_precision_trainer = AdvancedMixedPrecisionTrainer(mixed_precision_config)
        self.memory_optimizer = GPUMemoryOptimizer()
        self.performance_monitor = GPUPerformanceMonitor()
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def setup_training(self, model: nn.Module, input_shape: Tuple[int, ...]) -> Dict[str, Any]:
        """Setup complete training environment"""
        # Optimize model for GPU
        model = self.gpu_manager.optimize_model_for_gpu(model)
        
        # Optimize model memory
        model = self.memory_optimizer.optimize_model_memory(model, input_shape)
        
        # Find optimal batch size
        optimal_batch_size = self.memory_optimizer.optimize_batch_size(
            model, input_shape, target_memory_gb=8.0
        )
        
        # Profile memory usage
        memory_profile = self.memory_optimizer.profile_memory_usage(model, input_shape)
        
        # Start performance monitoring
        self.performance_monitor.start_monitoring(interval=2.0)
        
        setup_info = {
            "device": self.gpu_manager.device,
            "optimal_batch_size": optimal_batch_size,
            "memory_profile": memory_profile,
            "gpu_info": self.gpu_manager.gpu_info,
            "mixed_precision_enabled": self.mixed_precision_config.enabled
        }
        
        self.logger.info(f"Training setup completed: {setup_info}")
        return setup_info
    
    def training_loop(self, model: nn.Module, train_loader: DataLoader,
                     criterion: nn.Module, optimizer: torch.optim.Optimizer,
                     num_epochs: int) -> Dict[str, Any]:
        """Complete training loop with GPU optimization"""
        training_stats = {
            "epochs": [],
            "train_losses": [],
            "train_times": [],
            "memory_usage": [],
            "scaler_states": []
        }
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # Training epoch
            epoch_stats = self._train_epoch(model, train_loader, criterion, optimizer)
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Get current metrics
            latest_metrics = self.performance_monitor.get_latest_metrics()
            scaler_state = self.mixed_precision_trainer.mixed_precision_trainer.get_scaler_state()
            
            # Store statistics
            training_stats["epochs"].append(epoch + 1)
            training_stats["train_losses"].append(epoch_stats["avg_loss"])
            training_stats["train_times"].append(epoch_time)
            training_stats["memory_usage"].append(latest_metrics)
            training_stats["scaler_states"].append(scaler_state)
            
            # Log progress
            self.logger.info(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"Loss: {epoch_stats['avg_loss']:.4f}, "
                f"Time: {epoch_time:.2f}s, "
                f"Memory: {latest_metrics.get('memory_allocated_gb', 0):.2f}GB"
            )
            
            # Memory cleanup
            if epoch % 5 == 0:
                self.gpu_manager.clear_memory()
        
        # Stop monitoring
        self.performance_monitor.stop_monitoring()
        
        return training_stats
    
    def _train_epoch(self, model: nn.Module, train_loader: DataLoader,
                     criterion: nn.Module, optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """Train single epoch"""
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # Move data to GPU
            data = data.to(self.gpu_manager.device)
            target = target.to(self.gpu_manager.device)
            
            # Training step
            step_stats = self.mixed_precision_trainer.train_step(
                model, data, target, criterion, optimizer
            )
            
            total_loss += step_stats["loss"]
            num_batches += 1
            
            # Log batch progress
            if batch_idx % 100 == 0:
                self.logger.debug(
                    f"Batch {batch_idx}/{len(train_loader)} - "
                    f"Loss: {step_stats['loss']:.4f}, "
                    f"Time: {step_stats['step_time']:.4f}s"
                )
        
        return {
            "avg_loss": total_loss / num_batches,
            "total_batches": num_batches
        }
    
    def cleanup(self):
        """Cleanup resources"""
        self.performance_monitor.stop_monitoring()
        self.gpu_manager.clear_memory()
        self.logger.info("Training system cleanup completed")

# =============================================================================
# EXAMPLE USAGE AND DEMONSTRATION
# =============================================================================

def create_sample_model() -> nn.Module:
    """Create sample model for demonstration"""
    return nn.Sequential(
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 10)
    )

def create_sample_dataloader(batch_size: int = 32) -> DataLoader:
    """Create sample dataloader for demonstration"""
    # Create dummy dataset
    data = torch.randn(1000, 784)
    targets = torch.randint(0, 10, (1000,))
    
    dataset = torch.utils.data.TensorDataset(data, targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def main():
    """Example usage of GPU utilization and mixed precision system"""
    
    print("=== GPU Utilization and Mixed Precision Training System ===")
    
    # Configuration
    gpu_config = GPUConfig(
        device_ids=[0],  # Use first GPU
        memory_fraction=0.9,
        enable_mixed_precision=True,
        enable_gradient_checkpointing=True,
        enable_data_parallel=False,
        pin_memory=True,
        num_workers=4
    )
    
    mixed_precision_config = MixedPrecisionConfig(
        enabled=True,
        dtype="float16",
        loss_scaling=True,
        initial_scale=2**16,
        growth_factor=2.0
    )
    
    # Initialize integrated system
    training_system = IntegratedGPUTrainingSystem(gpu_config, mixed_precision_config)
    
    # Create model and data
    model = create_sample_model()
    input_shape = (32, 784)  # (batch_size, features)
    
    # Setup training
    setup_info = training_system.setup_training(model, input_shape)
    print(f"Setup completed: {setup_info}")
    
    # Create dataloader
    dataloader = create_sample_dataloader(setup_info["optimal_batch_size"])
    
    # Training components
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Run training
    print("\nStarting training...")
    training_stats = training_system.training_loop(
        model=model,
        train_loader=dataloader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=3
    )
    
    # Print results
    print("\n=== Training Results ===")
    print(f"Final Loss: {training_stats['train_losses'][-1]:.4f}")
    print(f"Total Training Time: {sum(training_stats['train_times']):.2f}s")
    print(f"Average Epoch Time: {np.mean(training_stats['train_times']):.2f}s")
    
    # Memory usage summary
    memory_usage = training_stats['memory_usage']
    if memory_usage:
        avg_memory = np.mean([m.get('memory_allocated_gb', 0) for m in memory_usage if m])
        print(f"Average GPU Memory Usage: {avg_memory:.2f}GB")
    
    # Cleanup
    training_system.cleanup()

if __name__ == "__main__":
    main()


