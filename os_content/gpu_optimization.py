from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Union, Tuple, Callable, TypeVar
from dataclasses import dataclass, field
from enum import Enum
import warnings
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.cuda.amp.grad_scaler import OptState
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import numpy as np
from transformers import get_linear_schedule_with_warmup
            import xformers
            from xformers.ops import memory_efficient_attention
            import pynvml
            import pynvml
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
GPU Optimization and Mixed Precision Training
Advanced GPU utilization, mixed precision training, and memory management
for optimal deep learning performance.
"""




# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")

class GPUConfig(Enum):
    """GPU configuration options"""
    SINGLE_GPU = "single_gpu"
    MULTI_GPU = "multi_gpu"
    DISTRIBUTED = "distributed"
    MIXED_PRECISION = "mixed_precision"
    GRADIENT_ACCUMULATION = "gradient_accumulation"
    MEMORY_EFFICIENT = "memory_efficient"

@dataclass
class GPUOptimizationConfig:
    """Configuration for GPU optimization"""
    # GPU settings
    gpu_config: GPUConfig = GPUConfig.SINGLE_GPU
    device_ids: List[int] = field(default_factory=lambda: [0])
    master_gpu: int = 0
    
    # Mixed precision settings
    use_amp: bool = True
    amp_dtype: torch.dtype = torch.float16
    amp_enabled: bool = True
    amp_autocast: bool = True
    
    # Memory optimization
    use_gradient_checkpointing: bool = False
    use_memory_efficient_attention: bool = True
    use_xformers: bool = True
    use_flash_attention: bool = True
    
    # Gradient accumulation
    use_gradient_accumulation: bool = True
    gradient_accumulation_steps: int = 4
    gradient_clipping: bool = True
    max_grad_norm: float = 1.0
    
    # Distributed training
    use_distributed: bool = False
    world_size: int = 1
    rank: int = 0
    backend: str = "nccl"
    
    # Performance monitoring
    profile_memory: bool = True
    log_memory_usage: bool = True
    memory_fraction: float = 0.9

class GPUMemoryManager:
    """Advanced GPU memory management"""
    
    def __init__(self, config: GPUOptimizationConfig):
        
    """__init__ function."""
self.config = config
        self.device = torch.device(f"cuda:{config.master_gpu}" if torch.cuda.is_available() else "cpu")
        self.memory_stats = {}
        
    def get_memory_info(self) -> Dict[str, Any]:
        """Get detailed GPU memory information"""
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
        
        memory_stats = {}
        for i in range(torch.cuda.device_count()):
            memory_stats[f"gpu_{i}"] = {
                "total_memory": torch.cuda.get_device_properties(i).total_memory,
                "allocated_memory": torch.cuda.memory_allocated(i),
                "cached_memory": torch.cuda.memory_reserved(i),
                "free_memory": torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i),
                "memory_fraction": torch.cuda.memory_allocated(i) / torch.cuda.get_device_properties(i).total_memory
            }
        
        return memory_stats
    
    def clear_cache(self) -> None:
        """Clear GPU cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared")
    
    def set_memory_fraction(self, fraction: float) -> None:
        """Set memory fraction for GPU"""
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(fraction)
            logger.info(f"GPU memory fraction set to {fraction}")
    
    def optimize_memory(self) -> None:
        """Apply memory optimization techniques"""
        if not torch.cuda.is_available():
            return
        
        # Set memory fraction
        self.set_memory_fraction(self.config.memory_fraction)
        
        # Clear cache
        self.clear_cache()
        
        # Enable memory efficient settings
        if self.config.use_memory_efficient_attention:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(False)
        
        # Prefer TF32 where supported for matmul speedups
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass
        
        logger.info("Memory optimization applied")
    
    def monitor_memory_usage(self, stage: str = "general") -> Dict[str, Any]:
        """Monitor memory usage during training"""
        memory_info = self.get_memory_info()
        
        if self.config.log_memory_usage:
            logger.info(f"Memory usage at {stage}: {memory_info}")
        
        self.memory_stats[stage] = memory_info
        return memory_info

class MixedPrecisionTrainer:
    """Advanced mixed precision training with proper GPU utilization"""
    
    def __init__(self, model: nn.Module, config: GPUOptimizationConfig):
        
    """__init__ function."""
self.config = config
        self.model = model
        self.device = torch.device(f"cuda:{config.master_gpu}" if torch.cuda.is_available() else "cpu")
        
        # Initialize mixed precision components
        self.scaler = GradScaler() if config.use_amp else None
        self.autocast_context = autocast if config.amp_autocast else None
        
        # Memory manager
        self.memory_manager = GPUMemoryManager(config)
        
        # Training state
        self.current_step = 0
        self.gradient_accumulation_step = 0
        
        # Performance metrics
        self.training_metrics = {
            'loss': [],
            'accuracy': [],
            'memory_usage': [],
            'training_time': []
        }
    
    def setup_model(self) -> nn.Module:
        """Setup model for GPU training"""
        # Move model to GPU
        self.model = self.model.to(self.device)
        
        # Apply memory optimizations
        if self.config.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        
        # Apply xformers optimizations
        if self.config.use_xformers:
            self._apply_xformers_optimizations()
        
        # Setup distributed training if needed
        if self.config.use_distributed:
            self.model = DDP(self.model, device_ids=[self.config.master_gpu])
            logger.info("Distributed training setup complete")
        
        return self.model
    
    def _apply_xformers_optimizations(self) -> None:
        """Apply xformers optimizations to the model"""
        try:
            
            # Replace attention layers with xformers
            for module in self.model.modules():
                if hasattr(module, 'attention'):
                    module.attention = memory_efficient_attention
            
            logger.info("Xformers optimizations applied")
        except ImportError:
            logger.warning("Xformers not available, skipping optimizations")
    
    def create_optimizer(self, learning_rate: float, weight_decay: float = 0.01) -> optim.Optimizer:
        """Create optimizer with proper GPU setup"""
        # Use AdamW for better performance
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            eps=1e-8,
            betas=(0.9, 0.999)
        )
        
        return optimizer
    
    def create_scheduler(self, optimizer: optim.Optimizer, num_training_steps: int) -> Any:
        """Create learning rate scheduler"""
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * num_training_steps),
            num_training_steps=num_training_steps
        )
    
    async def train_step(self, batch: Dict[str, torch.Tensor], 
                        optimizer: optim.Optimizer, 
                        scheduler: Any = None) -> Dict[str, float]:
        """Single training step with mixed precision"""
        start_time = time.time()
        
        # Move batch to GPU
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Mixed precision forward pass
        with autocast(enabled=self.config.use_amp):
            outputs = self.model(**batch)
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
            
            # Scale loss for gradient accumulation
            if self.config.use_gradient_accumulation:
                loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass with gradient scaling
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient accumulation
        self.gradient_accumulation_step += 1
        
        if self.gradient_accumulation_step >= self.config.gradient_accumulation_steps:
            # Gradient clipping
            if self.config.gradient_clipping:
                if self.scaler:
                    self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            
            # Optimizer step
            if self.scaler:
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                optimizer.step()
            
            # Scheduler step
            if scheduler:
                scheduler.step()
            
            # Reset gradients
            optimizer.zero_grad()
            self.gradient_accumulation_step = 0
        
        # Update metrics
        step_time = time.time() - start_time
        self.training_metrics['training_time'].append(step_time)
        
        # Monitor memory
        if self.config.profile_memory:
            memory_info = self.memory_manager.monitor_memory_usage(f"step_{self.current_step}")
            self.training_metrics['memory_usage'].append(memory_info)
        
        self.current_step += 1
        
        return {
            'loss': loss.item(),
            'step_time': step_time,
            'memory_usage': self.memory_manager.get_memory_info() if self.config.profile_memory else None
        }
    
    async def validate_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Validation step with mixed precision"""
        self.model.eval()
        
        with torch.no_grad():
            with autocast(enabled=self.config.use_amp):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                
                # Calculate accuracy for classification
                if hasattr(outputs, 'logits'):
                    predictions = torch.argmax(outputs.logits, dim=-1)
                    accuracy = (predictions == batch['labels']).float().mean().item()
                else:
                    accuracy = 0.0
        
        self.model.train()
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        return {
            'total_steps': self.current_step,
            'average_step_time': np.mean(self.training_metrics['training_time']) if self.training_metrics['training_time'] else 0,
            'memory_usage_history': self.training_metrics['memory_usage'],
            'gpu_config': self.config.gpu_config.value,
            'mixed_precision_enabled': self.config.use_amp,
            'gradient_accumulation_enabled': self.config.use_gradient_accumulation
        }

class GPUDataLoader:
    """Optimized data loader for GPU training"""
    
    def __init__(self, dataset: Dataset, config: GPUOptimizationConfig, **kwargs):
        
    """__init__ function."""
self.config = config
        self.device = torch.device(f"cuda:{config.master_gpu}" if torch.cuda.is_available() else "cpu")
        
        # Optimize data loader settings
        loader_kwargs = {
            'batch_size': kwargs.get('batch_size', 16),
            'shuffle': kwargs.get('shuffle', True),
            'num_workers': min(4, os.cpu_count() or 1),  # Optimize for GPU
            'pin_memory': True,  # Faster data transfer to GPU
            'persistent_workers': True,  # Keep workers alive
            'prefetch_factor': 2,  # Prefetch batches
            'drop_last': kwargs.get('drop_last', False)
        }
        
        self.data_loader = DataLoader(dataset, **loader_kwargs)
    
    def __iter__(self) -> Any:
        return iter(self.data_loader)
    
    def __len__(self) -> Any:
        return len(self.data_loader)

class GPUMonitoring:
    """Real-time GPU monitoring and profiling"""
    
    def __init__(self, config: GPUOptimizationConfig):
        
    """__init__ function."""
self.config = config
        self.monitoring_data = []
    
    def start_monitoring(self) -> None:
        """Start GPU monitoring"""
        if not torch.cuda.is_available():
            logger.warning("CUDA not available for monitoring")
            return
        
        logger.info("Starting GPU monitoring...")
    
    def log_gpu_stats(self, stage: str = "general") -> Dict[str, Any]:
        """Log current GPU statistics"""
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
        
        stats = {
            'timestamp': time.time(),
            'stage': stage,
            'gpu_count': torch.cuda.device_count(),
            'current_device': torch.cuda.current_device(),
            'devices': {}
        }
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            stats['devices'][f'gpu_{i}'] = {
                'name': props.name,
                'total_memory': props.total_memory,
                'allocated_memory': torch.cuda.memory_allocated(i),
                'cached_memory': torch.cuda.memory_reserved(i),
                'utilization': self._get_gpu_utilization(i),
                'temperature': self._get_gpu_temperature(i)
            }
        
        self.monitoring_data.append(stats)
        return stats
    
    def _get_gpu_utilization(self, device_id: int) -> float:
        """Get GPU utilization percentage"""
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return utilization.gpu
        except ImportError:
            return 0.0
    
    def _get_gpu_temperature(self, device_id: int) -> float:
        """Get GPU temperature"""
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            temperature = pynvml.nvmlDeviceGetTemperature(handle, 0)  # GPU temperature
            return temperature
        except ImportError:
            return 0.0
    
    def generate_monitoring_report(self) -> Dict[str, Any]:
        """Generate comprehensive monitoring report"""
        if not self.monitoring_data:
            return {"error": "No monitoring data available"}
        
        report = {
            'monitoring_duration': self.monitoring_data[-1]['timestamp'] - self.monitoring_data[0]['timestamp'],
            'total_samples': len(self.monitoring_data),
            'gpu_summary': {},
            'peak_memory_usage': {},
            'average_utilization': {}
        }
        
        # Calculate statistics for each GPU
        for device_id in range(torch.cuda.device_count()):
            device_key = f'gpu_{device_id}'
            
            # Peak memory usage
            peak_memory = max(
                sample['devices'][device_key]['allocated_memory'] 
                for sample in self.monitoring_data
            )
            
            # Average utilization
            avg_utilization = np.mean([
                sample['devices'][device_key]['utilization'] 
                for sample in self.monitoring_data
            ])
            
            report['gpu_summary'][device_key] = {
                'name': self.monitoring_data[0]['devices'][device_key]['name'],
                'peak_memory_mb': peak_memory / (1024 * 1024),
                'average_utilization': avg_utilization
            }
        
        return report

class OptimizedTrainingLoop:
    """Complete optimized training loop with GPU utilization"""
    
    def __init__(self, model: nn.Module, config: GPUOptimizationConfig):
        
    """__init__ function."""
self.config = config
        self.mixed_precision_trainer = MixedPrecisionTrainer(model, config)
        self.gpu_monitoring = GPUMonitoring(config)
        self.memory_manager = GPUMemoryManager(config)
        
        # Setup model
        self.model = self.mixed_precision_trainer.setup_model()
        
        # Performance tracking
        self.epoch_metrics = []
        self.best_metric = float('inf')
    
    async def train_epoch(self, train_loader: DataLoader, optimizer: optim.Optimizer, 
                         scheduler: Any = None) -> Dict[str, float]:
        """Train for one epoch with GPU optimization"""
        self.model.train()
        epoch_start_time = time.time()
        
        total_loss = 0
        total_steps = 0
        
        # Start monitoring
        self.gpu_monitoring.start_monitoring()
        
        for batch_idx, batch in enumerate(train_loader):
            # Training step
            step_metrics = await self.mixed_precision_trainer.train_step(batch, optimizer, scheduler)
            
            total_loss += step_metrics['loss']
            total_steps += 1
            
            # Log GPU stats periodically
            if batch_idx % 100 == 0:
                self.gpu_monitoring.log_gpu_stats(f"training_step_{batch_idx}")
        
        epoch_time = time.time() - epoch_start_time
        
        return {
            'loss': total_loss / total_steps,
            'epoch_time': epoch_time,
            'steps_per_second': total_steps / epoch_time
        }
    
    async def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        
        total_loss = 0
        total_accuracy = 0
        total_steps = 0
        
        with torch.no_grad():
            for batch in val_loader:
                step_metrics = await self.mixed_precision_trainer.validate_step(batch)
                
                total_loss += step_metrics['loss']
                total_accuracy += step_metrics['accuracy']
                total_steps += 1
        
        return {
            'loss': total_loss / total_steps,
            'accuracy': total_accuracy / total_steps
        }
    
    async def run_training(self, train_loader: DataLoader, val_loader: DataLoader,
                          num_epochs: int, learning_rate: float) -> Dict[str, Any]:
        """Run complete training with GPU optimization"""
        logger.info("Starting optimized GPU training...")
        
        # Create optimizer and scheduler
        optimizer = self.mixed_precision_trainer.create_optimizer(learning_rate)
        scheduler = self.mixed_precision_trainer.create_scheduler(
            optimizer, 
            len(train_loader) * num_epochs
        )
        
        # Training loop
        for epoch in range(num_epochs):
            logger.info(f"Training epoch {epoch + 1}/{num_epochs}")
            
            # Train epoch
            train_metrics = await self.train_epoch(train_loader, optimizer, scheduler)
            
            # Validate epoch
            val_metrics = await self.validate_epoch(val_loader)
            
            # Log metrics
            epoch_result = {
                'epoch': epoch + 1,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            }
            self.epoch_metrics.append(epoch_result)
            
            logger.info(f"Epoch {epoch + 1}: Train Loss: {train_metrics['loss']:.4f}, "
                       f"Val Loss: {val_metrics['loss']:.4f}, "
                       f"Val Accuracy: {val_metrics['accuracy']:.4f}")
            
            # Save best model
            if val_metrics['loss'] < self.best_metric:
                self.best_metric = val_metrics['loss']
                torch.save(self.model.state_dict(), f'models/best_model_epoch_{epoch + 1}.pt')
        
        # Generate final report
        performance_summary = self.mixed_precision_trainer.get_performance_summary()
        monitoring_report = self.gpu_monitoring.generate_monitoring_report()
        
        return {
            'training_history': self.epoch_metrics,
            'performance_summary': performance_summary,
            'monitoring_report': monitoring_report,
            'best_metric': self.best_metric
        }

# Utility functions
def setup_gpu_environment(config: GPUOptimizationConfig) -> None:
    """Setup GPU environment for optimal performance"""
    if not torch.cuda.is_available():
        logger.warning("CUDA not available")
        return
    
    # Set CUDA device
    torch.cuda.set_device(config.master_gpu)
    
    # Enable optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # Memory optimizations
    if config.use_memory_efficient_attention:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    
    logger.info(f"GPU environment setup complete. Using device: {config.master_gpu}")

def create_optimized_data_loader(dataset: Dataset, config: GPUOptimizationConfig, 
                               batch_size: int = 16) -> GPUDataLoader:
    """Create optimized data loader for GPU training"""
    return GPUDataLoader(dataset, config, batch_size=batch_size)

# Example usage
if __name__ == "__main__":
    # Example configuration
    config = GPUOptimizationConfig(
        gpu_config=GPUConfig.MIXED_PRECISION,
        use_amp=True,
        use_gradient_accumulation=True,
        gradient_accumulation_steps=4,
        use_memory_efficient_attention=True,
        profile_memory=True
    )
    
    # Setup GPU environment
    setup_gpu_environment(config)
    
    print("GPU optimization module ready!")
    print(f"Mixed precision: {config.use_amp}")
    print(f"Gradient accumulation: {config.use_gradient_accumulation}")
    print(f"Memory efficient attention: {config.use_memory_efficient_attention}") 