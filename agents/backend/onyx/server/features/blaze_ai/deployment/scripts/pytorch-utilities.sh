#!/usr/bin/env python3
"""
PyTorch Utilities for Blaze AI Deep Learning
Comprehensive PyTorch utilities, optimizations, and best practices
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.backends.cudnn as cudnn
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import warnings
import os

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class PyTorchConfiguration:
    """PyTorch-specific configuration and optimizations"""
    # CUDA optimizations
    enable_cudnn_benchmark: bool = True
    enable_cudnn_deterministic: bool = False
    enable_tf32: bool = True
    enable_flash_attention: bool = True
    memory_fraction: float = 0.9
    
    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = "float16"
    
    # Distributed training
    use_distributed: bool = False
    backend: str = "nccl"
    
    # Performance
    num_threads: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True


class PyTorchOptimizer:
    """PyTorch optimization utilities"""
    
    def __init__(self, config: PyTorchConfiguration):
        self.config = config
        self._apply_optimizations()
    
    def _apply_optimizations(self):
        """Apply PyTorch performance optimizations"""
        logger.info("Applying PyTorch optimizations...")
        
        # CUDA optimizations
        if torch.cuda.is_available():
            # Enable cuDNN benchmark for fixed input sizes
            cudnn.benchmark = self.config.enable_cudnn_benchmark
            cudnn.deterministic = self.config.enable_cudnn_deterministic
            
            # Enable TensorFloat-32 for faster training on Ampere GPUs
            torch.backends.cuda.matmul.allow_tf32 = self.config.enable_tf32
            torch.backends.cudnn.allow_tf32 = self.config.enable_tf32
            
            # Enable Flash Attention if available
            if self.config.enable_flash_attention:
                try:
                    torch.backends.cuda.enable_flash_sdp(True)
                    logger.info("Flash attention enabled")
                except:
                    logger.warning("Flash attention not available")
            
            # Set memory fraction
            torch.cuda.set_per_process_memory_fraction(self.config.memory_fraction)
            
            logger.info("CUDA optimizations applied")
        else:
            logger.info("CUDA not available, skipping CUDA optimizations")
        
        # Threading optimizations
        torch.set_num_threads(min(self.config.num_threads, torch.get_num_threads()))
        
        logger.info("PyTorch optimizations completed")
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get comprehensive device information"""
        device_info = {
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'cudnn_version': torch.backends.cudnn.version() if torch.cuda.is_available() else None,
            'num_threads': torch.get_num_threads(),
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
        
        if torch.cuda.is_available():
            device_info['gpu_devices'] = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                device_info['gpu_devices'].append({
                    'id': i,
                    'name': props.name,
                    'memory_total_gb': props.total_memory / (1024**3),
                    'compute_capability': f"{props.major}.{props.minor}",
                    'multi_processor_count': props.multi_processor_count
                })
        
        return device_info


class PyTorchDataLoader:
    """PyTorch DataLoader utilities and optimizations"""
    
    @staticmethod
    def create_optimized_loader(dataset: Dataset, 
                               batch_size: int,
                               num_workers: Optional[int] = None,
                               pin_memory: bool = True,
                               persistent_workers: bool = True,
                               **kwargs) -> DataLoader:
        """Create optimized PyTorch DataLoader"""
        
        # Auto-determine optimal number of workers
        if num_workers is None:
            num_workers = min(4, os.cpu_count() or 1)
        
        # Create optimized DataLoader
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            persistent_workers=persistent_workers and num_workers > 0,
            drop_last=True,
            **kwargs
        )
        
        logger.info(f"Optimized DataLoader created: {num_workers} workers, "
                   f"batch_size={batch_size}, pin_memory={pin_memory}")
        
        return loader
    
    @staticmethod
    def create_distributed_loader(dataset: Dataset,
                                batch_size: int,
                                world_size: int,
                                rank: int,
                                **kwargs) -> DataLoader:
        """Create distributed DataLoader for multi-GPU training"""
        
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            **kwargs
        )
        
        return loader


class PyTorchModelUtils:
    """PyTorch model utilities and helpers"""
    
    @staticmethod
    def count_parameters(model: nn.Module) -> Dict[str, int]:
        """Count model parameters with detailed breakdown"""
        total_params = 0
        trainable_params = 0
        non_trainable_params = 0
        
        for param in model.parameters():
            param_count = param.numel()
            total_params += param_count
            if param.requires_grad:
                trainable_params += param_count
            else:
                non_trainable_params += param_count
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': non_trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
        }
    
    @staticmethod
    def get_model_summary(model: nn.Module, input_size: Tuple[int, ...]) -> Dict[str, Any]:
        """Get comprehensive model summary"""
        model_info = PyTorchModelUtils.count_parameters(model)
        
        # Add input/output shape information
        try:
            with torch.no_grad():
                dummy_input = torch.randn(1, *input_size)
                if torch.cuda.is_available():
                    dummy_input = dummy_input.cuda()
                    model = model.cuda()
                
                output = model(dummy_input)
                model_info['input_shape'] = dummy_input.shape
                model_info['output_shape'] = output.shape
                
                # Move back to CPU if needed
                if not torch.cuda.is_available():
                    model = model.cpu()
                    
        except Exception as e:
            logger.warning(f"Could not determine input/output shapes: {e}")
            model_info['input_shape'] = None
            model_info['output_shape'] = None
        
        return model_info
    
    @staticmethod
    def save_model_checkpoint(model: nn.Module,
                            optimizer: optim.Optimizer,
                            scheduler: Optional[optim.lr_scheduler._LRScheduler],
                            epoch: int,
                            metrics: Dict[str, float],
                            filepath: str,
                            **kwargs):
        """Save comprehensive model checkpoint"""
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'pytorch_version': torch.__version__,
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            **kwargs
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        torch.save(checkpoint, filepath)
        logger.info(f"Model checkpoint saved: {filepath}")
    
    @staticmethod
    def load_model_checkpoint(model: nn.Module,
                            optimizer: optim.Optimizer,
                            scheduler: Optional[optim.lr_scheduler._LRScheduler],
                            filepath: str,
                            device: Optional[torch.device] = None) -> Dict[str, Any]:
        """Load model checkpoint with compatibility checks"""
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        checkpoint = torch.load(filepath, map_location=device)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if available
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"Model checkpoint loaded: {filepath}")
        logger.info(f"Resuming from epoch: {checkpoint['epoch']}")
        
        return checkpoint


class PyTorchTrainingUtils:
    """PyTorch training utilities and best practices"""
    
    @staticmethod
    def create_optimizer(model: nn.Module,
                        optimizer_type: str = "adamw",
                        learning_rate: float = 1e-4,
                        weight_decay: float = 1e-5,
                        **kwargs) -> optim.Optimizer:
        """Create optimized PyTorch optimizer"""
        
        optimizer_registry = {
            'adamw': optim.AdamW,
            'adam': optim.Adam,
            'sgd': optim.SGD,
            'rmsprop': optim.RMSprop
        }
        
        if optimizer_type not in optimizer_registry:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        
        optimizer_class = optimizer_registry[optimizer_type]
        
        # Group parameters for different learning rates
        param_groups = PyTorchTrainingUtils._group_parameters(model, weight_decay)
        
        optimizer = optimizer_class(
            param_groups,
            lr=learning_rate,
            **kwargs
        )
        
        logger.info(f"Optimizer created: {optimizer_type} with lr={learning_rate}")
        return optimizer
    
    @staticmethod
    def _group_parameters(model: nn.Module, weight_decay: float) -> List[Dict]:
        """Group parameters for different learning rates and weight decay"""
        
        # Parameters that should not have weight decay
        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
        
        param_groups = [
            {
                'params': [p for n, p in model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': weight_decay
            },
            {
                'params': [p for n, p in model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        
        return param_groups
    
    @staticmethod
    def create_scheduler(optimizer: optim.Optimizer,
                        scheduler_type: str = "cosine",
                        **kwargs) -> optim.lr_scheduler._LRScheduler:
        """Create PyTorch learning rate scheduler"""
        
        scheduler_registry = {
            'cosine': optim.lr_scheduler.CosineAnnealingLR,
            'step': optim.lr_scheduler.StepLR,
            'exponential': optim.lr_scheduler.ExponentialLR,
            'reduce_on_plateau': optim.lr_scheduler.ReduceLROnPlateau,
            'cosine_warm_restarts': optim.lr_scheduler.CosineAnnealingWarmRestarts
        }
        
        if scheduler_type not in scheduler_registry:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
        
        scheduler_class = scheduler_registry[scheduler_type]
        scheduler = scheduler_class(optimizer, **kwargs)
        
        logger.info(f"Scheduler created: {scheduler_type}")
        return scheduler
    
    @staticmethod
    def create_mixed_precision_scaler() -> Optional[GradScaler]:
        """Create mixed precision scaler if CUDA is available"""
        if torch.cuda.is_available():
            return GradScaler()
        return None


class PyTorchDistributedUtils:
    """PyTorch distributed training utilities"""
    
    @staticmethod
    def setup_distributed(rank: int, world_size: int, backend: str = "nccl"):
        """Setup distributed training environment"""
        
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, distributed training not supported")
            return False
        
        try:
            dist.init_process_group(
                backend=backend,
                init_method='env://',
                world_size=world_size,
                rank=rank
            )
            
            # Set device
            torch.cuda.set_device(rank)
            
            logger.info(f"Distributed training initialized: rank={rank}, world_size={world_size}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize distributed training: {e}")
            return False
    
    @staticmethod
    def cleanup_distributed():
        """Cleanup distributed training environment"""
        if dist.is_initialized():
            dist.destroy_process_group()
            logger.info("Distributed training cleaned up")
    
    @staticmethod
    def wrap_model_for_distributed(model: nn.Module, device: torch.device) -> nn.Module:
        """Wrap model for distributed training"""
        if dist.is_initialized():
            model = DDP(model, device_ids=[device])
            logger.info("Model wrapped for distributed training")
        
        return model


class PyTorchMemoryUtils:
    """PyTorch memory management utilities"""
    
    @staticmethod
    def get_memory_info() -> Dict[str, Any]:
        """Get comprehensive PyTorch memory information"""
        memory_info = {}
        
        if torch.cuda.is_available():
            gpu_memory = []
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i)
                reserved = torch.cuda.memory_reserved(i)
                total = torch.cuda.get_device_properties(i).total_memory
                
                gpu_memory.append({
                    'gpu_id': i,
                    'allocated_mb': allocated / (1024**2),
                    'reserved_mb': reserved / (1024**2),
                    'total_mb': total / (1024**2),
                    'free_mb': (total - reserved) / (1024**2),
                    'utilization_percent': (reserved / total) * 100
                })
            
            memory_info['gpu_memory'] = gpu_memory
        
        # System memory
        import psutil
        system_memory = psutil.virtual_memory()
        memory_info['system_memory'] = {
            'total_gb': system_memory.total / (1024**3),
            'available_gb': system_memory.available / (1024**3),
            'used_gb': system_memory.used / (1024**3),
            'percent': system_memory.percent
        }
        
        return memory_info
    
    @staticmethod
    def clear_gpu_cache():
        """Clear PyTorch GPU cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared")
    
    @staticmethod
    def optimize_memory_usage(model: nn.Module, 
                            input_size: Tuple[int, ...],
                            max_memory_usage: float = 0.8) -> int:
        """Find optimal batch size based on memory constraints"""
        
        if not torch.cuda.is_available():
            return 1
        
        device = next(model.parameters()).device
        batch_size = 1
        
        while True:
            try:
                # Clear cache
                torch.cuda.empty_cache()
                
                # Test batch size
                test_input = torch.randn(batch_size, *input_size).to(device)
                
                with torch.no_grad():
                    _ = model(test_input)
                
                # Check memory usage
                allocated = torch.cuda.memory_allocated(device)
                total = torch.cuda.get_device_properties(device).total_memory
                usage_ratio = allocated / total
                
                if usage_ratio <= max_memory_usage:
                    logger.info(f"Optimal batch size found: {batch_size}")
                    return batch_size
                
                batch_size *= 2
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    batch_size //= 2
                    torch.cuda.empty_cache()
                    logger.info(f"Memory limit reached, optimal batch size: {batch_size}")
                    return max(1, batch_size)
                else:
                    raise e


def main():
    """Main execution function"""
    logger.info("Starting PyTorch Utilities for Blaze AI...")
    
    # Create PyTorch configuration
    pytorch_config = PyTorchConfiguration(
        enable_cudnn_benchmark=True,
        enable_tf32=True,
        use_amp=True,
        memory_fraction=0.9
    )
    
    # Initialize PyTorch optimizer
    pytorch_optimizer = PyTorchOptimizer(pytorch_config)
    
    # Get device information
    device_info = pytorch_optimizer.get_device_info()
    logger.info(f"Device Information: {device_info}")
    
    # Get memory information
    memory_info = PyTorchMemoryUtils.get_memory_info()
    logger.info(f"Memory Information: {memory_info}")
    
    logger.info("PyTorch Utilities initialized successfully!")


if __name__ == "__main__":
    main()
