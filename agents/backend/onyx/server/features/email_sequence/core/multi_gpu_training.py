from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DataParallel, DistributedDataParallel
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import os
import time
import json
import subprocess
import socket
from pathlib import Path
from dataclasses import dataclass
import warnings
from core.training_logger import TrainingLogger, TrainingEventType, LogLevel
from core.error_handling import ErrorHandler, ModelError
    import torch.nn as nn
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Multi-GPU Training System

Comprehensive multi-GPU training support using DataParallel and DistributedDataParallel
for both single-machine multi-GPU and distributed multi-node training.
"""




@dataclass
class MultiGPUConfig:
    """Configuration for multi-GPU training"""
    
    # Training mode
    training_mode: str = "auto"  # "auto", "single_gpu", "data_parallel", "distributed"
    
    # DataParallel settings
    enable_data_parallel: bool = True
    device_ids: Optional[List[int]] = None  # None for all available GPUs
    
    # Distributed settings
    enable_distributed: bool = False
    backend: str = "nccl"  # "nccl" for GPU, "gloo" for CPU
    init_method: str = "env://"
    world_size: int = -1
    rank: int = -1
    local_rank: int = -1
    
    # Communication settings
    find_unused_parameters: bool = False
    broadcast_buffers: bool = True
    bucket_cap_mb: int = 25
    static_graph: bool = False
    
    # Performance settings
    enable_gradient_as_bucket_view: bool = False
    enable_find_unused_parameters: bool = False
    
    # Monitoring
    enable_gpu_monitoring: bool = True
    sync_bn: bool = False  # Synchronize batch normalization


class GPUMonitor:
    """GPU monitoring and analysis"""
    
    def __init__(self, logger: Optional[TrainingLogger] = None):
        
    """__init__ function."""
self.logger = logger
        self.gpu_metrics = {
            "memory_allocated": [],
            "memory_reserved": [],
            "utilization": [],
            "temperature": [],
            "power_usage": []
        }
        self.start_time = None
    
    def start_monitoring(self) -> Any:
        """Start GPU monitoring"""
        self.start_time = time.time()
        if self.logger:
            self.logger.log_info("GPU monitoring started")
    
    def record_gpu_metrics(self, device_ids: List[int] = None):
        """Record GPU metrics for specified devices"""
        
        if not torch.cuda.is_available():
            return
        
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        
        for device_id in device_ids:
            try:
                # Memory metrics
                memory_allocated = torch.cuda.memory_allocated(device_id)
                memory_reserved = torch.cuda.memory_reserved(device_id)
                
                # Utilization (if available)
                utilization = torch.cuda.utilization(device_id) if hasattr(torch.cuda, 'utilization') else 0
                
                # Store metrics
                self.gpu_metrics["memory_allocated"].append(memory_allocated)
                self.gpu_metrics["memory_reserved"].append(memory_reserved)
                self.gpu_metrics["utilization"].append(utilization)
                
                # Log metrics periodically
                if self.logger and len(self.gpu_metrics["memory_allocated"]) % 100 == 0:
                    self.logger.log_info(
                        f"GPU {device_id} - Memory: {memory_allocated/1024**3:.2f}GB, "
                        f"Utilization: {utilization}%"
                    )
                    
            except Exception as e:
                if self.logger:
                    self.logger.log_warning(f"Failed to record GPU {device_id} metrics: {e}")
    
    def get_gpu_summary(self) -> Dict[str, Any]:
        """Get GPU monitoring summary"""
        
        if not self.gpu_metrics["memory_allocated"]:
            return {}
        
        summary = {}
        for metric_name, values in self.gpu_metrics.items():
            if values:
                summary[f"{metric_name}_mean"] = sum(values) / len(values)
                summary[f"{metric_name}_max"] = max(values)
                summary[f"{metric_name}_min"] = min(values)
        
        # Overall statistics
        total_time = time.time() - self.start_time if self.start_time else 0
        summary["total_monitoring_time"] = total_time
        summary["total_recordings"] = len(self.gpu_metrics["memory_allocated"])
        
        return summary
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information"""
        
        if not torch.cuda.is_available():
            return {"cuda_available": False}
        
        gpu_info = {
            "cuda_available": True,
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
            "devices": {}
        }
        
        for i in range(torch.cuda.device_count()):
            try:
                gpu_info["devices"][i] = {
                    "name": torch.cuda.get_device_name(i),
                    "capability": torch.cuda.get_device_capability(i),
                    "total_memory": torch.cuda.get_device_properties(i).total_memory,
                    "multi_processor_count": torch.cuda.get_device_properties(i).multi_processor_count
                }
            except Exception as e:
                gpu_info["devices"][i] = {"error": str(e)}
        
        return gpu_info


class DataParallelManager:
    """DataParallel training manager"""
    
    def __init__(self, config: MultiGPUConfig, logger: Optional[TrainingLogger] = None):
        
    """__init__ function."""
self.config = config
        self.logger = logger
        self.is_initialized = False
    
    def setup_data_parallel(self, model: nn.Module) -> nn.Module:
        """Setup DataParallel for single machine multi-GPU training"""
        
        if not torch.cuda.is_available():
            if self.logger:
                self.logger.log_warning("CUDA not available, using CPU")
            return model
        
        device_count = torch.cuda.device_count()
        if device_count < 2:
            if self.logger:
                self.logger.log_info("Single GPU detected, skipping DataParallel")
            return model
        
        try:
            # Determine device IDs
            if self.config.device_ids is None:
                device_ids = list(range(device_count))
            else:
                device_ids = [i for i in self.config.device_ids if i < device_count]
            
            if len(device_ids) < 2:
                if self.logger:
                    self.logger.log_info("Less than 2 GPUs available, skipping DataParallel")
                return model
            
            # Move model to first GPU
            model = model.cuda(device_ids[0])
            
            # Wrap with DataParallel
            model = DataParallel(model, device_ids=device_ids)
            
            self.is_initialized = True
            
            if self.logger:
                self.logger.log_info(f"DataParallel initialized with {len(device_ids)} GPUs: {device_ids}")
            
            return model
            
        except Exception as e:
            if self.logger:
                self.logger.log_error(e, "DataParallel setup", "setup_data_parallel")
            return model
    
    def optimize_data_parallel(self, model: nn.Module) -> nn.Module:
        """Apply optimizations for DataParallel training"""
        
        if not isinstance(model, DataParallel):
            return model
        
        try:
            # Set DataParallel parameters
            model.broadcast_buffers = self.config.broadcast_buffers
            
            if self.logger:
                self.logger.log_info("DataParallel optimizations applied")
            
            return model
            
        except Exception as e:
            if self.logger:
                self.logger.log_error(e, "DataParallel optimization", "optimize_data_parallel")
            return model
    
    def get_data_parallel_info(self, model: nn.Module) -> Dict[str, Any]:
        """Get DataParallel information"""
        
        if not isinstance(model, DataParallel):
            return {"is_data_parallel": False}
        
        return {
            "is_data_parallel": True,
            "device_ids": model.device_ids,
            "output_device": model.output_device,
            "broadcast_buffers": model.broadcast_buffers,
            "dim": model.dim
        }


class DistributedManager:
    """Distributed training manager"""
    
    def __init__(self, config: MultiGPUConfig, logger: Optional[TrainingLogger] = None):
        
    """__init__ function."""
self.config = config
        self.logger = logger
        self.is_initialized = False
        self.world_size = 0
        self.rank = 0
        self.local_rank = 0
    
    def setup_distributed(self, rank: int = -1, world_size: int = -1, local_rank: int = -1):
        """Setup distributed training"""
        
        try:
            # Use environment variables if not provided
            if rank == -1:
                rank = int(os.environ.get("RANK", 0))
            if world_size == -1:
                world_size = int(os.environ.get("WORLD_SIZE", 1))
            if local_rank == -1:
                local_rank = int(os.environ.get("LOCAL_RANK", 0))
            
            self.rank = rank
            self.world_size = world_size
            self.local_rank = local_rank
            
            # Initialize process group
            if world_size > 1:
                dist.init_process_group(
                    backend=self.config.backend,
                    init_method=self.config.init_method,
                    world_size=world_size,
                    rank=rank
                )
                
                # Set device
                if torch.cuda.is_available():
                    torch.cuda.set_device(local_rank)
                
                self.is_initialized = True
                
                if self.logger:
                    self.logger.log_info(
                        f"Distributed training initialized - Rank: {rank}/{world_size}, "
                        f"Local Rank: {local_rank}, Backend: {self.config.backend}"
                    )
            else:
                if self.logger:
                    self.logger.log_warning("World size is 1, skipping distributed setup")
                    
        except Exception as e:
            if self.logger:
                self.logger.log_error(e, "Distributed setup", "setup_distributed")
            raise
    
    def setup_distributed_model(self, model: nn.Module) -> nn.Module:
        """Setup DistributedDataParallel model"""
        
        if not self.is_initialized:
            if self.logger:
                self.logger.log_warning("Distributed not initialized, returning original model")
            return model
        
        try:
            # Move model to device
            if torch.cuda.is_available():
                model = model.cuda(self.local_rank)
            
            # Synchronize batch normalization if enabled
            if self.config.sync_bn:
                model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
                if self.logger:
                    self.logger.log_info("Synchronized batch normalization enabled")
            
            # Wrap with DistributedDataParallel
            model = DistributedDataParallel(
                model,
                device_ids=[self.local_rank] if torch.cuda.is_available() else None,
                output_device=self.local_rank if torch.cuda.is_available() else None,
                find_unused_parameters=self.config.find_unused_parameters,
                broadcast_buffers=self.config.broadcast_buffers,
                bucket_cap_mb=self.config.bucket_cap_mb,
                static_graph=self.config.static_graph,
                gradient_as_bucket_view=self.config.enable_gradient_as_bucket_view
            )
            
            if self.logger:
                self.logger.log_info("DistributedDataParallel model setup completed")
            
            return model
            
        except Exception as e:
            if self.logger:
                self.logger.log_error(e, "Distributed model setup", "setup_distributed_model")
            return model
    
    def setup_distributed_dataloader(self, dataset, batch_size: int, **kwargs) -> DataLoader:
        """Setup distributed DataLoader with DistributedSampler"""
        
        if not self.is_initialized:
            return DataLoader(dataset, batch_size=batch_size, **kwargs)
        
        try:
            # Create distributed sampler
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=kwargs.get("shuffle", True)
            )
            
            # Create distributed DataLoader
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=kwargs.get("num_workers", 4),
                pin_memory=kwargs.get("pin_memory", True),
                persistent_workers=kwargs.get("persistent_workers", True),
                **{k: v for k, v in kwargs.items() if k not in ["sampler", "shuffle"]}
            )
            
            if self.logger:
                self.logger.log_info("Distributed DataLoader setup completed")
            
            return dataloader
            
        except Exception as e:
            if self.logger:
                self.logger.log_error(e, "Distributed DataLoader setup", "setup_distributed_dataloader")
            return DataLoader(dataset, batch_size=batch_size, **kwargs)
    
    def cleanup(self) -> Any:
        """Cleanup distributed training"""
        
        if self.is_initialized:
            dist.destroy_process_group()
            self.is_initialized = False
            
            if self.logger:
                self.logger.log_info("Distributed training cleanup completed")
    
    def get_distributed_info(self) -> Dict[str, Any]:
        """Get distributed training information"""
        
        return {
            "is_initialized": self.is_initialized,
            "world_size": self.world_size,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "backend": self.config.backend if self.is_initialized else None
        }


class MultiGPUTrainer:
    """Main multi-GPU training manager"""
    
    def __init__(
        self,
        config: MultiGPUConfig,
        logger: Optional[TrainingLogger] = None,
        device: str = "auto"
    ):
        
    """__init__ function."""
self.config = config
        self.logger = logger
        
        # Setup device
        match device:
    case "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Initialize components
        self.gpu_monitor = GPUMonitor(logger)
        self.data_parallel_manager = DataParallelManager(config, logger)
        self.distributed_manager = DistributedManager(config, logger)
        
        # Training state
        self.training_mode = "single_gpu"
        self.is_initialized = False
        
        if self.logger:
            self.logger.log_info(f"Multi-GPU trainer initialized on device: {self.device}")
    
    def auto_detect_training_mode(self) -> str:
        """Automatically detect the best training mode"""
        
        if not torch.cuda.is_available():
            return "single_gpu"
        
        device_count = torch.cuda.device_count()
        
        if device_count == 1:
            return "single_gpu"
        elif self.config.enable_distributed and device_count > 1:
            return "distributed"
        elif self.config.enable_data_parallel and device_count > 1:
            return "data_parallel"
        else:
            return "single_gpu"
    
    def initialize_training(self, model: nn.Module, rank: int = -1, world_size: int = -1):
        """Initialize multi-GPU training"""
        
        try:
            # Determine training mode
            if self.config.training_mode == "auto":
                self.training_mode = self.auto_detect_training_mode()
            else:
                self.training_mode = self.config.training_mode
            
            if self.logger:
                self.logger.log_info(f"Training mode: {self.training_mode}")
            
            # Initialize based on training mode
            if self.training_mode == "distributed":
                self.distributed_manager.setup_distributed(rank, world_size)
                model = self.distributed_manager.setup_distributed_model(model)
                
            elif self.training_mode == "data_parallel":
                model = self.data_parallel_manager.setup_data_parallel(model)
                model = self.data_parallel_manager.optimize_data_parallel(model)
            
            # Start GPU monitoring
            if self.config.enable_gpu_monitoring:
                self.gpu_monitor.start_monitoring()
            
            self.is_initialized = True
            
            if self.logger:
                self.logger.log_info("Multi-GPU training initialization completed")
            
            return model
            
        except Exception as e:
            if self.logger:
                self.logger.log_error(e, "Multi-GPU initialization", "initialize_training")
            raise
    
    def setup_dataloader(self, dataset, batch_size: int, **kwargs) -> DataLoader:
        """Setup DataLoader for multi-GPU training"""
        
        if self.training_mode == "distributed":
            return self.distributed_manager.setup_distributed_dataloader(
                dataset, batch_size, **kwargs
            )
        else:
            return DataLoader(dataset, batch_size=batch_size, **kwargs)
    
    def record_gpu_metrics(self) -> Any:
        """Record GPU metrics"""
        
        if not self.config.enable_gpu_monitoring:
            return
        
        device_ids = None
        if self.training_mode == "data_parallel":
            device_ids = self.data_parallel_manager.config.device_ids
        elif self.training_mode == "distributed":
            device_ids = [self.distributed_manager.local_rank]
        
        self.gpu_monitor.record_gpu_metrics(device_ids)
    
    def get_training_info(self) -> Dict[str, Any]:
        """Get comprehensive training information"""
        
        info = {
            "training_mode": self.training_mode,
            "is_initialized": self.is_initialized,
            "device": str(self.device),
            "gpu_info": self.gpu_monitor.get_gpu_info()
        }
        
        if self.training_mode == "data_parallel":
            info["data_parallel_info"] = self.data_parallel_manager.get_data_parallel_info(None)
        
        if self.training_mode == "distributed":
            info["distributed_info"] = self.distributed_manager.get_distributed_info()
        
        return info
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        
        return {
            "gpu_summary": self.gpu_monitor.get_gpu_summary(),
            "training_info": self.get_training_info()
        }
    
    def cleanup(self) -> Any:
        """Cleanup multi-GPU training"""
        
        if self.training_mode == "distributed":
            self.distributed_manager.cleanup()
        
        if self.logger:
            self.logger.log_info("Multi-GPU training cleanup completed")


# Utility functions for distributed training
def setup_distributed_environment(
    world_size: int,
    backend: str = "nccl",
    init_method: str = "env://"
) -> Dict[str, Any]:
    """Setup distributed training environment"""
    
    env_info = {
        "MASTER_ADDR": "localhost",
        "MASTER_PORT": "12355",
        "WORLD_SIZE": str(world_size),
        "BACKEND": backend,
        "INIT_METHOD": init_method
    }
    
    return env_info


def launch_distributed_training(
    script_path: str,
    world_size: int,
    backend: str = "nccl",
    master_addr: str = "localhost",
    master_port: str = "12355"
) -> subprocess.Popen:
    """Launch distributed training using torchrun"""
    
    cmd = [
        "torchrun",
        f"--nproc_per_node={world_size}",
        f"--nnodes=1",
        f"--node_rank=0",
        f"--master_addr={master_addr}",
        f"--master_port={master_port}",
        script_path
    ]
    
    return subprocess.Popen(cmd)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")


def get_free_port() -> int:
    """Get a free port for distributed training"""
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    
    return port


# Utility functions
def create_multi_gpu_trainer(
    logger: Optional[TrainingLogger] = None,
    device: str = "auto",
    **config_kwargs
) -> MultiGPUTrainer:
    """Create a multi-GPU trainer with default settings"""
    
    config = MultiGPUConfig(**config_kwargs)
    return MultiGPUTrainer(config, logger, device)


def optimize_model_for_multi_gpu(
    model: nn.Module,
    training_mode: str = "auto",
    enable_data_parallel: bool = True,
    enable_distributed: bool = False,
    logger: Optional[TrainingLogger] = None
) -> Tuple[nn.Module, MultiGPUTrainer]:
    """Optimize model for multi-GPU training"""
    
    config = MultiGPUConfig(
        training_mode=training_mode,
        enable_data_parallel=enable_data_parallel,
        enable_distributed=enable_distributed
    )
    
    trainer = MultiGPUTrainer(config, logger)
    optimized_model = trainer.initialize_training(model)
    
    return optimized_model, trainer


if __name__ == "__main__":
    # Example usage
    
    # Simple model for testing
    class TestModel(nn.Module):
        def __init__(self) -> Any:
            super().__init__()
            self.linear = nn.Linear(10, 2)
        
        def forward(self, x) -> Any:
            return self.linear(x)
    
    # Create multi-GPU trainer
    trainer = create_multi_gpu_trainer(
        training_mode="auto",
        enable_data_parallel=True,
        enable_distributed=False
    )
    
    # Create and optimize model
    model = TestModel()
    optimized_model, trainer = optimize_model_for_multi_gpu(
        model,
        training_mode="auto",
        enable_data_parallel=True
    )
    
    # Get training information
    info = trainer.get_training_info()
    print(f"Training info: {json.dumps(info, indent=2)}")
    
    # Get performance summary
    summary = trainer.get_performance_summary()
    print(f"Performance summary: {json.dumps(summary, indent=2)}") 