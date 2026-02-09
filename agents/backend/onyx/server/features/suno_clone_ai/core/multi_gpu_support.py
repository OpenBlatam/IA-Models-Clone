"""
Multi-GPU Support for Music Generation

Implements:
- DataParallel for single-node multi-GPU
- DistributedDataParallel for multi-node training
- Proper GPU device management
- Efficient batch distribution
"""

import logging
import os
from typing import Optional, Dict, Any, List
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import numpy as np

logger = logging.getLogger(__name__)


class MultiGPUGenerator:
    """
    Multi-GPU wrapper for music generation models.
    """
    
    def __init__(
        self,
        model: nn.Module,
        use_ddp: bool = False,
        device_ids: Optional[List[int]] = None
    ):
        """
        Initialize multi-GPU generator.
        
        Args:
            model: PyTorch model
            use_ddp: Use DistributedDataParallel instead of DataParallel
            device_ids: List of GPU device IDs to use
        """
        self.model = model
        self.use_ddp = use_ddp
        self.device_ids = device_ids or list(range(torch.cuda.device_count()))
        self.is_multi_gpu = len(self.device_ids) > 1
        
        if self.is_multi_gpu:
            self._setup_multi_gpu()
        else:
            self.wrapped_model = model
    
    def _setup_multi_gpu(self) -> None:
        """Setup multi-GPU configuration."""
        if self.use_ddp:
            # DistributedDataParallel setup
            if not torch.distributed.is_initialized():
                raise RuntimeError(
                    "DistributedDataParallel requires torch.distributed to be initialized. "
                    "Use torch.distributed.init_process_group() first."
                )
            
            self.wrapped_model = DistributedDataParallel(
                self.model,
                device_ids=[torch.distributed.get_rank()],
                output_device=torch.distributed.get_rank()
            )
            logger.info("Using DistributedDataParallel")
        else:
            # DataParallel setup
            self.wrapped_model = DataParallel(
                self.model,
                device_ids=self.device_ids
            )
            logger.info(f"Using DataParallel on devices: {self.device_ids}")
    
    def generate(self, *args, **kwargs) -> Any:
        """
        Generate using multi-GPU model.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Generation output
        """
        return self.wrapped_model(*args, **kwargs)
    
    def __call__(self, *args, **kwargs) -> Any:
        """Make model callable."""
        return self.generate(*args, **kwargs)


def setup_distributed_training(
    backend: str = "nccl",
    init_method: Optional[str] = None,
    rank: Optional[int] = None,
    world_size: Optional[int] = None
) -> Dict[str, int]:
    """
    Setup distributed training environment.
    
    Args:
        backend: Distributed backend (nccl, gloo, mpi)
        init_method: Initialization method
        rank: Process rank (if None, uses environment variables)
        world_size: World size (if None, uses environment variables)
        
    Returns:
        Dictionary with rank and world_size
    """
    if rank is None:
        rank = int(os.environ.get('RANK', 0))
    
    if world_size is None:
        world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    if init_method is None:
        init_method = os.environ.get('MASTER_ADDR', 'tcp://localhost:23456')
    
    torch.distributed.init_process_group(
        backend=backend,
        init_method=init_method,
        rank=rank,
        world_size=world_size
    )
    
    # Set device for this process
    local_rank = int(os.environ.get('LOCAL_RANK', rank))
    torch.cuda.set_device(local_rank)
    
    logger.info(
        f"Distributed training initialized: rank={rank}, "
        f"world_size={world_size}, local_rank={local_rank}"
    )
    
    return {
        'rank': rank,
        'world_size': world_size,
        'local_rank': local_rank
    }


def cleanup_distributed() -> None:
    """Cleanup distributed training."""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
        logger.info("Distributed training cleaned up")


class GradientAccumulator:
    """
    Gradient accumulation for large batch sizes.
    """
    
    def __init__(
        self,
        model: nn.Module,
        accumulation_steps: int = 4
    ):
        """
        Initialize gradient accumulator.
        
        Args:
            model: PyTorch model
            accumulation_steps: Number of steps to accumulate gradients
        """
        self.model = model
        self.accumulation_steps = accumulation_steps
        self.step_count = 0
    
    def accumulate(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer) -> None:
        """
        Accumulate gradients.
        
        Args:
            loss: Loss tensor
            optimizer: Optimizer
        """
        # Scale loss by accumulation steps
        scaled_loss = loss / self.accumulation_steps
        scaled_loss.backward()
        
        self.step_count += 1
        
        if self.step_count % self.accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            self.step_count = 0
    
    def zero_grad(self, optimizer: torch.optim.Optimizer) -> None:
        """
        Zero gradients if needed.
        
        Args:
            optimizer: Optimizer
        """
        if self.step_count == 0:
            optimizer.zero_grad()


def get_optimal_batch_size(
    model: nn.Module,
    input_shape: tuple,
    max_memory_gb: float = 24.0,
    device: Optional[str] = None
) -> int:
    """
    Find optimal batch size for given model and memory constraints.
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape
        max_memory_gb: Maximum GPU memory in GB
        device: Device to test on
        
    Returns:
        Optimal batch size
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    if device == "cpu":
        return 1
    
    model.eval()
    model.to(device)
    
    # Try increasing batch sizes
    batch_size = 1
    max_batch_size = 128
    
    while batch_size <= max_batch_size:
        try:
            # Clear cache
            torch.cuda.empty_cache()
            
            # Test with current batch size
            test_input = torch.randn(
                (batch_size,) + input_shape,
                device=device
            )
            
            with torch.no_grad():
                _ = model(test_input)
            
            # Check memory usage
            memory_used = torch.cuda.memory_allocated(device) / (1024**3)
            
            if memory_used > max_memory_gb * 0.9:  # Leave 10% buffer
                break
            
            batch_size *= 2
            
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            batch_size = batch_size // 2
            break
    
    torch.cuda.empty_cache()
    logger.info(f"Optimal batch size: {batch_size}")
    
    return max(batch_size, 1)


def profile_model(
    model: nn.Module,
    input_shape: tuple,
    device: Optional[str] = None,
    num_iterations: int = 10
) -> Dict[str, Any]:
    """
    Profile model performance.
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape
        device: Device to profile on
        num_iterations: Number of iterations for profiling
        
    Returns:
        Dictionary with profiling results
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Warmup
    test_input = torch.randn((1,) + input_shape, device=device)
    with torch.no_grad():
        for _ in range(5):
            _ = model(test_input)
    
    # Profile
    if device == "cuda":
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(test_input)
        end_event.record()
        
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) / num_iterations
    else:
        import time
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(test_input)
        elapsed_time = (time.time() - start_time) / num_iterations * 1000  # Convert to ms
    
    # Memory usage
    memory_stats = {}
    if device == "cuda":
        memory_stats = {
            'allocated_gb': torch.cuda.memory_allocated(device) / (1024**3),
            'reserved_gb': torch.cuda.memory_reserved(device) / (1024**3),
            'max_allocated_gb': torch.cuda.max_memory_allocated(device) / (1024**3)
        }
    
    results = {
        'avg_inference_time_ms': elapsed_time,
        'throughput_samples_per_sec': 1000.0 / elapsed_time if elapsed_time > 0 else 0,
        'memory_stats': memory_stats,
        'device': device
    }
    
    logger.info(f"Model profiling results: {results}")
    
    return results













