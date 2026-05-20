"""
Accelerator Utilities
=====================

Utilities for hardware acceleration and optimization.
"""

import logging
from typing import Optional, Dict, Any, List
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def setup_accelerator(device_type: str = 'auto') -> torch.device:
    """
    Setup accelerator (GPU, TPU, etc.).
    
    Args:
        device_type: Device type ('auto', 'cuda', 'mps', 'cpu')
        
    Returns:
        Device
    """
    if device_type == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            logger.info("Using MPS (Apple Silicon)")
        else:
            device = torch.device('cpu')
            logger.info("Using CPU")
    else:
        device = torch.device(device_type)
        logger.info(f"Using device: {device_type}")
    
    return device


def optimize_for_gpu(model: nn.Module, device: torch.device) -> nn.Module:
    """
    Optimize model for GPU.
    
    Args:
        model: PyTorch model
        device: GPU device
        
    Returns:
        Optimized model
    """
    model = model.to(device)
    
    # Enable cuDNN benchmarking for consistent input sizes
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        logger.info("GPU optimizations enabled")
    
    return model


def setup_multi_gpu(
    model: nn.Module,
    strategy: str = 'dataparallel',
    device_ids: Optional[List[int]] = None
) -> nn.Module:
    """
    Setup multi-GPU training.
    
    Args:
        model: PyTorch model
        strategy: Strategy ('dataparallel', 'distributed')
        device_ids: GPU device IDs
        
    Returns:
        Wrapped model
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, returning model as-is")
        return model
    
    num_gpus = torch.cuda.device_count()
    
    if num_gpus < 2:
        logger.warning(f"Only {num_gpus} GPU(s) available, returning model as-is")
        return model
    
    if strategy == 'dataparallel':
        if device_ids is None:
            device_ids = list(range(num_gpus))
        
        model = nn.DataParallel(model, device_ids=device_ids)
        logger.info(f"Model wrapped with DataParallel on GPUs: {device_ids}")
        
    elif strategy == 'distributed':
        # DistributedDataParallel requires more setup
        logger.warning("DistributedDataParallel requires additional setup, use training.distributed_training")
        return model
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return model


def get_accelerator_info() -> Dict[str, Any]:
    """
    Get accelerator information.
    
    Returns:
        Dictionary with accelerator info
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'mps_available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
    }
    
    if torch.cuda.is_available():
        info['cuda_version'] = torch.version.cuda
        info['cudnn_version'] = torch.backends.cudnn.version()
        info['cuda_devices'] = []
        
        for i in range(torch.cuda.device_count()):
            info['cuda_devices'].append({
                'id': i,
                'name': torch.cuda.get_device_name(i),
                'memory_total_gb': torch.cuda.get_device_properties(i).total_memory / 1024**3,
                'memory_allocated_gb': torch.cuda.memory_allocated(i) / 1024**3 if i == 0 else 0
            })
    
    return info


def enable_mixed_precision() -> Dict[str, Any]:
    """
    Enable mixed precision training.
    
    Returns:
        Configuration for mixed precision
    """
    return {
        'enabled': True,
        'dtype': torch.float16,
        'scaler': torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    }


def optimize_batch_size(
    model: nn.Module,
    input_shape: tuple,
    device: torch.device,
    start_batch_size: int = 1,
    max_batch_size: int = 128
) -> int:
    """
    Find optimal batch size for model.
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape
        device: Device to test on
        start_batch_size: Starting batch size
        max_batch_size: Maximum batch size
        
    Returns:
        Optimal batch size
    """
    model = model.to(device)
    model.eval()
    
    optimal_batch_size = start_batch_size
    
    for batch_size in range(start_batch_size, max_batch_size + 1, 2):
        try:
            dummy_input = torch.randn((batch_size, *input_shape)).to(device)
            
            with torch.no_grad():
                _ = model(dummy_input)
            
            optimal_batch_size = batch_size
            
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        
        except RuntimeError as e:
            if 'out of memory' in str(e):
                logger.info(f"OOM at batch size {batch_size}, optimal: {optimal_batch_size}")
                break
            else:
                raise e
    
    logger.info(f"Optimal batch size: {optimal_batch_size}")
    return optimal_batch_size



