"""Performance profiling utilities."""

import torch
from typing import Dict, Any, Optional
import time
import logging

try:
    from torch.profiler import profile, ProfilerActivity
    PROFILER_AVAILABLE = True
except ImportError:
    PROFILER_AVAILABLE = False

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)


def profile_training(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    num_batches: int = 10
) -> Dict[str, Any]:
    """
    Profile training performance.
    
    Args:
        model: Model to profile
        dataloader: Data loader
        device: Target device
        num_batches: Number of batches to profile
    
    Returns:
        Profiling results
    """
    if not PROFILER_AVAILABLE:
        logger.warning("Profiler not available")
        return {}
    
    model.train()
    results = {}
    
    try:
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True
        ) as prof:
            for i, (data, target) in enumerate(dataloader):
                if i >= num_batches:
                    break
                
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = torch.nn.functional.cross_entropy(output, target)
                loss.backward()
        
        # Get profiling results
        results = {
            "cpu_time": prof.key_averages().table(sort_by="cpu_time_total"),
            "cuda_time": prof.key_averages().table(sort_by="cuda_time_total"),
        }
        
        logger.info("✅ Training profiling completed")
        
    except Exception as e:
        logger.error(f"❌ Profiling error: {e}")
    
    return results


def profile_inference(
    model: torch.nn.Module,
    input_data: torch.Tensor,
    device: torch.device,
    num_iterations: int = 100
) -> Dict[str, Any]:
    """
    Profile inference performance.
    
    Args:
        model: Model to profile
        input_data: Sample input
        device: Target device
        num_iterations: Number of iterations
    
    Returns:
        Profiling results
    """
    model.eval()
    input_data = input_data.to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_data)
    
    # Profile
    torch.cuda.synchronize() if device.type == "cuda" else None
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(input_data)
    
    torch.cuda.synchronize() if device.type == "cuda" else None
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_iterations
    throughput = 1.0 / avg_time
    
    results = {
        "avg_inference_time": avg_time,
        "throughput": throughput,
        "device": str(device)
    }
    
    logger.info(f"✅ Inference profiling: {avg_time*1000:.2f}ms, {throughput:.2f} samples/s")
    
    return results



