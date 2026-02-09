"""
Training Profiler
Profile training loop performance
"""

from typing import Dict, Any, Optional
import logging
import time

logger = logging.getLogger(__name__)

try:
    import torch
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")


class TrainingProfiler:
    """Profile training performance"""
    
    def __init__(self):
        self.profiling_data: Dict[str, list] = {}
    
    def profile_epoch(
        self,
        model,
        dataloader: DataLoader,
        loss_fn,
        optimizer,
        device: str = "cuda"
    ) -> Dict[str, Any]:
        """
        Profile one training epoch
        
        Args:
            model: Model to profile
            dataloader: DataLoader
            loss_fn: Loss function
            optimizer: Optimizer
            device: Device
        
        Returns:
            Profiling results
        """
        if not TORCH_AVAILABLE:
            return {}
        
        model.train()
        
        times = {
            "data_loading": [],
            "forward": [],
            "backward": [],
            "optimizer_step": []
        }
        
        start_epoch = time.time()
        
        for batch_idx, batch in enumerate(dataloader):
            # Data loading time
            start_data = time.time()
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            times["data_loading"].append(time.time() - start_data)
            
            # Forward pass
            start_forward = time.time()
            outputs = model(batch)
            loss = loss_fn(outputs, batch)
            if device == "cuda":
                torch.cuda.synchronize()
            times["forward"].append(time.time() - start_forward)
            
            # Backward pass
            start_backward = time.time()
            loss.backward()
            if device == "cuda":
                torch.cuda.synchronize()
            times["backward"].append(time.time() - start_backward)
            
            # Optimizer step
            start_opt = time.time()
            optimizer.step()
            optimizer.zero_grad()
            if device == "cuda":
                torch.cuda.synchronize()
            times["optimizer_step"].append(time.time() - start_opt)
        
        total_time = time.time() - start_epoch
        
        # Calculate averages
        results = {
            "total_time_sec": total_time,
            "avg_data_loading_ms": sum(times["data_loading"]) / len(times["data_loading"]) * 1000,
            "avg_forward_ms": sum(times["forward"]) / len(times["forward"]) * 1000,
            "avg_backward_ms": sum(times["backward"]) / len(times["backward"]) * 1000,
            "avg_optimizer_step_ms": sum(times["optimizer_step"]) / len(times["optimizer_step"]) * 1000,
            "num_batches": len(times["forward"]),
            "samples_per_sec": len(dataloader.dataset) / total_time if total_time > 0 else 0.0
        }
        
        return results



