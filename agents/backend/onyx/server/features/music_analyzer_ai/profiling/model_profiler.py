"""
Model Profiler
Profile model forward pass and memory usage
"""

from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")


class ModelProfiler:
    """Profile model performance"""
    
    def __init__(self, model: nn.Module, device: str = "cuda"):
        self.model = model
        self.device = device
        self.hooks = []
    
    def profile_forward(
        self,
        input_shape: Tuple[int, ...],
        num_runs: int = 100,
        warmup_runs: int = 10
    ) -> Dict[str, Any]:
        """
        Profile forward pass
        
        Args:
            input_shape: Input tensor shape
            num_runs: Number of profiling runs
            warmup_runs: Number of warmup runs
        
        Returns:
            Profiling results
        """
        if not TORCH_AVAILABLE:
            return {}
        
        # Create dummy input
        dummy_input = torch.randn(input_shape).to(self.device)
        
        # Warmup
        self.model.eval()
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = self.model(dummy_input)
        
        # Profile
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        
        import time
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = self.model(dummy_input)
        
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        
        elapsed_time = time.time() - start_time
        
        avg_time = elapsed_time / num_runs
        throughput = 1.0 / avg_time if avg_time > 0 else 0.0
        
        return {
            "avg_forward_time_ms": avg_time * 1000,
            "total_time_ms": elapsed_time * 1000,
            "throughput_samples_per_sec": throughput,
            "num_runs": num_runs
        }
    
    def profile_memory(
        self,
        input_shape: Tuple[int, ...]
    ) -> Dict[str, float]:
        """
        Profile memory usage
        
        Args:
            input_shape: Input tensor shape
        
        Returns:
            Memory usage statistics
        """
        if not TORCH_AVAILABLE:
            return {}
        
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            
            dummy_input = torch.randn(input_shape).to(self.device)
            
            self.model.eval()
            with torch.no_grad():
                _ = self.model(dummy_input)
            
            torch.cuda.synchronize()
            
            peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
            current_memory = torch.cuda.memory_allocated() / 1024 / 1024
            
            return {
                "peak_memory_mb": peak_memory,
                "current_memory_mb": current_memory
            }
        
        return {}
    
    def profile_full(
        self,
        input_shape: Tuple[int, ...],
        num_runs: int = 100
    ) -> Dict[str, Any]:
        """Profile both forward pass and memory"""
        forward_stats = self.profile_forward(input_shape, num_runs)
        memory_stats = self.profile_memory(input_shape)
        
        return {
            **forward_stats,
            **memory_stats
        }



