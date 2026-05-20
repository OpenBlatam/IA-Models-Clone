"""
ML Optimizer - Performance optimizations for ML inference
"""

from typing import Optional, Dict, Any, Callable
import logging
import time
from functools import wraps
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for optimizations"""
    use_mixed_precision: bool = True
    use_tensor_cores: bool = True
    enable_cudnn_benchmark: bool = True
    max_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True
    compile_model: bool = True


class MLOptimizer:
    """
    ML Optimizer for performance improvements:
    - Mixed precision inference
    - Model compilation
    - Batch processing optimization
    - Memory management
    - GPU utilization
    """
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()
        self.optimized_models: Dict[str, Any] = {}
    
    def optimize_model(self, model: Any, model_id: str) -> Any:
        """Optimize a model for inference"""
        if model_id in self.optimized_models:
            return self.optimized_models[model_id]
        
        try:
            import torch
            
            # Compile model if supported
            if self.config.compile_model and hasattr(torch, "compile"):
                try:
                    model = torch.compile(model, mode="reduce-overhead")
                    logger.info(f"Compiled model: {model_id}")
                except Exception as e:
                    logger.warning(f"Could not compile model {model_id}: {str(e)}")
            
            # Enable cuDNN benchmark for consistent input sizes
            if self.config.enable_cudnn_benchmark:
                try:
                    torch.backends.cudnn.benchmark = True
                except AttributeError:
                    pass
            
            self.optimized_models[model_id] = model
            return model
        
        except ImportError:
            logger.warning("PyTorch not available, skipping optimizations")
            return model
    
    def optimize_inference(
        self,
        model: Any,
        input_data: Any,
        use_amp: Optional[bool] = None
    ) -> Any:
        """Run optimized inference"""
        use_amp = use_amp if use_amp is not None else self.config.use_mixed_precision
        
        try:
            import torch
            
            if use_amp and torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    return self._run_inference(model, input_data)
            else:
                return self._run_inference(model, input_data)
        
        except ImportError:
            return self._run_inference(model, input_data)
    
    def _run_inference(self, model: Any, input_data: Any) -> Any:
        """Run actual inference"""
        import torch
        
        with torch.no_grad():
            if hasattr(model, "__call__"):
                return model(input_data)
            else:
                return model.predict(input_data)
    
    def create_optimized_dataloader(
        self,
        dataset: Any,
        batch_size: int,
        shuffle: bool = False
    ) -> Any:
        """Create optimized DataLoader"""
        try:
            import torch
            
            return torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=self.config.max_workers,
                prefetch_factor=self.config.prefetch_factor,
                pin_memory=self.config.pin_memory,
                persistent_workers=True if self.config.max_workers > 0 else False
            )
        
        except ImportError:
            logger.warning("PyTorch not available")
            return None


def timing_decorator(func: Callable) -> Callable:
    """Decorator to measure function execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        logger.debug(f"{func.__name__} executed in {execution_time:.4f}s")
        return result
    return wrapper


def memory_profiler(func: Callable) -> Callable:
    """Decorator to profile memory usage"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / 1024 / 1024  # MB
            
            result = func(*args, **kwargs)
            
            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            mem_used = mem_after - mem_before
            
            logger.debug(f"{func.__name__} used {mem_used:.2f} MB")
            return result
        
        except ImportError:
            return func(*args, **kwargs)
    
    return wrapper


