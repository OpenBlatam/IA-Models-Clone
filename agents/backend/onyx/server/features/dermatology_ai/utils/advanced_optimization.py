"""
Advanced Performance Optimizations
Memory-efficient and compute-optimized techniques
"""

import torch
import torch.nn as nn
from typing import Optional, Callable, Any, Dict, List
import logging
import warnings

logger = logging.getLogger(__name__)


def enable_gradient_checkpointing(model: nn.Module, segments: int = 1):
    """
    Enable gradient checkpointing for memory efficiency
    Trades compute for memory (recomputes activations during backward)
    
    Args:
        model: PyTorch model
        segments: Number of segments to checkpoint (more = less memory, more compute)
    """
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")
    else:
        # Manual implementation for custom models
        def checkpoint_forward(module, input, output):
            return torch.utils.checkpoint.checkpoint(module, input)
        
        # Apply to transformer blocks if available
        for name, module in model.named_modules():
            if 'transformer' in name.lower() or 'block' in name.lower():
                if hasattr(module, 'forward'):
                    original_forward = module.forward
                    module.forward = lambda x: torch.utils.checkpoint.checkpoint(
                        original_forward, x
                    )
        logger.info("Gradient checkpointing applied manually")


def enable_flash_attention():
    """
    Enable Flash Attention for memory-efficient attention
    Requires compatible GPU and PyTorch 2.0+
    """
    try:
        if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
            torch.backends.cuda.enable_flash_sdp(True)
            logger.info("Flash Attention enabled")
        elif hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            # Flash attention is default in PyTorch 2.0+
            logger.info("Using scaled_dot_product_attention (Flash Attention compatible)")
    except Exception as e:
        logger.warning(f"Could not enable Flash Attention: {e}")


def enable_memory_efficient_attention():
    """Enable memory-efficient attention mechanisms"""
    try:
        # Enable SDPA with memory-efficient backend
        if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            logger.info("Memory-efficient attention enabled")
    except Exception as e:
        logger.warning(f"Could not enable memory-efficient attention: {e}")


class MemoryEfficientModel(nn.Module):
    """
    Wrapper for memory-efficient model operations
    Implements various memory optimization techniques
    """
    
    def __init__(
        self,
        model: nn.Module,
        use_gradient_checkpointing: bool = True,
        use_activation_checkpointing: bool = True,
        use_mixed_precision: bool = True
    ):
        super().__init__()
        self.model = model
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_activation_checkpointing = use_activation_checkpointing
        self.use_mixed_precision = use_mixed_precision
        
        if use_gradient_checkpointing:
            enable_gradient_checkpointing(self.model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with memory optimizations"""
        if self.use_mixed_precision and x.device.type == "cuda":
            with torch.cuda.amp.autocast():
                if self.use_activation_checkpointing:
                    return torch.utils.checkpoint.checkpoint(self.model, x)
                return self.model(x)
        else:
            if self.use_activation_checkpointing:
                return torch.utils.checkpoint.checkpoint(self.model, x)
            return self.model(x)


def optimize_tensor_operations():
    """Optimize tensor operations globally"""
    # Enable optimized tensor operations
    if torch.cuda.is_available():
        # Enable TensorFloat-32 for faster operations
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Enable cuDNN deterministic algorithms
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Enable memory-efficient algorithms
        torch.backends.cudnn.allow_tf32 = True
        
        logger.info("Tensor operations optimized")


class SmartBatchProcessor:
    """
    Intelligent batch processing with dynamic batching
    Automatically adjusts batch size based on available memory
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        initial_batch_size: int = 32,
        max_batch_size: int = 128,
        min_batch_size: int = 1
    ):
        self.model = model.to(device)
        self.device = torch.device(device)
        self.current_batch_size = initial_batch_size
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        self.model.eval()
        
        # Memory tracking
        self.memory_usage_history: List[float] = []
    
    def _estimate_memory_usage(self, batch_size: int) -> float:
        """Estimate memory usage for given batch size"""
        # Simple heuristic: assume linear relationship
        if not self.memory_usage_history:
            return batch_size * 0.1  # MB per sample (rough estimate)
        
        avg_memory_per_sample = sum(self.memory_usage_history) / len(self.memory_usage_history)
        return batch_size * avg_memory_per_sample
    
    def _get_available_memory(self) -> float:
        """Get available GPU memory in MB"""
        if self.device.type != "cuda":
            return float('inf')
        
        total = torch.cuda.get_device_properties(self.device).total_memory / (1024 ** 2)
        allocated = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(self.device) / (1024 ** 2)
        
        return total - reserved
    
    def _adjust_batch_size(self, success: bool):
        """Dynamically adjust batch size based on success"""
        if success:
            # Try to increase if we have memory
            available = self._get_available_memory()
            estimated = self._estimate_memory_usage(self.current_batch_size * 2)
            
            if estimated < available * 0.8 and self.current_batch_size * 2 <= self.max_batch_size:
                self.current_batch_size = min(self.current_batch_size * 2, self.max_batch_size)
                logger.debug(f"Increased batch size to {self.current_batch_size}")
        else:
            # Decrease on OOM
            self.current_batch_size = max(self.current_batch_size // 2, self.min_batch_size)
            logger.warning(f"Decreased batch size to {self.current_batch_size}")
    
    def process_batch(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """Process batch with dynamic sizing"""
        results = []
        
        # Process in chunks
        for i in range(0, len(inputs), self.current_batch_size):
            chunk = inputs[i:i + self.current_batch_size]
            
            try:
                # Stack and process
                batch_tensor = torch.stack(chunk).to(self.device, non_blocking=True)
                
                with torch.no_grad():
                    if self.device.type == "cuda":
                        with torch.cuda.amp.autocast():
                            outputs = self.model(batch_tensor)
                    else:
                        outputs = self.model(batch_tensor)
                
                results.extend(outputs.cpu().unbind(0))
                
                # Track memory
                if self.device.type == "cuda":
                    memory_used = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
                    self.memory_usage_history.append(memory_used / len(chunk))
                    if len(self.memory_usage_history) > 100:
                        self.memory_usage_history.pop(0)
                
                # Adjust batch size
                self._adjust_batch_size(success=True)
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    self._adjust_batch_size(success=False)
                    # Retry with smaller batch
                    return self.process_batch(inputs)
                else:
                    raise
        
        return results


class LazyModelLoader:
    """
    Lazy loading of models to reduce memory footprint
    Loads models on-demand and can unload when not in use
    """
    
    def __init__(self, model_factory: Callable, max_loaded: int = 1):
        self.model_factory = model_factory
        self.max_loaded = max_loaded
        self.loaded_models: Dict[str, nn.Module] = {}
        self.model_configs: Dict[str, Any] = {}
        self.access_times: Dict[str, float] = {}
        import time
        self.time = time
    
    def get_model(self, model_id: str, config: Optional[Dict] = None) -> nn.Module:
        """Get model, loading if necessary"""
        import time
        current_time = time.time()
        
        # Check if already loaded
        if model_id in self.loaded_models:
            self.access_times[model_id] = current_time
            return self.loaded_models[model_id]
        
        # Unload least recently used if at capacity
        if len(self.loaded_models) >= self.max_loaded:
            self._unload_lru()
        
        # Load new model
        model = self.model_factory(**(config or {}))
        self.loaded_models[model_id] = model
        self.model_configs[model_id] = config
        self.access_times[model_id] = current_time
        
        logger.info(f"Loaded model: {model_id}")
        return model
    
    def _unload_lru(self):
        """Unload least recently used model"""
        if not self.loaded_models:
            return
        
        lru_id = min(self.access_times.items(), key=lambda x: x[1])[0]
        
        # Move to CPU and clear
        model = self.loaded_models[lru_id]
        model.cpu()
        del self.loaded_models[lru_id]
        del self.access_times[lru_id]
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info(f"Unloaded model: {lru_id}")
    
    def unload_all(self):
        """Unload all models"""
        for model_id in list(self.loaded_models.keys()):
            self._unload_lru()


def optimize_preprocessing_pipeline(preprocessing_fn: Callable) -> Callable:
    """
    Optimize preprocessing pipeline with caching and batching
    """
    cache = {}
    cache_hits = 0
    cache_misses = 0
    
    def optimized_preprocess(input_data):
        nonlocal cache_hits, cache_misses
        
        # Create cache key (simplified - use hash in production)
        if isinstance(input_data, torch.Tensor):
            cache_key = hash(input_data.cpu().numpy().tobytes())
        else:
            cache_key = hash(str(input_data))
        
        # Check cache
        if cache_key in cache:
            cache_hits += 1
            return cache[cache[0]]
        
        # Process
        cache_misses += 1
        result = preprocessing_fn(input_data)
        
        # Cache result (limit cache size)
        if len(cache) < 1000:
            cache[cache_key] = result
        
        return result
    
    return optimized_preprocess


class TensorPool:
    """
    Tensor pooling for memory efficiency
    Reuses tensor memory to reduce allocations
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device)
        self.pools: Dict[tuple, List[torch.Tensor]] = {}
    
    def get_tensor(self, shape: tuple, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Get tensor from pool or create new"""
        key = (shape, dtype)
        
        if key in self.pools and self.pools[key]:
            tensor = self.pools[key].pop()
            tensor.zero_()  # Clear previous data
            return tensor
        
        # Create new tensor
        return torch.zeros(shape, dtype=dtype, device=self.device)
    
    def return_tensor(self, tensor: torch.Tensor):
        """Return tensor to pool"""
        key = (tensor.shape, tensor.dtype)
        
        if key not in self.pools:
            self.pools[key] = []
        
        # Limit pool size
        if len(self.pools[key]) < 10:
            self.pools[key].append(tensor.detach())
    
    def clear(self):
        """Clear all pools"""
        self.pools.clear()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


def enable_all_optimizations():
    """Enable all available optimizations"""
    optimize_tensor_operations()
    enable_flash_attention()
    enable_memory_efficient_attention()
    
    logger.info("All optimizations enabled")


class OptimizedDataLoader:
    """
    Highly optimized DataLoader with advanced features
    """
    
    def __init__(
        self,
        dataset,
        batch_size: int = 32,
        num_workers: int = None,
        pin_memory: bool = True,
        prefetch_factor: int = 4,
        persistent_workers: bool = True,
        use_tensor_pool: bool = True
    ):
        from torch.utils.data import DataLoader
        
        if num_workers is None:
            num_workers = min(8, torch.multiprocessing.cpu_count())
        
        self.tensor_pool = TensorPool() if use_tensor_pool else None
        
        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent_workers and num_workers > 0,
            drop_last=False
        )
    
    def __iter__(self):
        for batch in self.loader:
            yield batch
    
    def __len__(self):
        return len(self.loader)













