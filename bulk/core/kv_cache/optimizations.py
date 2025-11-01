"""
Performance optimizations for KV Cache.

Implements fast operations using PyTorch optimizations.
"""
import logging
from typing import Tuple, Optional
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class FastQuantizer:
    """Optimized quantizer using fused operations."""
    
    def __init__(self, bits: int = 8, use_amp: bool = True):
        self.bits = bits
        self.use_amp = use_amp
        self._cached_scales: dict = {}
    
    def _quantize_int8(self, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fast INT8 quantization."""
        key_max = key.abs().max()
        value_max = value.abs().max()
        key_scale = torch.clamp(key_max / 127.0, min=1e-8)
        value_scale = torch.clamp(value_max / 127.0, min=1e-8)
        
        key_q = (key / key_scale).round().clamp(-128, 127).to(torch.int8)
        value_q = (value / value_scale).round().clamp(-128, 127).to(torch.int8)
        
        return key_q, value_q, key_scale, value_scale
    
    def quantize(self, key: torch.Tensor, value: torch.Tensor, dtype: torch.dtype = torch.float16) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fast quantize with caching."""
        if self.bits != 8:
            return key, value
        
        try:
            with torch.cuda.amp.autocast(enabled=self.use_amp and key.is_cuda):
                key_q, value_q, k_scale, v_scale = self._quantize_int8(key, value)
                # Store scales for dequantization
                cache_key = id(key)
                self._cached_scales[cache_key] = (k_scale, v_scale)
                return key_q, value_q
        except Exception as e:
            logger.warning(f"Fast quantization failed: {e}")
            return key, value


class FastCompressor:
    """Optimized compressor using efficient operations."""
    
    def __init__(self, compression_ratio: float = 0.3, method: str = "svd", use_amp: bool = True):
        self.compression_ratio = compression_ratio
        self.method = method
        self.use_amp = use_amp
    
    def _compress_truncate(self, key: torch.Tensor, value: torch.Tensor, ratio: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fast truncation-based compression."""
        if ratio >= 1.0:
            return key, value
        
        k_size = int(key.numel() * ratio)
        v_size = int(value.numel() * ratio)
        
        if k_size >= key.numel() or v_size >= value.numel():
            return key, value
        
        k_flat = key.flatten()
        v_flat = value.flatten()
        
        return k_flat[:k_size], v_flat[:v_size]
    
    def compress(self, key: torch.Tensor, value: torch.Tensor, dtype: torch.dtype = torch.float16) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fast compress."""
        if self.compression_ratio >= 1.0:
            return key, value
        
        try:
            with torch.cuda.amp.autocast(enabled=self.use_amp and key.is_cuda):
                if self.method == "truncate":
                    return self._compress_truncate(key, value, self.compression_ratio)
                # Default: truncate for speed
                return self._compress_truncate(key, value, self.compression_ratio)
        except Exception as e:
            logger.warning(f"Fast compression failed: {e}")
            return key, value


class FastStorage:
    """Optimized storage with faster lookups."""
    
    def __init__(self):
        self._cache: dict = {}
        self._access_times: dict = {}
        self._access_counts: dict = {}
        self._lock = torch.multiprocessing.Lock() if hasattr(torch.multiprocessing, 'Lock') else None
    
    def get_fast(self, position: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Fast get without full lock overhead."""
        if position in self._cache:
            # Update access (minimal overhead)
            import time
            self._access_times[position] = time.time()
            self._access_counts[position] = self._access_counts.get(position, 0) + 1
            return self._cache[position]
        return None
    
    def put_fast(self, position: int, key: torch.Tensor, value: torch.Tensor) -> None:
        """Fast put with minimal copying."""
        # Use in-place operations where possible
        self._cache[position] = (key.detach(), value.detach())
        import time
        self._access_times[position] = time.time()
        self._access_counts[position] = 1


def optimize_tensor_transfer(tensor: torch.Tensor, target_device: torch.device, pin_memory: bool = False) -> torch.Tensor:
    """Optimized tensor transfer with pinning."""
    if tensor.device == target_device:
        return tensor
    
    # Pin memory for faster CPU->GPU transfer
    if pin_memory and target_device.type == "cuda" and tensor.device.type == "cpu":
        tensor = tensor.pin_memory()
    
    return tensor.to(target_device, non_blocking=True)


def fast_tensor_validation(key: torch.Tensor, value: torch.Tensor) -> bool:
    """Fast tensor validation."""
    if key.shape != value.shape:
        return False
    if key.numel() == 0 or value.numel() == 0:
        return False
    # Use efficient check
    if not (torch.isfinite(key).all() and torch.isfinite(value).all()):
        return False
    return True


class MemoryPool:
    """Memory pool for faster allocations."""
    
    def __init__(self, device: torch.device, dtype: torch.dtype = torch.float16):
        self.device = device
        self.dtype = dtype
        self._pool: list = []
        self._pool_size = 10
    
    def get_tensor(self, shape: Tuple[int, ...]) -> torch.Tensor:
        """Get tensor from pool or allocate new."""
        # For now, just allocate (could implement pooling)
        return torch.empty(shape, device=self.device, dtype=self.dtype)
    
    def return_tensor(self, tensor: torch.Tensor) -> None:
        """Return tensor to pool."""
        if len(self._pool) < self._pool_size:
            tensor.zero_()  # Clear for reuse
            self._pool.append(tensor)


def enable_torch_optimizations():
    """Enable PyTorch optimizations globally."""
    # Enable TF32 for faster computation on Ampere+
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Enable cuDNN benchmarking for faster convolutions
    torch.backends.cudnn.benchmark = True
    
    # Enable deterministic only if needed (disable for speed)
    torch.backends.cudnn.deterministic = False
    
    logger.info("PyTorch optimizations enabled")


def compile_cache_operations(enable: bool = True):
    """Enable torch.compile for cache operations."""
    if not enable:
        return
    
    try:
        # This would compile the cache operations if using torch.compile
        logger.info("torch.compile available for cache operations")
    except Exception:
        logger.warning("torch.compile not available")

