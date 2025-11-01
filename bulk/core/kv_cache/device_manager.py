"""
Device management module for KV Cache.

Handles device resolution, validation, and information.
"""
import logging
from typing import Optional
import torch

from kv_cache.config import CacheMode

logger = logging.getLogger(__name__)


class DeviceManager:
    """
    Manages device operations for KV cache.
    
    Responsibilities:
    - Resolve device from configuration
    - Validate device availability
    - Provide device information
    - Handle device fallbacks
    """
    
    def __init__(self, cache_mode: CacheMode, preferred_device: Optional[str] = None):
        """
        Initialize device manager.
        
        Args:
            cache_mode: Cache operation mode
            preferred_device: Preferred device ("cuda", "cpu", "mps", or None for auto)
        """
        self.cache_mode = cache_mode
        self.preferred_device = preferred_device
        self.device: Optional[torch.device] = None
        self._resolve_device()
    
    def _resolve_device(self) -> torch.device:
        """Resolve device from configuration with error handling."""
        # If preferred device is specified, try to use it
        if self.preferred_device:
            try:
                if self.preferred_device == "cuda":
                    if torch.cuda.is_available():
                        self.device = torch.device("cuda")
                        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
                        return self.device
                    else:
                        logger.warning("CUDA requested but not available, falling back to CPU")
                
                elif self.preferred_device == "mps":
                    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                        self.device = torch.device("mps")
                        logger.info("Using Apple MPS device")
                        return self.device
                    else:
                        logger.warning("MPS requested but not available, falling back to CPU")
                
                elif self.preferred_device == "cpu":
                    self.device = torch.device("cpu")
                    logger.info("Using CPU device")
                    return self.device
            except Exception as e:
                logger.warning(f"Error setting preferred device {self.preferred_device}: {e}")
        
        # Auto-resolve based on mode
        if self.cache_mode == CacheMode.TRAINING:
            # Training mode: prefer CUDA if available
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                logger.info(f"Training mode: Using CUDA device: {torch.cuda.get_device_name(0)}")
                return self.device
            logger.warning("Training mode: CUDA not available, falling back to CPU")
            self.device = torch.device("cpu")
            return self.device
        else:
            # Inference mode: can use any available device
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                logger.info(f"Inference mode: Using CUDA device: {torch.cuda.get_device_name(0)}")
                return self.device
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
                logger.info("Inference mode: Using Apple MPS device")
                return self.device
            self.device = torch.device("cpu")
            logger.info("Inference mode: Using CPU device")
            return self.device
    
    def get_device(self) -> torch.device:
        """Get resolved device."""
        if self.device is None:
            self._resolve_device()
        return self.device
    
    def get_device_info(self) -> dict:
        """Get detailed device information."""
        device = self.get_device()
        info = {
            "type": device.type,
            "index": device.index if device.index is not None else 0,
        }
        
        if device.type == "cuda" and torch.cuda.is_available():
            try:
                info["name"] = torch.cuda.get_device_name(device)
                info["capability"] = torch.cuda.get_device_capability(device)
                info["memory_allocated_mb"] = torch.cuda.memory_allocated(device) / (1024**2)
                info["memory_reserved_mb"] = torch.cuda.memory_reserved(device) / (1024**2)
            except Exception as e:
                logger.warning(f"Error getting CUDA device info: {e}")
        
        return info
    
    def supports_mixed_precision(self, dtype) -> bool:
        """
        Check if device supports mixed precision with given dtype.
        
        Args:
            dtype: torch.dtype (torch.float16 or torch.bfloat16)
            
        Returns:
            True if device supports the dtype
        """
        device = self.get_device()
        
        if device.type == "cuda":
            if dtype == torch.float16:
                return True
            elif dtype == torch.bfloat16:
                # BF16 requires compute capability 8.0+ (Ampere+)
                try:
                    capability = torch.cuda.get_device_capability(device)
                    return capability[0] >= 8
                except Exception:
                    return False
        elif device.type == "mps":
            # MPS supports float16
            return dtype == torch.float16
        
        return False
    
    def is_cuda(self) -> bool:
        """Check if device is CUDA."""
        return self.get_device().type == "cuda"
    
    def is_mps(self) -> bool:
        """Check if device is MPS."""
        return self.get_device().type == "mps"
    
    def is_cpu(self) -> bool:
        """Check if device is CPU."""
        return self.get_device().type == "cpu"

