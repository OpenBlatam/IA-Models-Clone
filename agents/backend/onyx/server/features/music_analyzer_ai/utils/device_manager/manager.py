"""
Device Manager Module

Base device manager class.
"""

from typing import Optional, Union
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")


class DeviceManager:
    """
    Centralized device management and optimization
    """
    
    def __init__(self, device: Optional[str] = None):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        
        self.device = device or self._get_best_device()
        self._setup_device()
    
    def _get_best_device(self) -> str:
        """Get best available device"""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _setup_device(self):
        """Setup device optimizations"""
        if self.device == "cuda":
            # Enable optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Log GPU info
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                logger.info(f"Using GPU: {gpu_name} ({gpu_memory:.2f} GB)")
    
    def move_to_device(self, obj: Union[torch.Tensor, torch.nn.Module]) -> Union[torch.Tensor, torch.nn.Module]:
        """Move object to device"""
        if hasattr(obj, 'to'):
            return obj.to(self.device, non_blocking=True)
        return obj
    
    def enable_mixed_precision(self) -> bool:
        """Check if mixed precision should be enabled"""
        return self.device == "cuda" and torch.cuda.is_available()
    
    def get_device(self) -> str:
        """Get current device"""
        return self.device
    
    def get_device_count(self) -> int:
        """Get number of available devices"""
        if self.device == "cuda":
            return torch.cuda.device_count()
        return 1



