"""
Device Manager Module
====================

Handles device detection and configuration for GPU, TPU, and CPU.
Provides automatic device selection and optimization settings.

Author: BUL System
Date: 2024
"""

import logging
import torch
from typing import Optional, Dict, Any, List
import warnings

logger = logging.getLogger(__name__)


class DeviceManager:
    """
    Manages device detection and configuration for training.
    
    Automatically detects and configures:
    - CUDA GPUs (NVIDIA)
    - TPUs (Tensor Processing Units)
    - Apple Silicon (MPS)
    - CPU (fallback)
    
    Attributes:
        device: The configured torch device
        device_info: Dictionary with device information
        supports_bf16: Whether device supports BF16 precision
        
    Example:
        >>> manager = DeviceManager()
        >>> device = manager.get_device()
        >>> print(f"Using device: {device}")
    """
    
    def __init__(self, preferred_device: Optional[str] = None):
        """
        Initialize DeviceManager and detect available devices.
        
        Args:
            preferred_device: Preferred device type ("cuda", "tpu", "mps", "cpu")
                             If None, auto-detects best available device.
        """
        self.device: Optional[torch.device] = None
        self.device_info: Dict[str, Any] = {}
        self.supports_bf16: bool = False
        self.preferred_device = preferred_device
        self._detect_device()
    
    def _detect_device(self) -> None:
        """Detect and configure the best available device."""
        # If preferred device is specified, try to use it first
        if self.preferred_device:
            if self._try_preferred_device():
                return
        
        # Check for TPU (XLA) - highest priority for training
        tpu_device = self._detect_tpu()
        if tpu_device:
            self.device = tpu_device
            self.device_info = {
                "type": "tpu",
                "device": str(tpu_device),
                "available": True,
                "priority": "high"
            }
            logger.info("TPU detected and configured")
            return
        
        # Check for CUDA/GPU - second priority
        cuda_device = self._detect_cuda()
        if cuda_device:
            self.device = cuda_device
            self.device_info = self._get_cuda_info()
            self.device_info["priority"] = "high"
            self.supports_bf16 = self._check_bf16_support()
            logger.info(f"CUDA available with {torch.cuda.device_count()} GPU(s)")
            if self.device_info.get("multi_gpu"):
                logger.info(f"Multi-GPU setup: {', '.join(self.device_info.get('device_names', []))}")
            return
        
        # Check for MPS (Apple Silicon) - third priority
        mps_device = self._detect_mps()
        if mps_device:
            self.device = mps_device
            self.device_info = {
                "type": "mps",
                "device": str(mps_device),
                "available": True,
                "priority": "medium"
            }
            logger.info("Apple Silicon (MPS) detected")
            return
        
        # Fallback to CPU
        self.device = torch.device("cpu")
        self.device_info = {
            "type": "cpu",
            "device": "cpu",
            "available": True,
            "cores": torch.get_num_threads(),
            "priority": "low"
        }
        logger.warning("No GPU/TPU detected, using CPU (training will be slow)")
        logger.info(f"CPU threads: {torch.get_num_threads()}")
    
    def _try_preferred_device(self) -> bool:
        """
        Try to use the preferred device if specified.
        
        Returns:
            True if preferred device is available and configured, False otherwise
        """
        if self.preferred_device == "tpu":
            tpu_device = self._detect_tpu()
            if tpu_device:
                self.device = tpu_device
                self.device_info = {"type": "tpu", "device": str(tpu_device), "available": True}
                logger.info("Using preferred device: TPU")
                return True
            else:
                logger.warning("Preferred device TPU not available, auto-detecting...")
        
        elif self.preferred_device == "cuda":
            cuda_device = self._detect_cuda()
            if cuda_device:
                self.device = cuda_device
                self.device_info = self._get_cuda_info()
                self.supports_bf16 = self._check_bf16_support()
                logger.info("Using preferred device: CUDA")
                return True
            else:
                logger.warning("Preferred device CUDA not available, auto-detecting...")
        
        elif self.preferred_device == "mps":
            mps_device = self._detect_mps()
            if mps_device:
                self.device = mps_device
                self.device_info = {"type": "mps", "device": str(mps_device), "available": True}
                logger.info("Using preferred device: MPS")
                return True
            else:
                logger.warning("Preferred device MPS not available, auto-detecting...")
        
        elif self.preferred_device == "cpu":
            self.device = torch.device("cpu")
            self.device_info = {"type": "cpu", "device": "cpu", "available": True, "cores": torch.get_num_threads()}
            logger.info("Using preferred device: CPU")
            return True
        
        return False
    
    def _detect_tpu(self) -> Optional[torch.device]:
        """
        Detect TPU availability.
        
        Returns:
            TPU device if available, None otherwise
        """
        try:
            import torch_xla.core.xla_model as xm
            device = xm.xla_device()
            if device:
                return device
        except (ImportError, RuntimeError):
            pass
        return None
    
    def _detect_cuda(self) -> Optional[torch.device]:
        """
        Detect CUDA GPU availability.
        
        Returns:
            CUDA device if available, None otherwise
        """
        if torch.cuda.is_available():
            return torch.device("cuda")
        return None
    
    def _detect_mps(self) -> Optional[torch.device]:
        """
        Detect Apple Silicon MPS availability.
        
        Returns:
            MPS device if available, None otherwise
        """
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return None
    
    def _get_cuda_info(self) -> Dict[str, Any]:
        """
        Get detailed CUDA device information.
        
        Returns:
            Dictionary with CUDA device information
        """
        info = {
            "type": "cuda",
            "device": "cuda",
            "available": True,
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
        }
        
        if torch.cuda.is_available():
            info["device_name"] = torch.cuda.get_device_name(0)
            info["capability"] = torch.cuda.get_device_capability(0)
            info["memory_total"] = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
            
            if torch.cuda.device_count() > 1:
                info["multi_gpu"] = True
                info["device_names"] = [
                    torch.cuda.get_device_name(i) 
                    for i in range(torch.cuda.device_count())
                ]
        
        return info
    
    def _check_bf16_support(self) -> bool:
        """
        Check if device supports BF16 precision.
        
        BF16 is available on Ampere+ GPUs (compute capability >= 8.0).
        
        Returns:
            True if BF16 is supported, False otherwise
        """
        if not torch.cuda.is_available():
            return False
        
        try:
            capability = torch.cuda.get_device_capability(0)
            # Ampere+ GPUs have compute capability >= 8.0
            return capability[0] >= 8
        except Exception:
            return False
    
    def get_device(self) -> torch.device:
        """
        Get the configured device.
        
        Returns:
            torch.device: The device to use for training
        """
        return self.device
    
    def get_device_info(self) -> Dict[str, Any]:
        """
        Get device information.
        
        Returns:
            Dictionary with device information
        """
        return self.device_info.copy()
    
    def is_cuda_available(self) -> bool:
        """Check if CUDA is available."""
        return self.device_info.get("type") == "cuda"
    
    def is_tpu_available(self) -> bool:
        """Check if TPU is available."""
        return self.device_info.get("type") == "tpu"
    
    def get_memory_info(self) -> Optional[Dict[str, float]]:
        """
        Get memory information for the current device.
        
        Returns:
            Dictionary with memory info (GB) or None if not available
        """
        if self.is_cuda_available():
            return {
                "allocated": torch.cuda.memory_allocated() / 1e9,
                "reserved": torch.cuda.memory_reserved() / 1e9,
                "total": self.device_info.get("memory_total", 0),
            }
        return None
    
    def get_recommended_batch_size(self, model_size_gb: float, sequence_length: int = 512) -> int:
        """
        Get recommended batch size based on device capabilities.
        
        Args:
            model_size_gb: Size of the model in GB
            sequence_length: Maximum sequence length (default: 512)
            
        Returns:
            Recommended batch size
        """
        if self.is_cuda_available():
            total_memory = self.device_info.get("memory_total", 8.0)
            # Use ~70% of available memory for batch
            available_memory = total_memory * 0.7
            # Estimate: each sample needs memory based on sequence length
            # Rough estimate: ~0.1 GB per 512 tokens
            estimated_per_sample = (sequence_length / 512) * 0.1
            recommended = max(1, int((available_memory - model_size_gb) / estimated_per_sample))
            return min(recommended, 32)  # Cap at 32
        
        # For CPU, use smaller batches
        return min(4, max(1, int(8 / (sequence_length / 512))))
    
    def clear_cache(self) -> None:
        """Clear GPU cache if using CUDA."""
        if self.is_cuda_available():
            torch.cuda.empty_cache()
            logger.debug("GPU cache cleared")
    
    def get_available_devices(self) -> List[str]:
        """
        Get list of all available device types.
        
        Returns:
            List of available device types
        """
        devices = []
        
        if self._detect_tpu():
            devices.append("tpu")
        if self._detect_cuda():
            devices.append("cuda")
        if self._detect_mps():
            devices.append("mps")
        devices.append("cpu")  # CPU is always available
        
        return devices
    
    def get_device_summary(self) -> str:
        """
        Get a human-readable summary of the device configuration.
        
        Returns:
            Summary string
        """
        info = self.device_info
        device_type = info.get("type", "unknown")
        
        if device_type == "cuda":
            name = info.get("device_name", "NVIDIA GPU")
            memory = info.get("memory_total", 0)
            count = info.get("device_count", 1)
            return f"CUDA: {count}x {name} ({memory:.1f} GB)"
        elif device_type == "tpu":
            return f"TPU: {info.get('device', 'TPU')}"
        elif device_type == "mps":
            return "Apple Silicon (MPS)"
        else:
            cores = info.get("cores", "unknown")
            return f"CPU ({cores} threads)"

