"""
Device management utilities for Blaze AI.

This module provides device detection, management, and optimization including:
- Device detection (CPU, GPU, MPS)
- Device information and capabilities
- Device-specific optimizations
- Memory management per device
"""

from __future__ import annotations

import os
import platform
import warnings
from typing import Any, Dict, List, Optional, Tuple

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available, device management limited")

try:
    import GPUtil
    GPU_UTIL_AVAILABLE = True
except ImportError:
    GPU_UTIL_AVAILABLE = False
    warnings.warn("GPUtil not available, GPU monitoring limited")

class DeviceInfo:
    """Information about a specific device."""
    
    def __init__(self, device_type: str, device_id: Optional[int] = None):
        self.device_type = device_type
        self.device_id = device_id
        self.name = self._get_device_name()
        self.capabilities = self._get_capabilities()
        self.memory_info = self._get_memory_info()
    
    def _get_device_name(self) -> str:
        """Get device name."""
        if self.device_type == "cpu":
            return platform.processor() or "CPU"
        elif self.device_type == "cuda" and TORCH_AVAILABLE and torch.cuda.is_available():
            if self.device_id is not None:
                return torch.cuda.get_device_name(self.device_id)
            return "CUDA Device"
        elif self.device_type == "mps" and TORCH_AVAILABLE and hasattr(torch.backends, 'mps'):
            return "Apple Silicon GPU (MPS)"
        else:
            return f"{self.device_type.upper()} Device"
    
    def _get_capabilities(self) -> Dict[str, Any]:
        """Get device capabilities."""
        capabilities = {
            "type": self.device_type,
            "id": self.device_id,
            "name": self.name
        }
        
        if self.device_type == "cpu":
            capabilities.update({
                "cores": os.cpu_count(),
                "architecture": platform.architecture()[0],
                "machine": platform.machine()
            })
        elif self.device_type == "cuda" and TORCH_AVAILABLE and torch.cuda.is_available():
            if self.device_id is not None:
                props = torch.cuda.get_device_properties(self.device_id)
                capabilities.update({
                    "compute_capability": f"{props.major}.{props.minor}",
                    "multiprocessor_count": props.multi_processor_count,
                    "max_threads_per_block": props.max_threads_per_block,
                    "max_shared_memory_per_block": props.max_shared_memory_per_block,
                    "total_memory_gb": props.total_memory / (1024**3)
                })
        elif self.device_type == "mps":
            capabilities.update({
                "metal_version": "2.0",  # Approximate
                "unified_memory": True
            })
        
        return capabilities
    
    def _get_memory_info(self) -> Dict[str, Any]:
        """Get device memory information."""
        memory_info = {}
        
        if self.device_type == "cpu":
            try:
                import psutil
                memory = psutil.virtual_memory()
                memory_info.update({
                    "total_gb": memory.total / (1024**3),
                    "available_gb": memory.available / (1024**3),
                    "used_gb": memory.used / (1024**3),
                    "percent_used": memory.percent
                })
            except ImportError:
                memory_info["error"] = "psutil not available"
        
        elif self.device_type == "cuda" and TORCH_AVAILABLE and torch.cuda.is_available():
            if self.device_id is not None:
                try:
                    allocated = torch.cuda.memory_allocated(self.device_id) / (1024**3)
                    reserved = torch.cuda.memory_reserved(self.device_id) / (1024**3)
                    total = torch.cuda.get_device_properties(self.device_id).total_memory / (1024**3)
                    
                    memory_info.update({
                        "total_gb": total,
                        "allocated_gb": allocated,
                        "reserved_gb": reserved,
                        "free_gb": total - reserved,
                        "utilization_percent": (allocated / total) * 100
                    })
                except Exception as e:
                    memory_info["error"] = str(e)
        
        return memory_info
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "device_type": self.device_type,
            "device_id": self.device_id,
            "name": self.name,
            "capabilities": self.capabilities,
            "memory_info": self.memory_info
        }

class DeviceManager:
    """Device management and optimization system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.devices: Dict[str, DeviceInfo] = {}
        self.primary_device: Optional[str] = None
        
        self._detect_devices()
        self._select_primary_device()
    
    def _detect_devices(self):
        """Detect available devices."""
        # Detect CPU
        self.devices["cpu"] = DeviceInfo("cpu")
        
        # Detect CUDA devices
        if TORCH_AVAILABLE and torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                device_key = f"cuda:{i}"
                self.devices[device_key] = DeviceInfo("cuda", i)
        
        # Detect MPS (Apple Silicon)
        if TORCH_AVAILABLE and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.devices["mps"] = DeviceInfo("mps")
    
    def _select_primary_device(self):
        """Select the primary device based on configuration and availability."""
        device_preference = self.config.get('device_preference', ['cuda', 'mps', 'cpu'])
        
        for device_type in device_preference:
            if device_type == "cuda":
                # Find first available CUDA device
                for key in self.devices:
                    if key.startswith("cuda:"):
                        self.primary_device = key
                        return
            elif device_type in self.devices:
                self.primary_device = device_type
                return
        
        # Fallback to CPU
        self.primary_device = "cpu"
    
    def get_device(self, device_spec: Optional[str] = None) -> str:
        """Get device specification string."""
        if device_spec:
            if device_spec in self.devices:
                return device_spec
            elif device_spec == "auto":
                return self.primary_device or "cpu"
            else:
                warnings.warn(f"Device {device_spec} not found, using primary device")
                return self.primary_device or "cpu"
        
        return self.primary_device or "cpu"
    
    def get_device_info(self, device_spec: Optional[str] = None) -> DeviceInfo:
        """Get device information."""
        device = self.get_device(device_spec)
        return self.devices[device]
    
    def get_available_devices(self) -> List[str]:
        """Get list of available device specifications."""
        return list(self.devices.keys())
    
    def get_device_capabilities(self, device_spec: Optional[str] = None) -> Dict[str, Any]:
        """Get device capabilities."""
        device_info = self.get_device_info(device_spec)
        return device_info.capabilities
    
    def get_memory_info(self, device_spec: Optional[str] = None) -> Dict[str, Any]:
        """Get device memory information."""
        device_info = self.get_device_info(device_spec)
        return device_info.memory_info
    
    def optimize_for_device(self, device_spec: Optional[str] = None) -> Dict[str, Any]:
        """Apply device-specific optimizations."""
        device = self.get_device(device_spec)
        optimizations = {}
        
        if device.startswith("cuda"):
            optimizations.update(self._optimize_cuda_device(device))
        elif device == "mps":
            optimizations.update(self._optimize_mps_device())
        elif device == "cpu":
            optimizations.update(self._optimize_cpu_device())
        
        return optimizations
    
    def _optimize_cuda_device(self, device: str) -> Dict[str, Any]:
        """Apply CUDA-specific optimizations."""
        if not TORCH_AVAILABLE:
            return {"error": "PyTorch not available"}
        
        optimizations = {}
        device_id = int(device.split(":")[1])
        
        # Set device
        torch.cuda.set_device(device_id)
        optimizations["device_set"] = device
        
        # Enable optimizations
        if hasattr(torch.backends, 'cudnn'):
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            optimizations["cudnn_benchmark"] = True
        
        # Enable TF32 for Ampere+ GPUs
        if hasattr(torch.backends, 'cuda'):
            if hasattr(torch.backends.cuda, 'matmul'):
                torch.backends.cuda.matmul.allow_tf32 = True
                optimizations["tf32_enabled"] = True
            if hasattr(torch.backends.cuda, 'cudnn'):
                torch.backends.cudnn.allow_tf32 = True
                optimizations["cudnn_tf32_enabled"] = True
        
        return optimizations
    
    def _optimize_mps_device(self) -> Dict[str, Any]:
        """Apply MPS-specific optimizations."""
        optimizations = {}
        
        # MPS optimizations are mostly automatic
        optimizations["mps_optimized"] = True
        
        return optimizations
    
    def _optimize_cpu_device(self) -> Dict[str, Any]:
        """Apply CPU-specific optimizations."""
        optimizations = {}
        
        # Set number of threads
        if TORCH_AVAILABLE:
            torch.set_num_threads(os.cpu_count())
            optimizations["num_threads"] = os.cpu_count()
        
        return optimizations
    
    def get_device_summary(self) -> Dict[str, Any]:
        """Get comprehensive device summary."""
        summary = {
            "primary_device": self.primary_device,
            "available_devices": self.get_available_devices(),
            "devices": {name: device.to_dict() for name, device in self.devices.items()}
        }
        
        # Add optimization recommendations
        summary["recommendations"] = self._get_optimization_recommendations()
        
        return summary
    
    def _get_optimization_recommendations(self) -> List[str]:
        """Get device optimization recommendations."""
        recommendations = []
        
        if self.primary_device and self.primary_device.startswith("cuda"):
            recommendations.append("Use mixed precision training for better performance")
            recommendations.append("Enable gradient checkpointing for memory efficiency")
            recommendations.append("Consider using torch.compile() for optimization")
        
        elif self.primary_device == "mps":
            recommendations.append("MPS provides good performance for Apple Silicon")
            recommendations.append("Use unified memory for large models")
        
        elif self.primary_device == "cpu":
            recommendations.append("Consider using smaller models for CPU inference")
            recommendations.append("Use batch processing for better CPU utilization")
        
        return recommendations

# Utility functions
def get_device_info(device_spec: Optional[str] = None) -> DeviceInfo:
    """Quick device information retrieval."""
    manager = DeviceManager()
    return manager.get_device_info(device_spec)

def get_available_devices() -> List[str]:
    """Get list of available devices."""
    manager = DeviceManager()
    return manager.get_available_devices()

def optimize_device(device_spec: Optional[str] = None) -> Dict[str, Any]:
    """Quick device optimization."""
    manager = DeviceManager()
    return manager.optimize_for_device(device_spec)

def create_device_manager(config: Optional[Dict[str, Any]] = None) -> DeviceManager:
    """Create a new device manager."""
    return DeviceManager(config)

# Export main classes
__all__ = [
    "DeviceInfo",
    "DeviceManager",
    "get_device_info",
    "get_available_devices",
    "optimize_device",
    "create_device_manager"
]


