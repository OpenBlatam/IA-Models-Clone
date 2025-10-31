#!/usr/bin/env python3
"""
Advanced Device Manager - Comprehensive device and memory management
Provides GPU/CPU management, memory optimization, and device utilities
"""

import torch
import torch.nn as nn
import psutil
import logging
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import numpy as np
from pathlib import Path
import json
import uuid
from datetime import datetime, timezone
import threading
import time
import gc

@dataclass
class DeviceInfo:
    """Information about a device."""
    device_id: str
    device_type: str  # cuda, cpu, mps
    name: str
    memory_total: float  # GB
    memory_available: float  # GB
    memory_used: float  # GB
    compute_capability: Optional[str] = None
    is_available: bool = True

@dataclass
class MemoryStats:
    """Memory statistics."""
    total_memory: float  # GB
    used_memory: float  # GB
    available_memory: float  # GB
    memory_usage_percent: float
    peak_memory: float  # GB
    memory_fragmentation: float

class DeviceManager:
    """Advanced device manager for GPU/CPU management."""
    
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
        self.devices = {}
        self.current_device = None
        self.memory_monitor = None
        self.memory_stats = {}
        
        # Initialize devices
        self._initialize_devices()
        self._start_memory_monitoring()
        
        self.logger.info("Device manager initialized")
    
    def _initialize_devices(self):
        """Initialize available devices."""
        # CPU device
        cpu_info = DeviceInfo(
            device_id="cpu",
            device_type="cpu",
            name="CPU",
            memory_total=psutil.virtual_memory().total / (1024**3),
            memory_available=psutil.virtual_memory().available / (1024**3),
            memory_used=psutil.virtual_memory().used / (1024**3),
            is_available=True
        )
        self.devices["cpu"] = cpu_info
        
        # CUDA devices
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                device_info = DeviceInfo(
                    device_id=f"cuda:{i}",
                    device_type="cuda",
                    name=torch.cuda.get_device_name(i),
                    memory_total=torch.cuda.get_device_properties(i).total_memory / (1024**3),
                    memory_available=torch.cuda.memory_reserved(i) / (1024**3),
                    memory_used=torch.cuda.memory_allocated(i) / (1024**3),
                    compute_capability=f"{torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}",
                    is_available=True
                )
                self.devices[f"cuda:{i}"] = device_info
        
        # MPS device (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            mps_info = DeviceInfo(
                device_id="mps",
                device_type="mps",
                name="Apple Silicon GPU",
                memory_total=psutil.virtual_memory().total / (1024**3),  # Shared with system
                memory_available=psutil.virtual_memory().available / (1024**3),
                memory_used=psutil.virtual_memory().used / (1024**3),
                is_available=True
            )
            self.devices["mps"] = mps_info
        
        # Set default device
        self.current_device = self._get_best_device()
        self.logger.info(f"Initialized {len(self.devices)} devices")
    
    def _get_best_device(self) -> str:
        """Get the best available device."""
        # Priority: CUDA > MPS > CPU
        if "cuda:0" in self.devices:
            return "cuda:0"
        elif "mps" in self.devices:
            return "mps"
        else:
            return "cpu"
    
    def _start_memory_monitoring(self):
        """Start memory monitoring thread."""
        self.memory_monitor = threading.Thread(target=self._monitor_memory, daemon=True)
        self.memory_monitor.start()
    
    def _monitor_memory(self):
        """Monitor memory usage."""
        while True:
            try:
                for device_id, device_info in self.devices.items():
                    if device_id == "cpu":
                        memory = psutil.virtual_memory()
                        self.memory_stats[device_id] = MemoryStats(
                            total_memory=memory.total / (1024**3),
                            used_memory=memory.used / (1024**3),
                            available_memory=memory.available / (1024**3),
                            memory_usage_percent=memory.percent,
                            peak_memory=memory.used / (1024**3),
                            memory_fragmentation=0.0
                        )
                    elif device_id.startswith("cuda"):
                        device_idx = int(device_id.split(":")[1])
                        memory_allocated = torch.cuda.memory_allocated(device_idx) / (1024**3)
                        memory_reserved = torch.cuda.memory_reserved(device_idx) / (1024**3)
                        memory_total = torch.cuda.get_device_properties(device_idx).total_memory / (1024**3)
                        
                        self.memory_stats[device_id] = MemoryStats(
                            total_memory=memory_total,
                            used_memory=memory_allocated,
                            available_memory=memory_total - memory_reserved,
                            memory_usage_percent=(memory_reserved / memory_total) * 100,
                            peak_memory=torch.cuda.max_memory_allocated(device_idx) / (1024**3),
                            memory_fragmentation=0.0
                        )
                
                time.sleep(1)  # Monitor every second
                
            except Exception as e:
                self.logger.error(f"Memory monitoring error: {e}")
                time.sleep(5)
    
    def get_device(self, device_id: str = None) -> torch.device:
        """Get PyTorch device."""
        if device_id is None:
            device_id = self.current_device
        
        if device_id not in self.devices:
            raise ValueError(f"Device {device_id} not available")
        
        return torch.device(device_id)
    
    def get_device_info(self, device_id: str = None) -> DeviceInfo:
        """Get device information."""
        if device_id is None:
            device_id = self.current_device
        
        if device_id not in self.devices:
            raise ValueError(f"Device {device_id} not available")
        
        return self.devices[device_id]
    
    def get_memory_stats(self, device_id: str = None) -> MemoryStats:
        """Get memory statistics for a device."""
        if device_id is None:
            device_id = self.current_device
        
        if device_id not in self.memory_stats:
            return MemoryStats(0, 0, 0, 0, 0, 0)
        
        return self.memory_stats[device_id]
    
    def get_available_devices(self) -> List[str]:
        """Get list of available devices."""
        return [device_id for device_id, device_info in self.devices.items() 
                if device_info.is_available]
    
    def get_best_device(self) -> str:
        """Get the best available device."""
        available_devices = self.get_available_devices()
        
        # Priority: CUDA > MPS > CPU
        for device_type in ["cuda", "mps", "cpu"]:
            for device_id in available_devices:
                if device_id.startswith(device_type):
                    return device_id
        
        return "cpu"
    
    def set_current_device(self, device_id: str):
        """Set current device."""
        if device_id not in self.devices:
            raise ValueError(f"Device {device_id} not available")
        
        self.current_device = device_id
        self.logger.info(f"Current device set to: {device_id}")
    
    def move_to_device(self, tensor_or_model, device_id: str = None):
        """Move tensor or model to device."""
        device = self.get_device(device_id)
        return tensor_or_model.to(device)
    
    def clear_memory(self, device_id: str = None):
        """Clear memory for a device."""
        if device_id is None:
            device_id = self.current_device
        
        if device_id.startswith("cuda"):
            device_idx = int(device_id.split(":")[1])
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            self.logger.info(f"Cleared CUDA memory for device {device_id}")
        elif device_id == "cpu":
            gc.collect()
            self.logger.info("Cleared CPU memory")
    
    def get_memory_usage(self, device_id: str = None) -> Dict[str, float]:
        """Get memory usage for a device."""
        if device_id is None:
            device_id = self.current_device
        
        stats = self.get_memory_stats(device_id)
        return {
            "total_memory": stats.total_memory,
            "used_memory": stats.used_memory,
            "available_memory": stats.available_memory,
            "memory_usage_percent": stats.memory_usage_percent,
            "peak_memory": stats.peak_memory
        }
    
    def is_memory_sufficient(self, required_memory: float, device_id: str = None) -> bool:
        """Check if device has sufficient memory."""
        if device_id is None:
            device_id = self.current_device
        
        stats = self.get_memory_stats(device_id)
        return stats.available_memory >= required_memory
    
    def optimize_memory(self, device_id: str = None):
        """Optimize memory usage."""
        if device_id is None:
            device_id = self.current_device
        
        # Clear cache
        self.clear_memory(device_id)
        
        # Set memory fraction if CUDA
        if device_id.startswith("cuda"):
            device_idx = int(device_id.split(":")[1])
            torch.cuda.set_per_process_memory_fraction(0.8, device_idx)
        
        self.logger.info(f"Optimized memory for device {device_id}")
    
    def get_device_capabilities(self, device_id: str = None) -> Dict[str, Any]:
        """Get device capabilities."""
        if device_id is None:
            device_id = self.current_device
        
        device_info = self.get_device_info(device_id)
        capabilities = {
            "device_id": device_id,
            "device_type": device_info.device_type,
            "name": device_info.name,
            "memory_total": device_info.memory_total,
            "compute_capability": device_info.compute_capability,
            "is_available": device_info.is_available
        }
        
        if device_id.startswith("cuda"):
            device_idx = int(device_id.split(":")[1])
            capabilities.update({
                "major": torch.cuda.get_device_properties(device_idx).major,
                "minor": torch.cuda.get_device_properties(device_idx).minor,
                "multi_processor_count": torch.cuda.get_device_properties(device_idx).multi_processor_count,
                "max_threads_per_block": torch.cuda.get_device_properties(device_idx).max_threads_per_block,
                "max_threads_per_multiprocessor": torch.cuda.get_device_properties(device_idx).max_threads_per_multiprocessor
            })
        
        return capabilities
    
    def benchmark_device(self, device_id: str = None, num_iterations: int = 100) -> Dict[str, float]:
        """Benchmark device performance."""
        if device_id is None:
            device_id = self.current_device
        
        device = self.get_device(device_id)
        
        # Create test tensors
        size = 1000
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        
        # Warmup
        for _ in range(10):
            _ = torch.matmul(a, b)
        
        # Benchmark
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.time()
        
        for _ in range(num_iterations):
            _ = torch.matmul(a, b)
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        end_time = time.time()
        
        # Calculate metrics
        total_time = end_time - start_time
        avg_time = total_time / num_iterations
        throughput = (num_iterations * size * size * size) / total_time  # FLOPS
        
        return {
            "total_time": total_time,
            "avg_time": avg_time,
            "throughput": throughput,
            "device": device_id
        }
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        return {
            "cpu_count": psutil.cpu_count(),
            "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            "memory_total": psutil.virtual_memory().total / (1024**3),
            "memory_available": psutil.virtual_memory().available / (1024**3),
            "python_version": torch.__version__,
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "mps_available": hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
            "devices": list(self.devices.keys())
        }
    
    def cleanup(self):
        """Cleanup device manager."""
        if self.memory_monitor:
            self.memory_monitor.join(timeout=1)
        
        # Clear all device memory
        for device_id in self.devices.keys():
            self.clear_memory(device_id)
        
        self.logger.info("Device manager cleaned up")
    
    def __del__(self):
        """Destructor."""
        self.cleanup()

class GPUManager(DeviceManager):
    """Specialized GPU manager."""
    
    def __init__(self, logger: logging.Logger = None):
        super().__init__(logger)
        self.gpu_devices = [device_id for device_id in self.devices.keys() 
                           if device_id.startswith("cuda")]
    
    def get_gpu_count(self) -> int:
        """Get number of available GPUs."""
        return len(self.gpu_devices)
    
    def get_gpu_memory_info(self, device_id: str = None) -> Dict[str, float]:
        """Get GPU memory information."""
        if device_id is None:
            device_id = self.current_device
        
        if not device_id.startswith("cuda"):
            raise ValueError(f"Device {device_id} is not a GPU")
        
        device_idx = int(device_id.split(":")[1])
        return {
            "total_memory": torch.cuda.get_device_properties(device_idx).total_memory / (1024**3),
            "allocated_memory": torch.cuda.memory_allocated(device_idx) / (1024**3),
            "reserved_memory": torch.cuda.memory_reserved(device_idx) / (1024**3),
            "free_memory": (torch.cuda.get_device_properties(device_idx).total_memory - 
                           torch.cuda.memory_reserved(device_idx)) / (1024**3)
        }
    
    def set_memory_fraction(self, fraction: float, device_id: str = None):
        """Set memory fraction for GPU."""
        if device_id is None:
            device_id = self.current_device
        
        if not device_id.startswith("cuda"):
            raise ValueError(f"Device {device_id} is not a GPU")
        
        device_idx = int(device_id.split(":")[1])
        torch.cuda.set_per_process_memory_fraction(fraction, device_idx)
        self.logger.info(f"Set memory fraction to {fraction} for device {device_id}")

class CPUManager(DeviceManager):
    """Specialized CPU manager."""
    
    def __init__(self, logger: logging.Logger = None):
        super().__init__(logger)
        self.cpu_device = "cpu"
    
    def get_cpu_count(self) -> int:
        """Get number of CPU cores."""
        return psutil.cpu_count()
    
    def get_cpu_usage(self) -> float:
        """Get CPU usage percentage."""
        return psutil.cpu_percent(interval=1)
    
    def get_cpu_memory_info(self) -> Dict[str, float]:
        """Get CPU memory information."""
        memory = psutil.virtual_memory()
        return {
            "total_memory": memory.total / (1024**3),
            "available_memory": memory.available / (1024**3),
            "used_memory": memory.used / (1024**3),
            "memory_usage_percent": memory.percent
        }
