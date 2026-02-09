"""
Resource Manager
================

Advanced resource management for upscaling operations.
"""

import logging
import psutil
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class ResourceStatus:
    """Current resource status."""
    cpu_percent: float
    memory_percent: float
    memory_available_mb: float
    gpu_available: bool
    gpu_memory_mb: Optional[float] = None
    timestamp: float = 0.0


class ResourceManager:
    """
    Resource manager for monitoring and optimization.
    
    Features:
    - CPU monitoring
    - Memory monitoring
    - GPU detection
    - Resource limits
    - Automatic throttling
    """
    
    def __init__(
        self,
        max_cpu_percent: float = 90.0,
        max_memory_percent: float = 85.0,
        max_memory_mb: Optional[float] = None,
        check_interval: float = 1.0
    ):
        """
        Initialize resource manager.
        
        Args:
            max_cpu_percent: Maximum CPU usage percentage
            max_memory_percent: Maximum memory usage percentage
            max_memory_mb: Maximum memory in MB
            check_interval: Resource check interval
        """
        self.max_cpu_percent = max_cpu_percent
        self.max_memory_percent = max_memory_percent
        self.max_memory_mb = max_memory_mb
        self.check_interval = check_interval
        
        # Resource history
        self.history: deque = deque(maxlen=100)
        
        # GPU detection
        self.gpu_available = self._detect_gpu()
        
        logger.info(f"ResourceManager initialized (GPU: {self.gpu_available})")
    
    def _detect_gpu(self) -> bool:
        """Detect GPU availability."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def get_status(self) -> ResourceStatus:
        """Get current resource status."""
        # CPU
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Memory
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_available_mb = memory.available / (1024 * 1024)
        
        # GPU
        gpu_memory_mb = None
        if self.gpu_available:
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
            except:
                pass
        
        status = ResourceStatus(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_available_mb=memory_available_mb,
            gpu_available=self.gpu_available,
            gpu_memory_mb=gpu_memory_mb,
            timestamp=time.time()
        )
        
        self.history.append(status)
        
        return status
    
    def is_resource_available(self) -> bool:
        """Check if resources are available for processing."""
        status = self.get_status()
        
        # Check CPU
        if status.cpu_percent > self.max_cpu_percent:
            logger.warning(f"CPU usage too high: {status.cpu_percent:.1f}%")
            return False
        
        # Check memory
        if status.memory_percent > self.max_memory_percent:
            logger.warning(f"Memory usage too high: {status.memory_percent:.1f}%")
            return False
        
        if self.max_memory_mb and status.memory_available_mb < self.max_memory_mb:
            logger.warning(f"Available memory too low: {status.memory_available_mb:.1f} MB")
            return False
        
        return True
    
    def wait_for_resources(self, timeout: float = 30.0) -> bool:
        """
        Wait for resources to become available.
        
        Args:
            timeout: Maximum time to wait
            
        Returns:
            True if resources available, False if timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.is_resource_available():
                return True
            
            time.sleep(self.check_interval)
        
        return False
    
    def get_recommended_batch_size(self, item_size_mb: float = 10.0) -> int:
        """
        Get recommended batch size based on available resources.
        
        Args:
            item_size_mb: Estimated size per item in MB
            
        Returns:
            Recommended batch size
        """
        status = self.get_status()
        
        # Calculate based on available memory
        if self.max_memory_mb:
            available = min(status.memory_available_mb, self.max_memory_mb)
        else:
            available = status.memory_available_mb
        
        # Reserve 20% for system
        usable_memory = available * 0.8
        
        # Calculate batch size
        batch_size = int(usable_memory / item_size_mb)
        
        # Limit based on CPU
        if status.cpu_percent > 70:
            batch_size = max(1, batch_size // 2)
        
        # Clamp to reasonable values
        batch_size = max(1, min(batch_size, 16))
        
        return batch_size
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get resource statistics."""
        if not self.history:
            return {
                "gpu_available": self.gpu_available,
                "samples": 0
            }
        
        cpu_values = [h.cpu_percent for h in self.history]
        memory_values = [h.memory_percent for h in self.history]
        
        return {
            "gpu_available": self.gpu_available,
            "samples": len(self.history),
            "cpu": {
                "current": cpu_values[-1] if cpu_values else 0.0,
                "avg": sum(cpu_values) / len(cpu_values) if cpu_values else 0.0,
                "max": max(cpu_values) if cpu_values else 0.0,
                "min": min(cpu_values) if cpu_values else 0.0
            },
            "memory": {
                "current": memory_values[-1] if memory_values else 0.0,
                "avg": sum(memory_values) / len(memory_values) if memory_values else 0.0,
                "max": max(memory_values) if memory_values else 0.0,
                "min": min(memory_values) if memory_values else 0.0,
                "available_mb": self.history[-1].memory_available_mb if self.history else 0.0
            }
        }


