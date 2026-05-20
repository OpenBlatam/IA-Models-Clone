"""
Resource Manager for Document Analyzer
=======================================

Advanced resource management for memory, CPU, and GPU optimization.
"""

import asyncio
import logging
import time
import gc
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class ResourceUsage:
    """Resource usage metrics"""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_mb: float = 0.0
    memory_available_mb: float = 0.0
    gpu_percent: float = 0.0
    gpu_memory_used_mb: float = 0.0
    gpu_memory_total_mb: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

class ResourceManager:
    """Advanced resource manager"""
    
    def __init__(
        self,
        memory_threshold: float = 85.0,
        cpu_threshold: float = 80.0,
        auto_cleanup: bool = True
    ):
        self.memory_threshold = memory_threshold
        self.cpu_threshold = cpu_threshold
        self.auto_cleanup = auto_cleanup
        self.usage_history: List[ResourceUsage] = []
        self.max_history = 100
        logger.info(f"ResourceManager initialized. Memory threshold: {memory_threshold}%, CPU threshold: {cpu_threshold}%")
    
    def get_current_usage(self) -> ResourceUsage:
        """Get current resource usage"""
        try:
            import psutil
            
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            usage = ResourceUsage(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / 1024 / 1024,
                memory_available_mb=memory.available / 1024 / 1024
            )
            
            # GPU usage if available
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory_used = torch.cuda.memory_allocated(0) / 1024 / 1024
                    gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
                    gpu_percent = (gpu_memory_used / gpu_memory_total) * 100 if gpu_memory_total > 0 else 0
                    
                    usage.gpu_percent = gpu_percent
                    usage.gpu_memory_used_mb = gpu_memory_used
                    usage.gpu_memory_total_mb = gpu_memory_total
            except ImportError:
                pass
            
            # Store in history
            self.usage_history.append(usage)
            if len(self.usage_history) > self.max_history:
                self.usage_history = self.usage_history[-self.max_history:]
            
            return usage
        except Exception as e:
            logger.error(f"Error getting resource usage: {e}")
            return ResourceUsage()
    
    def should_cleanup(self) -> bool:
        """Check if cleanup is needed"""
        usage = self.get_current_usage()
        return (
            usage.memory_percent > self.memory_threshold or
            usage.cpu_percent > self.cpu_threshold
        )
    
    def cleanup_memory(self):
        """Cleanup memory"""
        try:
            # Force garbage collection
            collected = gc.collect()
            logger.info(f"Garbage collection: {collected} objects collected")
            
            # GPU cleanup if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info("GPU cache cleared")
            except ImportError:
                pass
            
            return True
        except Exception as e:
            logger.error(f"Error cleaning up memory: {e}")
            return False
    
    async def auto_cleanup_if_needed(self):
        """Auto cleanup if thresholds are exceeded"""
        if self.auto_cleanup and self.should_cleanup():
            logger.warning("Resource thresholds exceeded, performing cleanup")
            self.cleanup_memory()
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        if not self.usage_history:
            return {}
        
        recent = self.usage_history[-10:]  # Last 10 measurements
        
        return {
            "current": {
                "cpu_percent": recent[-1].cpu_percent if recent else 0,
                "memory_percent": recent[-1].memory_percent if recent else 0,
                "gpu_percent": recent[-1].gpu_percent if recent else 0
            },
            "average": {
                "cpu_percent": sum(u.cpu_percent for u in recent) / len(recent) if recent else 0,
                "memory_percent": sum(u.memory_percent for u in recent) / len(recent) if recent else 0,
                "gpu_percent": sum(u.gpu_percent for u in recent) / len(recent) if recent else 0
            },
            "max": {
                "cpu_percent": max(u.cpu_percent for u in recent) if recent else 0,
                "memory_percent": max(u.memory_percent for u in recent) if recent else 0,
                "gpu_percent": max(u.gpu_percent for u in recent) if recent else 0
            }
        }

# Global instance
resource_manager = ResourceManager()
















