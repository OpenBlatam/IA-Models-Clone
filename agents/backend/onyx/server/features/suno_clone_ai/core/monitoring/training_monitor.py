"""
Training Monitoring

Monitors training progress, metrics, and system resources.
"""

import logging
import time
from typing import Dict, Any, Optional
import torch
import psutil
import os

logger = logging.getLogger(__name__)


class TrainingMonitor:
    """Monitor training progress and resources."""
    
    def __init__(self, log_interval: int = 10):
        """
        Initialize training monitor.
        
        Args:
            log_interval: Log every N batches
        """
        self.log_interval = log_interval
        self.start_time = None
        self.batch_count = 0
        self.epoch_count = 0
    
    def start_epoch(self) -> None:
        """Start monitoring epoch."""
        self.start_time = time.time()
        self.batch_count = 0
    
    def log_batch(
        self,
        loss: float,
        metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Log batch metrics.
        
        Args:
            loss: Batch loss
            metrics: Optional additional metrics
        """
        self.batch_count += 1
        
        if self.batch_count % self.log_interval == 0:
            elapsed = time.time() - self.start_time if self.start_time else 0
            batch_time = elapsed / self.batch_count if self.batch_count > 0 else 0
            
            log_msg = f"Batch {self.batch_count}: Loss={loss:.4f}, Time={batch_time:.3f}s/batch"
            
            if metrics:
                log_msg += f", Metrics={metrics}"
            
            logger.info(log_msg)
    
    def get_resource_usage(self) -> Dict[str, float]:
        """
        Get current resource usage.
        
        Returns:
            Dictionary with resource metrics
        """
        process = psutil.Process(os.getpid())
        
        metrics = {
            'cpu_percent': process.cpu_percent(),
            'memory_mb': process.memory_info().rss / (1024 ** 2)
        }
        
        if torch.cuda.is_available():
            metrics.update({
                'gpu_memory_allocated_gb': torch.cuda.memory_allocated() / (1024 ** 3),
                'gpu_memory_reserved_gb': torch.cuda.memory_reserved() / (1024 ** 3)
            })
        
        return metrics



