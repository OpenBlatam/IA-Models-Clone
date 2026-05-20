"""
Monitoring Utilities
Real-time monitoring and alerting
"""

import torch
import psutil
import logging
from typing import Dict, Any, Optional
from collections import deque
import time

logger = logging.getLogger(__name__)


class SystemMonitor:
    """
    Monitor system resources during training
    """
    
    def __init__(self, history_size: int = 100):
        """
        Initialize system monitor
        
        Args:
            history_size: Size of history buffer
        """
        self.history_size = history_size
        self.cpu_history = deque(maxlen=history_size)
        self.memory_history = deque(maxlen=history_size)
        self.gpu_history = deque(maxlen=history_size)
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get current system statistics
        
        Returns:
            Dictionary with system stats
        """
        stats = {
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_used_gb': psutil.virtual_memory().used / (1024 ** 3),
            'memory_total_gb': psutil.virtual_memory().total / (1024 ** 3),
        }
        
        # GPU stats if available
        if torch.cuda.is_available():
            stats['gpu_available'] = True
            stats['gpu_memory_allocated_gb'] = torch.cuda.memory_allocated() / (1024 ** 3)
            stats['gpu_memory_reserved_gb'] = torch.cuda.memory_reserved() / (1024 ** 3)
            stats['gpu_memory_total_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        else:
            stats['gpu_available'] = False
        
        return stats
    
    def update(self) -> None:
        """Update monitoring history"""
        stats = self.get_system_stats()
        self.cpu_history.append(stats['cpu_percent'])
        self.memory_history.append(stats['memory_percent'])
        
        if stats['gpu_available']:
            gpu_mem_percent = (stats['gpu_memory_allocated_gb'] / stats['gpu_memory_total_gb']) * 100
            self.gpu_history.append(gpu_mem_percent)
    
    def get_averages(self) -> Dict[str, float]:
        """Get average statistics"""
        averages = {}
        
        if self.cpu_history:
            averages['avg_cpu'] = sum(self.cpu_history) / len(self.cpu_history)
        
        if self.memory_history:
            averages['avg_memory'] = sum(self.memory_history) / len(self.memory_history)
        
        if self.gpu_history:
            averages['avg_gpu_memory'] = sum(self.gpu_history) / len(self.gpu_history)
        
        return averages


class TrainingMonitor:
    """
    Monitor training progress and metrics
    """
    
    def __init__(self):
        """Initialize training monitor"""
        self.metrics_history: Dict[str, list] = {}
        self.start_time = None
        self.system_monitor = SystemMonitor()
    
    def start(self) -> None:
        """Start monitoring"""
        self.start_time = time.time()
    
    def log_metrics(self, metrics: Dict[str, float], epoch: int) -> None:
        """
        Log training metrics
        
        Args:
            metrics: Dictionary of metrics
            epoch: Current epoch
        """
        for name, value in metrics.items():
            if name not in self.metrics_history:
                self.metrics_history[name] = []
            self.metrics_history[name].append(value)
        
        # Update system stats
        self.system_monitor.update()
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get monitoring summary
        
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'training_time': time.time() - self.start_time if self.start_time else 0,
            'metrics': {},
            'system': self.system_monitor.get_averages(),
        }
        
        for name, values in self.metrics_history.items():
            if values:
                summary['metrics'][name] = {
                    'latest': values[-1],
                    'mean': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                }
        
        return summary



