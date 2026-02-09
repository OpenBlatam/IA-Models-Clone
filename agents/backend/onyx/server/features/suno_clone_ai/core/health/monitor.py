"""
System Monitor

Continuous system monitoring.
"""

import logging
import time
import threading
from typing import Dict, Any, Optional, Callable
from collections import deque
import psutil

logger = logging.getLogger(__name__)


class SystemMonitor:
    """Monitor system resources continuously."""
    
    def __init__(
        self,
        interval: float = 1.0,
        max_history: int = 100
    ):
        """
        Initialize system monitor.
        
        Args:
            interval: Monitoring interval in seconds
            max_history: Maximum history size
        """
        self.interval = interval
        self.max_history = max_history
        self.monitoring = False
        self.monitor_thread = None
        self.history = {
            'cpu': deque(maxlen=max_history),
            'memory': deque(maxlen=max_history),
            'timestamp': deque(maxlen=max_history)
        }
        self.callbacks: list = []
    
    def start(self) -> None:
        """Start monitoring."""
        if self.monitoring:
            logger.warning("Monitor already running")
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("System monitor started")
    
    def stop(self) -> None:
        """Stop monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logger.info("System monitor stopped")
    
    def _monitor_loop(self) -> None:
        """Monitoring loop."""
        while self.monitoring:
            try:
                # Collect metrics
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()
                
                timestamp = time.time()
                
                # Store in history
                self.history['cpu'].append(cpu_percent)
                self.history['memory'].append(memory.percent)
                self.history['timestamp'].append(timestamp)
                
                # Call callbacks
                for callback in self.callbacks:
                    try:
                        callback({
                            'cpu': cpu_percent,
                            'memory': memory.percent,
                            'timestamp': timestamp
                        })
                    except Exception as e:
                        logger.error(f"Callback error: {e}")
                
                time.sleep(self.interval)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(self.interval)
    
    def add_callback(self, callback: Callable) -> None:
        """
        Add monitoring callback.
        
        Args:
            callback: Callback function
        """
        self.callbacks.append(callback)
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current system status.
        
        Returns:
            System status
        """
        if not self.history['cpu']:
            return {'status': 'no_data'}
        
        return {
            'cpu': {
                'current': self.history['cpu'][-1] if self.history['cpu'] else 0,
                'average': sum(self.history['cpu']) / len(self.history['cpu']) if self.history['cpu'] else 0,
                'max': max(self.history['cpu']) if self.history['cpu'] else 0
            },
            'memory': {
                'current': self.history['memory'][-1] if self.history['memory'] else 0,
                'average': sum(self.history['memory']) / len(self.history['memory']) if self.history['memory'] else 0,
                'max': max(self.history['memory']) if self.history['memory'] else 0
            },
            'monitoring': self.monitoring
        }
    
    def get_history(self) -> Dict[str, list]:
        """
        Get monitoring history.
        
        Returns:
            History dictionary
        """
        return {
            'cpu': list(self.history['cpu']),
            'memory': list(self.history['memory']),
            'timestamp': list(self.history['timestamp'])
        }


def monitor_system(
    interval: float = 1.0,
    **kwargs
) -> SystemMonitor:
    """Create and start system monitor."""
    monitor = SystemMonitor(interval, **kwargs)
    monitor.start()
    return monitor


def get_system_status() -> Dict[str, Any]:
    """Get current system status."""
    import psutil
    return {
        'cpu_percent': psutil.cpu_percent(interval=0.1),
        'memory_percent': psutil.virtual_memory().percent,
        'disk_percent': psutil.disk_usage('/').percent
    }



