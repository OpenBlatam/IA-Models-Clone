"""
CPU Optimizer
=============

Advanced CPU optimization techniques.
"""

import logging
import os
import psutil
from typing import Dict, Any, Optional
from dataclasses import dataclass
import multiprocessing

logger = logging.getLogger(__name__)


@dataclass
class CPUStats:
    """CPU statistics."""
    usage_percent: float
    cores: int
    frequency: float
    load_avg: float


class CPUOptimizer:
    """CPU optimizer with advanced techniques."""
    
    def __init__(self):
        self._affinity_set = False
        self._priority_set = False
    
    def set_cpu_affinity(self, cores: Optional[List[int]] = None):
        """Set CPU affinity for process."""
        try:
            process = psutil.Process(os.getpid())
            
            if cores is None:
                # Use all available cores
                cores = list(range(os.cpu_count()))
            
            process.cpu_affinity(cores)
            self._affinity_set = True
            logger.info(f"CPU affinity set to cores: {cores}")
            return True
        
        except Exception as e:
            logger.warning(f"Failed to set CPU affinity: {e}")
            return False
    
    def set_process_priority(self, priority: str = "high"):
        """Set process priority."""
        try:
            process = psutil.Process(os.getpid())
            
            priority_map = {
                "low": psutil.BELOW_NORMAL_PRIORITY_CLASS,
                "normal": psutil.NORMAL_PRIORITY_CLASS,
                "high": psutil.HIGH_PRIORITY_CLASS,
                "realtime": psutil.REALTIME_PRIORITY_CLASS
            }
            
            if priority in priority_map:
                process.nice(priority_map[priority])
                self._priority_set = True
                logger.info(f"Process priority set to: {priority}")
                return True
        
        except Exception as e:
            logger.warning(f"Failed to set process priority: {e}")
            return False
    
    def get_cpu_stats(self) -> CPUStats:
        """Get CPU statistics."""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_count = os.cpu_count() or multiprocessing.cpu_count()
        cpu_freq = psutil.cpu_freq()
        load_avg = os.getloadavg()[0] if hasattr(os, 'getloadavg') else 0.0
        
        return CPUStats(
            usage_percent=cpu_percent,
            cores=cpu_count,
            frequency=cpu_freq.current if cpu_freq else 0.0,
            load_avg=load_avg
        )
    
    def optimize_for_performance(self):
        """Optimize CPU for maximum performance."""
        # Set high priority
        self.set_process_priority("high")
        
        # Set CPU affinity to all cores
        self.set_cpu_affinity()
        
        logger.info("CPU optimized for performance")
    
    def get_cpu_info(self) -> Dict[str, Any]:
        """Get detailed CPU information."""
        cpu_count = os.cpu_count() or multiprocessing.cpu_count()
        cpu_freq = psutil.cpu_freq()
        
        return {
            "cores": cpu_count,
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "frequency_mhz": cpu_freq.current if cpu_freq else None,
            "min_frequency_mhz": cpu_freq.min if cpu_freq else None,
            "max_frequency_mhz": cpu_freq.max if cpu_freq else None,
            "usage_percent": psutil.cpu_percent(interval=0.1, percpu=True),
            "load_avg": os.getloadavg() if hasattr(os, 'getloadavg') else None
        }















