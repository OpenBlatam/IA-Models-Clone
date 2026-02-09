"""
System Monitor

System resource monitoring and alerting.
"""

import logging
import psutil
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class SystemMonitor:
    """
    Monitor system resources and performance.
    """
    
    def __init__(self):
        """Initialize system monitor."""
        self.warnings = []
        self.alerts = []
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get current system information.
        
        Returns:
            Dictionary with system info
        """
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Memory
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available = memory.available / (1024 ** 3)  # GB
            
            # Disk
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            disk_free = disk.free / (1024 ** 3)  # GB
            
            # Network (if available)
            try:
                network = psutil.net_io_counters()
                network_info = {
                    "bytes_sent": network.bytes_sent,
                    "bytes_recv": network.bytes_recv,
                }
            except:
                network_info = None
            
            return {
                "cpu": {
                    "percent": cpu_percent,
                    "count": cpu_count,
                },
                "memory": {
                    "percent": memory_percent,
                    "available_gb": round(memory_available, 2),
                    "total_gb": round(memory.total / (1024 ** 3), 2),
                },
                "disk": {
                    "percent": disk_percent,
                    "free_gb": round(disk_free, 2),
                    "total_gb": round(disk.total / (1024 ** 3), 2),
                },
                "network": network_info,
                "timestamp": datetime.utcnow().isoformat(),
            }
        
        except Exception as e:
            logger.error(f"Failed to get system info: {str(e)}", exc_info=True)
            return {
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }
    
    def check_resources(self) -> Dict[str, Any]:
        """
        Check system resources and generate warnings.
        
        Returns:
            Dictionary with resource status
        """
        info = self.get_system_info()
        
        if "error" in info:
            return info
        
        warnings = []
        alerts = []
        
        # Check CPU
        if info["cpu"]["percent"] > 90:
            alerts.append("CPU usage critical (>90%)")
        elif info["cpu"]["percent"] > 80:
            warnings.append("CPU usage high (>80%)")
        
        # Check Memory
        if info["memory"]["percent"] > 90:
            alerts.append("Memory usage critical (>90%)")
        elif info["memory"]["percent"] > 80:
            warnings.append("Memory usage high (>80%)")
        
        # Check Disk
        if info["disk"]["percent"] > 90:
            alerts.append("Disk usage critical (>90%)")
        elif info["disk"]["percent"] > 80:
            warnings.append("Disk usage high (>80%)")
        
        return {
            **info,
            "warnings": warnings,
            "alerts": alerts,
            "status": "critical" if alerts else ("warning" if warnings else "ok"),
        }
    
    def get_process_info(self) -> Dict[str, Any]:
        """
        Get current process information.
        
        Returns:
            Dictionary with process info
        """
        try:
            process = psutil.Process()
            
            return {
                "pid": process.pid,
                "memory_mb": round(process.memory_info().rss / (1024 ** 2), 2),
                "cpu_percent": process.cpu_percent(interval=0.1),
                "num_threads": process.num_threads(),
                "status": process.status(),
            }
        except Exception as e:
            logger.error(f"Failed to get process info: {str(e)}", exc_info=True)
            return {"error": str(e)}


# Global system monitor instance
_system_monitor: Optional[SystemMonitor] = None


def get_system_monitor() -> SystemMonitor:
    """
    Get global system monitor instance.
    
    Returns:
        SystemMonitor instance
    """
    global _system_monitor
    if _system_monitor is None:
        _system_monitor = SystemMonitor()
    return _system_monitor



