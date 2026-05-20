"""
System monitoring and health checks.
"""

import logging
import time
from typing import Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Optional dependency for system monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available, system monitoring will be limited")


@dataclass
class SystemHealth:
    """System health metrics."""
    status: str
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    uptime_seconds: float
    timestamp: datetime


class SystemMonitor:
    """
    System monitoring and health checks.
    """
    
    def __init__(self):
        self.start_time = time.time()
        self.health_history: list = []
    
    def get_system_health(self) -> SystemHealth:
        """
        Get current system health metrics.
        
        Returns:
            SystemHealth object with current metrics
        """
        if not PSUTIL_AVAILABLE:
            # Fallback to basic metrics
            uptime = time.time() - self.start_time
            return SystemHealth(
                status="unknown",
                cpu_percent=0.0,
                memory_percent=0.0,
                disk_percent=0.0,
                uptime_seconds=uptime,
                timestamp=datetime.now()
            )
        
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        uptime = time.time() - self.start_time
        
        # Determine overall status
        if cpu_percent > 90 or memory.percent > 90 or disk.percent > 90:
            status = "critical"
        elif cpu_percent > 70 or memory.percent > 70 or disk.percent > 70:
            status = "warning"
        else:
            status = "healthy"
        
        health = SystemHealth(
            status=status,
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            disk_percent=disk.percent,
            uptime_seconds=uptime,
            timestamp=datetime.now()
        )
        
        self.health_history.append(health)
        
        # Keep only last 100 entries
        if len(self.health_history) > 100:
            self.health_history = self.health_history[-100:]
        
        return health
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary statistics."""
        if not self.health_history:
            return {"status": "unknown", "message": "No health data available"}
        
        latest = self.health_history[-1]
        
        return {
            "status": latest.status,
            "cpu_percent": latest.cpu_percent,
            "memory_percent": latest.memory_percent,
            "disk_percent": latest.disk_percent,
            "uptime_seconds": latest.uptime_seconds,
            "uptime_formatted": self._format_uptime(latest.uptime_seconds),
            "timestamp": latest.timestamp.isoformat(),
            "history_count": len(self.health_history)
        }
    
    def _format_uptime(self, seconds: float) -> str:
        """Format uptime in human-readable format."""
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        minutes = int((seconds % 3600) // 60)
        
        if days > 0:
            return f"{days}d {hours}h {minutes}m"
        elif hours > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{minutes}m"
    
    def check_dependencies(self) -> Dict[str, Any]:
        """Check system dependencies."""
        dependencies = {
            "python_version": self._get_python_version(),
            "disk_space": self._check_disk_space(),
            "memory_available": self._check_memory(),
        }
        
        return dependencies
    
    def _get_python_version(self) -> str:
        """Get Python version."""
        import sys
        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    
    def _check_disk_space(self) -> Dict[str, Any]:
        """Check available disk space."""
        if not PSUTIL_AVAILABLE:
            return {"error": "psutil not available"}
        
        disk = psutil.disk_usage('/')
        return {
            "total_gb": round(disk.total / (1024**3), 2),
            "used_gb": round(disk.used / (1024**3), 2),
            "free_gb": round(disk.free / (1024**3), 2),
            "percent_used": disk.percent
        }
    
    def _check_memory(self) -> Dict[str, Any]:
        """Check available memory."""
        if not PSUTIL_AVAILABLE:
            return {"error": "psutil not available"}
        
        memory = psutil.virtual_memory()
        return {
            "total_gb": round(memory.total / (1024**3), 2),
            "available_gb": round(memory.available / (1024**3), 2),
            "used_gb": round(memory.used / (1024**3), 2),
            "percent_used": memory.percent
        }

