"""
Advanced Monitoring
===================

Advanced monitoring utilities for system metrics and health checks.
"""

import time
import psutil
import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """System metrics."""
    cpu_percent: float
    memory_percent: float
    memory_used: int
    memory_total: int
    disk_percent: float
    disk_used: int
    disk_total: int
    network_sent: int
    network_recv: int
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class HealthCheck:
    """Health check result."""
    name: str
    status: str  # "healthy", "degraded", "unhealthy"
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SystemMonitor:
    """System monitor for collecting metrics."""
    
    def __init__(self):
        """Initialize system monitor."""
        self.metrics_history: deque = deque(maxlen=1000)
        self.process = psutil.Process()
    
    def collect_metrics(self) -> SystemMetrics:
        """
        Collect current system metrics.
        
        Returns:
            System metrics
        """
        cpu_percent = self.process.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        network = psutil.net_io_counters()
        
        metrics = SystemMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used=memory.used,
            memory_total=memory.total,
            disk_percent=disk.percent,
            disk_used=disk.used,
            disk_total=disk.total,
            network_sent=network.bytes_sent,
            network_recv=network.bytes_recv
        )
        
        self.metrics_history.append(metrics)
        return metrics
    
    def get_average_metrics(self, period: Optional[timedelta] = None) -> Optional[SystemMetrics]:
        """
        Get average metrics over period.
        
        Args:
            period: Time period
            
        Returns:
            Average metrics or None
        """
        if not self.metrics_history:
            return None
        
        cutoff = datetime.now() - period if period else datetime.min
        recent_metrics = [
            m for m in self.metrics_history
            if m.timestamp >= cutoff
        ]
        
        if not recent_metrics:
            return None
        
        return SystemMetrics(
            cpu_percent=sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics),
            memory_percent=sum(m.memory_percent for m in recent_metrics) / len(recent_metrics),
            memory_used=sum(m.memory_used for m in recent_metrics) // len(recent_metrics),
            memory_total=recent_metrics[0].memory_total,
            disk_percent=sum(m.disk_percent for m in recent_metrics) / len(recent_metrics),
            disk_used=sum(m.disk_used for m in recent_metrics) // len(recent_metrics),
            disk_total=recent_metrics[0].disk_total,
            network_sent=sum(m.network_sent for m in recent_metrics) // len(recent_metrics),
            network_recv=sum(m.network_recv for m in recent_metrics) // len(recent_metrics)
        )
    
    def get_metrics_trend(self) -> Dict[str, Any]:
        """Get metrics trend."""
        if len(self.metrics_history) < 2:
            return {}
        
        recent = list(self.metrics_history)[-10:]
        
        return {
            "cpu_trend": "increasing" if recent[-1].cpu_percent > recent[0].cpu_percent else "decreasing",
            "memory_trend": "increasing" if recent[-1].memory_percent > recent[0].memory_percent else "decreasing",
            "disk_trend": "increasing" if recent[-1].disk_percent > recent[0].disk_percent else "decreasing"
        }


class HealthMonitor:
    """Health monitor for system health checks."""
    
    def __init__(self):
        """Initialize health monitor."""
        self.checks: Dict[str, Callable[[], HealthCheck]] = {}
        self.history: deque = deque(maxlen=100)
    
    def register_check(self, name: str, check_func: Callable[[], HealthCheck]):
        """
        Register health check.
        
        Args:
            name: Check name
            check_func: Check function
        """
        self.checks[name] = check_func
        logger.info(f"Registered health check: {name}")
    
    def run_checks(self) -> List[HealthCheck]:
        """
        Run all health checks.
        
        Returns:
            List of health check results
        """
        results = []
        for name, check_func in self.checks.items():
            try:
                result = check_func()
                results.append(result)
                self.history.append(result)
            except Exception as e:
                logger.error(f"Error running health check {name}: {e}")
                results.append(HealthCheck(
                    name=name,
                    status="unhealthy",
                    message=f"Check failed: {e}"
                ))
        
        return results
    
    def get_overall_health(self) -> str:
        """
        Get overall health status.
        
        Returns:
            Overall health status
        """
        if not self.checks:
            return "unknown"
        
        results = self.run_checks()
        
        if all(r.status == "healthy" for r in results):
            return "healthy"
        elif any(r.status == "unhealthy" for r in results):
            return "unhealthy"
        else:
            return "degraded"
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary."""
        results = self.run_checks()
        
        return {
            "overall": self.get_overall_health(),
            "checks": {
                r.name: {
                    "status": r.status,
                    "message": r.message,
                    "timestamp": r.timestamp.isoformat()
                }
                for r in results
            },
            "total_checks": len(results),
            "healthy": sum(1 for r in results if r.status == "healthy"),
            "degraded": sum(1 for r in results if r.status == "degraded"),
            "unhealthy": sum(1 for r in results if r.status == "unhealthy")
        }




