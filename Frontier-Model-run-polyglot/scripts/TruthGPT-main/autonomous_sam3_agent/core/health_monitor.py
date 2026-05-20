"""
Health Monitor
==============

Continuous health monitoring for 24/7 agent operation.
"""

import asyncio
import logging
import psutil
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthConfig:
    """Configuration for health monitoring."""
    check_interval_seconds: int = 30
    max_memory_percent: float = 90.0
    max_cpu_percent: float = 95.0
    max_gpu_percent: float = 95.0
    alert_on_degraded: bool = True
    alert_on_unhealthy: bool = True


class HealthMonitor:
    """
    Monitors health of all agent components.
    
    Features:
    - Periodic health checks
    - Memory/CPU/GPU monitoring
    - External service health (OpenRouter, SAM3)
    - Alert notifications
    - Health history tracking
    """
    
    def __init__(self, config: Optional[HealthConfig] = None):
        """
        Initialize health monitor.
        
        Args:
            config: Health monitoring configuration
        """
        self.config = config or HealthConfig()
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._health_history: List[Dict[str, HealthCheck]] = []
        self._max_history: int = 100
        self._alert_handlers: List[Callable[[HealthCheck], None]] = []
        self._external_checks: Dict[str, Callable] = {}
        
        logger.info(
            f"Initialized HealthMonitor "
            f"(interval={self.config.check_interval_seconds}s)"
        )
    
    def register_check(self, name: str, check_fn: Callable):
        """
        Register an external health check function.
        
        Args:
            name: Check name
            check_fn: Async function that returns HealthCheck
        """
        self._external_checks[name] = check_fn
        logger.debug(f"Registered health check: {name}")
    
    def register_alert_handler(self, handler: Callable[[HealthCheck], None]):
        """
        Register an alert handler for health issues.
        
        Args:
            handler: Callback function for alerts
        """
        self._alert_handlers.append(handler)
    
    async def start(self):
        """Start health monitoring."""
        if self._running:
            logger.warning("HealthMonitor is already running")
            return
        
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("HealthMonitor started")
    
    async def stop(self):
        """Stop health monitoring."""
        if not self._running:
            return
        
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("HealthMonitor stopped")
    
    async def _monitor_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                checks = await self.run_all_checks()
                self._process_results(checks)
                await asyncio.sleep(self.config.check_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"HealthMonitor error: {e}", exc_info=True)
                await asyncio.sleep(10)
    
    async def run_all_checks(self) -> Dict[str, HealthCheck]:
        """Run all health checks."""
        checks = {}
        
        # System checks
        checks["memory"] = self._check_memory()
        checks["cpu"] = self._check_cpu()
        checks["disk"] = self._check_disk()
        
        # GPU check (if available)
        gpu_check = self._check_gpu()
        if gpu_check:
            checks["gpu"] = gpu_check
        
        # External checks
        for name, check_fn in self._external_checks.items():
            try:
                if asyncio.iscoroutinefunction(check_fn):
                    checks[name] = await check_fn()
                else:
                    checks[name] = check_fn()
            except Exception as e:
                checks[name] = HealthCheck(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=str(e),
                )
        
        return checks
    
    def _check_memory(self) -> HealthCheck:
        """Check memory usage."""
        memory = psutil.virtual_memory()
        percent = memory.percent
        
        if percent > self.config.max_memory_percent:
            status = HealthStatus.UNHEALTHY
            message = f"Memory critical: {percent:.1f}%"
        elif percent > self.config.max_memory_percent * 0.8:
            status = HealthStatus.DEGRADED
            message = f"Memory high: {percent:.1f}%"
        else:
            status = HealthStatus.HEALTHY
            message = f"Memory OK: {percent:.1f}%"
        
        return HealthCheck(
            name="memory",
            status=status,
            message=message,
            details={
                "percent": percent,
                "available_gb": memory.available / (1024**3),
                "total_gb": memory.total / (1024**3),
            }
        )
    
    def _check_cpu(self) -> HealthCheck:
        """Check CPU usage."""
        percent = psutil.cpu_percent(interval=1)
        
        if percent > self.config.max_cpu_percent:
            status = HealthStatus.UNHEALTHY
            message = f"CPU critical: {percent:.1f}%"
        elif percent > self.config.max_cpu_percent * 0.8:
            status = HealthStatus.DEGRADED
            message = f"CPU high: {percent:.1f}%"
        else:
            status = HealthStatus.HEALTHY
            message = f"CPU OK: {percent:.1f}%"
        
        return HealthCheck(
            name="cpu",
            status=status,
            message=message,
            details={
                "percent": percent,
                "cores": psutil.cpu_count(),
            }
        )
    
    def _check_disk(self) -> HealthCheck:
        """Check disk usage."""
        disk = psutil.disk_usage("/")
        percent = disk.percent
        
        if percent > 95:
            status = HealthStatus.UNHEALTHY
            message = f"Disk critical: {percent:.1f}%"
        elif percent > 85:
            status = HealthStatus.DEGRADED
            message = f"Disk high: {percent:.1f}%"
        else:
            status = HealthStatus.HEALTHY
            message = f"Disk OK: {percent:.1f}%"
        
        return HealthCheck(
            name="disk",
            status=status,
            message=message,
            details={
                "percent": percent,
                "free_gb": disk.free / (1024**3),
                "total_gb": disk.total / (1024**3),
            }
        )
    
    def _check_gpu(self) -> Optional[HealthCheck]:
        """Check GPU usage (if available)."""
        try:
            import torch
            if not torch.cuda.is_available():
                return None
            
            # Get GPU memory usage
            gpu_memory_allocated = torch.cuda.memory_allocated() / (1024**3)
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            percent = (gpu_memory_allocated / gpu_memory_total) * 100
            
            if percent > self.config.max_gpu_percent:
                status = HealthStatus.UNHEALTHY
                message = f"GPU memory critical: {percent:.1f}%"
            elif percent > self.config.max_gpu_percent * 0.8:
                status = HealthStatus.DEGRADED
                message = f"GPU memory high: {percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"GPU memory OK: {percent:.1f}%"
            
            return HealthCheck(
                name="gpu",
                status=status,
                message=message,
                details={
                    "percent": percent,
                    "allocated_gb": gpu_memory_allocated,
                    "total_gb": gpu_memory_total,
                    "device": torch.cuda.get_device_name(0),
                }
            )
        except Exception:
            return None
    
    def _process_results(self, checks: Dict[str, HealthCheck]):
        """Process health check results."""
        # Store in history
        self._health_history.append(checks)
        if len(self._health_history) > self._max_history:
            self._health_history.pop(0)
        
        # Check for alerts
        for check in checks.values():
            should_alert = (
                (check.status == HealthStatus.DEGRADED and self.config.alert_on_degraded) or
                (check.status == HealthStatus.UNHEALTHY and self.config.alert_on_unhealthy)
            )
            
            if should_alert:
                logger.warning(f"Health alert: {check.name} - {check.message}")
                for handler in self._alert_handlers:
                    try:
                        handler(check)
                    except Exception as e:
                        logger.error(f"Alert handler error: {e}")
    
    def get_current_health(self) -> Dict[str, Any]:
        """Get current health status."""
        if not self._health_history:
            return {"status": "unknown", "checks": {}}
        
        latest = self._health_history[-1]
        
        # Determine overall status
        statuses = [c.status for c in latest.values()]
        if HealthStatus.UNHEALTHY in statuses:
            overall = HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            overall = HealthStatus.DEGRADED
        else:
            overall = HealthStatus.HEALTHY
        
        return {
            "status": overall.value,
            "timestamp": datetime.now().isoformat(),
            "checks": {
                name: {
                    "status": check.status.value,
                    "message": check.message,
                    "details": check.details,
                }
                for name, check in latest.items()
            }
        }
    
    def get_health_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get health check history."""
        history = []
        for checks in self._health_history[-limit:]:
            history.append({
                name: {
                    "status": check.status.value,
                    "message": check.message,
                    "timestamp": check.timestamp.isoformat(),
                }
                for name, check in checks.items()
            })
        return history
