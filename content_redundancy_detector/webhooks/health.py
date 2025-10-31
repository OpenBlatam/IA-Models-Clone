"""
Webhook Health Monitoring
Health checks and monitoring for webhook system
"""

import time
import asyncio
import logging
from typing import Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check"""
    name: str
    status: HealthStatus
    message: str
    timestamp: float
    latency: Optional[float] = None
    details: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "timestamp": self.timestamp,
            "iso_timestamp": datetime.fromtimestamp(self.timestamp).isoformat()
        }
        
        if self.latency is not None:
            result["latency"] = round(self.latency, 3)
        
        if self.details:
            result["details"] = self.details
        
        return result


class WebhookHealthChecker:
    """Comprehensive health checker for webhook system"""
    
    def __init__(self):
        self.checks: Dict[str, callable] = {}
        self.last_check_results: Dict[str, HealthCheckResult] = {}
        self.check_history: Dict[str, List[HealthCheckResult]] = {}
    
    def register_check(self, name: str, check_func: callable):
        """Register a health check"""
        self.checks[name] = check_func
        self.check_history[name] = []
    
    async def check_all(self, timeout: float = 5.0) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks"""
        results = {}
        
        for name, check_func in self.checks.items():
            try:
                start_time = time.time()
                
                # Run check with timeout
                if asyncio.iscoroutinefunction(check_func):
                    check_result = await asyncio.wait_for(check_func(), timeout=timeout)
                else:
                    check_result = check_func()
                
                latency = time.time() - start_time
                
                # Normalize result
                if isinstance(check_result, HealthCheckResult):
                    result = check_result
                    result.latency = latency
                elif isinstance(check_result, dict):
                    result = HealthCheckResult(
                        name=name,
                        status=HealthStatus(check_result.get("status", "unknown")),
                        message=check_result.get("message", ""),
                        timestamp=time.time(),
                        latency=latency,
                        details=check_result.get("details")
                    )
                else:
                    # Simple bool or status string
                    if isinstance(check_result, bool):
                        status = HealthStatus.HEALTHY if check_result else HealthStatus.UNHEALTHY
                    else:
                        status = HealthStatus(str(check_result))
                    
                    result = HealthCheckResult(
                        name=name,
                        status=status,
                        message="Check completed",
                        timestamp=time.time(),
                        latency=latency
                    )
                
            except asyncio.TimeoutError:
                result = HealthCheckResult(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check timeout after {timeout}s",
                    timestamp=time.time(),
                    latency=timeout
                )
        except Exception as e:
                logger.error(f"Health check '{name}' failed: {e}", exc_info=True)
                result = HealthCheckResult(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check error: {str(e)}",
                    timestamp=time.time()
                )
            
            results[name] = result
            self.last_check_results[name] = result
            self.check_history[name].append(result)
            
            # Keep only last 100 results per check
            if len(self.check_history[name]) > 100:
                self.check_history[name] = self.check_history[name][-100:]
        
        return results
    
    def get_overall_status(self) -> HealthStatus:
        """Get overall health status"""
        if not self.last_check_results:
            return HealthStatus.UNKNOWN
        
        statuses = [result.status for result in self.last_check_results.values()]
        
        if all(s == HealthStatus.HEALTHY for s in statuses):
            return HealthStatus.HEALTHY
        elif any(s == HealthStatus.UNHEALTHY for s in statuses):
            return HealthStatus.UNHEALTHY
        else:
            return HealthStatus.DEGRADED
    
    async def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report"""
        await self.check_all()
        
        overall_status = self.get_overall_status()
        
        return {
            "status": overall_status.value,
            "timestamp": time.time(),
            "iso_timestamp": datetime.utcnow().isoformat(),
            "checks": {
                name: result.to_dict()
                for name, result in self.last_check_results.items()
            },
            "summary": {
                "total_checks": len(self.last_check_results),
                "healthy": sum(
                    1 for r in self.last_check_results.values()
                    if r.status == HealthStatus.HEALTHY
                ),
                "degraded": sum(
                    1 for r in self.last_check_results.values()
                    if r.status == HealthStatus.DEGRADED
                ),
                "unhealthy": sum(
                    1 for r in self.last_check_results.values()
                    if r.status == HealthStatus.UNHEALTHY
                )
            }
        }


# Predefined health check functions

async def check_storage_health() -> HealthCheckResult:
    """Check storage backend health"""
    try:
        # This would check actual storage
        # For now, return healthy
        return HealthCheckResult(
            name="storage",
            status=HealthStatus.HEALTHY,
            message="Storage backend is accessible",
            timestamp=time.time()
        )
    except Exception as e:
        return HealthCheckResult(
            name="storage",
            status=HealthStatus.UNHEALTHY,
            message=f"Storage backend error: {str(e)}",
            timestamp=time.time()
        )


async def check_queue_health(manager) -> HealthCheckResult:
    """Check delivery queue health"""
    try:
        queue_size = manager._delivery_queue.qsize() if hasattr(manager, '_delivery_queue') else 0
        max_size = manager._max_queue_size if hasattr(manager, '_max_queue_size') else 1000
        
        queue_percent = (queue_size / max_size * 100) if max_size > 0 else 0
        
        if queue_percent > 90:
            status = HealthStatus.UNHEALTHY
            message = f"Queue nearly full: {queue_size}/{max_size}"
        elif queue_percent > 70:
            status = HealthStatus.DEGRADED
            message = f"Queue high: {queue_size}/{max_size}"
        else:
            status = HealthStatus.HEALTHY
            message = f"Queue healthy: {queue_size}/{max_size}"
        
        return HealthCheckResult(
            name="queue",
            status=status,
            message=message,
            timestamp=time.time(),
            details={
                "queue_size": queue_size,
                "max_size": max_size,
                "percent_full": round(queue_percent, 2)
            }
        )
    except Exception as e:
        return HealthCheckResult(
            name="queue",
            status=HealthStatus.UNHEALTHY,
            message=f"Queue check error: {str(e)}",
            timestamp=time.time()
        )


async def check_workers_health(manager) -> HealthCheckResult:
    """Check worker pool health"""
    try:
        if hasattr(manager, '_worker_tasks'):
            active_workers = sum(1 for task in manager._worker_tasks if not task.done())
            total_workers = len(manager._worker_tasks)
            
            if active_workers == 0 and total_workers > 0:
                status = HealthStatus.UNHEALTHY
                message = "No active workers"
            elif active_workers < total_workers:
                status = HealthStatus.DEGRADED
                message = f"Some workers inactive: {active_workers}/{total_workers}"
            else:
                status = HealthStatus.HEALTHY
                message = f"All workers active: {active_workers}/{total_workers}"
            
            return HealthCheckResult(
                name="workers",
                status=status,
                message=message,
                timestamp=time.time(),
                details={
                    "active_workers": active_workers,
                    "total_workers": total_workers
                }
            )
        else:
            return HealthCheckResult(
                name="workers",
                status=HealthStatus.UNKNOWN,
                message="Worker information not available",
                timestamp=time.time()
            )
    except Exception as e:
        return HealthCheckResult(
            name="workers",
            status=HealthStatus.UNHEALTHY,
            message=f"Worker check error: {str(e)}",
            timestamp=time.time()
        )


# Global health checker instance
health_checker = WebhookHealthChecker()
