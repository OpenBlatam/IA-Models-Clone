"""
Health check system for KV cache.

This module provides comprehensive health checking capabilities including
health status monitoring, dependency checks, and health endpoints.
"""

import time
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    DOWN = "down"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """A health check."""
    name: str
    check_function: Callable[[], bool]
    timeout: float = 5.0
    critical: bool = True
    description: str = ""


@dataclass
class HealthReport:
    """Health report."""
    status: HealthStatus
    timestamp: float
    checks: Dict[str, Dict[str, Any]]
    overall_health: float  # 0.0 to 1.0
    message: str
    dependencies: Dict[str, HealthStatus] = field(default_factory=dict)


class CacheHealthChecker:
    """Health checker for cache."""
    
    def __init__(self, cache: Any):
        self.cache = cache
        self._health_checks: Dict[str, HealthCheck] = {}
        self._lock = threading.Lock()
        
        # Register default health checks
        self._register_default_checks()
        
    def _register_default_checks(self) -> None:
        """Register default health checks."""
        # Cache availability check
        self.register_check(
            "cache_available",
            self._check_cache_available,
            critical=True,
            description="Check if cache is available"
        )
        
        # Memory check
        self.register_check(
            "memory_usage",
            self._check_memory_usage,
            critical=False,
            description="Check memory usage"
        )
        
        # Performance check
        self.register_check(
            "performance",
            self._check_performance,
            critical=False,
            description="Check cache performance"
        )
        
    def register_check(
        self,
        name: str,
        check_function: Callable[[], bool],
        timeout: float = 5.0,
        critical: bool = True,
        description: str = ""
    ) -> None:
        """Register a health check."""
        check = HealthCheck(
            name=name,
            check_function=check_function,
            timeout=timeout,
            critical=critical,
            description=description
        )
        
        with self._lock:
            self._health_checks[name] = check
            
    def _check_cache_available(self) -> bool:
        """Check if cache is available."""
        try:
            # Try to get a test key
            test_key = "__health_check__"
            test_value = self.cache.get(test_key)
            return True
        except Exception:
            return False
            
    def _check_memory_usage(self) -> bool:
        """Check memory usage."""
        try:
            if hasattr(self.cache, '_cache'):
                cache_size = len(self.cache._cache)
                if hasattr(self.cache, 'max_size'):
                    max_size = self.cache.max_size
                    usage_ratio = cache_size / max_size if max_size > 0 else 0.0
                    # Return True if usage is less than 90%
                    return usage_ratio < 0.9
            return True
        except Exception:
            return False
            
    def _check_performance(self) -> bool:
        """Check cache performance."""
        try:
            # Simple performance test
            start_time = time.time()
            test_key = f"__perf_test_{int(time.time())}__"
            self.cache.put(test_key, "test_value")
            self.cache.get(test_key)
            self.cache.delete(test_key)
            duration = time.time() - start_time
            
            # Return True if operations complete in less than 100ms
            return duration < 0.1
        except Exception:
            return False
            
    def run_checks(self) -> HealthReport:
        """Run all health checks."""
        results = {}
        critical_failures = 0
        total_checks = 0
        
        with self._lock:
            checks = dict(self._health_checks)
            
        for name, check in checks.items():
            total_checks += 1
            check_result = self._run_single_check(check)
            results[name] = check_result
            
            if not check_result['passed'] and check.critical:
                critical_failures += 1
                
        # Determine overall status
        if critical_failures > 0:
            status = HealthStatus.UNHEALTHY
            message = f"{critical_failures} critical check(s) failed"
        elif any(not r['passed'] for r in results.values()):
            status = HealthStatus.DEGRADED
            message = "Some non-critical checks failed"
        else:
            status = HealthStatus.HEALTHY
            message = "All checks passed"
            
        # Calculate overall health score
        passed_checks = sum(1 for r in results.values() if r['passed'])
        overall_health = passed_checks / total_checks if total_checks > 0 else 0.0
        
        return HealthReport(
            status=status,
            timestamp=time.time(),
            checks=results,
            overall_health=overall_health,
            message=message
        )
        
    def _run_single_check(self, check: HealthCheck) -> Dict[str, Any]:
        """Run a single health check."""
        start_time = time.time()
        
        try:
            # Run check with timeout
            result = check.check_function()
            duration = time.time() - start_time
            
            return {
                'passed': result,
                'duration': duration,
                'critical': check.critical,
                'description': check.description
            }
        except Exception as e:
            duration = time.time() - start_time
            return {
                'passed': False,
                'duration': duration,
                'error': str(e),
                'critical': check.critical,
                'description': check.description
            }
            
    def get_health_status(self) -> HealthStatus:
        """Get current health status."""
        report = self.run_checks()
        return report.status
        
    def is_healthy(self) -> bool:
        """Check if cache is healthy."""
        status = self.get_health_status()
        return status == HealthStatus.HEALTHY
        
    def get_detailed_report(self) -> HealthReport:
        """Get detailed health report."""
        return self.run_checks()


class HealthMonitor:
    """Continuous health monitoring."""
    
    def __init__(self, cache: Any, check_interval: float = 60.0):
        self.cache = cache
        self.check_interval = check_interval
        self.health_checker = CacheHealthChecker(cache)
        self._monitoring_thread: Optional[threading.Thread] = None
        self._running = False
        self._health_history: List[HealthReport] = []
        self._max_history = 100
        
    def start_monitoring(self) -> None:
        """Start continuous health monitoring."""
        if self._running:
            return
            
        self._running = True
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitoring_thread.start()
        
    def stop_monitoring(self) -> None:
        """Stop continuous health monitoring."""
        self._running = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5.0)
            
    def _monitoring_loop(self) -> None:
        """Monitoring loop."""
        while self._running:
            try:
                report = self.health_checker.run_checks()
                self._health_history.append(report)
                
                # Keep only last N reports
                if len(self._health_history) > self._max_history:
                    self._health_history = self._health_history[-self._max_history:]
                    
                # Alert if unhealthy
                if report.status == HealthStatus.UNHEALTHY:
                    print(f"WARNING: Cache is unhealthy: {report.message}")
                    
                time.sleep(self.check_interval)
            except Exception as e:
                print(f"Error in health monitoring: {e}")
                time.sleep(self.check_interval)
                
    def get_health_history(self) -> List[HealthReport]:
        """Get health check history."""
        return self._health_history.copy()
        
    def get_current_health(self) -> HealthReport:
        """Get current health status."""
        return self.health_checker.get_detailed_report()
