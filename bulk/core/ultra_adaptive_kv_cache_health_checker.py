"""
Advanced Health Checking and Self-Healing for Ultra-Adaptive K/V Cache Engine
"""

import asyncio
import time
from typing import Dict, Any, Optional, List, Callable
from enum import Enum
from dataclasses import dataclass
import logging

try:
    from ultra_adaptive_kv_cache_engine import UltraAdaptiveKVCacheEngine
except ImportError:
    UltraAdaptiveKVCacheEngine = None

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str
    timestamp: float
    details: Optional[Dict[str, Any]] = None


class ComponentHealthChecker:
    """Health checker for individual components."""
    
    def __init__(self, name: str, check_fn: Callable[[], bool], 
                 critical: bool = False, timeout: float = 5.0):
        self.name = name
        self.check_fn = check_fn
        self.critical = critical
        self.timeout = timeout
        self.last_check = None
        self.last_result = None
    
    async def check(self) -> HealthCheckResult:
        """Perform health check."""
        start_time = time.time()
        
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(self.check_fn) if not asyncio.iscoroutinefunction(self.check_fn) 
                else self.check_fn(),
                timeout=self.timeout
            )
            
            status = HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY
            message = f"{self.name} is {'operational' if result else 'not responding'}"
            
        except asyncio.TimeoutError:
            status = HealthStatus.CRITICAL if self.critical else HealthStatus.UNHEALTHY
            message = f"{self.name} check timed out after {self.timeout}s"
            result = False
        
        except Exception as e:
            status = HealthStatus.CRITICAL if self.critical else HealthStatus.UNHEALTHY
            message = f"{self.name} check failed: {str(e)}"
            result = False
        
        health_result = HealthCheckResult(
            name=self.name,
            status=status,
            message=message,
            timestamp=time.time(),
            details={'check_duration': time.time() - start_time, 'result': result}
        )
        
        self.last_check = time.time()
        self.last_result = health_result
        
        return health_result


class SystemHealthMonitor:
    """Comprehensive system health monitoring."""
    
    def __init__(self, engine: UltraAdaptiveKVCacheEngine):
        self.engine = engine
        self.health_checks: Dict[str, ComponentHealthChecker] = {}
        self.health_history = []
        self.monitoring = False
        self.check_interval = 30.0
    
    def register_check(self, name: str, check_fn: Callable[[], bool], 
                       critical: bool = False, timeout: float = 5.0):
        """Register a health check."""
        checker = ComponentHealthChecker(name, check_fn, critical, timeout)
        self.health_checks[name] = checker
    
    async def check_all(self) -> Dict[str, HealthCheckResult]:
        """Perform all health checks."""
        results = {}
        
        tasks = [checker.check() for checker in self.health_checks.values()]
        check_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, (name, checker) in enumerate(self.health_checks.items()):
            if isinstance(check_results[i], Exception):
                results[name] = HealthCheckResult(
                    name=name,
                    status=HealthStatus.CRITICAL,
                    message=f"Health check exception: {str(check_results[i])}",
                    timestamp=time.time()
                )
            else:
                results[name] = check_results[i]
        
        return results
    
    def get_overall_status(self, results: Dict[str, HealthCheckResult]) -> HealthStatus:
        """Get overall system status."""
        if not results:
            return HealthStatus.UNHEALTHY
        
        critical_failed = any(
            r.status == HealthStatus.CRITICAL 
            for r in results.values() 
            if self.health_checks[r.name].critical
        )
        
        if critical_failed:
            return HealthStatus.CRITICAL
        
        unhealthy_count = sum(1 for r in results.values() if r.status == HealthStatus.UNHEALTHY)
        total_critical = sum(1 for c in self.health_checks.values() if c.critical)
        
        if unhealthy_count > total_critical * 0.5:
            return HealthStatus.UNHEALTHY
        
        degraded_count = sum(1 for r in results.values() if r.status == HealthStatus.DEGRADED)
        if degraded_count > 0:
            return HealthStatus.DEGRADED
        
        return HealthStatus.HEALTHY
    
    async def start_monitoring(self):
        """Start continuous health monitoring."""
        self.monitoring = True
        logger.info("Starting health monitoring...")
        
        while self.monitoring:
            try:
                results = await self.check_all()
                overall = self.get_overall_status(results)
                
                self.health_history.append({
                    'timestamp': time.time(),
                    'overall_status': overall.value,
                    'checks': {k: {
                        'status': v.status.value,
                        'message': v.message
                    } for k, v in results.items()}
                })
                
                # Keep only last 100 health checks
                if len(self.health_history) > 100:
                    self.health_history = self.health_history[-100:]
                
                # Log if unhealthy
                if overall != HealthStatus.HEALTHY:
                    logger.warning(f"System health: {overall.value}")
                
                await asyncio.sleep(self.check_interval)
            
            except Exception as e:
                logger.error(f"Error in health monitoring: {e}")
                await asyncio.sleep(self.check_interval)
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        self.monitoring = False
        logger.info("Stopped health monitoring")
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report."""
        if not self.health_history:
            return {'status': 'no_data'}
        
        latest = self.health_history[-1]
        results = latest['checks']
        
        return {
            'timestamp': latest['timestamp'],
            'overall_status': latest['overall_status'],
            'components': {
                name: {
                    'status': check['status'],
                    'message': check['message']
                }
                for name, check in results.items()
            },
            'history_summary': {
                'total_checks': len(self.health_history),
                'healthy_percentage': sum(
                    1 for h in self.health_history 
                    if h['overall_status'] == 'healthy'
                ) / len(self.health_history) * 100 if self.health_history else 0
            }
        }


class SelfHealingManager:
    """Self-healing capabilities for the engine."""
    
    def __init__(self, engine: UltraAdaptiveKVCacheEngine):
        self.engine = engine
        self.recovery_actions = []
        self.recovery_history = []
    
    async def diagnose_and_heal(self, health_results: Dict[str, HealthCheckResult]):
        """Diagnose issues and attempt automatic recovery."""
        for name, result in health_results.items():
            if result.status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]:
                await self._attempt_recovery(name, result)
    
    async def _attempt_recovery(self, component: str, issue: HealthCheckResult):
        """Attempt to recover a component."""
        logger.info(f"Attempting recovery for {component}: {issue.message}")
        
        recovery_attempt = {
            'component': component,
            'issue': issue.message,
            'timestamp': time.time(),
            'actions_taken': []
        }
        
        # Memory issues
        if 'memory' in component.lower():
            recovery_attempt['actions_taken'].append('clear_cache')
            self.engine.clear_cache()
            
            recovery_attempt['actions_taken'].append('force_gc')
            import gc
            gc.collect()
        
        # Session issues
        if 'session' in component.lower():
            recovery_attempt['actions_taken'].append('cleanup_sessions')
            self.engine.cleanup_sessions(max_age=1800)  # 30 minutes
        
        # GPU issues
        if 'gpu' in component.lower():
            recovery_attempt['actions_taken'].append('reset_gpu_state')
            # Reset GPU workloads
            for gpu_id in self.engine.gpu_workloads:
                self.engine.gpu_workloads[gpu_id]['active_tasks'] = 0
        
        recovery_attempt['success'] = True
        self.recovery_history.append(recovery_attempt)
        
        # Keep only last 50 recovery attempts
        if len(self.recovery_history) > 50:
            self.recovery_history = self.recovery_history[-50:]
        
        logger.info(f"Recovery attempt for {component} completed: {recovery_attempt['actions_taken']}")


# Predefined health checks for common components

def create_engine_health_checks(engine: UltraAdaptiveKVCacheEngine) -> SystemHealthMonitor:
    """Create health monitor with predefined checks for engine."""
    monitor = SystemHealthMonitor(engine)
    
    # Memory health check
    def check_memory():
        try:
            mem_usage = engine._get_current_memory_usage()
            return mem_usage < 0.95  # Healthy if < 95%
        except:
            return False
    
    monitor.register_check("memory", check_memory, critical=True)
    
    # Cache health check
    def check_cache():
        try:
            stats = engine.get_performance_stats()
            cache_hit_rate = stats.get('engine_stats', {}).get('cache_hit_rate', 1.0)
            return cache_hit_rate > 0.3  # Healthy if hit rate > 30%
        except:
            return False
    
    monitor.register_check("cache", check_cache)
    
    # GPU health check
    def check_gpu():
        try:
            if engine.available_gpus:
                # Check if at least one GPU is available
                return len(engine.available_gpus) > 0
            return True  # CPU mode is fine
        except:
            return False
    
    monitor.register_check("gpu", check_gpu)
    
    # Error rate check
    def check_error_rate():
        try:
            stats = engine.get_performance_stats()
            error_rate = stats.get('engine_stats', {}).get('error_rate', 0)
            return error_rate < 0.1  # Healthy if error rate < 10%
        except:
            return False
    
    monitor.register_check("error_rate", check_error_rate, critical=True)
    
    # Response time check
    def check_response_time():
        try:
            stats = engine.get_performance_stats()
            avg_time = stats.get('engine_stats', {}).get('avg_response_time', 0)
            return avg_time < 5.0  # Healthy if avg < 5s
        except:
            return False
    
    monitor.register_check("response_time", check_response_time)
    
    return monitor

