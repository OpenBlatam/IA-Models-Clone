from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import time
import asyncio
from typing import Callable, Dict
from ...core.interfaces.health_interface import IHealthService
from ...core.entities.health import HealthStatus, ComponentHealth, HealthState
from ...shared.config import EnterpriseConfig
import logging
                import psutil
                import psutil
from typing import Any, List, Dict, Optional
"""
Health Check Service Implementation
==================================

Concrete implementation of health check service.
"""


logger = logging.getLogger(__name__)


class HealthCheckService(IHealthService):
    """Health check service implementation."""
    
    def __init__(self, config: EnterpriseConfig):
        
    """__init__ function."""
self.config = config
        self.checks: Dict[str, Callable] = {}
        self.last_check_time: Dict[str, float] = {}
        self.check_cache_ttl = 30  # seconds
        self.cached_results: Dict[str, ComponentHealth] = {}
        
    def register_check(self, name: str, check_func: Callable):
        """Register a health check function."""
        self.checks[name] = check_func
        logger.debug(f"Registered health check: {name}")
    
    async def run_checks(self) -> HealthStatus:
        """Run all registered health checks."""
        health_status = HealthStatus.create_healthy(self.config.app_version)
        
        for name, check_func in self.checks.items():
            try:
                # Use cached result if available and not expired
                if (name in self.last_check_time and 
                    time.time() - self.last_check_time[name] < self.check_cache_ttl and
                    name in self.cached_results):
                    component_health = self.cached_results[name]
                else:
                    # Run the check
                    if asyncio.iscoroutinefunction(check_func):
                        check_result = await asyncio.wait_for(check_func(), timeout=5.0)
                    else:
                        check_result = check_func()
                    
                    # Create component health
                    if isinstance(check_result, bool):
                        state = HealthState.HEALTHY if check_result else HealthState.UNHEALTHY
                        message = "OK" if check_result else "Check failed"
                        details = {}
                    elif isinstance(check_result, dict):
                        state = HealthState.HEALTHY if check_result.get("healthy", False) else HealthState.UNHEALTHY
                        message = check_result.get("message", "")
                        details = check_result.get("details", {})
                    else:
                        state = HealthState.HEALTHY if check_result else HealthState.UNHEALTHY
                        message = str(check_result)
                        details = {}
                    
                    component_health = ComponentHealth(
                        name=name,
                        state=state,
                        message=message,
                        details=details
                    )
                    
                    # Cache the result
                    self.cached_results[name] = component_health
                    self.last_check_time[name] = time.time()
                
                health_status.add_component_check(component_health)
                
            except asyncio.TimeoutError:
                logger.warning(f"Health check {name} timed out")
                component_health = ComponentHealth(
                    name=name,
                    state=HealthState.UNHEALTHY,
                    message="Health check timed out",
                    details={"error": "timeout", "timeout_seconds": 5.0}
                )
                health_status.add_component_check(component_health)
                
            except Exception as e:
                logger.error(f"Health check {name} failed: {e}")
                component_health = ComponentHealth(
                    name=name,
                    state=HealthState.UNHEALTHY,
                    message=f"Health check failed: {str(e)}",
                    details={"error": str(e), "error_type": type(e).__name__}
                )
                health_status.add_component_check(component_health)
        
        return health_status
    
    async def check_liveness(self) -> bool:
        """Check if the service is alive."""
        try:
            # Basic liveness check - just verify the service can respond
            return True
        except Exception as e:
            logger.error(f"Liveness check failed: {e}")
            return False
    
    async def check_readiness(self) -> bool:
        """Check if the service is ready to serve requests."""
        try:
            health_status = await self.run_checks()
            return health_status.is_ready()
        except Exception as e:
            logger.error(f"Readiness check failed: {e}")
            return False
    
    async def get_health_status(self) -> HealthStatus:
        """Get current health status."""
        return await self.run_checks()
    
    def register_default_checks(self) -> Any:
        """Register default health checks."""
        
        def basic_check():
            """Basic health check - always returns healthy."""
            return {"healthy": True, "message": "Service is running"}
        
        def memory_check():
            """Check memory usage."""
            try:
                memory_percent = psutil.virtual_memory().percent
                return {
                    "healthy": memory_percent < 90,
                    "message": f"Memory usage: {memory_percent:.1f}%",
                    "details": {"memory_percent": memory_percent}
                }
            except ImportError:
                return {"healthy": True, "message": "Memory check not available (psutil not installed)"}
        
        def disk_check():
            """Check disk usage."""
            try:
                disk_percent = psutil.disk_usage('/').percent
                return {
                    "healthy": disk_percent < 85,
                    "message": f"Disk usage: {disk_percent:.1f}%",
                    "details": {"disk_percent": disk_percent}
                }
            except (ImportError, Exception):
                return {"healthy": True, "message": "Disk check not available"}
        
        self.register_check("basic", basic_check)
        self.register_check("memory", memory_check)
        self.register_check("disk", disk_check) 