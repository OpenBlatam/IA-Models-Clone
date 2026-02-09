"""
Distributed Health Checks
Health checking across microservices
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime

from microservices.service_discovery import ServiceRegistry, get_service_registry
from microservices.service_client import get_service_client

logger = logging.getLogger(__name__)


class DistributedHealthChecker:
    """
    Distributed health checker for microservices
    
    Features:
    - Health checks across services
    - Dependency health tracking
    - Health aggregation
    - Health status propagation
    """
    
    def __init__(self):
        self.registry = get_service_registry()
        self._health_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ttl = 30  # seconds
    
    async def check_service_health(
        self,
        service_name: str,
        instance_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Check health of a service"""
        import time
        
        cache_key = f"{service_name}:{instance_id or 'all'}"
        
        # Check cache
        if cache_key in self._health_cache:
            cached_time = self._health_cache[cache_key].get("checked_at", 0)
            if time.time() - cached_time < self._cache_ttl:
                return self._health_cache[cache_key]
        
        # Perform health check
        if instance_id:
            # Check specific instance
            status = self.registry.check_health(service_name, instance_id)
            health_data = {
                "service": service_name,
                "instance": instance_id,
                "status": status.value,
                "checked_at": time.time()
            }
        else:
            # Check all instances
            info = self.registry.get_service_info(service_name)
            health_data = {
                "service": service_name,
                "status": info.get("status", "unknown"),
                "healthy_instances": info.get("healthy_instances", 0),
                "total_instances": info.get("total_instances", 0),
                "instances": info.get("instances", []),
                "checked_at": time.time()
            }
        
        # Cache result
        self._health_cache[cache_key] = health_data
        
        return health_data
    
    async def check_all_services(self) -> Dict[str, Any]:
        """Check health of all registered services"""
        services = self.registry.list_services()
        
        tasks = [
            self.check_service_health(service_name)
            for service_name in services
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        health_status = {}
        for service_name, result in zip(services, results):
            if isinstance(result, Exception):
                health_status[service_name] = {
                    "status": "error",
                    "error": str(result)
                }
            else:
                health_status[service_name] = result
        
        # Determine overall health
        all_healthy = all(
            s.get("status") == "healthy" or s.get("healthy_instances", 0) > 0
            for s in health_status.values()
        )
        
        return {
            "overall_status": "healthy" if all_healthy else "degraded",
            "services": health_status,
            "checked_at": datetime.utcnow().isoformat()
        }
    
    async def check_dependencies(
        self,
        service_name: str,
        dependencies: List[str]
    ) -> Dict[str, Any]:
        """Check health of service dependencies"""
        dependency_health = {}
        
        for dep in dependencies:
            try:
                health = await self.check_service_health(dep)
                dependency_health[dep] = health
            except Exception as e:
                dependency_health[dep] = {
                    "status": "error",
                    "error": str(e)
                }
        
        all_deps_healthy = all(
            d.get("status") == "healthy" or d.get("healthy_instances", 0) > 0
            for d in dependency_health.values()
        )
        
        return {
            "service": service_name,
            "dependencies": dependency_health,
            "all_healthy": all_deps_healthy
        }
    
    async def propagate_health(self, service_name: str) -> None:
        """Propagate health status to dependent services"""
        # In a real implementation, would notify dependent services
        # about health status changes
        logger.info(f"Propagating health status for: {service_name}")


# Global health checker
_health_checker: Optional[DistributedHealthChecker] = None


def get_distributed_health_checker() -> DistributedHealthChecker:
    """Get global distributed health checker"""
    global _health_checker
    if _health_checker is None:
        _health_checker = DistributedHealthChecker()
    return _health_checker















