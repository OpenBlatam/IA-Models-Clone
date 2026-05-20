"""
Health Check Service
===================

Service for comprehensive health checks of all components.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health status of a component"""
    name: str
    status: HealthStatus
    message: Optional[str] = None
    response_time: Optional[float] = None
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class HealthService:
    """
    Service for health checks.
    
    Features:
    - Component health checks
    - Response time tracking
    - Aggregated health status
    - Detailed component information
    """
    
    def __init__(self):
        """Initialize health service"""
        pass
    
    async def check_comfyui(self) -> ComponentHealth:
        """
        Check ComfyUI service health.
        
        Returns:
            ComponentHealth object
        """
        import time
        from services.comfyui_service import ComfyUIService
        
        start_time = time.time()
        try:
            service = ComfyUIService()
            queue_status = await service.get_queue_status()
            
            response_time = time.time() - start_time
            
            return ComponentHealth(
                name="comfyui",
                status=HealthStatus.HEALTHY,
                message="ComfyUI is accessible",
                response_time=response_time,
                details={
                    "queue_running": len(queue_status.get("queue_running", [])),
                    "queue_pending": len(queue_status.get("queue_pending", []))
                }
            )
        except Exception as e:
            response_time = time.time() - start_time
            logger.error(f"ComfyUI health check failed: {e}")
            return ComponentHealth(
                name="comfyui",
                status=HealthStatus.UNHEALTHY,
                message=f"ComfyUI check failed: {str(e)}",
                response_time=response_time
            )
    
    async def check_openrouter(self) -> ComponentHealth:
        """
        Check OpenRouter service health.
        
        Returns:
            ComponentHealth object
        """
        import time
        from config.settings import get_settings
        
        start_time = time.time()
        settings = get_settings()
        
        if not settings.openrouter_enabled:
            return ComponentHealth(
                name="openrouter",
                status=HealthStatus.UNKNOWN,
                message="OpenRouter is disabled"
            )
        
        try:
            from infrastructure.openrouter_client import get_openrouter_client
            client = get_openrouter_client()
            
            # Simple check - just verify client exists
            response_time = time.time() - start_time
            
            return ComponentHealth(
                name="openrouter",
                status=HealthStatus.HEALTHY,
                message="OpenRouter client is available",
                response_time=response_time
            )
        except Exception as e:
            response_time = time.time() - start_time
            logger.error(f"OpenRouter health check failed: {e}")
            return ComponentHealth(
                name="openrouter",
                status=HealthStatus.DEGRADED,
                message=f"OpenRouter check failed: {str(e)}",
                response_time=response_time
            )
    
    async def check_truthgpt(self) -> ComponentHealth:
        """
        Check TruthGPT service health.
        
        Returns:
            ComponentHealth object
        """
        import time
        from config.settings import get_settings
        
        start_time = time.time()
        settings = get_settings()
        
        if not settings.truthgpt_enabled:
            return ComponentHealth(
                name="truthgpt",
                status=HealthStatus.UNKNOWN,
                message="TruthGPT is disabled"
            )
        
        try:
            from infrastructure.truthgpt_client import TruthGPTClient
            client = TruthGPTClient()
            
            # Simple check - just verify client exists
            response_time = time.time() - start_time
            
            return ComponentHealth(
                name="truthgpt",
                status=HealthStatus.HEALTHY,
                message="TruthGPT client is available",
                response_time=response_time
            )
        except Exception as e:
            response_time = time.time() - start_time
            logger.error(f"TruthGPT health check failed: {e}")
            return ComponentHealth(
                name="truthgpt",
                status=HealthStatus.DEGRADED,
                message=f"TruthGPT check failed: {str(e)}",
                response_time=response_time
            )
    
    async def check_cache(self) -> ComponentHealth:
        """
        Check cache service health.
        
        Returns:
            ComponentHealth object
        """
        import time
        from services.cache_service import get_cache_service
        
        start_time = time.time()
        try:
            cache = get_cache_service()
            stats = cache.get_stats()
            
            response_time = time.time() - start_time
            
            return ComponentHealth(
                name="cache",
                status=HealthStatus.HEALTHY,
                message="Cache service is operational",
                response_time=response_time,
                details={
                    "size": stats["size"],
                    "max_size": stats["max_size"],
                    "hit_rate": stats["hit_rate"]
                }
            )
        except Exception as e:
            response_time = time.time() - start_time
            logger.error(f"Cache health check failed: {e}")
            return ComponentHealth(
                name="cache",
                status=HealthStatus.DEGRADED,
                message=f"Cache check failed: {str(e)}",
                response_time=response_time
            )
    
    async def check_all(self) -> Dict[str, Any]:
        """
        Check health of all components.
        
        Returns:
            Dictionary with overall health and component details
        """
        components = [
            self.check_comfyui(),
            self.check_openrouter(),
            self.check_truthgpt(),
            self.check_cache()
        ]
        
        results = await asyncio.gather(*components, return_exceptions=True)
        
        component_healths = []
        healthy_count = 0
        degraded_count = 0
        unhealthy_count = 0
        
        for result in results:
            if isinstance(result, Exception):
                component_healths.append(ComponentHealth(
                    name="unknown",
                    status=HealthStatus.UNKNOWN,
                    message=f"Check failed: {str(result)}"
                ))
                unhealthy_count += 1
            else:
                component_healths.append(result)
                if result.status == HealthStatus.HEALTHY:
                    healthy_count += 1
                elif result.status == HealthStatus.DEGRADED:
                    degraded_count += 1
                elif result.status == HealthStatus.UNHEALTHY:
                    unhealthy_count += 1
        
        # Determine overall status
        if unhealthy_count > 0:
            overall_status = HealthStatus.UNHEALTHY
        elif degraded_count > 0:
            overall_status = HealthStatus.DEGRADED
        elif healthy_count == len(component_healths):
            overall_status = HealthStatus.HEALTHY
        else:
            overall_status = HealthStatus.UNKNOWN
        
        return {
            "status": overall_status.value,
            "timestamp": datetime.now().isoformat(),
            "components": [
                {
                    "name": ch.name,
                    "status": ch.status.value,
                    "message": ch.message,
                    "response_time": ch.response_time,
                    "details": ch.details,
                    "timestamp": ch.timestamp.isoformat() if ch.timestamp else None
                }
                for ch in component_healths
            ],
            "summary": {
                "total": len(component_healths),
                "healthy": healthy_count,
                "degraded": degraded_count,
                "unhealthy": unhealthy_count
            }
        }


# Global health service instance
_health_service: Optional[HealthService] = None


def get_health_service() -> HealthService:
    """Get or create health service instance"""
    global _health_service
    if _health_service is None:
        _health_service = HealthService()
    return _health_service

