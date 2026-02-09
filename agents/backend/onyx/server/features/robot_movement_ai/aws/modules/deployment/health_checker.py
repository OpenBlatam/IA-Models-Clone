"""
Deployment Health Checker
=========================

Health checker for deployments.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional
from aws.modules.observability.health_check import HealthChecker, HealthStatus

logger = logging.getLogger(__name__)


class DeploymentHealthChecker:
    """Health checker for deployments."""
    
    def __init__(self, service_url: str, check_interval: float = 5.0):
        self.service_url = service_url
        self.check_interval = check_interval
        self._health_status: Optional[HealthStatus] = None
        self._check_task: Optional[asyncio.Task] = None
    
    async def check_health(self) -> Dict[str, Any]:
        """Check service health."""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.service_url}/health")
                if response.status_code == 200:
                    self._health_status = HealthStatus.HEALTHY
                    return {"status": "healthy", "response": response.json()}
                else:
                    self._health_status = HealthStatus.UNHEALTHY
                    return {"status": "unhealthy", "status_code": response.status_code}
        except Exception as e:
            self._health_status = HealthStatus.UNHEALTHY
            logger.error(f"Health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}
    
    async def start_monitoring(self):
        """Start health monitoring."""
        async def monitor():
            while True:
                await self.check_health()
                await asyncio.sleep(self.check_interval)
        
        self._check_task = asyncio.create_task(monitor())
        logger.info(f"Started health monitoring for {self.service_url}")
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        if self._check_task:
            self._check_task.cancel()
            logger.info("Stopped health monitoring")
    
    def is_healthy(self) -> bool:
        """Check if service is healthy."""
        return self._health_status == HealthStatus.HEALTHY















