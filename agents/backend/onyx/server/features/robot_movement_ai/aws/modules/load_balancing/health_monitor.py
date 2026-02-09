"""
Health Monitor
==============

Advanced health monitoring for load balancing.
"""

import logging
import asyncio
import httpx
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class HealthCheck:
    """Health check result."""
    server_id: str
    healthy: bool
    response_time: float
    status_code: Optional[int] = None
    error: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class HealthMonitor:
    """Advanced health monitor."""
    
    def __init__(
        self,
        check_interval: float = 10.0,
        timeout: float = 5.0,
        failure_threshold: int = 3
    ):
        self.check_interval = check_interval
        self.timeout = timeout
        self.failure_threshold = failure_threshold
        self._servers: Dict[str, Dict[str, Any]] = {}
        self._health_history: Dict[str, List[HealthCheck]] = {}
        self._failure_counts: Dict[str, int] = {}
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
    
    def register_server(
        self,
        server_id: str,
        health_check_url: str,
        expected_status: int = 200
    ):
        """Register server for health monitoring."""
        self._servers[server_id] = {
            "url": health_check_url,
            "expected_status": expected_status,
            "healthy": True
        }
        self._health_history[server_id] = []
        self._failure_counts[server_id] = 0
        logger.info(f"Registered server for health monitoring: {server_id}")
    
    async def check_health(self, server_id: str) -> HealthCheck:
        """Check single server health."""
        if server_id not in self._servers:
            raise ValueError(f"Server {server_id} not registered")
        
        server = self._servers[server_id]
        start_time = asyncio.get_event_loop().time()
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(server["url"])
                response_time = asyncio.get_event_loop().time() - start_time
                
                healthy = response.status_code == server["expected_status"]
                
                check = HealthCheck(
                    server_id=server_id,
                    healthy=healthy,
                    response_time=response_time,
                    status_code=response.status_code
                )
                
                if not healthy:
                    self._failure_counts[server_id] += 1
                else:
                    self._failure_counts[server_id] = 0
                
        except Exception as e:
            response_time = asyncio.get_event_loop().time() - start_time
            self._failure_counts[server_id] += 1
            
            check = HealthCheck(
                server_id=server_id,
                healthy=False,
                response_time=response_time,
                error=str(e)
            )
        
        # Update server health
        if self._failure_counts[server_id] >= self.failure_threshold:
            server["healthy"] = False
            logger.warning(f"Server {server_id} marked as unhealthy")
        else:
            server["healthy"] = check.healthy
        
        # Store in history
        self._health_history[server_id].append(check)
        if len(self._health_history[server_id]) > 100:
            self._health_history[server_id].pop(0)
        
        return check
    
    async def check_all(self) -> Dict[str, HealthCheck]:
        """Check health of all servers."""
        tasks = [
            self.check_health(server_id)
            for server_id in self._servers.keys()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        health_checks = {}
        for i, (server_id, result) in enumerate(zip(self._servers.keys(), results)):
            if isinstance(result, Exception):
                logger.error(f"Health check failed for {server_id}: {result}")
                continue
            health_checks[server_id] = result
        
        return health_checks
    
    def start_monitoring(self):
        """Start continuous health monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        
        async def monitor():
            while self._monitoring:
                await self.check_all()
                await asyncio.sleep(self.check_interval)
        
        self._monitor_task = asyncio.create_task(monitor())
        logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
        logger.info("Health monitoring stopped")
    
    def is_healthy(self, server_id: str) -> bool:
        """Check if server is healthy."""
        if server_id not in self._servers:
            return False
        return self._servers[server_id]["healthy"]
    
    def get_health_stats(self) -> Dict[str, Any]:
        """Get health statistics."""
        return {
            "total_servers": len(self._servers),
            "healthy_servers": sum(1 for s in self._servers.values() if s["healthy"]),
            "unhealthy_servers": sum(1 for s in self._servers.values() if not s["healthy"]),
            "server_health": {
                server_id: {
                    "healthy": server["healthy"],
                    "failures": self._failure_counts.get(server_id, 0)
                }
                for server_id, server in self._servers.items()
            }
        }















