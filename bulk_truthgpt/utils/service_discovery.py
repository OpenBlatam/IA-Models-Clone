"""
Service Discovery
=================

Advanced service discovery and health checking for microservices.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import aiohttp

logger = logging.getLogger(__name__)

@dataclass
class ServiceInstance:
    """Service instance definition."""
    service_id: str
    service_name: str
    host: str
    port: int
    health_check_url: Optional[str] = None
    metadata: Dict[str, Any] = None
    last_health_check: Optional[datetime] = None
    is_healthy: bool = True
    weight: int = 1

class ServiceDiscovery:
    """Advanced service discovery."""
    
    def __init__(self):
        self.services: Dict[str, List[ServiceInstance]] = {}
        self.health_check_interval = 30.0  # seconds
        self.health_check_task = None
        self.is_running = False
    
    def register_service(
        self,
        service_name: str,
        service_id: str,
        host: str,
        port: int,
        health_check_url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        weight: int = 1
    ) -> ServiceInstance:
        """Register a service instance."""
        instance = ServiceInstance(
            service_id=service_id,
            service_name=service_name,
            host=host,
            port=port,
            health_check_url=health_check_url,
            metadata=metadata or {},
            weight=weight
        )
        
        if service_name not in self.services:
            self.services[service_name] = []
        
        # Check if instance already exists
        existing = next(
            (i for i in self.services[service_name] if i.service_id == service_id),
            None
        )
        
        if existing:
            # Update existing
            self.services[service_name].remove(existing)
        
        self.services[service_name].append(instance)
        logger.info(f"Service registered: {service_name} ({service_id}) at {host}:{port}")
        
        return instance
    
    def deregister_service(self, service_name: str, service_id: str):
        """Deregister a service instance."""
        if service_name in self.services:
            self.services[service_name] = [
                i for i in self.services[service_name]
                if i.service_id != service_id
            ]
            logger.info(f"Service deregistered: {service_name} ({service_id})")
    
    def get_instances(self, service_name: str, healthy_only: bool = True) -> List[ServiceInstance]:
        """Get service instances."""
        instances = self.services.get(service_name, [])
        
        if healthy_only:
            instances = [i for i in instances if i.is_healthy]
        
        return instances
    
    def get_instance(self, service_name: str, strategy: str = "round_robin") -> Optional[ServiceInstance]:
        """Get a service instance using load balancing strategy."""
        instances = self.get_instances(service_name, healthy_only=True)
        
        if not instances:
            return None
        
        if strategy == "round_robin":
            # Simple round robin (would need state for real round robin)
            return instances[0]
        elif strategy == "random":
            import random
            return random.choice(instances)
        elif strategy == "weighted":
            # Weighted selection
            total_weight = sum(i.weight for i in instances)
            import random
            r = random.uniform(0, total_weight)
            cumulative = 0
            for instance in instances:
                cumulative += instance.weight
                if r <= cumulative:
                    return instance
            return instances[0]
        else:
            return instances[0]
    
    async def health_check(self, instance: ServiceInstance) -> bool:
        """Perform health check on service instance."""
        try:
            if instance.health_check_url:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        instance.health_check_url,
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as response:
                        instance.is_healthy = response.status == 200
            else:
                # Try to connect to host:port
                try:
                    reader, writer = await asyncio.wait_for(
                        asyncio.open_connection(instance.host, instance.port),
                        timeout=5.0
                    )
                    writer.close()
                    await writer.wait_closed()
                    instance.is_healthy = True
                except:
                    instance.is_healthy = False
            
            instance.last_health_check = datetime.now()
            return instance.is_healthy
            
        except Exception as e:
            logger.error(f"Health check failed for {instance.service_name} ({instance.service_id}): {e}")
            instance.is_healthy = False
            instance.last_health_check = datetime.now()
            return False
    
    async def _health_check_loop(self):
        """Background health check loop."""
        while self.is_running:
            try:
                for service_name, instances in self.services.items():
                    for instance in instances:
                        await self.health_check(instance)
                
                await asyncio.sleep(self.health_check_interval)
                
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(self.health_check_interval)
    
    async def start(self):
        """Start service discovery."""
        if self.is_running:
            return
        
        self.is_running = True
        self.health_check_task = asyncio.create_task(self._health_check_loop())
        logger.info("Service discovery started")
    
    async def stop(self):
        """Stop service discovery."""
        self.is_running = False
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
        logger.info("Service discovery stopped")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get service discovery statistics."""
        total_instances = sum(len(instances) for instances in self.services.values())
        healthy_instances = sum(
            sum(1 for i in instances if i.is_healthy)
            for instances in self.services.values()
        )
        
        return {
            "total_services": len(self.services),
            "total_instances": total_instances,
            "healthy_instances": healthy_instances,
            "unhealthy_instances": total_instances - healthy_instances,
            "services": {
                name: {
                    "total": len(instances),
                    "healthy": sum(1 for i in instances if i.is_healthy),
                    "instances": [
                        {
                            "id": i.service_id,
                            "host": i.host,
                            "port": i.port,
                            "healthy": i.is_healthy,
                            "last_check": i.last_health_check.isoformat() if i.last_health_check else None
                        }
                        for i in instances
                    ]
                }
                for name, instances in self.services.items()
            }
        }

# Global instance
service_discovery = ServiceDiscovery()

















