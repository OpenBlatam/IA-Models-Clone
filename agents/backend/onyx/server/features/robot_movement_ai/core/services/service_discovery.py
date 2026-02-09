"""
Service Discovery System
========================

Sistema de descubrimiento de servicios.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ServiceStatus(Enum):
    """Estado del servicio."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class Service:
    """Servicio."""
    service_id: str
    name: str
    address: str
    port: int
    version: str = "1.0.0"
    status: ServiceStatus = ServiceStatus.UNKNOWN
    metadata: Dict[str, Any] = field(default_factory=dict)
    registered_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_health_check: Optional[str] = None


class ServiceDiscovery:
    """
    Descubrimiento de servicios.
    
    Gestiona registro y descubrimiento de servicios.
    """
    
    def __init__(self):
        """Inicializar descubrimiento de servicios."""
        self.services: Dict[str, Service] = {}
        self.service_groups: Dict[str, List[str]] = {}  # group_name -> service_ids
        self.health_check_interval: float = 30.0
        self.health_check_task: Optional[asyncio.Task] = None
    
    def register_service(
        self,
        service_id: str,
        name: str,
        address: str,
        port: int,
        version: str = "1.0.0",
        group: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Service:
        """
        Registrar servicio.
        
        Args:
            service_id: ID único del servicio
            name: Nombre del servicio
            address: Dirección IP
            port: Puerto
            version: Versión
            group: Grupo del servicio (opcional)
            metadata: Metadata adicional
            
        Returns:
            Servicio registrado
        """
        service = Service(
            service_id=service_id,
            name=name,
            address=address,
            port=port,
            version=version,
            metadata=metadata or {}
        )
        
        self.services[service_id] = service
        
        if group:
            if group not in self.service_groups:
                self.service_groups[group] = []
            if service_id not in self.service_groups[group]:
                self.service_groups[group].append(service_id)
        
        logger.info(f"Registered service: {name} ({service_id}) at {address}:{port}")
        
        return service
    
    def unregister_service(self, service_id: str) -> bool:
        """Desregistrar servicio."""
        if service_id in self.services:
            del self.services[service_id]
            
            # Remover de grupos
            for group_services in self.service_groups.values():
                if service_id in group_services:
                    group_services.remove(service_id)
            
            logger.info(f"Unregistered service: {service_id}")
            return True
        return False
    
    def get_service(self, service_id: str) -> Optional[Service]:
        """Obtener servicio por ID."""
        return self.services.get(service_id)
    
    def find_services(
        self,
        name: Optional[str] = None,
        group: Optional[str] = None,
        status: Optional[ServiceStatus] = None
    ) -> List[Service]:
        """
        Buscar servicios.
        
        Args:
            name: Filtrar por nombre
            group: Filtrar por grupo
            status: Filtrar por estado
            
        Returns:
            Lista de servicios
        """
        services = list(self.services.values())
        
        if name:
            services = [s for s in services if s.name == name]
        
        if group:
            group_service_ids = self.service_groups.get(group, [])
            services = [s for s in services if s.service_id in group_service_ids]
        
        if status:
            services = [s for s in services if s.status == status]
        
        return services
    
    async def check_service_health(self, service: Service) -> bool:
        """
        Verificar salud del servicio.
        
        Args:
            service: Servicio a verificar
            
        Returns:
            True si está saludable, False si no
        """
        try:
            import aiohttp
            url = f"http://{service.address}:{service.port}/health"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                    is_healthy = response.status == 200
                    service.status = ServiceStatus.HEALTHY if is_healthy else ServiceStatus.UNHEALTHY
                    service.last_health_check = datetime.now().isoformat()
                    return is_healthy
        except Exception as e:
            logger.warning(f"Health check failed for {service.name}: {e}")
            service.status = ServiceStatus.UNHEALTHY
            service.last_health_check = datetime.now().isoformat()
            return False
    
    async def start_health_checks(self) -> None:
        """Iniciar verificaciones de salud periódicas."""
        if self.health_check_task:
            return
        
        async def health_check_loop():
            while True:
                try:
                    for service in list(self.services.values()):
                        await self.check_service_health(service)
                    await asyncio.sleep(self.health_check_interval)
                except Exception as e:
                    logger.error(f"Error in health check loop: {e}")
                    await asyncio.sleep(self.health_check_interval)
        
        self.health_check_task = asyncio.create_task(health_check_loop())
        logger.info("Started health check loop")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas del descubrimiento de servicios."""
        total_services = len(self.services)
        healthy_services = sum(1 for s in self.services.values() if s.status == ServiceStatus.HEALTHY)
        unhealthy_services = sum(1 for s in self.services.values() if s.status == ServiceStatus.UNHEALTHY)
        
        return {
            "total_services": total_services,
            "healthy_services": healthy_services,
            "unhealthy_services": unhealthy_services,
            "service_groups": len(self.service_groups),
            "services_by_group": {
                group: len(service_ids)
                for group, service_ids in self.service_groups.items()
            }
        }


# Instancia global
_service_discovery: Optional[ServiceDiscovery] = None


def get_service_discovery() -> ServiceDiscovery:
    """Obtener instancia global del descubrimiento de servicios."""
    global _service_discovery
    if _service_discovery is None:
        _service_discovery = ServiceDiscovery()
    return _service_discovery






