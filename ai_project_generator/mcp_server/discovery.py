"""
MCP Service Discovery - Descubrimiento de servicios
====================================================
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from enum import Enum

logger = logging.getLogger(__name__)


class MCPServiceStatus(str, Enum):
    """Estados de servicio MCP"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ServiceInfo(BaseModel):
    """Información de un servicio"""
    service_id: str = Field(..., description="ID único del servicio")
    name: str = Field(..., description="Nombre del servicio")
    endpoint: str = Field(..., description="Endpoint del servicio")
    version: str = Field(default="1.0.0", description="Versión del servicio")
    status: MCPServiceStatus = Field(default=MCPServiceStatus.UNKNOWN, description="Estado del servicio")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata adicional")
    last_seen: datetime = Field(default_factory=datetime.utcnow, description="Última vez visto")
    heartbeat_interval: int = Field(default=30, description="Intervalo de heartbeat en segundos")


class ServiceRegistry:
    """
    Registry de servicios MCP
    
    Permite registrar y descubrir servicios disponibles.
    """
    
    def __init__(self, heartbeat_timeout: int = 60):
        """
        Args:
            heartbeat_timeout: Timeout en segundos para considerar servicio muerto
        """
        self.heartbeat_timeout = heartbeat_timeout
        self._services: Dict[str, ServiceInfo] = {}
        self._running = False
        self._cleanup_task: Optional[asyncio.Task] = None
    
    def register(self, service: ServiceInfo):
        """
        Registra un servicio
        
        Args:
            service: Información del servicio
        """
        self._services[service.service_id] = service
        service.last_seen = datetime.utcnow()
        logger.info(f"Registered service: {service.service_id} at {service.endpoint}")
    
    def unregister(self, service_id: str):
        """
        Elimina un servicio del registry
        
        Args:
            service_id: ID del servicio
        """
        if service_id in self._services:
            del self._services[service_id]
            logger.info(f"Unregistered service: {service_id}")
    
    def get_service(self, service_id: str) -> Optional[ServiceInfo]:
        """
        Obtiene información de un servicio
        
        Args:
            service_id: ID del servicio
            
        Returns:
            ServiceInfo o None
        """
        return self._services.get(service_id)
    
    def list_services(
        self,
        status: Optional[MCPServiceStatus] = None,
        name: Optional[str] = None,
    ) -> List[ServiceInfo]:
        """
        Lista servicios
        
        Args:
            status: Filtrar por estado (opcional)
            name: Filtrar por nombre (opcional)
            
        Returns:
            Lista de servicios
        """
        services = list(self._services.values())
        
        if status:
            services = [s for s in services if s.status == status]
        
        if name:
            services = [s for s in services if s.name == name]
        
        return services
    
    def update_heartbeat(self, service_id: str):
        """
        Actualiza heartbeat de un servicio
        
        Args:
            service_id: ID del servicio
        """
        service = self._services.get(service_id)
        if service:
            service.last_seen = datetime.utcnow()
            service.status = MCPServiceStatus.HEALTHY
    
    async def start_cleanup(self):
        """Inicia tarea de limpieza de servicios muertos"""
        if self._running:
            return
        
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Service discovery cleanup started")
    
    async def stop_cleanup(self):
        """Detiene tarea de limpieza"""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        logger.info("Service discovery cleanup stopped")
    
    async def _cleanup_loop(self):
        """Loop de limpieza de servicios muertos"""
        while self._running:
            try:
                await asyncio.sleep(self.heartbeat_timeout)
                
                now = datetime.utcnow()
                services_to_remove = []
                
                for service_id, service in self._services.items():
                    elapsed = (now - service.last_seen).total_seconds()
                    
                    if elapsed > self.heartbeat_timeout:
                        service.status = MCPServiceStatus.UNHEALTHY
                        logger.warning(
                            f"Service {service_id} hasn't sent heartbeat in {elapsed:.0f}s"
                        )
                    
                    # Remover si está muerto por mucho tiempo
                    if elapsed > self.heartbeat_timeout * 2:
                        services_to_remove.append(service_id)
                
                for service_id in services_to_remove:
                    self.unregister(service_id)
                    logger.info(f"Removed dead service: {service_id}")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    def discover_service(self, name: str) -> Optional[ServiceInfo]:
        """
        Descubre un servicio por nombre
        
        Args:
            name: Nombre del servicio
            
        Returns:
            ServiceInfo del servicio más reciente o None
        """
        services = self.list_services(name=name, status=MCPServiceStatus.HEALTHY)
        
        if not services:
            return None
        
        # Retornar el más reciente
        return max(services, key=lambda s: s.last_seen)

