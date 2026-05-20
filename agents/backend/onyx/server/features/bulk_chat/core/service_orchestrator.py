"""
Service Orchestrator - Orquestador de Servicios
===============================================

Sistema de orquestación de servicios con gestión de dependencias, circuitos y health checks.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


class ServiceStatus(Enum):
    """Estado de servicio."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"
    UNKNOWN = "unknown"


class ServiceDependency(Enum):
    """Tipo de dependencia."""
    REQUIRED = "required"
    OPTIONAL = "optional"
    CRITICAL = "critical"


@dataclass
class Service:
    """Servicio."""
    service_id: str
    name: str
    endpoint: str
    status: ServiceStatus = ServiceStatus.UNKNOWN
    health_check_url: Optional[str] = None
    health_check_interval: int = 30
    last_health_check: Optional[datetime] = None
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ServiceHealth:
    """Salud de servicio."""
    service_id: str
    status: ServiceStatus
    response_time: float
    error_count: int = 0
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    checked_at: datetime = field(default_factory=datetime.now)


class ServiceOrchestrator:
    """Orquestador de servicios."""
    
    def __init__(self):
        self.services: Dict[str, Service] = {}
        self.health_records: Dict[str, ServiceHealth] = {}
        self.health_checkers: Dict[str, Callable] = {}
        self._lock = asyncio.Lock()
    
    def register_service(
        self,
        service_id: str,
        name: str,
        endpoint: str,
        health_check_url: Optional[str] = None,
        dependencies: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Registrar servicio."""
        service = Service(
            service_id=service_id,
            name=name,
            endpoint=endpoint,
            health_check_url=health_check_url,
            dependencies=dependencies or [],
            metadata=metadata or {},
        )
        
        async def save_service():
            async with self._lock:
                self.services[service_id] = service
                
                # Inicializar health record
                self.health_records[service_id] = ServiceHealth(
                    service_id=service_id,
                    status=ServiceStatus.UNKNOWN,
                    response_time=0.0,
                )
        
        asyncio.create_task(save_service())
        
        # Iniciar health checks
        if health_check_url:
            asyncio.create_task(self._start_health_checks(service_id))
        
        logger.info(f"Registered service: {service_id} - {name}")
        return service_id
    
    def register_health_checker(
        self,
        service_id: str,
        checker: Callable,
    ):
        """Registrar health checker personalizado."""
        self.health_checkers[service_id] = checker
    
    async def _start_health_checks(self, service_id: str):
        """Iniciar health checks."""
        service = self.services.get(service_id)
        if not service:
            return
        
        while service_id in self.services:
            try:
                await self._perform_health_check(service_id)
            except Exception as e:
                logger.error(f"Error in health check for {service_id}: {e}")
            
            await asyncio.sleep(service.health_check_interval)
    
    async def _perform_health_check(self, service_id: str):
        """Realizar health check."""
        service = self.services.get(service_id)
        if not service:
            return
        
        start_time = datetime.now()
        status = ServiceStatus.UNHEALTHY
        error = None
        
        try:
            # Verificar dependencias
            for dep_id in service.dependencies:
                dep_service = self.services.get(dep_id)
                if not dep_service or dep_service.status != ServiceStatus.HEALTHY:
                    status = ServiceStatus.DEGRADED
                    break
            
            # Health check personalizado o por URL
            if service_id in self.health_checkers:
                checker = self.health_checkers[service_id]
                if asyncio.iscoroutinefunction(checker):
                    result = await checker()
                else:
                    result = checker()
                
                if result:
                    status = ServiceStatus.HEALTHY
            elif service.health_check_url:
                # Health check HTTP simple (en producción usar httpx)
                import urllib.request
                try:
                    urllib.request.urlopen(service.health_check_url, timeout=5)
                    status = ServiceStatus.HEALTHY
                except Exception as e:
                    error = str(e)
                    status = ServiceStatus.UNHEALTHY
            else:
                status = ServiceStatus.UNKNOWN
            
        except Exception as e:
            error = str(e)
            status = ServiceStatus.UNHEALTHY
        
        response_time = (datetime.now() - start_time).total_seconds()
        
        async with self._lock:
            service.status = status
            service.last_health_check = datetime.now()
            
            health = self.health_records.get(service_id)
            if health:
                health.status = status
                health.response_time = response_time
                health.checked_at = datetime.now()
                
                if status == ServiceStatus.HEALTHY:
                    health.last_success = datetime.now()
                    health.error_count = 0
                else:
                    health.error_count += 1
                    health.last_failure = datetime.now()
    
    def get_service(self, service_id: str) -> Optional[Dict[str, Any]]:
        """Obtener servicio."""
        service = self.services.get(service_id)
        if not service:
            return None
        
        health = self.health_records.get(service_id)
        
        return {
            "service_id": service.service_id,
            "name": service.name,
            "endpoint": service.endpoint,
            "status": service.status.value,
            "health_check_url": service.health_check_url,
            "dependencies": service.dependencies,
            "last_health_check": service.last_health_check.isoformat() if service.last_health_check else None,
            "health": {
                "response_time": health.response_time if health else 0.0,
                "error_count": health.error_count if health else 0,
                "last_success": health.last_success.isoformat() if health and health.last_success else None,
                "last_failure": health.last_failure.isoformat() if health and health.last_failure else None,
            },
        }
    
    def get_healthy_services(self) -> List[Dict[str, Any]]:
        """Obtener servicios saludables."""
        healthy = [
            self.get_service(sid)
            for sid, service in self.services.items()
            if service.status == ServiceStatus.HEALTHY
        ]
        
        return [s for s in healthy if s]
    
    def get_service_orchestrator_summary(self) -> Dict[str, Any]:
        """Obtener resumen del orquestador."""
        by_status: Dict[str, int] = defaultdict(int)
        
        for service in self.services.values():
            by_status[service.status.value] += 1
        
        return {
            "total_services": len(self.services),
            "services_by_status": dict(by_status),
            "healthy_services": len([s for s in self.services.values() if s.status == ServiceStatus.HEALTHY]),
        }



