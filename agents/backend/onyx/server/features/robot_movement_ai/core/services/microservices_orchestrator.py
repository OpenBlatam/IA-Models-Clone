"""
Microservices Orchestrator System
==================================

Sistema de orquestación de microservicios.
"""

import logging
import uuid
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ServiceStatus(Enum):
    """Estado de servicio."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


@dataclass
class Microservice:
    """Microservicio."""
    service_id: str
    name: str
    endpoint: str
    version: str = "1.0.0"
    status: ServiceStatus = ServiceStatus.UNKNOWN
    last_health_check: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ServiceCall:
    """Llamada a servicio."""
    call_id: str
    service_id: str
    method: str
    endpoint: str
    payload: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    response: Optional[Any] = None
    error: Optional[str] = None
    duration_ms: Optional[float] = None


class MicroservicesOrchestrator:
    """
    Orquestador de microservicios.
    
    Gestiona registro, descubrimiento y llamadas a microservicios.
    """
    
    def __init__(self):
        """Inicializar orquestador."""
        self.services: Dict[str, Microservice] = {}
        self.service_calls: List[ServiceCall] = []
        self.max_calls_history = 10000
        self.health_check_interval = 30.0  # segundos
    
    def register_service(
        self,
        name: str,
        endpoint: str,
        version: str = "1.0.0",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Registrar microservicio.
        
        Args:
            name: Nombre del servicio
            endpoint: Endpoint del servicio
            version: Versión del servicio
            metadata: Metadata adicional
            
        Returns:
            ID del servicio
        """
        service_id = str(uuid.uuid4())
        
        service = Microservice(
            service_id=service_id,
            name=name,
            endpoint=endpoint,
            version=version,
            metadata=metadata or {}
        )
        
        self.services[service_id] = service
        logger.info(f"Registered service: {name} ({service_id})")
        
        return service_id
    
    def deregister_service(self, service_id: str) -> bool:
        """Desregistrar microservicio."""
        if service_id in self.services:
            del self.services[service_id]
            logger.info(f"Deregistered service: {service_id}")
            return True
        return False
    
    def get_service(self, service_id: str) -> Optional[Microservice]:
        """Obtener servicio por ID."""
        return self.services.get(service_id)
    
    def find_service(self, name: str) -> List[Microservice]:
        """Buscar servicios por nombre."""
        return [s for s in self.services.values() if s.name == name]
    
    async def call_service(
        self,
        service_id: str,
        method: str,
        endpoint: str,
        payload: Dict[str, Any],
        timeout: float = 30.0
    ) -> Any:
        """
        Llamar a servicio.
        
        Args:
            service_id: ID del servicio
            method: Método HTTP (GET, POST, etc.)
            endpoint: Endpoint relativo
            payload: Datos de la petición
            timeout: Timeout en segundos
            
        Returns:
            Respuesta del servicio
        """
        import time
        import aiohttp
        
        service = self.get_service(service_id)
        if not service:
            raise ValueError(f"Service not found: {service_id}")
        
        call_id = str(uuid.uuid4())
        full_url = f"{service.endpoint.rstrip('/')}/{endpoint.lstrip('/')}"
        
        start_time = time.time()
        
        service_call = ServiceCall(
            call_id=call_id,
            service_id=service_id,
            method=method,
            endpoint=endpoint,
            payload=payload
        )
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method=method,
                    url=full_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=timeout)
                ) as response:
                    result = await response.json()
                    service_call.response = result
                    service_call.duration_ms = (time.time() - start_time) * 1000
                    
                    self.service_calls.append(service_call)
                    if len(self.service_calls) > self.max_calls_history:
                        self.service_calls = self.service_calls[-self.max_calls_history:]
                    
                    return result
        except Exception as e:
            service_call.error = str(e)
            service_call.duration_ms = (time.time() - start_time) * 1000
            self.service_calls.append(service_call)
            logger.error(f"Error calling service {service.name}: {e}", exc_info=True)
            raise
    
    async def health_check(self, service_id: str) -> bool:
        """
        Verificar salud de servicio.
        
        Args:
            service_id: ID del servicio
            
        Returns:
            True si está saludable
        """
        service = self.get_service(service_id)
        if not service:
            return False
        
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{service.endpoint}/health",
                    timeout=aiohttp.ClientTimeout(total=5.0)
                ) as response:
                    is_healthy = response.status == 200
                    service.status = ServiceStatus.HEALTHY if is_healthy else ServiceStatus.UNHEALTHY
                    service.last_health_check = datetime.now().isoformat()
                    return is_healthy
        except Exception as e:
            service.status = ServiceStatus.UNHEALTHY
            service.last_health_check = datetime.now().isoformat()
            logger.warning(f"Health check failed for {service.name}: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas del orquestador."""
        status_counts = {}
        for service in self.services.values():
            status_counts[service.status.value] = status_counts.get(service.status.value, 0) + 1
        
        successful_calls = sum(1 for call in self.service_calls if call.error is None)
        failed_calls = sum(1 for call in self.service_calls if call.error is not None)
        
        avg_duration = 0.0
        if self.service_calls:
            durations = [c.duration_ms for c in self.service_calls if c.duration_ms]
            if durations:
                avg_duration = sum(durations) / len(durations)
        
        return {
            "total_services": len(self.services),
            "status_counts": status_counts,
            "total_calls": len(self.service_calls),
            "successful_calls": successful_calls,
            "failed_calls": failed_calls,
            "average_duration_ms": avg_duration
        }


# Instancia global
_microservices_orchestrator: Optional[MicroservicesOrchestrator] = None


def get_microservices_orchestrator() -> MicroservicesOrchestrator:
    """Obtener instancia global del orquestador."""
    global _microservices_orchestrator
    if _microservices_orchestrator is None:
        _microservices_orchestrator = MicroservicesOrchestrator()
    return _microservices_orchestrator


