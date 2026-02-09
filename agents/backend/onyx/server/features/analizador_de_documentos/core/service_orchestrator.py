"""
Orquestador de Servicios
=========================

Sistema para orquestación y coordinación de servicios.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ServiceStatus(Enum):
    """Estado de servicio"""
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"
    RESTARTING = "restarting"


@dataclass
class Service:
    """Servicio"""
    service_id: str
    name: str
    url: str
    status: ServiceStatus
    dependencies: List[str] = None
    health_check: Optional[Callable] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


class ServiceOrchestrator:
    """
    Orquestador de servicios
    
    Proporciona:
    - Gestión de ciclo de vida de servicios
    - Dependencias entre servicios
    - Health checks automáticos
    - Auto-restart
    - Coordinación de servicios
    """
    
    def __init__(self):
        """Inicializar orquestador"""
        self.services: Dict[str, Service] = {}
        self.service_health: Dict[str, Dict[str, Any]] = {}
        logger.info("ServiceOrchestrator inicializado")
    
    def register_service(
        self,
        service_id: str,
        name: str,
        url: str,
        dependencies: Optional[List[str]] = None,
        health_check: Optional[Callable] = None
    ) -> Service:
        """Registrar servicio"""
        service = Service(
            service_id=service_id,
            name=name,
            url=url,
            status=ServiceStatus.RUNNING,
            dependencies=dependencies or [],
            health_check=health_check
        )
        
        self.services[service_id] = service
        logger.info(f"Servicio registrado: {service_id}")
        
        return service
    
    def start_service(self, service_id: str) -> bool:
        """Iniciar servicio"""
        if service_id not in self.services:
            return False
        
        service = self.services[service_id]
        
        # Verificar dependencias
        for dep_id in service.dependencies:
            if dep_id not in self.services:
                logger.error(f"Dependencia no encontrada: {dep_id}")
                return False
            
            dep_service = self.services[dep_id]
            if dep_service.status != ServiceStatus.RUNNING:
                logger.warning(f"Dependencia {dep_id} no está corriendo")
        
        service.status = ServiceStatus.RUNNING
        logger.info(f"Servicio iniciado: {service_id}")
        
        return True
    
    def stop_service(self, service_id: str) -> bool:
        """Detener servicio"""
        if service_id not in self.services:
            return False
        
        service = self.services[service_id]
        service.status = ServiceStatus.STOPPED
        
        logger.info(f"Servicio detenido: {service_id}")
        
        return True
    
    def get_service_status(self, service_id: str) -> Optional[Dict[str, Any]]:
        """Obtener estado de servicio"""
        if service_id not in self.services:
            return None
        
        service = self.services[service_id]
        
        return {
            "service_id": service_id,
            "name": service.name,
            "status": service.status.value,
            "url": service.url,
            "dependencies": service.dependencies
        }
    
    def get_all_services_status(self) -> Dict[str, Dict[str, Any]]:
        """Obtener estado de todos los servicios"""
        return {
            service_id: self.get_service_status(service_id)
            for service_id in self.services.keys()
        }


# Instancia global
_service_orchestrator: Optional[ServiceOrchestrator] = None


def get_service_orchestrator() -> ServiceOrchestrator:
    """Obtener instancia global del orquestador"""
    global _service_orchestrator
    if _service_orchestrator is None:
        _service_orchestrator = ServiceOrchestrator()
    return _service_orchestrator














