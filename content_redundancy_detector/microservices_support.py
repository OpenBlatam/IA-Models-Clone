"""
Microservices Support for Distributed Architecture
Sistema de microservicios para arquitectura distribuida ultra-optimizada
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.routing import APIRouter

logger = logging.getLogger(__name__)


class ServiceStatus(Enum):
    """Estados de servicio"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    STARTING = "starting"
    STOPPING = "stopping"
    MAINTENANCE = "maintenance"


class ServiceType(Enum):
    """Tipos de servicio"""
    ANALYSIS = "analysis"
    SIMILARITY = "similarity"
    QUALITY = "quality"
    AI_ML = "ai_ml"
    CACHE = "cache"
    DATABASE = "database"
    NOTIFICATION = "notification"
    AUTHENTICATION = "authentication"


@dataclass
class ServiceInfo:
    """Información de servicio"""
    id: str
    name: str
    type: ServiceType
    version: str
    host: str
    port: int
    status: ServiceStatus
    health_endpoint: str
    created_at: float
    last_health_check: float
    metadata: Dict[str, Any]


@dataclass
class ServiceRequest:
    """Request a servicio"""
    service_id: str
    endpoint: str
    method: str
    data: Dict[str, Any]
    headers: Dict[str, str]
    timeout: float = 30.0


@dataclass
class ServiceResponse:
    """Response de servicio"""
    service_id: str
    status_code: int
    data: Dict[str, Any]
    headers: Dict[str, str]
    response_time: float
    timestamp: float


@dataclass
class LoadBalancerConfig:
    """Configuración de load balancer"""
    algorithm: str  # round_robin, least_connections, weighted
    health_check_interval: float = 30.0
    max_retries: int = 3
    timeout: float = 30.0


class ServiceRegistry:
    """Registry de servicios"""
    
    def __init__(self):
        self.services: Dict[str, ServiceInfo] = {}
        self.service_types: Dict[ServiceType, List[str]] = {}
        self._lock = asyncio.Lock()
    
    async def register_service(self, service_info: ServiceInfo):
        """Registrar servicio"""
        async with self._lock:
            self.services[service_info.id] = service_info
            
            if service_info.type not in self.service_types:
                self.service_types[service_info.type] = []
            
            if service_info.id not in self.service_types[service_info.type]:
                self.service_types[service_info.type].append(service_info.id)
            
            logger.info(f"Service registered: {service_info.id} ({service_info.name})")
    
    async def unregister_service(self, service_id: str):
        """Desregistrar servicio"""
        async with self._lock:
            if service_id in self.services:
                service = self.services[service_id]
                
                # Remover de tipos de servicio
                if service.type in self.service_types:
                    if service_id in self.service_types[service.type]:
                        self.service_types[service.type].remove(service_id)
                
                del self.services[service_id]
                logger.info(f"Service unregistered: {service_id}")
    
    async def get_service(self, service_id: str) -> Optional[ServiceInfo]:
        """Obtener servicio por ID"""
        return self.services.get(service_id)
    
    async def get_services_by_type(self, service_type: ServiceType) -> List[ServiceInfo]:
        """Obtener servicios por tipo"""
        service_ids = self.service_types.get(service_type, [])
        return [self.services[service_id] for service_id in service_ids if service_id in self.services]
    
    async def get_healthy_services(self, service_type: ServiceType) -> List[ServiceInfo]:
        """Obtener servicios saludables por tipo"""
        services = await self.get_services_by_type(service_type)
        return [service for service in services if service.status == ServiceStatus.HEALTHY]
    
    async def update_service_status(self, service_id: str, status: ServiceStatus):
        """Actualizar estado de servicio"""
        async with self._lock:
            if service_id in self.services:
                self.services[service_id].status = status
                self.services[service_id].last_health_check = time.time()
    
    async def get_all_services(self) -> List[ServiceInfo]:
        """Obtener todos los servicios"""
        return list(self.services.values())
    
    async def get_registry_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del registry"""
        total_services = len(self.services)
        healthy_services = sum(1 for service in self.services.values() if service.status == ServiceStatus.HEALTHY)
        
        service_type_counts = {}
        for service_type, service_ids in self.service_types.items():
            service_type_counts[service_type.value] = len(service_ids)
        
        return {
            "total_services": total_services,
            "healthy_services": healthy_services,
            "unhealthy_services": total_services - healthy_services,
            "service_types": service_type_counts
        }


class LoadBalancer:
    """Load balancer para servicios"""
    
    def __init__(self, config: LoadBalancerConfig):
        self.config = config
        self.service_registry = ServiceRegistry()
        self.current_index: Dict[ServiceType, int] = {}
        self.connection_counts: Dict[str, int] = {}
    
    async def get_service(self, service_type: ServiceType) -> Optional[ServiceInfo]:
        """Obtener servicio usando algoritmo de load balancing"""
        healthy_services = await self.service_registry.get_healthy_services(service_type)
        
        if not healthy_services:
            return None
        
        if self.config.algorithm == "round_robin":
            return await self._round_robin_selection(service_type, healthy_services)
        elif self.config.algorithm == "least_connections":
            return await self._least_connections_selection(healthy_services)
        elif self.config.algorithm == "weighted":
            return await self._weighted_selection(healthy_services)
        else:
            return healthy_services[0]  # Default: first available
    
    async def _round_robin_selection(self, service_type: ServiceType, services: List[ServiceInfo]) -> ServiceInfo:
        """Selección round robin"""
        if service_type not in self.current_index:
            self.current_index[service_type] = 0
        
        service = services[self.current_index[service_type]]
        self.current_index[service_type] = (self.current_index[service_type] + 1) % len(services)
        return service
    
    async def _least_connections_selection(self, services: List[ServiceInfo]) -> ServiceInfo:
        """Selección por menor número de conexiones"""
        min_connections = float('inf')
        selected_service = services[0]
        
        for service in services:
            connections = self.connection_counts.get(service.id, 0)
            if connections < min_connections:
                min_connections = connections
                selected_service = service
        
        return selected_service
    
    async def _weighted_selection(self, services: List[ServiceInfo]) -> ServiceInfo:
        """Selección ponderada"""
        # Por simplicidad, usar round robin
        return services[0]
    
    async def increment_connections(self, service_id: str):
        """Incrementar contador de conexiones"""
        self.connection_counts[service_id] = self.connection_counts.get(service_id, 0) + 1
    
    async def decrement_connections(self, service_id: str):
        """Decrementar contador de conexiones"""
        if service_id in self.connection_counts:
            self.connection_counts[service_id] = max(0, self.connection_counts[service_id] - 1)


class CircuitBreaker:
    """Circuit breaker para servicios"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_counts: Dict[str, int] = {}
        self.last_failure_times: Dict[str, float] = {}
        self.circuit_states: Dict[str, str] = {}  # closed, open, half_open
    
    async def can_execute(self, service_id: str) -> bool:
        """Verificar si se puede ejecutar request"""
        current_time = time.time()
        
        if service_id not in self.circuit_states:
            self.circuit_states[service_id] = "closed"
            return True
        
        state = self.circuit_states[service_id]
        
        if state == "closed":
            return True
        elif state == "open":
            last_failure = self.last_failure_times.get(service_id, 0)
            if current_time - last_failure > self.recovery_timeout:
                self.circuit_states[service_id] = "half_open"
                return True
            return False
        elif state == "half_open":
            return True
        
        return False
    
    async def record_success(self, service_id: str):
        """Registrar éxito"""
        self.failure_counts[service_id] = 0
        self.circuit_states[service_id] = "closed"
    
    async def record_failure(self, service_id: str):
        """Registrar fallo"""
        self.failure_counts[service_id] = self.failure_counts.get(service_id, 0) + 1
        self.last_failure_times[service_id] = time.time()
        
        if self.failure_counts[service_id] >= self.failure_threshold:
            self.circuit_states[service_id] = "open"
            logger.warning(f"Circuit breaker opened for service: {service_id}")


class ServiceClient:
    """Cliente para servicios"""
    
    def __init__(self, load_balancer: LoadBalancer, circuit_breaker: CircuitBreaker):
        self.load_balancer = load_balancer
        self.circuit_breaker = circuit_breaker
        self.http_client = httpx.AsyncClient(timeout=30.0)
    
    async def make_request(self, service_type: ServiceType, endpoint: str, 
                          method: str = "GET", data: Optional[Dict[str, Any]] = None,
                          headers: Optional[Dict[str, str]] = None) -> ServiceResponse:
        """Hacer request a servicio"""
        service = await self.load_balancer.get_service(service_type)
        
        if not service:
            raise HTTPException(status_code=503, detail=f"No healthy services available for type: {service_type.value}")
        
        if not await self.circuit_breaker.can_execute(service.id):
            raise HTTPException(status_code=503, detail=f"Circuit breaker open for service: {service.id}")
        
        try:
            await self.load_balancer.increment_connections(service.id)
            
            url = f"http://{service.host}:{service.port}{endpoint}"
            start_time = time.time()
            
            if method.upper() == "GET":
                response = await self.http_client.get(url, headers=headers)
            elif method.upper() == "POST":
                response = await self.http_client.post(url, json=data, headers=headers)
            elif method.upper() == "PUT":
                response = await self.http_client.put(url, json=data, headers=headers)
            elif method.upper() == "DELETE":
                response = await self.http_client.delete(url, headers=headers)
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported method: {method}")
            
            response_time = time.time() - start_time
            
            await self.circuit_breaker.record_success(service.id)
            
            return ServiceResponse(
                service_id=service.id,
                status_code=response.status_code,
                data=response.json() if response.content else {},
                headers=dict(response.headers),
                response_time=response_time,
                timestamp=time.time()
            )
            
        except Exception as e:
            await self.circuit_breaker.record_failure(service.id)
            logger.error(f"Error making request to service {service.id}: {e}")
            raise HTTPException(status_code=500, detail=f"Service request failed: {str(e)}")
        
        finally:
            await self.load_balancer.decrement_connections(service.id)
    
    async def close(self):
        """Cerrar cliente"""
        await self.http_client.aclose()


class HealthChecker:
    """Health checker para servicios"""
    
    def __init__(self, service_registry: ServiceRegistry):
        self.service_registry = service_registry
        self.http_client = httpx.AsyncClient(timeout=10.0)
        self.is_running = False
    
    async def start_health_checks(self, interval: float = 30.0):
        """Iniciar health checks"""
        self.is_running = True
        
        while self.is_running:
            try:
                await self._check_all_services()
                await asyncio.sleep(interval)
            except Exception as e:
                logger.error(f"Error in health check: {e}")
                await asyncio.sleep(interval)
    
    async def stop_health_checks(self):
        """Detener health checks"""
        self.is_running = False
    
    async def _check_all_services(self):
        """Verificar todos los servicios"""
        services = await self.service_registry.get_all_services()
        
        for service in services:
            try:
                await self._check_service_health(service)
            except Exception as e:
                logger.error(f"Error checking health for service {service.id}: {e}")
                await self.service_registry.update_service_status(service.id, ServiceStatus.UNHEALTHY)
    
    async def _check_service_health(self, service: ServiceInfo):
        """Verificar salud de servicio específico"""
        try:
            url = f"http://{service.host}:{service.port}{service.health_endpoint}"
            response = await self.http_client.get(url, timeout=5.0)
            
            if response.status_code == 200:
                await self.service_registry.update_service_status(service.id, ServiceStatus.HEALTHY)
            else:
                await self.service_registry.update_service_status(service.id, ServiceStatus.UNHEALTHY)
                
        except Exception as e:
            logger.warning(f"Health check failed for service {service.id}: {e}")
            await self.service_registry.update_service_status(service.id, ServiceStatus.UNHEALTHY)
    
    async def close(self):
        """Cerrar health checker"""
        await self.http_client.aclose()


class MicroservicesManager:
    """Manager de microservicios"""
    
    def __init__(self):
        self.service_registry = ServiceRegistry()
        self.load_balancer = LoadBalancer(LoadBalancerConfig(
            algorithm="round_robin",
            health_check_interval=30.0
        ))
        self.circuit_breaker = CircuitBreaker()
        self.service_client = ServiceClient(self.load_balancer, self.circuit_breaker)
        self.health_checker = HealthChecker(self.service_registry)
        self.is_running = False
    
    async def start(self):
        """Iniciar manager de microservicios"""
        try:
            self.is_running = True
            
            # Iniciar health checker
            asyncio.create_task(self.health_checker.start_health_checks())
            
            logger.info("Microservices manager started")
            
        except Exception as e:
            logger.error(f"Error starting microservices manager: {e}")
            raise
    
    async def stop(self):
        """Detener manager de microservicios"""
        try:
            self.is_running = False
            
            # Detener health checker
            await self.health_checker.stop_health_checks()
            
            # Cerrar cliente
            await self.service_client.close()
            
            # Cerrar health checker
            await self.health_checker.close()
            
            logger.info("Microservices manager stopped")
            
        except Exception as e:
            logger.error(f"Error stopping microservices manager: {e}")
    
    async def register_service(self, service_info: ServiceInfo):
        """Registrar servicio"""
        await self.service_registry.register_service(service_info)
    
    async def unregister_service(self, service_id: str):
        """Desregistrar servicio"""
        await self.service_registry.unregister_service(service_id)
    
    async def make_service_request(self, service_type: ServiceType, endpoint: str,
                                  method: str = "GET", data: Optional[Dict[str, Any]] = None,
                                  headers: Optional[Dict[str, str]] = None) -> ServiceResponse:
        """Hacer request a servicio"""
        return await self.service_client.make_request(service_type, endpoint, method, data, headers)
    
    async def get_service_info(self, service_id: str) -> Optional[ServiceInfo]:
        """Obtener información de servicio"""
        return await self.service_registry.get_service(service_id)
    
    async def get_services_by_type(self, service_type: ServiceType) -> List[ServiceInfo]:
        """Obtener servicios por tipo"""
        return await self.service_registry.get_services_by_type(service_type)
    
    async def get_healthy_services(self, service_type: ServiceType) -> List[ServiceInfo]:
        """Obtener servicios saludables por tipo"""
        return await self.service_registry.get_healthy_services(service_type)
    
    async def get_manager_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del manager"""
        registry_stats = await self.service_registry.get_registry_stats()
        
        return {
            "is_running": self.is_running,
            "registry_stats": registry_stats,
            "load_balancer_config": {
                "algorithm": self.load_balancer.config.algorithm,
                "health_check_interval": self.load_balancer.config.health_check_interval
            },
            "circuit_breaker_stats": {
                "failure_counts": self.circuit_breaker.failure_counts,
                "circuit_states": self.circuit_breaker.circuit_states
            }
        }


# Instancia global del manager de microservicios
microservices_manager = MicroservicesManager()


# Router para endpoints de microservicios
microservices_router = APIRouter()


@microservices_router.post("/services/register")
async def register_service_endpoint(service_data: dict):
    """Registrar servicio"""
    try:
        service_info = ServiceInfo(
            id=service_data["id"],
            name=service_data["name"],
            type=ServiceType(service_data["type"]),
            version=service_data["version"],
            host=service_data["host"],
            port=service_data["port"],
            status=ServiceStatus.STARTING,
            health_endpoint=service_data.get("health_endpoint", "/health"),
            created_at=time.time(),
            last_health_check=time.time(),
            metadata=service_data.get("metadata", {})
        )
        
        await microservices_manager.register_service(service_info)
        
        return {
            "message": "Service registered successfully",
            "service_id": service_info.id
        }
        
    except Exception as e:
        logger.error(f"Error registering service: {e}")
        raise HTTPException(status_code=500, detail="Failed to register service")


@microservices_router.delete("/services/{service_id}")
async def unregister_service_endpoint(service_id: str):
    """Desregistrar servicio"""
    try:
        await microservices_manager.unregister_service(service_id)
        return {"message": "Service unregistered successfully", "service_id": service_id}
    except Exception as e:
        logger.error(f"Error unregistering service: {e}")
        raise HTTPException(status_code=500, detail="Failed to unregister service")


@microservices_router.get("/services")
async def get_services_endpoint():
    """Obtener todos los servicios"""
    try:
        services = await microservices_manager.service_registry.get_all_services()
        return {
            "services": [
                {
                    "id": service.id,
                    "name": service.name,
                    "type": service.type.value,
                    "version": service.version,
                    "host": service.host,
                    "port": service.port,
                    "status": service.status.value,
                    "created_at": service.created_at,
                    "last_health_check": service.last_health_check
                }
                for service in services
            ]
        }
    except Exception as e:
        logger.error(f"Error getting services: {e}")
        raise HTTPException(status_code=500, detail="Failed to get services")


@microservices_router.get("/services/{service_id}")
async def get_service_endpoint(service_id: str):
    """Obtener servicio por ID"""
    try:
        service = await microservices_manager.get_service_info(service_id)
        if not service:
            raise HTTPException(status_code=404, detail="Service not found")
        
        return {
            "id": service.id,
            "name": service.name,
            "type": service.type.value,
            "version": service.version,
            "host": service.host,
            "port": service.port,
            "status": service.status.value,
            "health_endpoint": service.health_endpoint,
            "created_at": service.created_at,
            "last_health_check": service.last_health_check,
            "metadata": service.metadata
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting service: {e}")
        raise HTTPException(status_code=500, detail="Failed to get service")


@microservices_router.get("/services/type/{service_type}")
async def get_services_by_type_endpoint(service_type: str):
    """Obtener servicios por tipo"""
    try:
        service_type_enum = ServiceType(service_type)
        services = await microservices_manager.get_services_by_type(service_type_enum)
        
        return {
            "service_type": service_type,
            "services": [
                {
                    "id": service.id,
                    "name": service.name,
                    "version": service.version,
                    "host": service.host,
                    "port": service.port,
                    "status": service.status.value,
                    "created_at": service.created_at
                }
                for service in services
            ]
        }
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid service type")
    except Exception as e:
        logger.error(f"Error getting services by type: {e}")
        raise HTTPException(status_code=500, detail="Failed to get services by type")


@microservices_router.get("/services/healthy/{service_type}")
async def get_healthy_services_endpoint(service_type: str):
    """Obtener servicios saludables por tipo"""
    try:
        service_type_enum = ServiceType(service_type)
        services = await microservices_manager.get_healthy_services(service_type_enum)
        
        return {
            "service_type": service_type,
            "healthy_services": [
                {
                    "id": service.id,
                    "name": service.name,
                    "version": service.version,
                    "host": service.host,
                    "port": service.port,
                    "created_at": service.created_at
                }
                for service in services
            ]
        }
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid service type")
    except Exception as e:
        logger.error(f"Error getting healthy services: {e}")
        raise HTTPException(status_code=500, detail="Failed to get healthy services")


@microservices_router.get("/stats")
async def get_microservices_stats_endpoint():
    """Obtener estadísticas de microservicios"""
    try:
        stats = await microservices_manager.get_manager_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting microservices stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get microservices stats")


# Funciones de utilidad para integración
async def start_microservices_manager():
    """Iniciar manager de microservicios"""
    await microservices_manager.start()


async def stop_microservices_manager():
    """Detener manager de microservicios"""
    await microservices_manager.stop()


async def register_microservice(service_info: ServiceInfo):
    """Registrar microservicio"""
    await microservices_manager.register_service(service_info)


async def make_microservice_request(service_type: ServiceType, endpoint: str,
                                   method: str = "GET", data: Optional[Dict[str, Any]] = None,
                                   headers: Optional[Dict[str, str]] = None) -> ServiceResponse:
    """Hacer request a microservicio"""
    return await microservices_manager.make_service_request(service_type, endpoint, method, data, headers)


def get_microservices_stats() -> Dict[str, Any]:
    """Obtener estadísticas de microservicios"""
    return asyncio.run(microservices_manager.get_manager_stats())


logger.info("Microservices support module loaded successfully")

