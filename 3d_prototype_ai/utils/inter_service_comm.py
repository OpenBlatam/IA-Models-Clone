"""
Inter-Service Communication - Patrones de comunicación entre servicios
======================================================================

Soporta:
- HTTP/REST
- gRPC
- Message queues
- Event-driven
"""

import logging
import aiohttp
from typing import Dict, Optional, List, Any, Callable
from enum import Enum

logger = logging.getLogger(__name__)


class CommunicationPattern(Enum):
    """Patrón de comunicación"""
    REST = "rest"
    GRPC = "grpc"
    MESSAGE_QUEUE = "message_queue"
    EVENT_DRIVEN = "event_driven"


class RESTClient:
    """Cliente REST para comunicación entre servicios"""
    
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Obtiene o crea sesión HTTP"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(timeout=self.timeout)
        return self.session
    
    async def get(self, path: str, headers: Optional[Dict] = None) -> Dict:
        """GET request"""
        session = await self._get_session()
        url = f"{self.base_url}{path}"
        
        try:
            async with session.get(url, headers=headers) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error(f"REST GET error: {e}")
            raise
    
    async def post(self, path: str, data: Dict, headers: Optional[Dict] = None) -> Dict:
        """POST request"""
        session = await self._get_session()
        url = f"{self.base_url}{path}"
        
        try:
            async with session.post(url, json=data, headers=headers) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error(f"REST POST error: {e}")
            raise
    
    async def close(self):
        """Cierra la sesión"""
        if self.session and not self.session.closed:
            await self.session.close()


class ServiceClient:
    """Cliente para comunicación con otros servicios"""
    
    def __init__(self, service_name: str, 
                 discovery: Optional[Any] = None,
                 pattern: CommunicationPattern = CommunicationPattern.REST):
        self.service_name = service_name
        self.discovery = discovery
        self.pattern = pattern
        self.clients: List[RESTClient] = []
        self._setup_clients()
    
    def _setup_clients(self):
        """Configura clientes basado en service discovery"""
        if self.discovery:
            instances = self.discovery.discover_service(self.service_name)
            for instance in instances:
                url = f"http://{instance['address']}:{instance.get('port', 8030)}"
                self.clients.append(RESTClient(url))
        else:
            # Fallback a URL directa
            url = f"http://{self.service_name}:8030"
            self.clients.append(RESTClient(url))
    
    async def call(self, method: str, path: str, 
                  data: Optional[Dict] = None,
                  headers: Optional[Dict] = None) -> Dict:
        """Llama a un servicio"""
        if not self.clients:
            raise Exception(f"No clients available for {self.service_name}")
        
        # Round-robin entre clientes
        client = self.clients[0]  # Simplificado, en producción usaría load balancing
        
        if method.upper() == "GET":
            return await client.get(path, headers=headers)
        elif method.upper() == "POST":
            return await client.post(path, data or {}, headers=headers)
        else:
            raise ValueError(f"Unsupported method: {method}")
    
    async def close(self):
        """Cierra todos los clientes"""
        for client in self.clients:
            await client.close()


class ServiceRegistry:
    """Registro de servicios disponibles"""
    
    def __init__(self):
        self.services: Dict[str, ServiceClient] = {}
    
    def register_service(self, name: str, client: ServiceClient):
        """Registra un servicio"""
        self.services[name] = client
        logger.info(f"Service {name} registered")
    
    def get_service(self, name: str) -> Optional[ServiceClient]:
        """Obtiene un servicio"""
        return self.services.get(name)
    
    async def call_service(self, service_name: str, method: str, path: str,
                          data: Optional[Dict] = None) -> Dict:
        """Llama a un servicio registrado"""
        service = self.get_service(service_name)
        if not service:
            raise Exception(f"Service {service_name} not found")
        
        return await service.call(method, path, data)
    
    async def close_all(self):
        """Cierra todos los servicios"""
        for service in self.services.values():
            await service.close()


# Instancia global
service_registry = ServiceRegistry()




