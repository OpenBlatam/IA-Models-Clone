"""
API Gateway System
==================

Sistema de gateway para gestión de APIs externas.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class APIStatus(Enum):
    """Estado de API."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"


@dataclass
class APIEndpoint:
    """Endpoint de API."""
    endpoint_id: str
    name: str
    base_url: str
    path: str
    method: str  # GET, POST, PUT, DELETE
    description: str
    headers: Dict[str, str] = field(default_factory=dict)
    auth_type: Optional[str] = None  # None, "api_key", "bearer", "basic"
    auth_config: Dict[str, Any] = field(default_factory=dict)
    status: APIStatus = APIStatus.ACTIVE
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class APIRequest:
    """Request a API."""
    request_id: str
    endpoint_id: str
    method: str
    url: str
    headers: Dict[str, str]
    params: Optional[Dict[str, Any]] = None
    data: Optional[Dict[str, Any]] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class APIGateway:
    """
    Gateway de APIs.
    
    Gestiona endpoints de APIs externas.
    """
    
    def __init__(self):
        """Inicializar gateway de APIs."""
        self.endpoints: Dict[str, APIEndpoint] = {}
        self.requests: List[APIRequest] = []
        self.max_requests = 10000
    
    def register_endpoint(
        self,
        endpoint_id: str,
        name: str,
        base_url: str,
        path: str,
        method: str,
        description: str,
        headers: Optional[Dict[str, str]] = None,
        auth_type: Optional[str] = None,
        auth_config: Optional[Dict[str, Any]] = None
    ) -> APIEndpoint:
        """
        Registrar endpoint de API.
        
        Args:
            endpoint_id: ID único del endpoint
            name: Nombre
            base_url: URL base
            path: Path del endpoint
            method: Método HTTP
            description: Descripción
            headers: Headers adicionales
            auth_type: Tipo de autenticación
            auth_config: Configuración de autenticación
            
        Returns:
            Endpoint registrado
        """
        endpoint = APIEndpoint(
            endpoint_id=endpoint_id,
            name=name,
            base_url=base_url,
            path=path,
            method=method.upper(),
            description=description,
            headers=headers or {},
            auth_type=auth_type,
            auth_config=auth_config or {}
        )
        
        self.endpoints[endpoint_id] = endpoint
        logger.info(f"Registered API endpoint: {name} ({endpoint_id})")
        
        return endpoint
    
    def get_endpoint(self, endpoint_id: str) -> Optional[APIEndpoint]:
        """Obtener endpoint por ID."""
        return self.endpoints.get(endpoint_id)
    
    def list_endpoints(
        self,
        status: Optional[APIStatus] = None
    ) -> List[APIEndpoint]:
        """
        Listar endpoints.
        
        Args:
            status: Filtrar por estado
            
        Returns:
            Lista de endpoints
        """
        endpoints = list(self.endpoints.values())
        
        if status:
            endpoints = [e for e in endpoints if e.status == status]
        
        return endpoints
    
    def build_url(
        self,
        endpoint_id: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Construir URL del endpoint.
        
        Args:
            endpoint_id: ID del endpoint
            params: Parámetros de query
            
        Returns:
            URL construida o None
        """
        endpoint = self.get_endpoint(endpoint_id)
        if not endpoint:
            return None
        
        url = f"{endpoint.base_url.rstrip('/')}/{endpoint.path.lstrip('/')}"
        
        if params:
            from urllib.parse import urlencode
            url += f"?{urlencode(params)}"
        
        return url
    
    def build_headers(
        self,
        endpoint_id: str,
        additional_headers: Optional[Dict[str, str]] = None
    ) -> Optional[Dict[str, str]]:
        """
        Construir headers del endpoint.
        
        Args:
            endpoint_id: ID del endpoint
            additional_headers: Headers adicionales
            
        Returns:
            Headers construidos o None
        """
        endpoint = self.get_endpoint(endpoint_id)
        if not endpoint:
            return None
        
        headers = endpoint.headers.copy()
        
        # Agregar autenticación
        if endpoint.auth_type == "api_key":
            api_key = endpoint.auth_config.get("api_key")
            key_name = endpoint.auth_config.get("key_name", "X-API-Key")
            if api_key:
                headers[key_name] = api_key
        elif endpoint.auth_type == "bearer":
            token = endpoint.auth_config.get("token")
            if token:
                headers["Authorization"] = f"Bearer {token}"
        elif endpoint.auth_type == "basic":
            username = endpoint.auth_config.get("username")
            password = endpoint.auth_config.get("password")
            if username and password:
                import base64
                credentials = base64.b64encode(
                    f"{username}:{password}".encode()
                ).decode()
                headers["Authorization"] = f"Basic {credentials}"
        
        if additional_headers:
            headers.update(additional_headers)
        
        return headers
    
    def record_request(
        self,
        endpoint_id: str,
        method: str,
        url: str,
        headers: Dict[str, str],
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None
    ) -> APIRequest:
        """
        Registrar request.
        
        Args:
            endpoint_id: ID del endpoint
            method: Método HTTP
            url: URL
            headers: Headers
            params: Parámetros
            data: Datos
            
        Returns:
            Request registrado
        """
        request_id = f"req_{len(self.requests)}"
        request = APIRequest(
            request_id=request_id,
            endpoint_id=endpoint_id,
            method=method,
            url=url,
            headers=headers,
            params=params,
            data=data
        )
        
        self.requests.append(request)
        if len(self.requests) > self.max_requests:
            self.requests = self.requests[-self.max_requests:]
        
        return request
    
    def get_requests(
        self,
        endpoint_id: Optional[str] = None,
        limit: int = 100
    ) -> List[APIRequest]:
        """
        Obtener requests.
        
        Args:
            endpoint_id: Filtrar por endpoint
            limit: Límite de resultados
            
        Returns:
            Lista de requests
        """
        requests = self.requests
        
        if endpoint_id:
            requests = [r for r in requests if r.endpoint_id == endpoint_id]
        
        requests.sort(key=lambda x: x.timestamp, reverse=True)
        return requests[:limit]


# Instancia global
_api_gateway: Optional[APIGateway] = None


def get_api_gateway() -> APIGateway:
    """Obtener instancia global del gateway de APIs."""
    global _api_gateway
    if _api_gateway is None:
        _api_gateway = APIGateway()
    return _api_gateway






