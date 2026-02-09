"""
API Gateway Service - Gateway de API
=====================================

Sistema de API Gateway para enrutamiento y gestión de requests.
"""

import logging
import httpx
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class RouteMethod(str, Enum):
    """Métodos HTTP"""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"


@dataclass
class Route:
    """Ruta del gateway"""
    path: str
    method: RouteMethod
    target_url: str
    timeout: int = 30
    retries: int = 3
    rate_limit: Optional[int] = None
    authentication_required: bool = True
    headers: Dict[str, str] = field(default_factory=dict)


@dataclass
class GatewayRequest:
    """Request del gateway"""
    route: Route
    path_params: Dict[str, str]
    query_params: Dict[str, Any]
    headers: Dict[str, str]
    body: Optional[Any] = None
    user_id: Optional[str] = None


class APIGatewayService:
    """Servicio de API Gateway"""
    
    def __init__(self):
        """Inicializar servicio"""
        self.routes: Dict[str, Route] = {}  # path:method -> route
        logger.info("APIGatewayService initialized")
    
    def register_route(
        self,
        path: str,
        method: RouteMethod,
        target_url: str,
        timeout: int = 30,
        retries: int = 3,
        rate_limit: Optional[int] = None,
        authentication_required: bool = True
    ) -> Route:
        """Registrar ruta"""
        route_key = f"{path}:{method.value}"
        
        route = Route(
            path=path,
            method=method,
            target_url=target_url,
            timeout=timeout,
            retries=retries,
            rate_limit=rate_limit,
            authentication_required=authentication_required,
        )
        
        self.routes[route_key] = route
        
        logger.info(f"Route registered: {route_key} -> {target_url}")
        return route
    
    async def forward_request(
        self,
        path: str,
        method: str,
        path_params: Dict[str, str],
        query_params: Dict[str, Any],
        headers: Dict[str, str],
        body: Optional[Any] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Reenviar request a servicio destino"""
        route_key = f"{path}:{method.upper()}"
        route = self.routes.get(route_key)
        
        if not route:
            raise ValueError(f"Route not found: {route_key}")
        
        # Construir URL destino
        target_url = route.target_url
        for param, value in path_params.items():
            target_url = target_url.replace(f"{{{param}}}", str(value))
        
        # Agregar query params
        if query_params:
            query_string = "&".join(f"{k}={v}" for k, v in query_params.items())
            target_url = f"{target_url}?{query_string}"
        
        # Preparar headers
        forward_headers = {**route.headers, **headers}
        if user_id:
            forward_headers["X-User-ID"] = user_id
        
        # Enviar request
        try:
            async with httpx.AsyncClient(timeout=route.timeout) as client:
                if method.upper() == "GET":
                    response = await client.get(target_url, headers=forward_headers)
                elif method.upper() == "POST":
                    response = await client.post(target_url, json=body, headers=forward_headers)
                elif method.upper() == "PUT":
                    response = await client.put(target_url, json=body, headers=forward_headers)
                elif method.upper() == "DELETE":
                    response = await client.delete(target_url, headers=forward_headers)
                else:
                    response = await client.request(method.upper(), target_url, json=body, headers=forward_headers)
                
                return {
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                    "body": response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text,
                }
        
        except Exception as e:
            logger.error(f"Gateway request failed: {e}")
            raise
    
    def get_routes(self) -> List[Dict[str, Any]]:
        """Obtener todas las rutas"""
        return [
            {
                "path": r.path,
                "method": r.method.value,
                "target_url": r.target_url,
                "timeout": r.timeout,
                "retries": r.retries,
            }
            for r in self.routes.values()
        ]




