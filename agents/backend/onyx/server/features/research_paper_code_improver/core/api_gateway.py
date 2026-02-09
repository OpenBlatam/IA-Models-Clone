"""
API Gateway - Gateway para routing avanzado y gestión de APIs
==============================================================
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import time

logger = logging.getLogger(__name__)


class RouteMethod(Enum):
    """Métodos HTTP"""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    OPTIONS = "OPTIONS"
    HEAD = "HEAD"


@dataclass
class Route:
    """Ruta del gateway"""
    path: str
    method: RouteMethod
    handler: Callable
    service_name: Optional[str] = None
    service_url: Optional[str] = None
    timeout: float = 30.0
    retry_count: int = 3
    rate_limit: Optional[int] = None  # requests per minute
    authentication_required: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RouteRule:
    """Regla de routing"""
    pattern: str
    target_service: str
    target_url: str
    priority: int = 0
    conditions: Dict[str, Any] = field(default_factory=dict)


class APIGateway:
    """API Gateway para routing y gestión"""
    
    def __init__(self):
        self.routes: List[Route] = []
        self.rules: List[RouteRule] = []
        self.middleware: List[Callable] = []
        self.service_registry: Dict[str, Dict[str, Any]] = {}
        self.request_stats: Dict[str, Dict[str, Any]] = {}
    
    def register_route(
        self,
        path: str,
        method: RouteMethod,
        handler: Callable,
        **kwargs
    ) -> Route:
        """Registra una ruta"""
        route = Route(
            path=path,
            method=method,
            handler=handler,
            service_name=kwargs.get("service_name"),
            service_url=kwargs.get("service_url"),
            timeout=kwargs.get("timeout", 30.0),
            retry_count=kwargs.get("retry_count", 3),
            rate_limit=kwargs.get("rate_limit"),
            authentication_required=kwargs.get("authentication_required", True),
            metadata=kwargs.get("metadata", {})
        )
        self.routes.append(route)
        logger.info(f"Ruta registrada: {method.value} {path}")
        return route
    
    def add_rule(self, rule: RouteRule):
        """Agrega una regla de routing"""
        self.rules.append(rule)
        self.rules.sort(key=lambda r: r.priority, reverse=True)
    
    def register_service(
        self,
        service_name: str,
        base_url: str,
        health_check_url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Registra un servicio"""
        self.service_registry[service_name] = {
            "base_url": base_url,
            "health_check_url": health_check_url,
            "metadata": metadata or {},
            "registered_at": datetime.now().isoformat(),
            "status": "healthy"
        }
    
    def add_middleware(self, middleware: Callable):
        """Agrega middleware"""
        self.middleware.append(middleware)
    
    async def route_request(
        self,
        method: str,
        path: str,
        headers: Dict[str, str],
        body: Optional[Any] = None,
        query_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Rutea una petición"""
        start_time = time.time()
        
        try:
            # Aplicar middleware
            for mw in self.middleware:
                result = await mw(method, path, headers, body, query_params)
                if result:  # Si middleware retorna algo, usarlo
                    return result
            
            # Buscar ruta
            route = self._find_route(method, path)
            if not route:
                return {
                    "status_code": 404,
                    "body": {"error": "Route not found"}
                }
            
            # Verificar rate limit
            if route.rate_limit:
                if not self._check_rate_limit(path, route.rate_limit):
                    return {
                        "status_code": 429,
                        "body": {"error": "Rate limit exceeded"}
                    }
            
            # Ejecutar handler o proxy a servicio
            if route.service_url:
                result = await self._proxy_to_service(route, method, path, headers, body, query_params)
            else:
                result = await self._execute_handler(route, method, path, headers, body, query_params)
            
            # Registrar estadísticas
            duration = time.time() - start_time
            self._record_stats(path, method, duration, result.get("status_code", 200))
            
            return result
        except Exception as e:
            logger.error(f"Error routing request: {e}")
            return {
                "status_code": 500,
                "body": {"error": str(e)}
            }
    
    def _find_route(self, method: str, path: str) -> Optional[Route]:
        """Encuentra una ruta"""
        method_enum = RouteMethod(method.upper())
        
        for route in self.routes:
            if route.method == method_enum and self._path_matches(route.path, path):
                return route
        
        return None
    
    def _path_matches(self, pattern: str, path: str) -> bool:
        """Verifica si un path coincide con un patrón"""
        # Soporte básico para wildcards
        if pattern == path:
            return True
        
        # Convertir patrón a regex simple
        pattern_regex = pattern.replace("*", ".*")
        import re
        return bool(re.match(pattern_regex, path))
    
    async def _execute_handler(
        self,
        route: Route,
        method: str,
        path: str,
        headers: Dict[str, str],
        body: Optional[Any],
        query_params: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Ejecuta un handler"""
        try:
            if asyncio.iscoroutinefunction(route.handler):
                result = await asyncio.wait_for(
                    route.handler(method, path, headers, body, query_params),
                    timeout=route.timeout
                )
            else:
                result = route.handler(method, path, headers, body, query_params)
            
            return {
                "status_code": 200,
                "body": result
            }
        except asyncio.TimeoutError:
            return {
                "status_code": 504,
                "body": {"error": "Request timeout"}
            }
        except Exception as e:
            logger.error(f"Error ejecutando handler: {e}")
            return {
                "status_code": 500,
                "body": {"error": str(e)}
            }
    
    async def _proxy_to_service(
        self,
        route: Route,
        method: str,
        path: str,
        headers: Dict[str, str],
        body: Optional[Any],
        query_params: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Proxy a un servicio externo"""
        import httpx
        
        url = f"{route.service_url}{path}"
        
        try:
            async with httpx.AsyncClient(timeout=route.timeout) as client:
                response = await client.request(
                    method=method,
                    url=url,
                    headers=headers,
                    json=body,
                    params=query_params
                )
                
                return {
                    "status_code": response.status_code,
                    "body": response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text,
                    "headers": dict(response.headers)
                }
        except Exception as e:
            logger.error(f"Error proxying to service: {e}")
            return {
                "status_code": 502,
                "body": {"error": f"Service unavailable: {str(e)}"}
            }
    
    def _check_rate_limit(self, path: str, limit: int) -> bool:
        """Verifica rate limit"""
        now = time.time()
        minute_ago = now - 60
        
        if path not in self.request_stats:
            self.request_stats[path] = {"requests": []}
        
        stats = self.request_stats[path]
        stats["requests"] = [t for t in stats["requests"] if t > minute_ago]
        
        if len(stats["requests"]) >= limit:
            return False
        
        stats["requests"].append(now)
        return True
    
    def _record_stats(self, path: str, method: str, duration: float, status_code: int):
        """Registra estadísticas"""
        if path not in self.request_stats:
            self.request_stats[path] = {
                "requests": [],
                "total_requests": 0,
                "total_duration": 0,
                "status_codes": {}
            }
        
        stats = self.request_stats[path]
        stats["total_requests"] += 1
        stats["total_duration"] += duration
        stats["status_codes"][status_code] = stats["status_codes"].get(status_code, 0) + 1
    
    def get_stats(self, path: Optional[str] = None) -> Dict[str, Any]:
        """Obtiene estadísticas"""
        if path:
            return self.request_stats.get(path, {})
        return self.request_stats
    
    def get_service_health(self, service_name: str) -> Dict[str, Any]:
        """Obtiene el estado de salud de un servicio"""
        if service_name not in self.service_registry:
            return {"status": "not_found"}
        
        service = self.service_registry[service_name]
        return {
            "service_name": service_name,
            "status": service["status"],
            "base_url": service["base_url"],
            "registered_at": service["registered_at"]
        }




