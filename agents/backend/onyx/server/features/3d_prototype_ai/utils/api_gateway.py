"""
API Gateway - Sistema de API gateway
====================================
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class RouteMethod(str, Enum):
    """Métodos HTTP"""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"


class APIGateway:
    """Sistema de API Gateway"""
    
    def __init__(self):
        self.routes: Dict[str, Dict[str, Any]] = {}
        self.middlewares: List[callable] = []
        self.rate_limits: Dict[str, Dict[str, int]] = {}
        self.circuit_breakers: Dict[str, Any] = {}
    
    def register_route(self, path: str, method: RouteMethod,
                      handler: callable, service_name: str,
                      rate_limit: Optional[Dict[str, int]] = None):
        """Registra una ruta"""
        route_key = f"{method.value}:{path}"
        
        self.routes[route_key] = {
            "path": path,
            "method": method.value,
            "handler": handler,
            "service_name": service_name,
            "registered_at": datetime.now().isoformat()
        }
        
        if rate_limit:
            self.rate_limits[route_key] = rate_limit
        
        logger.info(f"Ruta registrada: {route_key} -> {service_name}")
    
    def add_middleware(self, middleware: callable, priority: int = 0):
        """Agrega middleware"""
        self.middlewares.append({
            "func": middleware,
            "priority": priority
        })
        
        # Ordenar por prioridad
        self.middlewares.sort(key=lambda x: x["priority"], reverse=True)
        
        logger.info(f"Middleware agregado con prioridad {priority}")
    
    async def route_request(self, path: str, method: str,
                          headers: Dict[str, str], body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Enruta una solicitud"""
        route_key = f"{method.upper()}:{path}"
        route = self.routes.get(route_key)
        
        if not route:
            # Buscar ruta con parámetros
            route = self._find_parameterized_route(path, method)
        
        if not route:
            return {
                "status": 404,
                "error": "Route not found"
            }
        
        # Ejecutar middlewares
        context = {
            "path": path,
            "method": method,
            "headers": headers,
            "body": body,
            "route": route
        }
        
        for middleware in self.middlewares:
            try:
                if hasattr(middleware["func"], '__call__'):
                    result = await middleware["func"](context) if hasattr(middleware["func"], '__await__') else middleware["func"](context)
                    if result and result.get("stop"):
                        return result
            except Exception as e:
                logger.error(f"Error en middleware: {e}")
        
        # Ejecutar handler
        try:
            handler = route["handler"]
            if hasattr(handler, '__await__'):
                result = await handler(context)
            else:
                result = handler(context)
            
            return {
                "status": 200,
                "data": result,
                "service": route["service_name"]
            }
        except Exception as e:
            logger.error(f"Error ejecutando handler: {e}")
            return {
                "status": 500,
                "error": str(e)
            }
    
    def _find_parameterized_route(self, path: str, method: str) -> Optional[Dict[str, Any]]:
        """Busca ruta con parámetros"""
        # Implementación simplificada
        # En producción usaría regex o path matching más sofisticado
        for route_key, route in self.routes.items():
            if route["method"] == method.upper():
                # Verificar si la ruta coincide (simplificado)
                route_path = route["path"]
                if route_path.endswith("*") and path.startswith(route_path[:-1]):
                    return route
        
        return None
    
    def get_routes(self) -> List[Dict[str, Any]]:
        """Obtiene todas las rutas"""
        return [
            {
                "path": r["path"],
                "method": r["method"],
                "service": r["service_name"],
                "registered_at": r["registered_at"]
            }
            for r in self.routes.values()
        ]
    
    def get_gateway_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del gateway"""
        return {
            "total_routes": len(self.routes),
            "total_middlewares": len(self.middlewares),
            "services": list(set(r["service_name"] for r in self.routes.values()))
        }
    
    def add_rate_limit(self, route_key: str, requests: int, window_seconds: int):
        """Agrega rate limiting a una ruta"""
        self.rate_limits[route_key] = {
            "requests": requests,
            "window_seconds": window_seconds
        }
        logger.info(f"Rate limit added to {route_key}: {requests} requests per {window_seconds}s")
    
    def add_request_transformation(self, route_key: str, transform_func: callable):
        """Agrega transformación de requests"""
        if route_key in self.routes:
            self.routes[route_key]["request_transform"] = transform_func
            logger.info(f"Request transformation added to {route_key}")
    
    def add_response_transformation(self, route_key: str, transform_func: callable):
        """Agrega transformación de responses"""
        if route_key in self.routes:
            self.routes[route_key]["response_transform"] = transform_func
            logger.info(f"Response transformation added to {route_key}")
    
    def add_security_filter(self, route_key: str, filter_func: callable):
        """Agrega filtro de seguridad"""
        if route_key in self.routes:
            if "security_filters" not in self.routes[route_key]:
                self.routes[route_key]["security_filters"] = []
            self.routes[route_key]["security_filters"].append(filter_func)
            logger.info(f"Security filter added to {route_key}")

