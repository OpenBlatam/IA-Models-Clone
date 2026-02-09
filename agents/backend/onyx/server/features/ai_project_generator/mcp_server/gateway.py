"""
MCP API Gateway - Features de API Gateway
===========================================
"""

import logging
from typing import Dict, Any, List, Optional, Callable
from fastapi import APIRouter, Request, Response
from fastapi.routing import APIRoute
from datetime import datetime

logger = logging.getLogger(__name__)


class GatewayRoute(BaseModel):
    """Ruta del gateway"""
    path: str
    method: str
    target_url: str
    timeout: int = 30
    retries: int = 3
    rate_limit: Optional[int] = None
    authentication_required: bool = True


class APIGateway:
    """
    API Gateway
    
    Proporciona features de gateway como routing, rate limiting, auth, etc.
    """
    
    def __init__(self):
        self._routes: List[GatewayRoute] = []
        self._middleware: List[Callable] = []
        self.router = APIRouter()
    
    def add_route(self, route: GatewayRoute):
        """
        Agrega ruta al gateway
        
        Args:
            route: Ruta del gateway
        """
        self._routes.append(route)
        logger.info(f"Added gateway route: {route.method} {route.path} -> {route.target_url}")
    
    def add_middleware(self, middleware: Callable):
        """
        Agrega middleware al gateway
        
        Args:
            middleware: Función middleware
        """
        self._middleware.append(middleware)
        logger.info(f"Added gateway middleware: {middleware.__name__}")
    
    async def proxy_request(
        self,
        request: Request,
        route: GatewayRoute,
    ) -> Response:
        """
        Proxea request a backend
        
        Args:
            request: Request original
            route: Ruta del gateway
            
        Returns:
            Response del backend
        """
        import httpx
        
        # Aplicar middleware
        for middleware in self._middleware:
            request = await middleware(request)
        
        # Construir URL target
        target_url = f"{route.target_url}{request.url.path}"
        
        # Proxear request
        async with httpx.AsyncClient(timeout=route.timeout) as client:
            try:
                response = await client.request(
                    method=route.method,
                    url=target_url,
                    headers=dict(request.headers),
                    params=dict(request.query_params),
                    content=await request.body(),
                )
                
                return Response(
                    content=response.content,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                )
                
            except Exception as e:
                logger.error(f"Error proxying request to {target_url}: {e}")
                return Response(
                    content=f"Gateway error: {str(e)}",
                    status_code=502,
                )
    
    def get_router(self) -> APIRouter:
        """Retorna router del gateway"""
        return self.router
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del gateway"""
        return {
            "routes": len(self._routes),
            "middleware": len(self._middleware),
            "routes_detail": [r.dict() for r in self._routes],
        }

