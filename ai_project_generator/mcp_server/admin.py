"""
MCP Admin - Endpoints administrativos
======================================
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from .security import MCPSecurityManager, Scope
from .exceptions import MCPAuthenticationError, MCPAuthorizationError

logger = logging.getLogger(__name__)
security = HTTPBearer()


class MCPAdmin:
    """
    Endpoints administrativos para MCP
    
    Proporciona endpoints para administración y monitoreo.
    Requiere permisos de administrador.
    """
    
    def __init__(
        self,
        security_manager: MCPSecurityManager,
        require_admin: bool = True,
    ):
        """
        Args:
            security_manager: Gestor de seguridad
            require_admin: Requerir permisos de admin
        """
        self.security_manager = security_manager
        self.require_admin = require_admin
        self.router = APIRouter(prefix="/mcp/v1/admin", tags=["admin"])
        self._setup_routes()
    
    def _check_admin(self, credentials: HTTPAuthorizationCredentials) -> Dict[str, Any]:
        """Verifica permisos de administrador"""
        try:
            user = self.security_manager.verify_token(credentials.credentials)
        except ValueError as e:
            raise MCPAuthenticationError(f"Invalid token: {e}")
        
        if self.require_admin:
            user_id = user.get("sub", "unknown")
            scopes = user.get("scopes", [])
            if "admin" not in scopes and Scope.ADMIN.value not in scopes:
                raise MCPAuthorizationError("Admin access required")
        
        return user
    
    def _setup_routes(self):
        """Configura rutas administrativas"""
        
        @self.router.get("/stats")
        async def get_stats(
            credentials: HTTPAuthorizationCredentials = Depends(security)
        ) -> Dict[str, Any]:
            """Obtiene estadísticas del servidor"""
            self._check_admin(credentials)
            
            return {
                "status": "ok",
                "message": "Admin stats endpoint - implement with actual stats",
            }
        
        @self.router.get("/webhooks")
        async def list_webhooks(
            credentials: HTTPAuthorizationCredentials = Depends(security)
        ) -> List[Dict[str, Any]]:
            """Lista webhooks registrados"""
            self._check_admin(credentials)
            
            # Implementar con webhook manager
            return []
        
        @self.router.post("/webhooks")
        async def register_webhook(
            webhook_data: Dict[str, Any],
            credentials: HTTPAuthorizationCredentials = Depends(security)
        ) -> Dict[str, Any]:
            """Registra un nuevo webhook"""
            self._check_admin(credentials)
            
            # Implementar con webhook manager
            return {"status": "registered"}
        
        @self.router.delete("/webhooks/{webhook_id}")
        async def delete_webhook(
            webhook_id: str,
            credentials: HTTPAuthorizationCredentials = Depends(security)
        ) -> Dict[str, Any]:
            """Elimina un webhook"""
            self._check_admin(credentials)
            
            # Implementar con webhook manager
            return {"status": "deleted"}
        
        @self.router.get("/cache/stats")
        async def get_cache_stats(
            credentials: HTTPAuthorizationCredentials = Depends(security)
        ) -> Dict[str, Any]:
            """Obtiene estadísticas del cache"""
            self._check_admin(credentials)
            
            # Implementar con cache
            return {"status": "ok"}
        
        @self.router.post("/cache/invalidate")
        async def invalidate_cache(
            resource_id: Optional[str] = None,
            credentials: HTTPAuthorizationCredentials = Depends(security)
        ) -> Dict[str, Any]:
            """Invalida cache"""
            self._check_admin(credentials)
            
            # Implementar con cache
            return {"status": "invalidated"}
        
        @self.router.get("/rate-limits")
        async def get_rate_limit_stats(
            credentials: HTTPAuthorizationCredentials = Depends(security)
        ) -> Dict[str, Any]:
            """Obtiene estadísticas de rate limiting"""
            self._check_admin(credentials)
            
            # Implementar con rate limiter
            return {"status": "ok"}
        
        @self.router.post("/rate-limits/reset")
        async def reset_rate_limits(
            key: Optional[str] = None,
            credentials: HTTPAuthorizationCredentials = Depends(security)
        ) -> Dict[str, Any]:
            """Resetea rate limits"""
            self._check_admin(credentials)
            
            # Implementar con rate limiter
            return {"status": "reset"}
    
    def get_router(self) -> APIRouter:
        """Retorna el router administrativo"""
        return self.router

