"""
MCP Multi-Tenancy - Soporte multi-tenant
=========================================
"""

import logging
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from datetime import datetime

from .exceptions import MCPError

logger = logging.getLogger(__name__)


class Tenant(BaseModel):
    """Información de un tenant"""
    tenant_id: str = Field(..., description="ID único del tenant")
    name: str = Field(..., description="Nombre del tenant")
    enabled: bool = Field(default=True, description="Tenant habilitado")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata del tenant")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    limits: Dict[str, Any] = Field(
        default_factory=lambda: {
            "max_resources": 100,
            "max_requests_per_minute": 1000,
            "max_storage_mb": 1024,
        },
        description="Límites del tenant"
    )


class TenantContext(BaseModel):
    """Contexto de tenant para requests"""
    tenant_id: str = Field(..., description="ID del tenant")
    user_id: Optional[str] = Field(None, description="ID del usuario")
    scopes: List[str] = Field(default_factory=list, description="Scopes del usuario")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata adicional")


class TenantManager:
    """
    Gestor de multi-tenancy
    
    Permite aislar recursos y operaciones por tenant.
    """
    
    def __init__(self):
        self._tenants: Dict[str, Tenant] = {}
        self._default_tenant_id: Optional[str] = None
    
    def register_tenant(self, tenant: Tenant):
        """
        Registra un tenant
        
        Args:
            tenant: Información del tenant
        """
        self._tenants[tenant.tenant_id] = tenant
        logger.info(f"Registered tenant: {tenant.tenant_id}")
    
    def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        """
        Obtiene información de un tenant
        
        Args:
            tenant_id: ID del tenant
            
        Returns:
            Tenant o None
        """
        return self._tenants.get(tenant_id)
    
    def list_tenants(self) -> List[Tenant]:
        """
        Lista todos los tenants
        
        Returns:
            Lista de tenants
        """
        return list(self._tenants.values())
    
    def extract_tenant_from_request(self, request: Any) -> Optional[str]:
        """
        Extrae tenant_id desde request
        
        Args:
            request: Request de FastAPI
            
        Returns:
            tenant_id o None
        """
        # Intentar desde header
        tenant_id = request.headers.get("X-Tenant-ID")
        if tenant_id:
            return tenant_id
        
        # Intentar desde query params
        tenant_id = request.query_params.get("tenant_id")
        if tenant_id:
            return tenant_id
        
        # Usar default si está configurado
        return self._default_tenant_id
    
    def validate_tenant_access(
        self,
        tenant_id: str,
        resource_id: str,
    ) -> bool:
        """
        Valida acceso de tenant a recurso
        
        Args:
            tenant_id: ID del tenant
            resource_id: ID del recurso
            
        Returns:
            True si tiene acceso
        """
        tenant = self.get_tenant(tenant_id)
        if not tenant or not tenant.enabled:
            return False
        
        # Verificar límites
        # Implementar validación de límites según necesidad
        
        return True
    
    def get_tenant_context(self, tenant_id: str, user_id: Optional[str] = None) -> TenantContext:
        """
        Crea contexto de tenant
        
        Args:
            tenant_id: ID del tenant
            user_id: ID del usuario (opcional)
            
        Returns:
            TenantContext
        """
        return TenantContext(
            tenant_id=tenant_id,
            user_id=user_id,
            metadata={},
        )


def tenant_middleware(tenant_manager: TenantManager):
    """
    Middleware para extraer y validar tenant
    
    Args:
        tenant_manager: Instancia de TenantManager
    """
    from fastapi import Request
    from starlette.middleware.base import BaseHTTPMiddleware
    
    class TenantMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            tenant_id = tenant_manager.extract_tenant_from_request(request)
            
            if tenant_id:
                request.state.tenant_id = tenant_id
                request.state.tenant_context = tenant_manager.get_tenant_context(tenant_id)
            
            return await call_next(request)
    
    return TenantMiddleware

