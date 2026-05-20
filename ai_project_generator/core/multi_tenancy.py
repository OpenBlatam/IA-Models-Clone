"""
Multi-Tenancy - Multi-Tenancy Support
====================================

Soporte para multi-tenancy:
- Tenant isolation
- Tenant-specific configuration
- Tenant data segregation
- Tenant management
"""

import logging
from typing import Optional, Dict, Any, List, Callable
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class TenantIsolation(str, Enum):
    """Estrategias de aislamiento"""
    DATABASE = "database"  # Base de datos separada
    SCHEMA = "schema"  # Schema separado
    ROW_LEVEL = "row_level"  # Row-level security
    APPLICATION = "application"  # Aislamiento en aplicación


class Tenant:
    """Tenant"""
    
    def __init__(
        self,
        tenant_id: str,
        name: str,
        isolation: TenantIsolation = TenantIsolation.ROW_LEVEL,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        self.tenant_id = tenant_id
        self.name = name
        self.isolation = isolation
        self.config = config or {}
        self.created_at = datetime.now()
        self.active = True
    
    def get_database_name(self) -> str:
        """Obtiene nombre de base de datos para tenant"""
        if self.isolation == TenantIsolation.DATABASE:
            return f"tenant_{self.tenant_id}"
        return "shared_database"
    
    def get_schema_name(self) -> str:
        """Obtiene nombre de schema para tenant"""
        if self.isolation == TenantIsolation.SCHEMA:
            return f"tenant_{self.tenant_id}"
        return "public"


class TenantManager:
    """
    Gestor de tenants.
    """
    
    def __init__(self) -> None:
        self.tenants: Dict[str, Tenant] = {}
        self.tenant_middleware: Optional[Callable] = None
    
    def register_tenant(
        self,
        tenant_id: str,
        name: str,
        isolation: TenantIsolation = TenantIsolation.ROW_LEVEL,
        config: Optional[Dict[str, Any]] = None
    ) -> Tenant:
        """Registra un tenant"""
        tenant = Tenant(tenant_id, name, isolation, config)
        self.tenants[tenant_id] = tenant
        logger.info(f"Tenant registered: {tenant_id}")
        return tenant
    
    def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        """Obtiene tenant"""
        return self.tenants.get(tenant_id)
    
    def get_current_tenant(self, request: Any) -> Optional[Tenant]:
        """Obtiene tenant actual del request"""
        # Extraer tenant_id del header, subdomain, o path
        tenant_id = (
            request.headers.get("X-Tenant-ID") or
            request.headers.get("Tenant-ID") or
            getattr(request.state, "tenant_id", None)
        )
        
        if tenant_id:
            return self.get_tenant(tenant_id)
        
        return None
    
    def set_tenant_context(self, tenant_id: str, context: Dict[str, Any]) -> None:
        """Establece contexto de tenant"""
        tenant = self.get_tenant(tenant_id)
        if tenant:
            tenant.config.update(context)
    
    def list_tenants(self) -> List[Dict[str, Any]]:
        """Lista todos los tenants"""
        return [
            {
                "tenant_id": tenant.tenant_id,
                "name": tenant.name,
                "isolation": tenant.isolation.value,
                "active": tenant.active,
                "created_at": tenant.created_at.isoformat()
            }
            for tenant in self.tenants.values()
        ]


from datetime import datetime


def get_tenant_manager() -> TenantManager:
    """Obtiene gestor de tenants"""
    return TenantManager()

