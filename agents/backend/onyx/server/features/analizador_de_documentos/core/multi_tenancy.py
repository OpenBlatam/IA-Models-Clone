"""
Sistema de Multi-Tenancy
========================

Sistema para soportar múltiples tenants/usuarios con aislamiento de datos.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class Tenant:
    """Tenant/Usuario"""
    tenant_id: str
    name: str
    config: Dict[str, Any]
    created_at: str = None
    active: bool = True
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()


class MultiTenancyManager:
    """
    Gestor de multi-tenancy
    
    Proporciona:
    - Aislamiento de datos por tenant
    - Configuración por tenant
    - Límites y quotas
    - Estadísticas por tenant
    """
    
    def __init__(self):
        """Inicializar gestor"""
        self.tenants: Dict[str, Tenant] = {}
        self.tenant_data: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.tenant_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "documents_processed": 0,
            "analyses_performed": 0,
            "storage_used": 0,
            "api_calls": 0
        })
        logger.info("MultiTenancyManager inicializado")
    
    def register_tenant(
        self,
        tenant_id: str,
        name: str,
        config: Optional[Dict[str, Any]] = None
    ) -> Tenant:
        """
        Registrar nuevo tenant
        
        Args:
            tenant_id: ID único del tenant
            name: Nombre del tenant
            config: Configuración específica
        
        Returns:
            Tenant creado
        """
        tenant = Tenant(
            tenant_id=tenant_id,
            name=name,
            config=config or {}
        )
        
        self.tenants[tenant_id] = tenant
        logger.info(f"Tenant registrado: {tenant_id}")
        
        return tenant
    
    def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        """Obtener tenant"""
        return self.tenants.get(tenant_id)
    
    def get_tenant_config(self, tenant_id: str) -> Dict[str, Any]:
        """Obtener configuración del tenant"""
        tenant = self.get_tenant(tenant_id)
        if tenant:
            return tenant.config
        return {}
    
    def store_tenant_data(
        self,
        tenant_id: str,
        key: str,
        value: Any
    ):
        """Almacenar datos del tenant"""
        self.tenant_data[tenant_id][key] = value
    
    def get_tenant_data(
        self,
        tenant_id: str,
        key: str
    ) -> Any:
        """Obtener datos del tenant"""
        return self.tenant_data[tenant_id].get(key)
    
    def increment_stat(
        self,
        tenant_id: str,
        stat_name: str,
        value: int = 1
    ):
        """Incrementar estadística del tenant"""
        if stat_name in self.tenant_stats[tenant_id]:
            self.tenant_stats[tenant_id][stat_name] += value
    
    def get_tenant_stats(self, tenant_id: str) -> Dict[str, Any]:
        """Obtener estadísticas del tenant"""
        return self.tenant_stats[tenant_id].copy()
    
    def check_quota(
        self,
        tenant_id: str,
        resource: str,
        requested_amount: int = 1
    ) -> bool:
        """
        Verificar si tenant tiene quota disponible
        
        Args:
            tenant_id: ID del tenant
            resource: Recurso a verificar (documents, api_calls, storage)
            requested_amount: Cantidad solicitada
        
        Returns:
            True si hay quota disponible
        """
        tenant = self.get_tenant(tenant_id)
        if not tenant:
            return False
        
        config = tenant.config
        quota = config.get(f"{resource}_quota", float('inf'))
        current = self.tenant_stats[tenant_id].get(f"{resource}_used", 0)
        
        return current + requested_amount <= quota
    
    def list_tenants(self) -> List[Dict[str, Any]]:
        """Listar todos los tenants"""
        return [
            {
                "tenant_id": t.tenant_id,
                "name": t.name,
                "active": t.active,
                "stats": self.tenant_stats[t.tenant_id]
            }
            for t in self.tenants.values()
        ]


# Instancia global
_multi_tenancy_manager: Optional[MultiTenancyManager] = None


def get_multi_tenancy_manager() -> MultiTenancyManager:
    """Obtener instancia global del gestor"""
    global _multi_tenancy_manager
    if _multi_tenancy_manager is None:
        _multi_tenancy_manager = MultiTenancyManager()
    return _multi_tenancy_manager
















