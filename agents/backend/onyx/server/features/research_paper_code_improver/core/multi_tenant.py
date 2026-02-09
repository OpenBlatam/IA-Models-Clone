"""
Multi-Tenant System - Sistema de multi-tenancy
===============================================
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import json
from pathlib import Path
import uuid

logger = logging.getLogger(__name__)


class MultiTenantManager:
    """
    Gestiona multi-tenancy para aislar datos por organización/usuario.
    """
    
    def __init__(self, tenants_dir: str = "data/tenants"):
        """
        Inicializar gestor de multi-tenancy.
        
        Args:
            tenants_dir: Directorio para datos de tenants
        """
        self.tenants_dir = Path(tenants_dir)
        self.tenants_dir.mkdir(parents=True, exist_ok=True)
        
        self.tenants: Dict[str, Dict[str, Any]] = {}
        self._load_tenants()
    
    def create_tenant(
        self,
        name: str,
        owner_id: str,
        plan: str = "free",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Crea un nuevo tenant.
        
        Args:
            name: Nombre del tenant
            owner_id: ID del propietario
            plan: Plan (free, pro, enterprise)
            metadata: Metadata adicional (opcional)
            
        Returns:
            Información del tenant creado
        """
        tenant_id = str(uuid.uuid4())
        
        tenant = {
            "tenant_id": tenant_id,
            "name": name,
            "owner_id": owner_id,
            "plan": plan,
            "metadata": metadata or {},
            "created_at": datetime.now().isoformat(),
            "active": True,
            "limits": self._get_plan_limits(plan)
        }
        
        self.tenants[tenant_id] = tenant
        self._save_tenant(tenant)
        
        # Crear directorio del tenant
        tenant_dir = self.tenants_dir / tenant_id
        tenant_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Tenant creado: {tenant_id} ({name})")
        
        return tenant
    
    def get_tenant(self, tenant_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene información de un tenant"""
        return self.tenants.get(tenant_id)
    
    def get_tenant_by_user(self, user_id: str) -> List[Dict[str, Any]]:
        """Obtiene tenants de un usuario"""
        return [
            tenant for tenant in self.tenants.values()
            if tenant.get("owner_id") == user_id
        ]
    
    def check_tenant_limit(
        self,
        tenant_id: str,
        resource_type: str,
        current_usage: int
    ) -> tuple[bool, Optional[str]]:
        """
        Verifica si un tenant excede sus límites.
        
        Args:
            tenant_id: ID del tenant
            resource_type: Tipo de recurso (papers, improvements, etc.)
            current_usage: Uso actual
            
        Returns:
            (allowed, reason)
        """
        tenant = self.get_tenant(tenant_id)
        if not tenant:
            return False, "Tenant no encontrado"
        
        limits = tenant.get("limits", {})
        limit = limits.get(resource_type, float("inf"))
        
        if current_usage >= limit:
            return False, f"Límite de {resource_type} excedido: {current_usage}/{limit}"
        
        return True, None
    
    def _get_plan_limits(self, plan: str) -> Dict[str, int]:
        """Obtiene límites según el plan"""
        limits_map = {
            "free": {
                "papers": 10,
                "improvements_per_month": 50,
                "models": 1,
                "storage_mb": 100
            },
            "pro": {
                "papers": 100,
                "improvements_per_month": 1000,
                "models": 5,
                "storage_mb": 1000
            },
            "enterprise": {
                "papers": float("inf"),
                "improvements_per_month": float("inf"),
                "models": float("inf"),
                "storage_mb": float("inf")
            }
        }
        
        return limits_map.get(plan, limits_map["free"])
    
    def _load_tenants(self):
        """Carga tenants desde disco"""
        for tenant_file in self.tenants_dir.glob("*.json"):
            try:
                with open(tenant_file, "r", encoding="utf-8") as f:
                    tenant = json.load(f)
                    self.tenants[tenant["tenant_id"]] = tenant
            except Exception as e:
                logger.warning(f"Error cargando tenant {tenant_file}: {e}")
    
    def _save_tenant(self, tenant: Dict[str, Any]):
        """Guarda tenant en disco"""
        try:
            tenant_file = self.tenants_dir / f"{tenant['tenant_id']}.json"
            with open(tenant_file, "w", encoding="utf-8") as f:
                json.dump(tenant, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error guardando tenant: {e}")




