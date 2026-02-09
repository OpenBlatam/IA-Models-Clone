"""
Tenant Manager
===============

Multi-tenant management.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class Tenant:
    """Tenant definition."""
    id: str
    name: str
    domain: Optional[str] = None
    status: str = "active"  # active, suspended, deleted
    created_at: datetime = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.metadata is None:
            self.metadata = {}


class TenantManager:
    """Multi-tenant manager."""
    
    def __init__(self):
        self._tenants: Dict[str, Tenant] = {}
        self._tenant_data: Dict[str, Dict[str, Any]] = {}
    
    def create_tenant(
        self,
        tenant_id: str,
        name: str,
        domain: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tenant:
        """Create new tenant."""
        tenant = Tenant(
            id=tenant_id,
            name=name,
            domain=domain,
            metadata=metadata or {}
        )
        
        self._tenants[tenant_id] = tenant
        self._tenant_data[tenant_id] = {}
        
        logger.info(f"Created tenant: {tenant_id}")
        return tenant
    
    def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        """Get tenant by ID."""
        return self._tenants.get(tenant_id)
    
    def get_tenant_by_domain(self, domain: str) -> Optional[Tenant]:
        """Get tenant by domain."""
        for tenant in self._tenants.values():
            if tenant.domain == domain:
                return tenant
        return None
    
    def update_tenant(
        self,
        tenant_id: str,
        name: Optional[str] = None,
        domain: Optional[str] = None,
        status: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update tenant."""
        if tenant_id not in self._tenants:
            return False
        
        tenant = self._tenants[tenant_id]
        
        if name:
            tenant.name = name
        if domain:
            tenant.domain = domain
        if status:
            tenant.status = status
        if metadata:
            tenant.metadata.update(metadata)
        
        logger.info(f"Updated tenant: {tenant_id}")
        return True
    
    def delete_tenant(self, tenant_id: str) -> bool:
        """Delete tenant."""
        if tenant_id not in self._tenants:
            return False
        
        # Mark as deleted instead of removing
        self._tenants[tenant_id].status = "deleted"
        
        # Clean up tenant data
        if tenant_id in self._tenant_data:
            del self._tenant_data[tenant_id]
        
        logger.info(f"Deleted tenant: {tenant_id}")
        return True
    
    def list_tenants(self, status: Optional[str] = None) -> List[Tenant]:
        """List tenants."""
        tenants = list(self._tenants.values())
        
        if status:
            tenants = [t for t in tenants if t.status == status]
        
        return tenants
    
    def get_tenant_stats(self) -> Dict[str, Any]:
        """Get tenant statistics."""
        return {
            "total_tenants": len(self._tenants),
            "active_tenants": sum(1 for t in self._tenants.values() if t.status == "active"),
            "suspended_tenants": sum(1 for t in self._tenants.values() if t.status == "suspended"),
            "deleted_tenants": sum(1 for t in self._tenants.values() if t.status == "deleted")
        }















