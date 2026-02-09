"""
Resource Quota
==============

Resource quota management for tenants.
"""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Quota:
    """Resource quota definition."""
    tenant_id: str
    resource_type: str
    limit: float
    current_usage: float = 0.0
    unit: str = "count"
    
    def is_exceeded(self) -> bool:
        """Check if quota is exceeded."""
        return self.current_usage >= self.limit
    
    def get_usage_percentage(self) -> float:
        """Get usage percentage."""
        if self.limit == 0:
            return 0.0
        return (self.current_usage / self.limit) * 100


class ResourceQuota:
    """Resource quota manager."""
    
    def __init__(self):
        self._quotas: Dict[str, Dict[str, Quota]] = {}  # tenant_id -> resource_type -> Quota
    
    def set_quota(
        self,
        tenant_id: str,
        resource_type: str,
        limit: float,
        unit: str = "count"
    ):
        """Set quota for tenant."""
        if tenant_id not in self._quotas:
            self._quotas[tenant_id] = {}
        
        quota = Quota(
            tenant_id=tenant_id,
            resource_type=resource_type,
            limit=limit,
            unit=unit
        )
        
        self._quotas[tenant_id][resource_type] = quota
        logger.info(f"Set quota for {tenant_id}: {resource_type} = {limit} {unit}")
    
    def get_quota(self, tenant_id: str, resource_type: str) -> Optional[Quota]:
        """Get quota for tenant."""
        return self._quotas.get(tenant_id, {}).get(resource_type)
    
    def check_quota(self, tenant_id: str, resource_type: str, amount: float = 1.0) -> bool:
        """Check if quota allows operation."""
        quota = self.get_quota(tenant_id, resource_type)
        
        if quota is None:
            return True  # No quota set
        
        return (quota.current_usage + amount) <= quota.limit
    
    def use_quota(self, tenant_id: str, resource_type: str, amount: float = 1.0) -> bool:
        """Use quota."""
        quota = self.get_quota(tenant_id, resource_type)
        
        if quota is None:
            return True
        
        if not self.check_quota(tenant_id, resource_type, amount):
            return False
        
        quota.current_usage += amount
        return True
    
    def release_quota(self, tenant_id: str, resource_type: str, amount: float = 1.0):
        """Release quota."""
        quota = self.get_quota(tenant_id, resource_type)
        
        if quota:
            quota.current_usage = max(0, quota.current_usage - amount)
    
    def get_tenant_quotas(self, tenant_id: str) -> Dict[str, Quota]:
        """Get all quotas for tenant."""
        return self._quotas.get(tenant_id, {}).copy()
    
    def get_quota_stats(self, tenant_id: str) -> Dict[str, Any]:
        """Get quota statistics for tenant."""
        quotas = self.get_tenant_quotas(tenant_id)
        
        return {
            "tenant_id": tenant_id,
            "total_quotas": len(quotas),
            "quotas": {
                resource_type: {
                    "limit": quota.limit,
                    "current_usage": quota.current_usage,
                    "usage_percentage": quota.get_usage_percentage(),
                    "exceeded": quota.is_exceeded(),
                    "unit": quota.unit
                }
                for resource_type, quota in quotas.items()
            }
        }















