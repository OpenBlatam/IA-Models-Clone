"""
Cache multi-tenancy support.

Provides multi-tenancy capabilities for cache.
"""
from __future__ import annotations

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Tenant:
    """Tenant information."""
    id: str
    name: str
    quota: Dict[str, Any]
    metadata: Dict[str, Any]


class CacheMultiTenancy:
    """
    Cache multi-tenancy manager.
    
    Provides multi-tenancy support.
    """
    
    def __init__(self):
        """Initialize multi-tenancy."""
        self.tenants: Dict[str, Tenant] = {}
        self.tenant_caches: Dict[str, Any] = {}
        self.tenant_stats: Dict[str, Dict[str, Any]] = {}
    
    def register_tenant(
        self,
        tenant_id: str,
        tenant_name: str,
        cache: Any,
        quota: Optional[Dict[str, Any]] = None
    ) -> Tenant:
        """
        Register tenant.
        
        Args:
            tenant_id: Tenant ID
            tenant_name: Tenant name
            cache: Cache instance for tenant
            quota: Optional quota limits
            
        Returns:
            Tenant instance
        """
        tenant = Tenant(
            id=tenant_id,
            name=tenant_name,
            quota=quota or {},
            metadata={}
        )
        
        self.tenants[tenant_id] = tenant
        self.tenant_caches[tenant_id] = cache
        self.tenant_stats[tenant_id] = {}
        
        logger.info(f"Registered tenant: {tenant_id}")
        
        return tenant
    
    def get_tenant_cache(self, tenant_id: str) -> Optional[Any]:
        """
        Get cache for tenant.
        
        Args:
            tenant_id: Tenant ID
            
        Returns:
            Cache instance or None
        """
        return self.tenant_caches.get(tenant_id)
    
    def check_quota(self, tenant_id: str, operation: str) -> bool:
        """
        Check if tenant has quota for operation.
        
        Args:
            tenant_id: Tenant ID
            operation: Operation type
            
        Returns:
            True if quota available
        """
        if tenant_id not in self.tenants:
            return False
        
        tenant = self.tenants[tenant_id]
        quota = tenant.quota
        
        # Check operation-specific quota
        if operation in quota:
            current_usage = self.tenant_stats[tenant_id].get(operation, 0)
            limit = quota[operation]
            
            if current_usage >= limit:
                return False
        
        return True
    
    def record_usage(self, tenant_id: str, operation: str, amount: float = 1.0) -> None:
        """
        Record usage for tenant.
        
        Args:
            tenant_id: Tenant ID
            operation: Operation type
            amount: Usage amount
        """
        if tenant_id not in self.tenant_stats:
            self.tenant_stats[tenant_id] = {}
        
        if operation not in self.tenant_stats[tenant_id]:
            self.tenant_stats[tenant_id][operation] = 0.0
        
        self.tenant_stats[tenant_id][operation] += amount
    
    def get_tenant_stats(self, tenant_id: str) -> Dict[str, Any]:
        """
        Get statistics for tenant.
        
        Args:
            tenant_id: Tenant ID
            
        Returns:
            Tenant statistics
        """
        if tenant_id not in self.tenants:
            return {}
        
        tenant = self.tenants[tenant_id]
        stats = self.tenant_stats.get(tenant_id, {})
        
        cache = self.tenant_caches.get(tenant_id)
        cache_stats = cache.get_stats() if cache else {}
        
        return {
            "tenant_id": tenant_id,
            "tenant_name": tenant.name,
            "quota": tenant.quota,
            "usage": stats,
            "cache_stats": cache_stats
        }
    
    def get_all_tenant_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all tenants.
        
        Returns:
            Dictionary of tenant stats
        """
        return {
            tenant_id: self.get_tenant_stats(tenant_id)
            for tenant_id in self.tenants.keys()
        }
    
    def isolate_tenant(self, tenant_id: str) -> bool:
        """
        Isolate tenant (prevent access).
        
        Args:
            tenant_id: Tenant ID
            
        Returns:
            True if isolated
        """
        if tenant_id not in self.tenants:
            return False
        
        # In production: would implement actual isolation
        logger.info(f"Isolated tenant: {tenant_id}")
        return True
    
    def unisolate_tenant(self, tenant_id: str) -> bool:
        """
        Unisolate tenant (allow access).
        
        Args:
            tenant_id: Tenant ID
            
        Returns:
            True if unisolated
        """
        if tenant_id not in self.tenants:
            return False
        
        # In production: would implement actual unisolation
        logger.info(f"Unisolated tenant: {tenant_id}")
        return True


class TenantCacheRouter:
    """
    Tenant cache router.
    
    Routes requests to appropriate tenant cache.
    """
    
    def __init__(self, multi_tenancy: CacheMultiTenancy):
        """
        Initialize router.
        
        Args:
            multi_tenancy: Multi-tenancy manager
        """
        self.multi_tenancy = multi_tenancy
    
    def route(
        self,
        tenant_id: str,
        operation: str,
        *args,
        **kwargs
    ) -> Any:
        """
        Route operation to tenant cache.
        
        Args:
            tenant_id: Tenant ID
            operation: Operation name
            *args: Operation arguments
            **kwargs: Operation keyword arguments
            
        Returns:
            Operation result
        """
        # Check quota
        if not self.multi_tenancy.check_quota(tenant_id, operation):
            raise RuntimeError(f"Quota exceeded for tenant {tenant_id}")
        
        # Get tenant cache
        cache = self.multi_tenancy.get_tenant_cache(tenant_id)
        if cache is None:
            raise ValueError(f"Tenant {tenant_id} not found")
        
        # Execute operation
        if operation == "get":
            result = cache.get(*args, **kwargs)
            self.multi_tenancy.record_usage(tenant_id, "get")
        elif operation == "put":
            cache.put(*args, **kwargs)
            self.multi_tenancy.record_usage(tenant_id, "put")
            result = None
        elif operation == "clear":
            cache.clear()
            self.multi_tenancy.record_usage(tenant_id, "clear")
            result = None
        else:
            # Generic operation
            if hasattr(cache, operation):
                result = getattr(cache, operation)(*args, **kwargs)
                self.multi_tenancy.record_usage(tenant_id, operation)
            else:
                raise ValueError(f"Unknown operation: {operation}")
        
        return result

