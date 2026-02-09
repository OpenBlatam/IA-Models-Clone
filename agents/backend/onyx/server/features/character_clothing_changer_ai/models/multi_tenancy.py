"""
Multi-Tenancy System for Flux2 Clothing Changer
================================================

Multi-tenant support with isolation and resource management.
"""

import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from collections import defaultdict
import logging
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class Tenant:
    """Tenant information."""
    tenant_id: str
    name: str
    plan: str  # "free", "basic", "premium", "enterprise"
    max_requests_per_day: int = 1000
    max_requests_per_month: int = 30000
    max_storage_mb: float = 1024.0
    features: List[str] = None
    metadata: Dict[str, Any] = None
    created_at: float = 0.0
    
    def __post_init__(self):
        if self.features is None:
            self.features = []
        if self.metadata is None:
            self.metadata = {}
        if self.created_at == 0.0:
            self.created_at = time.time()


@dataclass
class TenantUsage:
    """Tenant usage statistics."""
    tenant_id: str
    requests_today: int = 0
    requests_this_month: int = 0
    storage_mb: float = 0.0
    last_request: float = 0.0


class MultiTenancy:
    """Multi-tenancy management system."""
    
    def __init__(self):
        """Initialize multi-tenancy system."""
        self.tenants: Dict[str, Tenant] = {}
        self.usage: Dict[str, TenantUsage] = {}
        self.tenant_resources: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Default plans
        self.plans = {
            "free": {
                "max_requests_per_day": 100,
                "max_requests_per_month": 3000,
                "max_storage_mb": 100.0,
                "features": ["basic_processing"],
            },
            "basic": {
                "max_requests_per_day": 1000,
                "max_requests_per_month": 30000,
                "max_storage_mb": 1024.0,
                "features": ["basic_processing", "batch_processing"],
            },
            "premium": {
                "max_requests_per_day": 10000,
                "max_requests_per_month": 300000,
                "max_storage_mb": 10240.0,
                "features": ["basic_processing", "batch_processing", "priority_support", "api_access"],
            },
            "enterprise": {
                "max_requests_per_day": -1,  # Unlimited
                "max_requests_per_month": -1,  # Unlimited
                "max_storage_mb": -1.0,  # Unlimited
                "features": ["all"],
            },
        }
    
    def create_tenant(
        self,
        tenant_id: str,
        name: str,
        plan: str = "free",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tenant:
        """
        Create a new tenant.
        
        Args:
            tenant_id: Unique tenant identifier
            name: Tenant name
            plan: Plan name
            metadata: Optional metadata
            
        Returns:
            Created tenant
        """
        if plan not in self.plans:
            raise ValueError(f"Unknown plan: {plan}")
        
        plan_config = self.plans[plan]
        
        tenant = Tenant(
            tenant_id=tenant_id,
            name=name,
            plan=plan,
            max_requests_per_day=plan_config["max_requests_per_day"],
            max_requests_per_month=plan_config["max_requests_per_month"],
            max_storage_mb=plan_config["max_storage_mb"],
            features=plan_config["features"],
            metadata=metadata or {},
        )
        
        self.tenants[tenant_id] = tenant
        self.usage[tenant_id] = TenantUsage(tenant_id=tenant_id)
        
        logger.info(f"Created tenant: {tenant_id} with plan {plan}")
        return tenant
    
    def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        """Get tenant information."""
        return self.tenants.get(tenant_id)
    
    def check_quota(
        self,
        tenant_id: str,
        operation: str = "request",
    ) -> Dict[str, Any]:
        """
        Check if tenant has quota for operation.
        
        Args:
            tenant_id: Tenant identifier
            operation: Operation type
            
        Returns:
            Quota check result
        """
        if tenant_id not in self.tenants:
            return {
                "allowed": False,
                "reason": "Tenant not found",
            }
        
        tenant = self.tenants[tenant_id]
        usage = self.usage[tenant_id]
        
        # Check daily quota
        if tenant.max_requests_per_day > 0:
            if usage.requests_today >= tenant.max_requests_per_day:
                return {
                    "allowed": False,
                    "reason": "Daily quota exceeded",
                    "quota": tenant.max_requests_per_day,
                    "used": usage.requests_today,
                }
        
        # Check monthly quota
        if tenant.max_requests_per_month > 0:
            if usage.requests_this_month >= tenant.max_requests_per_month:
                return {
                    "allowed": False,
                    "reason": "Monthly quota exceeded",
                    "quota": tenant.max_requests_per_month,
                    "used": usage.requests_this_month,
                }
        
        return {
            "allowed": True,
            "daily_remaining": (
                tenant.max_requests_per_day - usage.requests_today
                if tenant.max_requests_per_day > 0 else -1
            ),
            "monthly_remaining": (
                tenant.max_requests_per_month - usage.requests_this_month
                if tenant.max_requests_per_month > 0 else -1
            ),
        }
    
    def record_usage(
        self,
        tenant_id: str,
        operation: str = "request",
        storage_mb: float = 0.0,
    ) -> None:
        """
        Record tenant usage.
        
        Args:
            tenant_id: Tenant identifier
            operation: Operation type
            storage_mb: Storage used in MB
        """
        if tenant_id not in self.usage:
            self.usage[tenant_id] = TenantUsage(tenant_id=tenant_id)
        
        usage = self.usage[tenant_id]
        
        if operation == "request":
            usage.requests_today += 1
            usage.requests_this_month += 1
        
        usage.storage_mb += storage_mb
        usage.last_request = time.time()
    
    def has_feature(
        self,
        tenant_id: str,
        feature: str,
    ) -> bool:
        """
        Check if tenant has access to feature.
        
        Args:
            tenant_id: Tenant identifier
            feature: Feature name
            
        Returns:
            True if tenant has feature
        """
        if tenant_id not in self.tenants:
            return False
        
        tenant = self.tenants[tenant_id]
        
        if "all" in tenant.features:
            return True
        
        return feature in tenant.features
    
    def upgrade_tenant(
        self,
        tenant_id: str,
        new_plan: str,
    ) -> bool:
        """
        Upgrade tenant plan.
        
        Args:
            tenant_id: Tenant identifier
            new_plan: New plan name
            
        Returns:
            True if upgraded
        """
        if tenant_id not in self.tenants:
            return False
        
        if new_plan not in self.plans:
            return False
        
        tenant = self.tenants[tenant_id]
        plan_config = self.plans[new_plan]
        
        tenant.plan = new_plan
        tenant.max_requests_per_day = plan_config["max_requests_per_day"]
        tenant.max_requests_per_month = plan_config["max_requests_per_month"]
        tenant.max_storage_mb = plan_config["max_storage_mb"]
        tenant.features = plan_config["features"]
        
        logger.info(f"Upgraded tenant {tenant_id} to plan {new_plan}")
        return True
    
    def get_tenant_statistics(self, tenant_id: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a tenant."""
        if tenant_id not in self.tenants:
            return None
        
        tenant = self.tenants[tenant_id]
        usage = self.usage[tenant_id]
        
        return {
            "tenant_id": tenant_id,
            "name": tenant.name,
            "plan": tenant.plan,
            "usage": {
                "requests_today": usage.requests_today,
                "requests_this_month": usage.requests_this_month,
                "storage_mb": usage.storage_mb,
                "last_request": usage.last_request,
            },
            "quotas": {
                "max_requests_per_day": tenant.max_requests_per_day,
                "max_requests_per_month": tenant.max_requests_per_month,
                "max_storage_mb": tenant.max_storage_mb,
            },
            "features": tenant.features,
        }
    
    def get_all_statistics(self) -> Dict[str, Any]:
        """Get statistics for all tenants."""
        return {
            "total_tenants": len(self.tenants),
            "by_plan": {
                plan: sum(1 for t in self.tenants.values() if t.plan == plan)
                for plan in self.plans.keys()
            },
            "total_usage": {
                "total_requests_today": sum(u.requests_today for u in self.usage.values()),
                "total_requests_month": sum(u.requests_this_month for u in self.usage.values()),
                "total_storage_mb": sum(u.storage_mb for u in self.usage.values()),
            },
        }
    
    def reset_daily_usage(self) -> None:
        """Reset daily usage counters (call daily)."""
        for usage in self.usage.values():
            usage.requests_today = 0
        logger.info("Daily usage counters reset")
    
    def reset_monthly_usage(self) -> None:
        """Reset monthly usage counters (call monthly)."""
        for usage in self.usage.values():
            usage.requests_this_month = 0
        logger.info("Monthly usage counters reset")


