"""
Multi-Tenancy
=============

Multi-tenancy support modules.
"""

from aws.modules.multitenancy.tenant_manager import TenantManager, Tenant
from aws.modules.multitenancy.tenant_isolation import TenantIsolation
from aws.modules.multitenancy.resource_quota import ResourceQuota, Quota

__all__ = [
    "TenantManager",
    "Tenant",
    "TenantIsolation",
    "ResourceQuota",
    "Quota",
]

