"""
API Versioning
==============

API versioning modules.
"""

from aws.modules.api_versioning.version_manager import VersionManager, APIVersion, VersionStatus
from aws.modules.api_versioning.version_router import VersionRouter
from aws.modules.api_versioning.deprecation_manager import DeprecationManager, DeprecationNotice

__all__ = [
    "VersionManager",
    "APIVersion",
    "VersionStatus",
    "VersionRouter",
    "DeprecationManager",
    "DeprecationNotice",
]

