from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from __future__ import annotations
from typing import Any, Dict, List, Optional, Set, Callable, TypeVar
from dataclasses import dataclass, field
from datetime import datetime
from .base_types import PermissionType, PermissionStatus
import msgspec
import numpy as np
import time
    import pandas as pd
    from datadog import api as dd_api
    import sentry_sdk
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Permission Mixin - Onyx Integration
Permission handling functionality for models.
"""
try:
except ImportError:
    pd = None
try:
except ImportError:
    dd_api = None
try:
except ImportError:
    sentry_sdk = None

T = TypeVar("T")

class Permission(msgspec.Struct, frozen=True, slots=True):
    id: str
    type: str
    status: str
    granted_at: str = msgspec.field(default_factory=lambda: datetime.utcnow().isoformat())
    expires_at: str | None = None
    metadata: dict = msgspec.field(default_factory=dict)

    def as_tuple(self) -> tuple:
        return (self.id, self.type, self.status, self.granted_at, self.expires_at, self.metadata)

    @staticmethod
    def batch_encode(items: List["Permission"]) -> bytes:
        start = time.time()
        out = msgspec.json.encode(items)
        Permission._log_metric("batch_encode", len(items), time.time() - start)
        return out

    @staticmethod
    def batch_decode(data: bytes) -> List["Permission"]:
        start = time.time()
        out = msgspec.json.decode(data, type=List[Permission])
        Permission._log_metric("batch_decode", len(out), time.time() - start)
        return out

    @staticmethod
    def batch_deduplicate(items: List["Permission"], key: Callable[[Any], Any] = lambda x: x.id) -> List["Permission"]:
        start = time.time()
        seen = set()
        out = []
        for item in items:
            k = key(item)
            if k not in seen:
                seen.add(k)
                out.append(item)
        Permission._log_metric("batch_deduplicate", len(items), time.time() - start)
        return out

    @staticmethod
    def batch_validate_unique(items: List["Permission"], key: Callable[[Any], Any] = lambda x: x.id) -> None:
        seen = set()
        for item in items:
            k = key(item)
            if k in seen:
                Permission._log_error("duplicate_key", k)
                raise ValueError(f"Duplicate key found: {k}")
            seen.add(k)

    @staticmethod
    def batch_filter(items: List["Permission"], predicate: Callable[["Permission"], bool]) -> List["Permission"]:
        return [item for item in items if predicate(item)]

    @staticmethod
    def batch_map(items: List["Permission"], func: Callable[["Permission"], T]) -> List[T]:
        return [func(item) for item in items]

    @staticmethod
    def batch_groupby(items: List["Permission"], key: Callable[["Permission"], Any]) -> Dict[Any, List["Permission"]]:
        groups = {}
        for item in items:
            k = key(item)
            groups.setdefault(k, []).append(item)
        return groups

    @staticmethod
    def batch_sort(items: List["Permission"], key: Callable[["Permission"], Any], reverse: bool = False) -> List["Permission"]:
        return sorted(items, key=key, reverse=reverse)

    @staticmethod
    def batch_to_dicts(items: List["Permission"]) -> List[dict]:
        return [item.__dict__ for item in items]

    @staticmethod
    def batch_from_dicts(dicts: List[dict]) -> List["Permission"]:
        return [Permission(**d) for d in dicts]

    @staticmethod
    def batch_to_numpy(items: List["Permission"]):
        
    """batch_to_numpy function."""
arr = np.array([item.as_tuple() for item in items], dtype=object)
        return arr

    @staticmethod
    def batch_to_pandas(items: List["Permission"]):
        
    """batch_to_pandas function."""
if pd is None:
            raise ImportError("pandas is not installed")
        return pd.DataFrame(Permission.batch_to_dicts(items))

    @staticmethod
    def batch_to_parquet(items: List["Permission"], path: str):
        
    """batch_to_parquet function."""
if pd is None:
            raise ImportError("pandas is not installed")
        Permission.batch_to_pandas(items).to_parquet(path)

    @staticmethod
    def batch_from_parquet(path: str) -> List["Permission"]:
        if pd is None:
            raise ImportError("pandas is not installed")
        df = pd.read_parquet(path)
        return Permission.batch_from_dicts(df.to_dict(orient="records"))

    @staticmethod
    def validate_batch(items: List["Permission"]) -> None:
        for item in items:
            if not item.id or not item.type or not item.status:
                Permission._log_error("invalid_permission", item.id)
                raise ValueError(f"Invalid Permission: {item}")

    @staticmethod
    def _log_metric(operation: str, count: int, duration: float):
        
    """_log_metric function."""
if dd_api:
            dd_api.Metric.send(
                metric=f"permission.{operation}.duration",
                points=duration,
                tags=[f"count:{count}"]
            )

    @staticmethod
    def _log_error(error_type: str, value: Any):
        
    """_log_error function."""
if sentry_sdk:
            sentry_sdk.capture_message(f"Permission error: {error_type} - {value}")

class PermissionMixin:
    """Mixin for permission handling functionality."""
    
    _permissions: Dict[str, Dict[PermissionType, Permission]] = {}
    
    def grant_permission(self, entity_id: str, permission_type: PermissionType, expires_at: Optional[datetime] = None, metadata: Optional[Dict[str, Any]] = None) -> Permission:
        """Grant a permission."""
        if entity_id not in self._permissions:
            self._permissions[entity_id] = {}
        
        permission = Permission(
            type=permission_type,
            expires_at=expires_at,
            metadata=metadata or {}
        )
        
        self._permissions[entity_id][permission_type] = permission
        return permission
    
    def revoke_permission(self, entity_id: str, permission_type: PermissionType) -> None:
        """Revoke a permission."""
        if entity_id in self._permissions and permission_type in self._permissions[entity_id]:
            del self._permissions[entity_id][permission_type]
            if not self._permissions[entity_id]:
                del self._permissions[entity_id]
    
    def has_permission(self, entity_id: str, permission_type: PermissionType) -> bool:
        """Check if an entity has a permission."""
        if entity_id not in self._permissions:
            return False
        
        permission = self._permissions[entity_id].get(permission_type)
        if not permission:
            return False
        
        if permission.status != PermissionStatus.ACTIVE:
            return False
        
        if permission.expires_at and permission.expires_at < datetime.utcnow():
            return False
        
        return True
    
    def get_permissions(self, entity_id: str) -> Dict[PermissionType, Permission]:
        """Get all permissions for an entity."""
        return self._permissions.get(entity_id, {}).copy()
    
    def get_all_permissions(self) -> Dict[str, Dict[PermissionType, Permission]]:
        """Get all permissions."""
        return {entity_id: permissions.copy() for entity_id, permissions in self._permissions.items()}
    
    def clear_permissions(self, entity_id: str) -> None:
        """Clear all permissions for an entity."""
        if entity_id in self._permissions:
            del self._permissions[entity_id]
    
    def clear_all_permissions(self) -> None:
        """Clear all permissions."""
        self._permissions.clear()
    
    def update_permission_status(self, entity_id: str, permission_type: PermissionType, status: PermissionStatus) -> None:
        """Update permission status."""
        if entity_id in self._permissions and permission_type in self._permissions[entity_id]:
            self._permissions[entity_id][permission_type].status = status
    
    def update_permission_metadata(self, entity_id: str, permission_type: PermissionType, metadata: Dict[str, Any]) -> None:
        """Update permission metadata."""
        if entity_id in self._permissions and permission_type in self._permissions[entity_id]:
            self._permissions[entity_id][permission_type].metadata.update(metadata) 