from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

from __future__ import annotations
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from .base_types import StatusType, StatusCategory
import msgspec
import numpy as np
    import pandas as pd
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Status Mixin - Onyx Integration
Status handling functionality for models.
"""
try:
except ImportError:
    pd = None

class Status(msgspec.Struct, frozen=True, slots=True):
    id: str
    type: str
    category: str
    timestamp: str = msgspec.field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: dict = msgspec.field(default_factory=dict)

    def as_tuple(self) -> tuple:
        return (self.id, self.type, self.category, self.timestamp, self.metadata)

    @staticmethod
    def batch_encode(items: List["Status"]) -> bytes:
        return msgspec.json.encode(items)

    @staticmethod
    def batch_decode(data: bytes) -> List["Status"]:
        return msgspec.json.decode(data, type=List[Status])

    @staticmethod
    def batch_deduplicate(items: List["Status"], key: Callable[[Any], Any] = lambda x: x.id) -> List["Status"]:
        seen = set()
        out = []
        for item in items:
            k = key(item)
            if k not in seen:
                seen.add(k)
                out.append(item)
        return out

    @staticmethod
    def batch_to_dicts(items: List["Status"]) -> List[dict]:
        return [item.__dict__ for item in items]

    @staticmethod
    def batch_from_dicts(dicts: List[dict]) -> List["Status"]:
        return [Status(**d) for d in dicts]

    @staticmethod
    def batch_to_numpy(items: List["Status"]):
        
    """batch_to_numpy function."""
arr = np.array([item.as_tuple() for item in items], dtype=object)
        return arr

    @staticmethod
    def batch_to_pandas(items: List["Status"]):
        
    """batch_to_pandas function."""
if pd is None:
            raise ImportError("pandas is not installed")
        return pd.DataFrame(Status.batch_to_dicts(items))

    @staticmethod
    def batch_to_parquet(items: List["Status"], path: str):
        
    """batch_to_parquet function."""
if pd is None:
            raise ImportError("pandas is not installed")
        Status.batch_to_pandas(items).to_parquet(path)

    @staticmethod
    def batch_from_parquet(path: str) -> List["Status"]:
        if pd is None:
            raise ImportError("pandas is not installed")
        df = pd.read_parquet(path)
        return Status.batch_from_dicts(df.to_dict(orient="records"))

    @staticmethod
    def validate_batch(items: List["Status"]) -> None:
        for item in items:
            if not item.id or not item.type or not item.category:
                raise ValueError(f"Invalid Status: {item}")

class StatusMixin:
    """Mixin for status handling functionality."""
    
    _status_history: List[Status] = []
    _max_history_size: int = 1000
    
    def set_status(self, status_type: StatusType, category: StatusCategory, metadata: Optional[Dict[str, Any]] = None) -> Status:
        """Set a status."""
        status = Status(
            type=status_type,
            category=category,
            metadata=metadata or {}
        )
        
        self._status_history.append(status)
        if len(self._status_history) > self._max_history_size:
            self._status_history.pop(0)
        
        return status
    
    def get_current_status(self) -> Optional[Status]:
        """Get current status."""
        if not self._status_history:
            return None
        return self._status_history[-1]
    
    def get_status_history(self, status_type: Optional[StatusType] = None, category: Optional[StatusCategory] = None, limit: Optional[int] = None) -> List[Status]:
        """Get status history."""
        history = self._status_history
        
        if status_type:
            history = [s for s in history if s.type == status_type]
        
        if category:
            history = [s for s in history if s.category == category]
        
        if limit:
            history = history[-limit:]
        
        return history
    
    def clear_status_history(self) -> None:
        """Clear status history."""
        self._status_history.clear()
    
    def update_status_metadata(self, metadata: Dict[str, Any]) -> None:
        """Update current status metadata."""
        if self._status_history:
            self._status_history[-1].metadata.update(metadata)
    
    def get_status_by_category(self, category: StatusCategory) -> Optional[Status]:
        """Get most recent status by category."""
        for status in reversed(self._status_history):
            if status.category == category:
                return status
        return None
    
    def get_status_by_type(self, status_type: StatusType) -> Optional[Status]:
        """Get most recent status by type."""
        for status in reversed(self._status_history):
            if status.type == status_type:
                return status
        return None 