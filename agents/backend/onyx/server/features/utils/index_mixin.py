from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from __future__ import annotations
from typing import Any, Dict, List, Optional, Set, Type, Callable, TypeVar
from dataclasses import dataclass, field
from datetime import datetime
from .base_types import IndexType, IndexStatus
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
Index Mixin - Onyx Integration
Indexing functionality for models.
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

class Index(msgspec.Struct, frozen=True, slots=True):
    id: str
    name: str
    type: str
    fields: List[str]
    status: str
    created_at: str = msgspec.field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = msgspec.field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: dict = msgspec.field(default_factory=dict)

    def as_tuple(self) -> tuple:
        return (self.id, self.name, self.type, self.fields, self.status, self.created_at, self.updated_at, self.metadata)

    @staticmethod
    def batch_encode(items: List["Index"]) -> bytes:
        start = time.time()
        out = msgspec.json.encode(items)
        Index._log_metric("batch_encode", len(items), time.time() - start)
        return out

    @staticmethod
    def batch_decode(data: bytes) -> List["Index"]:
        start = time.time()
        out = msgspec.json.decode(data, type=List[Index])
        Index._log_metric("batch_decode", len(out), time.time() - start)
        return out

    @staticmethod
    def batch_deduplicate(items: List["Index"], key: Callable[[Any], Any] = lambda x: x.id) -> List["Index"]:
        start = time.time()
        seen = set()
        out = []
        for item in items:
            k = key(item)
            if k not in seen:
                seen.add(k)
                out.append(item)
        Index._log_metric("batch_deduplicate", len(items), time.time() - start)
        return out

    @staticmethod
    def batch_validate_unique(items: List["Index"], key: Callable[[Any], Any] = lambda x: x.id) -> None:
        seen = set()
        for item in items:
            k = key(item)
            if k in seen:
                Index._log_error("duplicate_key", k)
                raise ValueError(f"Duplicate key found: {k}")
            seen.add(k)

    @staticmethod
    def batch_filter(items: List["Index"], predicate: Callable[["Index"], bool]) -> List["Index"]:
        return [item for item in items if predicate(item)]

    @staticmethod
    def batch_map(items: List["Index"], func: Callable[["Index"], T]) -> List[T]:
        return [func(item) for item in items]

    @staticmethod
    def batch_groupby(items: List["Index"], key: Callable[["Index"], Any]) -> Dict[Any, List["Index"]]:
        groups = {}
        for item in items:
            k = key(item)
            groups.setdefault(k, []).append(item)
        return groups

    @staticmethod
    def batch_sort(items: List["Index"], key: Callable[["Index"], Any], reverse: bool = False) -> List["Index"]:
        return sorted(items, key=key, reverse=reverse)

    @staticmethod
    def batch_to_dicts(items: List["Index"]) -> List[dict]:
        return [item.__dict__ for item in items]

    @staticmethod
    def batch_from_dicts(dicts: List[dict]) -> List["Index"]:
        return [Index(**d) for d in dicts]

    @staticmethod
    def batch_to_numpy(items: List["Index"]):
        
    """batch_to_numpy function."""
arr = np.array([item.as_tuple() for item in items], dtype=object)
        return arr

    @staticmethod
    def batch_to_pandas(items: List["Index"]):
        
    """batch_to_pandas function."""
if pd is None:
            raise ImportError("pandas is not installed")
        return pd.DataFrame(Index.batch_to_dicts(items))

    @staticmethod
    def batch_to_parquet(items: List["Index"], path: str):
        
    """batch_to_parquet function."""
if pd is None:
            raise ImportError("pandas is not installed")
        Index.batch_to_pandas(items).to_parquet(path)

    @staticmethod
    def batch_from_parquet(path: str) -> List["Index"]:
        if pd is None:
            raise ImportError("pandas is not installed")
        df = pd.read_parquet(path)
        return Index.batch_from_dicts(df.to_dict(orient="records"))

    @staticmethod
    def validate_batch(items: List["Index"]) -> None:
        for item in items:
            if not item.id or not item.name or not item.type:
                Index._log_error("invalid_index", item.id)
                raise ValueError(f"Invalid Index: {item}")

    @staticmethod
    def _log_metric(operation: str, count: int, duration: float):
        
    """_log_metric function."""
if dd_api:
            dd_api.Metric.send(
                metric=f"index.{operation}.duration",
                points=duration,
                tags=[f"count:{count}"]
            )

    @staticmethod
    def _log_error(error_type: str, value: Any):
        
    """_log_error function."""
if sentry_sdk:
            sentry_sdk.capture_message(f"Index error: {error_type} - {value}")

class IndexMixin:
    """Mixin for indexing functionality."""
    
    _indexes: Dict[str, Index] = {}
    _index_values: Dict[str, Dict[str, Set[str]]] = {}
    
    def create_index(self, name: str, type: IndexType, fields: List[str], metadata: Optional[Dict[str, Any]] = None) -> Index:
        """Create an index."""
        if name in self._indexes:
            raise ValueError(f"Index {name} already exists")
        
        index = Index(
            name=name,
            type=type,
            fields=fields,
            metadata=metadata or {}
        )
        
        self._indexes[name] = index
        self._index_values[name] = {}
        
        return index
    
    def drop_index(self, name: str) -> None:
        """Drop an index."""
        if name not in self._indexes:
            raise ValueError(f"Index {name} does not exist")
        
        del self._indexes[name]
        del self._index_values[name]
    
    def get_index(self, name: str) -> Optional[Index]:
        """Get an index."""
        return self._indexes.get(name)
    
    def get_indexes(self) -> Dict[str, Index]:
        """Get all indexes."""
        return self._indexes.copy()
    
    def add_to_index(self, index_name: str, value: str, key: str) -> None:
        """Add a value to an index."""
        if index_name not in self._indexes:
            raise ValueError(f"Index {index_name} does not exist")
        
        if value not in self._index_values[index_name]:
            self._index_values[index_name][value] = set()
        
        self._index_values[index_name][value].add(key)
    
    def remove_from_index(self, index_name: str, value: str, key: str) -> None:
        """Remove a value from an index."""
        if index_name not in self._indexes:
            raise ValueError(f"Index {index_name} does not exist")
        
        if value in self._index_values[index_name]:
            self._index_values[index_name][value].discard(key)
            if not self._index_values[index_name][value]:
                del self._index_values[index_name][value]
    
    def search_index(self, index_name: str, value: str) -> Set[str]:
        """Search an index."""
        if index_name not in self._indexes:
            raise ValueError(f"Index {index_name} does not exist")
        
        return self._index_values[index_name].get(value, set())
    
    def get_index_values(self, index_name: str) -> Dict[str, Set[str]]:
        """Get all values in an index."""
        if index_name not in self._indexes:
            raise ValueError(f"Index {index_name} does not exist")
        
        return self._index_values[index_name].copy()
    
    def clear_index(self, index_name: str) -> None:
        """Clear an index."""
        if index_name not in self._indexes:
            raise ValueError(f"Index {index_name} does not exist")
        
        self._index_values[index_name].clear() 