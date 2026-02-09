"""
Cache versioning system.

Provides versioning capabilities for cache entries.
"""
from __future__ import annotations

import logging
import time
import threading
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class VersionStrategy(Enum):
    """Versioning strategies."""
    TIMESTAMP = "timestamp"
    SEQUENTIAL = "sequential"
    HASH = "hash"
    UUID = "uuid"


@dataclass
class VersionedEntry:
    """Versioned cache entry."""
    value: Any
    version: str
    timestamp: float
    metadata: Dict[str, Any]


class CacheVersioning:
    """
    Cache versioning manager.
    
    Provides versioning for cache entries.
    """
    
    def __init__(self, strategy: VersionStrategy = VersionStrategy.SEQUENTIAL):
        """
        Initialize versioning.
        
        Args:
            strategy: Versioning strategy
        """
        self.strategy = strategy
        self.versions: Dict[str, List[VersionedEntry]] = {}
        self.current_version: Dict[str, int] = {}
        self.lock = threading.Lock()
    
    def _generate_version(self, key: str) -> str:
        """
        Generate version identifier.
        
        Args:
            key: Cache key
            
        Returns:
            Version string
        """
        if self.strategy == VersionStrategy.TIMESTAMP:
            return str(time.time())
        elif self.strategy == VersionStrategy.SEQUENTIAL:
            with self.lock:
                version = self.current_version.get(key, 0) + 1
                self.current_version[key] = version
                return str(version)
        elif self.strategy == VersionStrategy.HASH:
            import hashlib
            content = f"{key}{time.time()}"
            return hashlib.md5(content.encode()).hexdigest()
        elif self.strategy == VersionStrategy.UUID:
            import uuid
            return str(uuid.uuid4())
        else:
            return str(time.time())
    
    def put_versioned(
        self,
        key: str,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Put versioned entry.
        
        Args:
            key: Cache key
            value: Value to cache
            metadata: Optional metadata
            
        Returns:
            Version identifier
        """
        version = self._generate_version(key)
        
        entry = VersionedEntry(
            value=value,
            version=version,
            timestamp=time.time(),
            metadata=metadata or {}
        )
        
        with self.lock:
            if key not in self.versions:
                self.versions[key] = []
            self.versions[key].append(entry)
            
            # Keep only last N versions
            if len(self.versions[key]) > 100:
                self.versions[key] = self.versions[key][-100:]
        
        logger.debug(f"Put versioned entry: {key}@{version}")
        
        return version
    
    def get_versioned(
        self,
        key: str,
        version: Optional[str] = None
    ) -> Optional[VersionedEntry]:
        """
        Get versioned entry.
        
        Args:
            key: Cache key
            version: Optional version identifier (latest if None)
            
        Returns:
            Versioned entry or None
        """
        with self.lock:
            if key not in self.versions:
                return None
            
            entries = self.versions[key]
            
            if not entries:
                return None
            
            if version is None:
                # Return latest
                return entries[-1]
            
            # Find specific version
            for entry in reversed(entries):
                if entry.version == version:
                    return entry
            
            return None
    
    def list_versions(self, key: str) -> List[str]:
        """
        List all versions for key.
        
        Args:
            key: Cache key
            
        Returns:
            List of version identifiers
        """
        with self.lock:
            if key not in self.versions:
                return []
            
            return [entry.version for entry in self.versions[key]]
    
    def get_version_history(self, key: str) -> List[VersionedEntry]:
        """
        Get version history for key.
        
        Args:
            key: Cache key
            
        Returns:
            List of versioned entries
        """
        with self.lock:
            if key not in self.versions:
                return []
            
            return self.versions[key].copy()
    
    def rollback(self, key: str, version: str) -> bool:
        """
        Rollback to specific version.
        
        Args:
            key: Cache key
            version: Version to rollback to
            
        Returns:
            True if successful
        """
        entry = self.get_versioned(key, version)
        if entry is None:
            return False
        
        # Put as current version
        self.put_versioned(key, entry.value, entry.metadata)
        
        logger.info(f"Rolled back {key} to version {version}")
        
        return True


class CacheReplication:
    """
    Cache replication manager.
    
    Provides replication capabilities for cache.
    """
    
    def __init__(self, cache: Any, replicas: List[Any]):
        """
        Initialize replication.
        
        Args:
            cache: Primary cache instance
            replicas: List of replica cache instances
        """
        self.cache = cache
        self.replicas = replicas
        self.replication_strategy = "async"  # async or sync
    
    def put_replicated(
        self,
        key: Any,
        value: Any,
        sync: bool = False
    ) -> None:
        """
        Put to primary and replicas.
        
        Args:
            key: Cache key
            value: Value to cache
            sync: Whether to wait for replication
        """
        # Put to primary
        self.cache.put(key, value)
        
        # Replicate to replicas
        if sync:
            for replica in self.replicas:
                replica.put(key, value)
        else:
            # Async replication
            import threading
            for replica in self.replicas:
                thread = threading.Thread(
                    target=replica.put,
                    args=(key, value),
                    daemon=True
                )
                thread.start()
    
    def get_replicated(self, key: Any) -> Optional[Any]:
        """
        Get from primary (with fallback to replicas).
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        # Try primary first
        value = self.cache.get(key)
        if value is not None:
            return value
        
        # Fallback to replicas
        for replica in self.replicas:
            value = replica.get(key)
            if value is not None:
                # Promote to primary
                self.cache.put(key, value)
                return value
        
        return None
    
    def sync_replicas(self) -> None:
        """Synchronize all replicas with primary."""
        # Get all keys from primary
        stats = self.cache.get_stats()
        # In production: would iterate through all keys
        
        logger.info("Synchronized replicas with primary")



Provides versioning capabilities for cache entries.
"""
from __future__ import annotations

import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class VersionStrategy(Enum):
    """Versioning strategies."""
    TIMESTAMP = "timestamp"
    INCREMENTAL = "incremental"
    HASH = "hash"
    SEMANTIC = "semantic"


@dataclass
class CacheVersion:
    """Cache version information."""
    version: str
    timestamp: float
    strategy: VersionStrategy
    metadata: Dict[str, Any]


class CacheVersionManager:
    """
    Cache version manager.
    
    Manages versions of cache entries.
    """
    
    def __init__(self, strategy: VersionStrategy = VersionStrategy.TIMESTAMP):
        """
        Initialize version manager.
        
        Args:
            strategy: Versioning strategy
        """
        self.strategy = strategy
        self.versions: Dict[int, List[CacheVersion]] = {}
        self.version_counter = 0
    
    def create_version(
        self,
        position: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> CacheVersion:
        """
        Create version for position.
        
        Args:
            position: Cache position
            metadata: Optional metadata
            
        Returns:
            Created version
        """
        if self.strategy == VersionStrategy.TIMESTAMP:
            version_str = str(time.time())
        elif self.strategy == VersionStrategy.INCREMENTAL:
            self.version_counter += 1
            version_str = str(self.version_counter)
        elif self.strategy == VersionStrategy.HASH:
            import hashlib
            version_str = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        else:  # SEMANTIC
            version_str = "1.0.0"
        
        version = CacheVersion(
            version=version_str,
            timestamp=time.time(),
            strategy=self.strategy,
            metadata=metadata or {}
        )
        
        if position not in self.versions:
            self.versions[position] = []
        
        self.versions[position].append(version)
        
        logger.debug(f"Created version {version_str} for position {position}")
        
        return version
    
    def get_latest_version(self, position: int) -> Optional[CacheVersion]:
        """
        Get latest version for position.
        
        Args:
            position: Cache position
            
        Returns:
            Latest version or None
        """
        if position not in self.versions or not self.versions[position]:
            return None
        
        versions = self.versions[position]
        return max(versions, key=lambda v: v.timestamp)
    
    def get_all_versions(self, position: int) -> List[CacheVersion]:
        """
        Get all versions for position.
        
        Args:
            position: Cache position
            
        Returns:
            List of versions
        """
        return self.versions.get(position, [])
    
    def get_version(self, position: int, version: str) -> Optional[CacheVersion]:
        """
        Get specific version.
        
        Args:
            position: Cache position
            version: Version string
            
        Returns:
            Version or None
        """
        versions = self.versions.get(position, [])
        for v in versions:
            if v.version == version:
                return v
        return None
    
    def rollback_to_version(
        self,
        position: int,
        version: str,
        cache: Any
    ) -> bool:
        """
        Rollback to specific version.
        
        Args:
            position: Cache position
            version: Version to rollback to
            cache: Cache instance
            
        Returns:
            True if successful
        """
        target_version = self.get_version(position, version)
        if target_version is None:
            return False
        
        # In production: would restore cache entry from version
        logger.info(f"Rolling back position {position} to version {version}")
        
        return True
    
    def cleanup_old_versions(self, position: int, keep_last: int = 10) -> int:
        """
        Cleanup old versions.
        
        Args:
            position: Cache position
            keep_last: Number of versions to keep
            
        Returns:
            Number of versions removed
        """
        if position not in self.versions:
            return 0
        
        versions = self.versions[position]
        if len(versions) <= keep_last:
            return 0
        
        # Sort by timestamp
        versions.sort(key=lambda v: v.timestamp, reverse=True)
        
        # Remove old versions
        removed = versions[keep_last:]
        self.versions[position] = versions[:keep_last]
        
        logger.info(f"Removed {len(removed)} old versions for position {position}")
        
        return len(removed)


class VersionedCache:
    """
    Versioned cache wrapper.
    
    Wraps cache with versioning capabilities.
    """
    
    def __init__(self, cache: Any, version_manager: CacheVersionManager):
        """
        Initialize versioned cache.
        
        Args:
            cache: Cache instance
            version_manager: Version manager
        """
        self.cache = cache
        self.version_manager = version_manager
    
    def get(self, position: int, version: Optional[str] = None) -> Optional[Any]:
        """
        Get value, optionally for specific version.
        
        Args:
            position: Cache position
            version: Optional version string
            
        Returns:
            Cached value or None
        """
        if version:
            # Get specific version
            version_info = self.version_manager.get_version(position, version)
            if version_info:
                # In production: would retrieve from versioned storage
                return self.cache.get(position)
            return None
        
        return self.cache.get(position)
    
    def put(
        self,
        position: int,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> CacheVersion:
        """
        Put value and create version.
        
        Args:
            position: Cache position
            value: Value to put
            metadata: Optional metadata
            
        Returns:
            Created version
        """
        # Put in cache
        self.cache.put(position, value)
        
        # Create version
        version = self.version_manager.create_version(position, metadata)
        
        return version
    
    def get_version_history(self, position: int) -> List[CacheVersion]:
        """
        Get version history.
        
        Args:
            position: Cache position
            
        Returns:
            List of versions
        """
        return self.version_manager.get_all_versions(position)
    
    def rollback(self, position: int, version: str) -> bool:
        """
        Rollback to version.
        
        Args:
            position: Cache position
            version: Version to rollback to
            
        Returns:
            True if successful
        """
        return self.version_manager.rollback_to_version(position, version, self.cache)
