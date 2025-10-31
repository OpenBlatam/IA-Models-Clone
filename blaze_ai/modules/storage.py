"""
Blaze AI Storage Module v7.2.0

This module provides ultra-compact storage capabilities with intelligent compression,
data deduplication, and efficient data management through the modular system.
"""

import asyncio
import logging
import hashlib
import json
import pickle
import zlib
import lz4.frame
import snappy
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import time
import threading

from .base import BaseModule, ModuleConfig, ModuleStatus

logger = logging.getLogger(__name__)

# ============================================================================
# STORAGE MODULE CONFIGURATION
# ============================================================================

class CompressionType(Enum):
    """Available compression types."""
    NONE = "none"
    ZLIB = "zlib"
    LZ4 = "lz4"
    SNAPPY = "snappy"
    AUTO = "auto"

class StorageStrategy(Enum):
    """Storage strategies for different data types."""
    MEMORY = "memory"
    DISK = "disk"
    HYBRID = "hybrid"
    DISTRIBUTED = "distributed"

class StorageModuleConfig(ModuleConfig):
    """Configuration for the Storage Module."""
    
    def __init__(self, **kwargs):
        super().__init__(
            name="storage",
            module_type="STORAGE",
            priority=2,  # High priority for data management
            **kwargs
        )
        
        # Storage-specific configurations
        self.storage_path: str = kwargs.get("storage_path", "./blaze_storage")
        self.max_memory_size: int = kwargs.get("max_memory_size", 1024 * 1024 * 1024)  # 1GB
        self.max_disk_size: int = kwargs.get("max_disk_size", 10 * 1024 * 1024 * 1024)  # 10GB
        self.default_compression: CompressionType = kwargs.get("default_compression", CompressionType.AUTO)
        self.storage_strategy: StorageStrategy = kwargs.get("storage_strategy", StorageStrategy.HYBRID)
        self.enable_deduplication: bool = kwargs.get("enable_deduplication", True)
        self.enable_encryption: bool = kwargs.get("enable_encryption", False)
        self.encryption_key: Optional[str] = kwargs.get("encryption_key")
        self.cleanup_interval: float = kwargs.get("cleanup_interval", 300.0)  # 5 minutes
        self.max_file_age: float = kwargs.get("max_file_age", 86400 * 7)  # 7 days

class StorageMetrics:
    """Metrics specific to storage operations."""
    
    def __init__(self):
        self.total_stored: int = 0
        self.total_retrieved: int = 0
        self.total_deleted: int = 0
        self.memory_usage: int = 0
        self.disk_usage: int = 0
        self.compression_ratio: float = 1.0
        self.deduplication_savings: int = 0
        self.encryption_overhead: float = 0.0
        self.average_store_time: float = 0.0
        self.average_retrieve_time: float = 0.0

# ============================================================================
# STORAGE IMPLEMENTATIONS
# ============================================================================

class CompressionManager:
    """Manages data compression and decompression."""
    
    @staticmethod
    def compress(data: bytes, compression_type: CompressionType) -> Tuple[bytes, CompressionType]:
        """Compress data using the specified method."""
        if compression_type == CompressionType.AUTO:
            # Auto-select best compression
            compression_type = CompressionManager._select_best_compression(data)
        
        try:
            if compression_type == CompressionType.ZLIB:
                compressed = zlib.compress(data, level=9)
            elif compression_type == CompressionType.LZ4:
                compressed = lz4.frame.compress(data, compression_level=9)
            elif compression_type == CompressionType.SNAPPY:
                compressed = snappy.compress(data)
            else:
                compressed = data
            
            return compressed, compression_type
            
        except Exception as e:
            logger.warning(f"Compression failed with {compression_type}, falling back to none: {e}")
            return data, CompressionType.NONE
    
    @staticmethod
    def decompress(data: bytes, compression_type: CompressionType) -> bytes:
        """Decompress data using the specified method."""
        try:
            if compression_type == CompressionType.ZLIB:
                return zlib.decompress(data)
            elif compression_type == CompressionType.LZ4:
                return lz4.frame.decompress(data)
            elif compression_type == CompressionType.SNAPPY:
                return snappy.decompress(data)
            else:
                return data
                
        except Exception as e:
            logger.error(f"Decompression failed with {compression_type}: {e}")
            raise
    
    @staticmethod
    def _select_best_compression(data: bytes) -> CompressionType:
        """Select the best compression method for the data."""
        if len(data) < 1024:  # Small data, no compression
            return CompressionType.NONE
        
        # Test different compression methods
        try:
            zlib_size = len(zlib.compress(data, level=9))
            lz4_size = len(lz4.frame.compress(data, compression_level=9))
            snappy_size = len(snappy.compress(data))
            
            sizes = [
                (zlib_size, CompressionType.ZLIB),
                (lz4_size, CompressionType.LZ4),
                (snappy_size, CompressionType.SNAPPY)
            ]
            
            # Return the method with smallest compressed size
            return min(sizes, key=lambda x: x[0])[1]
            
        except Exception:
            return CompressionType.NONE

class DataDeduplicator:
    """Handles data deduplication to save storage space."""
    
    def __init__(self):
        self.content_hashes: Dict[str, str] = {}  # hash -> key
        self.duplicate_count: int = 0
        self.saved_bytes: int = 0
    
    def add_content(self, key: str, content: bytes) -> bool:
        """Add content and check for duplicates."""
        content_hash = hashlib.sha256(content).hexdigest()
        
        if content_hash in self.content_hashes:
            # Duplicate found
            self.duplicate_count += 1
            self.saved_bytes += len(content)
            return True
        else:
            # New content
            self.content_hashes[content_hash] = key
            return False
    
    def get_duplicate_key(self, content: bytes) -> Optional[str]:
        """Get the key of duplicate content if it exists."""
        content_hash = hashlib.sha256(content).hexdigest()
        return self.content_hashes.get(content_hash)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get deduplication statistics."""
        return {
            "duplicate_count": self.duplicate_count,
            "saved_bytes": self.saved_bytes,
            "unique_content_count": len(self.content_hashes)
        }

class StorageEntry:
    """Represents a storage entry with metadata."""
    
    def __init__(self, key: str, data: Any, compression_type: CompressionType):
        self.key = key
        self.original_data = data
        self.compression_type = compression_type
        self.created_at = time.time()
        self.last_accessed = time.time()
        self.access_count = 0
        self.size = 0
        self.compressed_size = 0
        self._compressed_data = None
    
    def compress_data(self) -> bytes:
        """Compress the data and return compressed bytes."""
        if self._compressed_data is None:
            # Serialize data
            if isinstance(self.original_data, (dict, list, str, int, float, bool)):
                serialized = json.dumps(self.original_data).encode('utf-8')
            else:
                serialized = pickle.dumps(self.original_data)
            
            self.size = len(serialized)
            
            # Compress
            compressed, actual_compression = CompressionManager.compress(serialized, self.compression_type)
            self.compression_type = actual_compression
            self.compressed_size = len(compressed)
            self._compressed_data = compressed
        
        return self._compressed_data
    
    def decompress_data(self) -> Any:
        """Decompress and return the original data."""
        if self._compressed_data is None:
            return self.original_data
        
        # Decompress
        decompressed = CompressionManager.decompress(self._compressed_data, self.compression_type)
        
        # Deserialize
        try:
            if self.compression_type == CompressionType.NONE:
                # Try to deserialize as JSON first
                try:
                    return json.loads(decompressed.decode('utf-8'))
                except:
                    return pickle.loads(decompressed)
            else:
                # For compressed data, assume it was serialized with pickle
                return pickle.loads(decompressed)
        except Exception as e:
            logger.error(f"Failed to deserialize data for key {self.key}: {e}")
            return self.original_data
    
    def update_access(self):
        """Update access statistics."""
        self.last_accessed = time.time()
        self.access_count += 1

# ============================================================================
# STORAGE MODULE IMPLEMENTATION
# ============================================================================

class StorageModule(BaseModule):
    """
    Storage Module - Provides ultra-compact storage with intelligent compression.
    
    This module provides:
    - Intelligent data compression
    - Data deduplication
    - Memory and disk storage
    - Automatic cleanup and optimization
    - Encryption support
    """
    
    def __init__(self, config: StorageModuleConfig):
        super().__init__(config)
        self.storage_path = Path(config.storage_path)
        self.memory_storage: Dict[str, StorageEntry] = {}
        self.disk_storage: Dict[str, StorageEntry] = {}
        self.storage_metrics = StorageMetrics()
        self.deduplicator = DataDeduplicator()
        self.cleanup_task: Optional[asyncio.Task] = None
        self._lock = threading.RLock()
        
    async def initialize(self) -> bool:
        """Initialize the Storage Module."""
        try:
            logger.info("Initializing Storage Module...")
            
            # Create storage directory
            self.storage_path.mkdir(parents=True, exist_ok=True)
            
            # Start cleanup task
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            self.status = ModuleStatus.ACTIVE
            logger.info("Storage Module initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Storage Module: {e}")
            self.status = ModuleStatus.ERROR
            return False
    
    async def shutdown(self) -> bool:
        """Shutdown the Storage Module."""
        try:
            logger.info("Shutting down Storage Module...")
            
            # Stop cleanup task
            if self.cleanup_task:
                self.cleanup_task.cancel()
                try:
                    await self.cleanup_task
                except asyncio.CancelledError:
                    pass
            
            # Save any remaining data to disk
            await self._persist_memory_data()
            
            self.status = ModuleStatus.SHUTDOWN
            logger.info("Storage Module shutdown successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error during Storage Module shutdown: {e}")
            return False
    
    async def store(self, key: str, data: Any, compression_type: Optional[CompressionType] = None) -> bool:
        """Store data with the specified key."""
        try:
            start_time = time.time()
            
            if compression_type is None:
                compression_type = self.config.default_compression
            
            # Check for duplicates if deduplication is enabled
            if self.config.enable_deduplication:
                if isinstance(data, (dict, list, str, int, float, bool)):
                    serialized = json.dumps(data).encode('utf-8')
                else:
                    serialized = pickle.dumps(data)
                
                duplicate_key = self.deduplicator.get_duplicate_key(serialized)
                if duplicate_key:
                    # Store reference to duplicate
                    with self._lock:
                        self.memory_storage[key] = StorageEntry(
                            key, {"duplicate_of": duplicate_key}, CompressionType.NONE
                        )
                    self.storage_metrics.total_stored += 1
                    return True
            
            # Create storage entry
            entry = StorageEntry(key, data, compression_type)
            
            # Determine storage location
            if self.config.storage_strategy == StorageStrategy.MEMORY:
                storage_target = self.memory_storage
            elif self.config.storage_strategy == StorageStrategy.DISK:
                storage_target = self.disk_storage
            else:  # HYBRID
                # Store in memory if small, otherwise on disk
                if len(str(data)) < 1024 * 1024:  # 1MB
                    storage_target = self.memory_storage
                else:
                    storage_target = self.disk_storage
            
            # Store the entry
            with self._lock:
                storage_target[key] = entry
                
                # Update metrics
                if storage_target == self.memory_storage:
                    self.storage_metrics.memory_usage += entry.size
                else:
                    self.storage_metrics.disk_usage += entry.size
            
            # Persist to disk if using disk storage
            if storage_target == self.disk_storage:
                await self._persist_entry_to_disk(entry)
            
            # Update metrics
            store_time = time.time() - start_time
            self.storage_metrics.total_stored += 1
            self.storage_metrics.average_store_time = (
                (self.storage_metrics.average_store_time * (self.storage_metrics.total_stored - 1) + store_time) /
                self.storage_metrics.total_stored
            )
            
            logger.info(f"Data stored successfully: {key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store data for key {key}: {e}")
            return False
    
    async def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve data by key."""
        try:
            start_time = time.time()
            
            # Check memory storage first
            entry = self.memory_storage.get(key)
            if entry:
                entry.update_access()
                data = entry.decompress_data()
                self.storage_metrics.total_retrieved += 1
                return data
            
            # Check disk storage
            entry = self.disk_storage.get(key)
            if entry:
                entry.update_access()
                data = entry.decompress_data()
                self.storage_metrics.total_retrieved += 1
                return data
            
            # Check for duplicate references
            entry = self.memory_storage.get(key)
            if entry and isinstance(entry.original_data, dict) and "duplicate_of" in entry.original_data:
                duplicate_key = entry.original_data["duplicate_of"]
                return await self.retrieve(duplicate_key)
            
            logger.warning(f"Key not found: {key}")
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve data for key {key}: {e}")
            return None
        finally:
            # Update metrics
            retrieve_time = time.time() - start_time
            if self.storage_metrics.total_retrieved > 0:
                self.storage_metrics.average_retrieve_time = (
                    (self.storage_metrics.average_retrieve_time * (self.storage_metrics.total_retrieved - 1) + retrieve_time) /
                    self.storage_metrics.total_retrieved
                )
    
    async def delete(self, key: str) -> bool:
        """Delete data by key."""
        try:
            deleted = False
            
            # Check memory storage
            if key in self.memory_storage:
                entry = self.memory_storage[key]
                self.storage_metrics.memory_usage -= entry.size
                del self.memory_storage[key]
                deleted = True
            
            # Check disk storage
            if key in self.disk_storage:
                entry = self.disk_storage[key]
                self.storage_metrics.disk_usage -= entry.size
                del self.disk_storage[key]
                deleted = True
            
            if deleted:
                self.storage_metrics.total_deleted += 1
                logger.info(f"Data deleted successfully: {key}")
                return True
            else:
                logger.warning(f"Key not found for deletion: {key}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete data for key {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if a key exists."""
        return key in self.memory_storage or key in self.disk_storage
    
    async def list_keys(self, pattern: Optional[str] = None) -> List[str]:
        """List all keys, optionally filtered by pattern."""
        all_keys = list(self.memory_storage.keys()) + list(self.disk_storage.keys())
        
        if pattern:
            import fnmatch
            return [key for key in all_keys if fnmatch.fnmatch(key, pattern)]
        
        return all_keys
    
    async def get_storage_info(self) -> Dict[str, Any]:
        """Get comprehensive storage information."""
        with self._lock:
            return {
                "memory_entries": len(self.memory_storage),
                "disk_entries": len(self.disk_storage),
                "total_entries": len(self.memory_storage) + len(self.disk_storage),
                "memory_usage_bytes": self.storage_metrics.memory_usage,
                "disk_usage_bytes": self.storage_metrics.disk_usage,
                "total_usage_bytes": self.storage_metrics.memory_usage + self.storage_metrics.disk_usage,
                "compression_ratio": self.storage_metrics.compression_ratio,
                "deduplication_stats": self.deduplicator.get_stats(),
                "storage_strategy": self.config.storage_strategy.value,
                "default_compression": self.config.default_compression.value
            }
    
    async def _persist_entry_to_disk(self, entry: StorageEntry):
        """Persist a storage entry to disk."""
        try:
            file_path = self.storage_path / f"{entry.key}.blaze"
            
            # Compress and serialize the entry
            compressed_data = entry.compress_data()
            
            # Save to disk
            with open(file_path, 'wb') as f:
                f.write(compressed_data)
                
        except Exception as e:
            logger.error(f"Failed to persist entry {entry.key} to disk: {e}")
    
    async def _persist_memory_data(self):
        """Persist all memory data to disk."""
        try:
            for entry in self.memory_storage.values():
                await self._persist_entry_to_disk(entry)
                
        except Exception as e:
            logger.error(f"Failed to persist memory data: {e}")
    
    async def _cleanup_loop(self):
        """Background cleanup loop."""
        while self.status == ModuleStatus.ACTIVE:
            try:
                await asyncio.sleep(self.config.cleanup_interval)
                await self._perform_cleanup()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    async def _perform_cleanup(self):
        """Perform storage cleanup operations."""
        try:
            current_time = time.time()
            
            # Clean up old entries
            for storage_dict in [self.memory_storage, self.disk_storage]:
                keys_to_remove = []
                
                for key, entry in storage_dict.items():
                    if current_time - entry.created_at > self.config.max_file_age:
                        keys_to_remove.append(key)
                
                for key in keys_to_remove:
                    await self.delete(key)
            
            # Update compression ratio
            total_original = sum(entry.size for entry in self.memory_storage.values()) + \
                           sum(entry.size for entry in self.disk_storage.values())
            total_compressed = sum(entry.compressed_size for entry in self.memory_storage.values()) + \
                             sum(entry.compressed_size for entry in self.disk_storage.values())
            
            if total_original > 0:
                self.storage_metrics.compression_ratio = total_compressed / total_original
            
            logger.debug("Storage cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get module metrics."""
        return {
            "module": "storage",
            "status": self.status.value,
            "storage_metrics": {
                "total_stored": self.storage_metrics.total_stored,
                "total_retrieved": self.storage_metrics.total_retrieved,
                "total_deleted": self.storage_metrics.total_deleted,
                "memory_usage": self.storage_metrics.memory_usage,
                "disk_usage": self.storage_metrics.disk_usage,
                "compression_ratio": self.storage_metrics.compression_ratio,
                "deduplication_savings": self.storage_metrics.deduplication_savings,
                "average_store_time": self.storage_metrics.average_store_time,
                "average_retrieve_time": self.storage_metrics.average_retrieve_time
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check module health."""
        try:
            health_status = "healthy"
            issues = []
            
            # Check storage directory
            if not self.storage_path.exists():
                health_status = "unhealthy"
                issues.append("Storage directory does not exist")
            
            # Check memory usage
            if self.storage_metrics.memory_usage > self.config.max_memory_size:
                health_status = "warning"
                issues.append(f"Memory usage exceeds limit: {self.storage_metrics.memory_usage} > {self.config.max_memory_size}")
            
            # Check disk usage
            if self.storage_metrics.disk_usage > self.config.max_disk_size:
                health_status = "warning"
                issues.append(f"Disk usage exceeds limit: {self.storage_metrics.disk_usage} > {self.config.max_disk_size}")
            
            return {
                "status": health_status,
                "issues": issues,
                "total_entries": len(self.memory_storage) + len(self.disk_storage),
                "memory_usage": self.storage_metrics.memory_usage,
                "disk_usage": self.storage_metrics.disk_usage,
                "uptime": self.get_uptime()
            }
            
        except Exception as e:
            return {
                "status": "error",
                "issues": [f"Health check failed: {e}"],
                "error": str(e)
            }

# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_storage_module(**kwargs) -> StorageModule:
    """Create a Storage Module instance."""
    config = StorageModuleConfig(**kwargs)
    return StorageModule(config)

def create_storage_module_with_defaults() -> StorageModule:
    """Create a Storage Module with default configurations."""
    return create_storage_module(
        storage_path="./blaze_storage",
        max_memory_size=512 * 1024 * 1024,  # 512MB
        max_disk_size=5 * 1024 * 1024 * 1024,  # 5GB
        default_compression=CompressionType.AUTO,
        storage_strategy=StorageStrategy.HYBRID,
        enable_deduplication=True,
        cleanup_interval=600.0  # 10 minutes
    )

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "StorageModule",
    "StorageModuleConfig",
    "StorageMetrics",
    "CompressionType",
    "StorageStrategy",
    "CompressionManager",
    "DataDeduplicator",
    "StorageEntry",
    "create_storage_module",
    "create_storage_module_with_defaults"
]
