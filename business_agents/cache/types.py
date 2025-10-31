"""
Cache Types and Definitions
===========================

Type definitions for advanced caching components.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union, Callable
from datetime import datetime, timedelta
import uuid

class CacheBackend(Enum):
    """Cache backend types."""
    REDIS = "redis"
    MEMCACHED = "memcached"
    MEMORY = "memory"
    DISK = "disk"
    CDN = "cdn"
    DATABASE = "database"

class CacheStrategy(Enum):
    """Cache strategy types."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In, First Out
    LIFO = "lifo"  # Last In, First Out
    TTL = "ttl"  # Time To Live
    WRITE_THROUGH = "write_through"
    WRITE_BACK = "write_back"
    WRITE_AROUND = "write_around"
    READ_THROUGH = "read_through"
    CACHE_ASIDE = "cache_aside"

class CachePolicy(Enum):
    """Cache eviction policies."""
    NO_EVICTION = "no_eviction"
    ALLKEYS_LRU = "allkeys_lru"
    ALLKEYS_LFU = "allkeys_lfu"
    ALLKEYS_RANDOM = "allkeys_random"
    VOLATILE_LRU = "volatile_lru"
    VOLATILE_LFU = "volatile_lfu"
    VOLATILE_TTL = "volatile_ttl"
    VOLATILE_RANDOM = "volatile_random"

@dataclass
class CacheKey:
    """Cache key definition."""
    key: str
    namespace: str = "default"
    tags: List[str] = field(default_factory=list)
    version: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    def get_full_key(self) -> str:
        """Get full cache key."""
        parts = [self.namespace, self.key]
        if self.version:
            parts.append(self.version)
        return ":".join(parts)

@dataclass
class CacheValue:
    """Cache value definition."""
    data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    compressed: bool = False
    serialized: bool = False
    checksum: Optional[str] = None

@dataclass
class CacheEntry:
    """Cache entry definition."""
    key: CacheKey
    value: CacheValue
    ttl: Optional[timedelta] = None
    created_at: datetime = field(default_factory=datetime.now)
    accessed_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    size_bytes: int = 0
    priority: int = 0
    tags: List[str] = field(default_factory=list)
    
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if not self.ttl:
            return False
        return datetime.now() - self.created_at > self.ttl
    
    def update_access(self):
        """Update access information."""
        self.accessed_at = datetime.now()
        self.access_count += 1

@dataclass
class CacheStats:
    """Cache statistics."""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    evictions: int = 0
    total_size_bytes: int = 0
    entry_count: int = 0
    hit_rate: float = 0.0
    miss_rate: float = 0.0
    last_reset: datetime = field(default_factory=datetime.now)
    
    def calculate_rates(self):
        """Calculate hit and miss rates."""
        total = self.hits + self.misses
        if total > 0:
            self.hit_rate = self.hits / total
            self.miss_rate = self.misses / total

@dataclass
class CacheMetrics:
    """Cache performance metrics."""
    response_time_ms: float = 0.0
    throughput_ops_per_sec: float = 0.0
    memory_usage_percent: float = 0.0
    cpu_usage_percent: float = 0.0
    network_io_bytes: int = 0
    disk_io_bytes: int = 0
    error_rate: float = 0.0
    connection_count: int = 0
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class CacheTier:
    """Cache tier definition."""
    name: str
    backend: CacheBackend
    capacity_mb: int
    ttl_seconds: int
    strategy: CacheStrategy
    policy: CachePolicy
    priority: int = 0
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ClusterNode:
    """Cache cluster node definition."""
    id: str
    host: str
    port: int
    role: str = "master"  # master, slave, sentinel
    status: str = "online"  # online, offline, maintenance
    weight: int = 1
    slots: List[int] = field(default_factory=list)
    last_heartbeat: datetime = field(default_factory=datetime.now)
    metrics: CacheMetrics = field(default_factory=CacheMetrics)

@dataclass
class ClusterConfig:
    """Cache cluster configuration."""
    name: str
    nodes: List[ClusterNode] = field(default_factory=list)
    replication_factor: int = 1
    sharding_strategy: str = "consistent_hash"
    failover_strategy: str = "automatic"
    health_check_interval: int = 30
    max_retries: int = 3
    timeout_seconds: int = 5

@dataclass
class CDNProvider:
    """CDN provider definition."""
    name: str
    provider_type: str  # aws_cloudfront, cloudflare, azure_cdn, etc.
    endpoint: str
    api_key: str
    region: str
    config: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CDNConfig:
    """CDN configuration."""
    provider: CDNProvider
    cache_ttl: int = 3600
    compression: bool = True
    gzip: bool = True
    brotli: bool = True
    cache_headers: Dict[str, str] = field(default_factory=dict)
    purge_on_update: bool = True
    edge_locations: List[str] = field(default_factory=list)

@dataclass
class CacheNode:
    """Distributed cache node."""
    id: str
    address: str
    port: int
    role: str = "member"
    status: str = "active"
    capacity_gb: float = 1.0
    used_gb: float = 0.0
    last_seen: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ReplicationStrategy:
    """Cache replication strategy."""
    strategy_type: str  # master_slave, master_master, ring
    replication_factor: int = 2
    consistency_level: str = "eventual"  # strong, eventual, weak
    conflict_resolution: str = "last_write_wins"
    sync_interval: int = 60
    async_replication: bool = True

@dataclass
class CacheWarmup:
    """Cache warmup configuration."""
    enabled: bool = True
    strategies: List[str] = field(default_factory=lambda: ["preload", "lazy_load"])
    preload_keys: List[str] = field(default_factory=list)
    preload_patterns: List[str] = field(default_factory=list)
    warmup_interval: int = 3600
    batch_size: int = 100

@dataclass
class CacheInvalidation:
    """Cache invalidation configuration."""
    strategy: str = "tag_based"  # tag_based, key_based, pattern_based
    invalidation_events: List[str] = field(default_factory=list)
    cascade_invalidation: bool = True
    invalidation_timeout: int = 30
    retry_attempts: int = 3

@dataclass
class CacheCompression:
    """Cache compression configuration."""
    enabled: bool = True
    algorithm: str = "gzip"  # gzip, lz4, snappy, zstd
    compression_level: int = 6
    min_size_bytes: int = 1024
    max_size_bytes: int = 10485760  # 10MB

@dataclass
class CacheEncryption:
    """Cache encryption configuration."""
    enabled: bool = False
    algorithm: str = "AES-256-GCM"
    key_rotation_interval: int = 86400  # 24 hours
    encrypt_at_rest: bool = True
    encrypt_in_transit: bool = True

@dataclass
class CacheMonitoring:
    """Cache monitoring configuration."""
    enabled: bool = True
    metrics_interval: int = 60
    alert_thresholds: Dict[str, float] = field(default_factory=dict)
    health_checks: List[str] = field(default_factory=lambda: ["connectivity", "performance", "capacity"])
    logging_level: str = "INFO"

@dataclass
class AdvancedCacheConfig:
    """Advanced cache configuration."""
    tiers: List[CacheTier] = field(default_factory=list)
    cluster: Optional[ClusterConfig] = None
    cdn: Optional[CDNConfig] = None
    replication: Optional[ReplicationStrategy] = None
    warmup: CacheWarmup = field(default_factory=CacheWarmup)
    invalidation: CacheInvalidation = field(default_factory=CacheInvalidation)
    compression: CacheCompression = field(default_factory=CacheCompression)
    encryption: CacheEncryption = field(default_factory=CacheEncryption)
    monitoring: CacheMonitoring = field(default_factory=CacheMonitoring)
