from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import asyncio
import time
import logging
import json
import pickle
import sqlite3
import threading
import weakref
import gc
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager
from collections import defaultdict, OrderedDict
from pathlib import Path
import hashlib
from datetime import datetime, timedelta
import signal
import sys
from concurrent.futures import ThreadPoolExecutor
import functools
import importlib
import importlib.util
import sys
import os
import dns.resolver
import dns.exception
import aiohttp
import asyncio
import aiofiles
import sqlite3
import redis
import psutil
from typing import Any, List, Dict, Optional
"""
Lazy Loading and Caching Examples

This module provides comprehensive examples for lazy-loading heavy modules and caching
DNS lookups and vulnerability database queries to improve performance and reduce
redundant network requests.

Key Features:
- Lazy loading of heavy modules (exploit databases, ML models, etc.)
- DNS lookup caching with TTL management
- Vulnerability database query caching
- Memory-efficient caching strategies
- Cache invalidation and cleanup
- Performance monitoring and metrics
- Thread-safe caching operations
"""


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CacheType(Enum):
    """Cache types for different use cases"""
    MEMORY = "memory"
    DISK = "disk"
    REDIS = "redis"
    HYBRID = "hybrid"


class LazyLoadStrategy(Enum):
    """Lazy loading strategies"""
    ON_DEMAND = "on_demand"
    BACKGROUND = "background"
    PRELOAD = "preload"
    CONDITIONAL = "conditional"


@dataclass
class CacheConfig:
    """Configuration for caching systems"""
    cache_type: CacheType = CacheType.MEMORY
    max_size: int = 1000
    ttl_seconds: int = 3600  # 1 hour
    cleanup_interval: int = 300  # 5 minutes
    enable_compression: bool = True
    enable_persistence: bool = True
    cache_dir: str = "cache"
    redis_url: Optional[str] = None
    enable_metrics: bool = True
    thread_safe: bool = True
    max_memory_mb: int = 512


@dataclass
class LazyLoadConfig:
    """Configuration for lazy loading"""
    strategy: LazyLoadStrategy = LazyLoadStrategy.ON_DEMAND
    preload_modules: List[str] = field(default_factory=list)
    background_workers: int = 2
    load_timeout: float = 30.0
    enable_monitoring: bool = True
    cache_loaded_modules: bool = True
    memory_limit_mb: int = 256


class LazyModuleLoader:
    """Lazy loading system for heavy modules"""
    
    def __init__(self, config: LazyLoadConfig):
        
    """__init__ function."""
self.config = config
        self._loaded_modules: Dict[str, Any] = {}
        self._loading_modules: Dict[str, asyncio.Task] = {}
        self._module_metadata: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock() if config.cache_loaded_modules else None
        self._background_queue: asyncio.Queue = asyncio.Queue()
        self._background_workers: List[asyncio.Task] = []
        self._monitoring_data: Dict[str, Any] = {
            "load_times": {},
            "memory_usage": {},
            "load_count": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        if config.strategy == LazyLoadStrategy.BACKGROUND:
            self._start_background_workers()
    
    async def __aenter__(self) -> Any:
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> Any:
        """Async context manager exit"""
        await self.cleanup()
    
    def _start_background_workers(self) -> Any:
        """Start background workers for preloading"""
        for i in range(self.config.background_workers):
            worker = asyncio.create_task(self._background_worker(f"worker-{i}"))
            self._background_workers.append(worker)
    
    async def _background_worker(self, worker_id: str):
        """Background worker for preloading modules"""
        logger.debug(f"Background worker {worker_id} started")
        
        while True:
            try:
                module_name = await asyncio.wait_for(
                    self._background_queue.get(),
                    timeout=1.0
                )
                
                if module_name is None:  # Shutdown signal
                    break
                
                await self._load_module_internal(module_name)
                self._background_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Background worker {worker_id} error: {e}")
                continue
        
        logger.debug(f"Background worker {worker_id} stopped")
    
    async def load_module(self, module_name: str, force_reload: bool = False) -> Any:
        """Load a module lazily"""
        if not module_name:
            raise ValueError("Module name cannot be empty")
        
        # Check if already loaded
        if not force_reload and module_name in self._loaded_modules:
            self._monitoring_data["cache_hits"] += 1
            return self._loaded_modules[module_name]
        
        self._monitoring_data["cache_misses"] += 1
        
        # Check if currently loading
        if module_name in self._loading_modules:
            try:
                await self._loading_modules[module_name]
                return self._loaded_modules[module_name]
            except Exception as e:
                logger.error(f"Module {module_name} loading failed: {e}")
                del self._loading_modules[module_name]
        
        # Load module based on strategy
        if self.config.strategy == LazyLoadStrategy.BACKGROUND:
            return await self._load_module_background(module_name)
        else:
            return await self._load_module_internal(module_name)
    
    async def _load_module_internal(self, module_name: str) -> Any:
        """Internal module loading logic"""
        start_time = time.time()
        
        try:
            # Check memory usage
            if self._is_memory_limit_exceeded():
                await self._cleanup_memory()
            
            # Load module
            module = await self._import_module(module_name)
            
            # Store module
            if self.config.cache_loaded_modules:
                async with self._lock:
                    self._loaded_modules[module_name] = module
                    self._module_metadata[module_name] = {
                        "load_time": time.time(),
                        "memory_usage": self._get_memory_usage(),
                        "size": self._estimate_module_size(module)
                    }
            
            # Update monitoring
            load_time = time.time() - start_time
            self._monitoring_data["load_times"][module_name] = load_time
            self._monitoring_data["load_count"] += 1
            
            logger.info(f"Module {module_name} loaded in {load_time:.3f}s")
            return module
            
        except Exception as e:
            logger.error(f"Failed to load module {module_name}: {e}")
            raise
    
    async def _load_module_background(self, module_name: str) -> Any:
        """Load module in background"""
        # Add to background queue
        await self._background_queue.put(module_name)
        
        # Create loading task
        loading_task = asyncio.create_task(self._load_module_internal(module_name))
        self._loading_modules[module_name] = loading_task
        
        try:
            await asyncio.wait_for(loading_task, timeout=self.config.load_timeout)
            return self._loaded_modules[module_name]
        except asyncio.TimeoutError:
            logger.error(f"Module {module_name} loading timed out")
            raise TimeoutError(f"Module {module_name} loading timed out")
    
    async def _import_module(self, module_name: str) -> Any:
        """Import a module with error handling"""
        try:
            # Try standard import first
            module = importlib.import_module(module_name)
            return module
        except ImportError:
            # Try alternative import methods
            module = await self._import_module_alternative(module_name)
            return module
    
    async def _import_module_alternative(self, module_name: str) -> Any:
        """Alternative module import methods"""
        # Check if it's a file path
        if os.path.exists(module_name):
            spec = importlib.util.spec_from_file_location("module", module_name)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module
        
        # Try with different extensions
        for ext in ['.py', '.pyc', '.so', '.dll']:
            try:
                full_name = module_name + ext
                if os.path.exists(full_name):
                    spec = importlib.util.spec_from_file_location("module", full_name)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        return module
            except Exception:
                continue
        
        raise ImportError(f"Could not import module: {module_name}")
    
    def _is_memory_limit_exceeded(self) -> bool:
        """Check if memory limit is exceeded"""
        if not self.config.enable_monitoring:
            return False
        
        current_memory = self._get_memory_usage()
        return current_memory > self.config.memory_limit_mb
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / 1024 / 1024
        except Exception:
            return 0.0
    
    def _estimate_module_size(self, module: Any) -> int:
        """Estimate module size in bytes"""
        try:
            return sys.getsizeof(module)
        except Exception:
            return 0
    
    async def _cleanup_memory(self) -> Any:
        """Clean up memory by unloading least used modules"""
        if not self.config.cache_loaded_modules:
            return
        
        async with self._lock:
            # Sort modules by last access time
            sorted_modules = sorted(
                self._module_metadata.items(),
                key=lambda x: x[1].get("load_time", 0)
            )
            
            # Remove oldest modules until memory usage is acceptable
            for module_name, metadata in sorted_modules:
                if self._is_memory_limit_exceeded():
                    del self._loaded_modules[module_name]
                    del self._module_metadata[module_name]
                    logger.info(f"Unloaded module {module_name} to free memory")
                else:
                    break
    
    async def preload_modules(self, module_names: List[str]):
        """Preload specified modules"""
        if not module_names:
            return
        
        logger.info(f"Preloading {len(module_names)} modules")
        
        for module_name in module_names:
            try:
                await self.load_module(module_name)
            except Exception as e:
                logger.error(f"Failed to preload {module_name}: {e}")
    
    def get_loaded_modules(self) -> List[str]:
        """Get list of loaded modules"""
        return list(self._loaded_modules.keys())
    
    def get_monitoring_data(self) -> Dict[str, Any]:
        """Get monitoring data"""
        return {
            **self._monitoring_data,
            "loaded_modules": len(self._loaded_modules),
            "loading_modules": len(self._loading_modules),
            "current_memory_mb": self._get_memory_usage()
        }
    
    async def cleanup(self) -> Any:
        """Clean up resources"""
        # Stop background workers
        for _ in self._background_workers:
            await self._background_queue.put(None)
        
        if self._background_workers:
            await asyncio.gather(*self._background_workers, return_exceptions=True)
        
        # Clear loaded modules
        self._loaded_modules.clear()
        self._module_metadata.clear()
        self._loading_modules.clear()
        
        logger.info("Lazy module loader cleaned up")


class DNSCache:
    """DNS lookup caching system"""
    
    def __init__(self, config: CacheConfig):
        
    """__init__ function."""
self.config = config
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock() if config.thread_safe else None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "total_queries": 0,
            "failed_queries": 0
        }
        
        if config.cache_type == CacheType.DISK:
            self._init_disk_cache()
        elif config.cache_type == CacheType.REDIS:
            self._init_redis_cache()
    
    def _init_disk_cache(self) -> Any:
        """Initialize disk cache"""
        cache_dir = Path(self.config.cache_dir)
        cache_dir.mkdir(exist_ok=True)
        self._cache_file = cache_dir / "dns_cache.db"
        
        # Initialize SQLite database
        with sqlite3.connect(self._cache_file) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS dns_cache (
                    hostname TEXT PRIMARY KEY,
                    ip_addresses TEXT,
                    record_type TEXT,
                    ttl INTEGER,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
    
    def _init_redis_cache(self) -> Any:
        """Initialize Redis cache"""
        if not self.config.redis_url:
            raise ValueError("Redis URL required for Redis cache")
        
        self._redis_client = redis.from_url(self.config.redis_url)
    
    async def resolve(self, hostname: str, record_type: str = "A") -> List[str]:
        """Resolve hostname with caching"""
        if not hostname:
            raise ValueError("Hostname cannot be empty")
        
        self._stats["total_queries"] += 1
        
        # Check cache first
        cached_result = await self._get_cached_result(hostname, record_type)
        if cached_result:
            self._stats["cache_hits"] += 1
            return cached_result
        
        self._stats["cache_misses"] += 1
        
        # Perform DNS lookup
        try:
            result = await self._perform_dns_lookup(hostname, record_type)
            await self._cache_result(hostname, record_type, result)
            return result
        except Exception as e:
            self._stats["failed_queries"] += 1
            logger.error(f"DNS lookup failed for {hostname}: {e}")
            raise
    
    async def _get_cached_result(self, hostname: str, record_type: str) -> Optional[List[str]]:
        """Get cached DNS result"""
        cache_key = f"{hostname}:{record_type}"
        
        if self.config.cache_type == CacheType.MEMORY:
            return self._get_memory_cached_result(cache_key)
        elif self.config.cache_type == CacheType.DISK:
            return await self._get_disk_cached_result(cache_key)
        elif self.config.cache_type == CacheType.REDIS:
            return await self._get_redis_cached_result(cache_key)
        else:
            return None
    
    def _get_memory_cached_result(self, cache_key: str) -> Optional[List[str]]:
        """Get result from memory cache"""
        if self._lock:
            with self._lock:
                return self._get_memory_cached_result_internal(cache_key)
        else:
            return self._get_memory_cached_result_internal(cache_key)
    
    def _get_memory_cached_result_internal(self, cache_key: str) -> Optional[List[str]]:
        """Internal memory cache lookup"""
        if cache_key not in self._cache:
            return None
        
        cache_entry = self._cache[cache_key]
        if self._is_cache_entry_expired(cache_entry):
            del self._cache[cache_key]
            return None
        
        return cache_entry["result"]
    
    async def _get_disk_cached_result(self, cache_key: str) -> Optional[List[str]]:
        """Get result from disk cache"""
        try:
            with sqlite3.connect(self._cache_file) as conn:
                cursor = conn.execute(
                    "SELECT ip_addresses, ttl, timestamp FROM dns_cache WHERE hostname = ?",
                    (cache_key,)
                )
                row = cursor.fetchone()
                
                if not row:
                    return None
                
                ip_addresses, ttl, timestamp = row
                
                # Check if expired
                cache_time = datetime.fromisoformat(timestamp)
                if datetime.now() - cache_time > timedelta(seconds=ttl):
                    # Remove expired entry
                    conn.execute("DELETE FROM dns_cache WHERE hostname = ?", (cache_key,))
                    conn.commit()
                    return None
                
                return json.loads(ip_addresses)
        except Exception as e:
            logger.error(f"Failed to get disk cached result: {e}")
            return None
    
    async def _get_redis_cached_result(self, cache_key: str) -> Optional[List[str]]:
        """Get result from Redis cache"""
        try:
            result = self._redis_client.get(cache_key)
            if result:
                return json.loads(result)
            return None
        except Exception as e:
            logger.error(f"Failed to get Redis cached result: {e}")
            return None
    
    async def _perform_dns_lookup(self, hostname: str, record_type: str) -> List[str]:
        """Perform actual DNS lookup"""
        loop = asyncio.get_event_loop()
        
        def dns_lookup():
            
    """dns_lookup function."""
try:
                resolver = dns.resolver.Resolver()
                resolver.timeout = 5.0
                resolver.lifetime = 10.0
                
                answers = resolver.resolve(hostname, record_type)
                return [str(answer) for answer in answers]
            except dns.exception.DNSException as e:
                raise Exception(f"DNS resolution failed: {e}")
        
        return await loop.run_in_executor(None, dns_lookup)
    
    async def _cache_result(self, hostname: str, record_type: str, result: List[str]):
        """Cache DNS result"""
        cache_key = f"{hostname}:{record_type}"
        cache_entry = {
            "result": result,
            "timestamp": datetime.now(),
            "ttl": self.config.ttl_seconds
        }
        
        if self.config.cache_type == CacheType.MEMORY:
            self._cache_memory_result(cache_key, cache_entry)
        elif self.config.cache_type == CacheType.DISK:
            await self._cache_disk_result(cache_key, cache_entry)
        elif self.config.cache_type == CacheType.REDIS:
            await self._cache_redis_result(cache_key, cache_entry)
    
    def _cache_memory_result(self, cache_key: str, cache_entry: Dict[str, Any]):
        """Cache result in memory"""
        if self._lock:
            with self._lock:
                self._cache_memory_result_internal(cache_key, cache_entry)
        else:
            self._cache_memory_result_internal(cache_key, cache_entry)
    
    def _cache_memory_result_internal(self, cache_key: str, cache_entry: Dict[str, Any]):
        """Internal memory cache storage"""
        # Implement LRU eviction if cache is full
        if len(self._cache) >= self.config.max_size:
            self._evict_oldest_entry()
        
        self._cache[cache_key] = cache_entry
    
    async def _cache_disk_result(self, cache_key: str, cache_entry: Dict[str, Any]):
        """Cache result on disk"""
        try:
            with sqlite3.connect(self._cache_file) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO dns_cache 
                    (hostname, ip_addresses, record_type, ttl, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    cache_key,
                    json.dumps(cache_entry["result"]),
                    "A",  # Default record type
                    cache_entry["ttl"],
                    cache_entry["timestamp"].isoformat()
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to cache disk result: {e}")
    
    async def _cache_redis_result(self, cache_key: str, cache_entry: Dict[str, Any]):
        """Cache result in Redis"""
        try:
            self._redis_client.setex(
                cache_key,
                cache_entry["ttl"],
                json.dumps(cache_entry["result"])
            )
        except Exception as e:
            logger.error(f"Failed to cache Redis result: {e}")
    
    def _is_cache_entry_expired(self, cache_entry: Dict[str, Any]) -> bool:
        """Check if cache entry is expired"""
        timestamp = cache_entry["timestamp"]
        ttl = cache_entry["ttl"]
        return datetime.now() - timestamp > timedelta(seconds=ttl)
    
    def _evict_oldest_entry(self) -> Any:
        """Evict oldest cache entry"""
        if not self._cache:
            return
        
        oldest_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k]["timestamp"]
        )
        del self._cache[oldest_key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            **self._stats,
            "cache_size": len(self._cache),
            "hit_rate": self._stats["cache_hits"] / max(self._stats["total_queries"], 1)
        }
    
    async def cleanup(self) -> Any:
        """Clean up expired entries"""
        if self.config.cache_type == CacheType.MEMORY:
            self._cleanup_memory_cache()
        elif self.config.cache_type == CacheType.DISK:
            await self._cleanup_disk_cache()
        elif self.config.cache_type == CacheType.REDIS:
            await self._cleanup_redis_cache()
    
    def _cleanup_memory_cache(self) -> Any:
        """Clean up memory cache"""
        if self._lock:
            with self._lock:
                self._cleanup_memory_cache_internal()
        else:
            self._cleanup_memory_cache_internal()
    
    def _cleanup_memory_cache_internal(self) -> Any:
        """Internal memory cache cleanup"""
        expired_keys = [
            key for key, entry in self._cache.items()
            if self._is_cache_entry_expired(entry)
        ]
        
        for key in expired_keys:
            del self._cache[key]
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired DNS cache entries")
    
    async def _cleanup_disk_cache(self) -> Any:
        """Clean up disk cache"""
        try:
            with sqlite3.connect(self._cache_file) as conn:
                conn.execute("""
                    DELETE FROM dns_cache 
                    WHERE datetime(timestamp) < datetime('now', '-1 hour')
                """)
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to cleanup disk cache: {e}")
    
    async def _cleanup_redis_cache(self) -> Any:
        """Clean up Redis cache"""
        # Redis handles TTL automatically, no manual cleanup needed
        pass


class VulnerabilityDBCache:
    """Vulnerability database query caching system"""
    
    def __init__(self, config: CacheConfig):
        
    """__init__ function."""
self.config = config
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock() if config.thread_safe else None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "total_queries": 0,
            "failed_queries": 0
        }
        
        if config.cache_type == CacheType.DISK:
            self._init_disk_cache()
        elif config.cache_type == CacheType.REDIS:
            self._init_redis_cache()
    
    def _init_disk_cache(self) -> Any:
        """Initialize disk cache"""
        cache_dir = Path(self.config.cache_dir)
        cache_dir.mkdir(exist_ok=True)
        self._cache_file = cache_dir / "vuln_cache.db"
        
        # Initialize SQLite database
        with sqlite3.connect(self._cache_file) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS vuln_cache (
                    query_hash TEXT PRIMARY KEY,
                    query_params TEXT,
                    result_data TEXT,
                    ttl INTEGER,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
    
    def _init_redis_cache(self) -> Any:
        """Initialize Redis cache"""
        if not self.config.redis_url:
            raise ValueError("Redis URL required for Redis cache")
        
        self._redis_client = redis.from_url(self.config.redis_url)
    
    async def query_vulnerability_db(self, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """Query vulnerability database with caching"""
        if not query_params:
            raise ValueError("Query parameters cannot be empty")
        
        self._stats["total_queries"] += 1
        
        # Generate query hash
        query_hash = self._generate_query_hash(query_params)
        
        # Check cache first
        cached_result = await self._get_cached_result(query_hash)
        if cached_result:
            self._stats["cache_hits"] += 1
            return cached_result
        
        self._stats["cache_misses"] += 1
        
        # Perform actual query
        try:
            result = await self._perform_vuln_query(query_params)
            await self._cache_result(query_hash, query_params, result)
            return result
        except Exception as e:
            self._stats["failed_queries"] += 1
            logger.error(f"Vulnerability query failed: {e}")
            raise
    
    def _generate_query_hash(self, query_params: Dict[str, Any]) -> str:
        """Generate hash for query parameters"""
        query_str = json.dumps(query_params, sort_keys=True)
        return hashlib.md5(query_str.encode()).hexdigest()
    
    async def _get_cached_result(self, query_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached vulnerability query result"""
        if self.config.cache_type == CacheType.MEMORY:
            return self._get_memory_cached_result(query_hash)
        elif self.config.cache_type == CacheType.DISK:
            return await self._get_disk_cached_result(query_hash)
        elif self.config.cache_type == CacheType.REDIS:
            return await self._get_redis_cached_result(query_hash)
        else:
            return None
    
    def _get_memory_cached_result(self, query_hash: str) -> Optional[Dict[str, Any]]:
        """Get result from memory cache"""
        if self._lock:
            with self._lock:
                return self._get_memory_cached_result_internal(query_hash)
        else:
            return self._get_memory_cached_result_internal(query_hash)
    
    def _get_memory_cached_result_internal(self, query_hash: str) -> Optional[Dict[str, Any]]:
        """Internal memory cache lookup"""
        if query_hash not in self._cache:
            return None
        
        cache_entry = self._cache[query_hash]
        if self._is_cache_entry_expired(cache_entry):
            del self._cache[query_hash]
            return None
        
        return cache_entry["result"]
    
    async def _get_disk_cached_result(self, query_hash: str) -> Optional[Dict[str, Any]]:
        """Get result from disk cache"""
        try:
            with sqlite3.connect(self._cache_file) as conn:
                cursor = conn.execute(
                    "SELECT result_data, ttl, timestamp FROM vuln_cache WHERE query_hash = ?",
                    (query_hash,)
                )
                row = cursor.fetchone()
                
                if not row:
                    return None
                
                result_data, ttl, timestamp = row
                
                # Check if expired
                cache_time = datetime.fromisoformat(timestamp)
                if datetime.now() - cache_time > timedelta(seconds=ttl):
                    # Remove expired entry
                    conn.execute("DELETE FROM vuln_cache WHERE query_hash = ?", (query_hash,))
                    conn.commit()
                    return None
                
                return json.loads(result_data)
        except Exception as e:
            logger.error(f"Failed to get disk cached result: {e}")
            return None
    
    async def _get_redis_cached_result(self, query_hash: str) -> Optional[Dict[str, Any]]:
        """Get result from Redis cache"""
        try:
            result = self._redis_client.get(query_hash)
            if result:
                return json.loads(result)
            return None
        except Exception as e:
            logger.error(f"Failed to get Redis cached result: {e}")
            return None
    
    async def _perform_vuln_query(self, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform actual vulnerability database query"""
        # Simulate vulnerability database query
        await asyncio.sleep(0.1)  # Simulate network delay
        
        # Mock response based on query parameters
        cve_id = query_params.get("cve_id", "")
        product = query_params.get("product", "")
        version = query_params.get("version", "")
        
        return {
            "cve_id": cve_id,
            "product": product,
            "version": version,
            "severity": "HIGH",
            "description": f"Vulnerability in {product} {version}",
            "cvss_score": 8.5,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _cache_result(self, query_hash: str, query_params: Dict[str, Any], result: Dict[str, Any]):
        """Cache vulnerability query result"""
        cache_entry = {
            "result": result,
            "timestamp": datetime.now(),
            "ttl": self.config.ttl_seconds
        }
        
        if self.config.cache_type == CacheType.MEMORY:
            self._cache_memory_result(query_hash, cache_entry)
        elif self.config.cache_type == CacheType.DISK:
            await self._cache_disk_result(query_hash, query_params, cache_entry)
        elif self.config.cache_type == CacheType.REDIS:
            await self._cache_redis_result(query_hash, cache_entry)
    
    def _cache_memory_result(self, query_hash: str, cache_entry: Dict[str, Any]):
        """Cache result in memory"""
        if self._lock:
            with self._lock:
                self._cache_memory_result_internal(query_hash, cache_entry)
        else:
            self._cache_memory_result_internal(query_hash, cache_entry)
    
    def _cache_memory_result_internal(self, query_hash: str, cache_entry: Dict[str, Any]):
        """Internal memory cache storage"""
        # Implement LRU eviction if cache is full
        if len(self._cache) >= self.config.max_size:
            self._evict_oldest_entry()
        
        self._cache[query_hash] = cache_entry
    
    async def _cache_disk_result(self, query_hash: str, query_params: Dict[str, Any], cache_entry: Dict[str, Any]):
        """Cache result on disk"""
        try:
            with sqlite3.connect(self._cache_file) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO vuln_cache 
                    (query_hash, query_params, result_data, ttl, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    query_hash,
                    json.dumps(query_params),
                    json.dumps(cache_entry["result"]),
                    cache_entry["ttl"],
                    cache_entry["timestamp"].isoformat()
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to cache disk result: {e}")
    
    async def _cache_redis_result(self, query_hash: str, cache_entry: Dict[str, Any]):
        """Cache result in Redis"""
        try:
            self._redis_client.setex(
                query_hash,
                cache_entry["ttl"],
                json.dumps(cache_entry["result"])
            )
        except Exception as e:
            logger.error(f"Failed to cache Redis result: {e}")
    
    def _is_cache_entry_expired(self, cache_entry: Dict[str, Any]) -> bool:
        """Check if cache entry is expired"""
        timestamp = cache_entry["timestamp"]
        ttl = cache_entry["ttl"]
        return datetime.now() - timestamp > timedelta(seconds=ttl)
    
    def _evict_oldest_entry(self) -> Any:
        """Evict oldest cache entry"""
        if not self._cache:
            return
        
        oldest_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k]["timestamp"]
        )
        del self._cache[oldest_key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            **self._stats,
            "cache_size": len(self._cache),
            "hit_rate": self._stats["cache_hits"] / max(self._stats["total_queries"], 1)
        }
    
    async def cleanup(self) -> Any:
        """Clean up expired entries"""
        if self.config.cache_type == CacheType.MEMORY:
            self._cleanup_memory_cache()
        elif self.config.cache_type == CacheType.DISK:
            await self._cleanup_disk_cache()
        elif self.config.cache_type == CacheType.REDIS:
            await self._cleanup_redis_cache()
    
    def _cleanup_memory_cache(self) -> Any:
        """Clean up memory cache"""
        if self._lock:
            with self._lock:
                self._cleanup_memory_cache_internal()
        else:
            self._cleanup_memory_cache_internal()
    
    def _cleanup_memory_cache_internal(self) -> Any:
        """Internal memory cache cleanup"""
        expired_keys = [
            key for key, entry in self._cache.items()
            if self._is_cache_entry_expired(entry)
        ]
        
        for key in expired_keys:
            del self._cache[key]
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired vulnerability cache entries")
    
    async def _cleanup_disk_cache(self) -> Any:
        """Clean up disk cache"""
        try:
            with sqlite3.connect(self._cache_file) as conn:
                conn.execute("""
                    DELETE FROM vuln_cache 
                    WHERE datetime(timestamp) < datetime('now', '-1 hour')
                """)
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to cleanup disk cache: {e}")
    
    async def _cleanup_redis_cache(self) -> Any:
        """Clean up Redis cache"""
        # Redis handles TTL automatically, no manual cleanup needed
        pass


# Example usage and demonstration functions

async def demonstrate_lazy_loading():
    """Demonstrate lazy loading capabilities"""
    logger.info("Starting lazy loading demonstration")
    
    # Configuration
    config = LazyLoadConfig(
        strategy=LazyLoadStrategy.ON_DEMAND,
        enable_monitoring=True,
        cache_loaded_modules=True,
        memory_limit_mb=256
    )
    
    # Create lazy loader
    async with LazyModuleLoader(config) as loader:
        # Define heavy modules to load
        heavy_modules = [
            "numpy",
            "pandas",
            "matplotlib",
            "scikit-learn",
            "tensorflow"
        ]
        
        for module_name in heavy_modules:
            try:
                logger.info(f"Loading module: {module_name}")
                module = await loader.load_module(module_name)
                logger.info(f"Successfully loaded: {module_name}")
                
                # Get monitoring data
                monitoring_data = loader.get_monitoring_data()
                logger.info(f"Memory usage: {monitoring_data['current_memory_mb']:.1f}MB")
                
            except Exception as e:
                logger.error(f"Failed to load {module_name}: {e}")
        
        # Print final statistics
        final_stats = loader.get_monitoring_data()
        logger.info(f"Final statistics: {json.dumps(final_stats, indent=2)}")


async def demonstrate_dns_caching():
    """Demonstrate DNS caching capabilities"""
    logger.info("Starting DNS caching demonstration")
    
    # Configuration
    config = CacheConfig(
        cache_type=CacheType.MEMORY,
        max_size=100,
        ttl_seconds=3600,
        enable_metrics=True,
        thread_safe=True
    )
    
    # Create DNS cache
    dns_cache = DNSCache(config)
    
    # Test domains
    test_domains = [
        "google.com",
        "github.com",
        "stackoverflow.com",
        "example.com",
        "microsoft.com"
    ]
    
    for domain in test_domains:
        try:
            logger.info(f"Resolving: {domain}")
            
            # First resolution (cache miss)
            start_time = time.time()
            ips = await dns_cache.resolve(domain)
            first_time = time.time() - start_time
            
            logger.info(f"First resolution: {ips} (took {first_time:.3f}s)")
            
            # Second resolution (cache hit)
            start_time = time.time()
            cached_ips = await dns_cache.resolve(domain)
            second_time = time.time() - start_time
            
            logger.info(f"Cached resolution: {cached_ips} (took {second_time:.3f}s)")
            
        except Exception as e:
            logger.error(f"Failed to resolve {domain}: {e}")
    
    # Print cache statistics
    stats = dns_cache.get_stats()
    logger.info(f"DNS cache statistics: {json.dumps(stats, indent=2)}")
    
    # Cleanup
    await dns_cache.cleanup()


async def demonstrate_vulnerability_caching():
    """Demonstrate vulnerability database caching"""
    logger.info("Starting vulnerability caching demonstration")
    
    # Configuration
    config = CacheConfig(
        cache_type=CacheType.MEMORY,
        max_size=50,
        ttl_seconds=1800,  # 30 minutes
        enable_metrics=True,
        thread_safe=True
    )
    
    # Create vulnerability cache
    vuln_cache = VulnerabilityDBCache(config)
    
    # Test queries
    test_queries = [
        {"cve_id": "CVE-2021-44228", "product": "log4j", "version": "2.14.1"},
        {"cve_id": "CVE-2021-34527", "product": "windows", "version": "10"},
        {"cve_id": "CVE-2021-26855", "product": "exchange", "version": "2019"},
        {"product": "apache", "version": "2.4.49"},
        {"product": "nginx", "version": "1.20.0"}
    ]
    
    for query in test_queries:
        try:
            logger.info(f"Querying: {query}")
            
            # First query (cache miss)
            start_time = time.time()
            result = await vuln_cache.query_vulnerability_db(query)
            first_time = time.time() - start_time
            
            logger.info(f"First query result: {result} (took {first_time:.3f}s)")
            
            # Second query (cache hit)
            start_time = time.time()
            cached_result = await vuln_cache.query_vulnerability_db(query)
            second_time = time.time() - start_time
            
            logger.info(f"Cached query result: {cached_result} (took {second_time:.3f}s)")
            
        except Exception as e:
            logger.error(f"Failed to query vulnerability DB: {e}")
    
    # Print cache statistics
    stats = vuln_cache.get_stats()
    logger.info(f"Vulnerability cache statistics: {json.dumps(stats, indent=2)}")
    
    # Cleanup
    await vuln_cache.cleanup()


async def demonstrate_hybrid_caching():
    """Demonstrate hybrid caching with multiple cache types"""
    logger.info("Starting hybrid caching demonstration")
    
    # Memory cache configuration
    memory_config = CacheConfig(
        cache_type=CacheType.MEMORY,
        max_size=100,
        ttl_seconds=1800,
        enable_metrics=True
    )
    
    # Disk cache configuration
    disk_config = CacheConfig(
        cache_type=CacheType.DISK,
        max_size=1000,
        ttl_seconds=7200,  # 2 hours
        enable_metrics=True,
        cache_dir="cache"
    )
    
    # Create caches
    dns_memory_cache = DNSCache(memory_config)
    dns_disk_cache = DNSCache(disk_config)
    vuln_memory_cache = VulnerabilityDBCache(memory_config)
    vuln_disk_cache = VulnerabilityDBCache(disk_config)
    
    # Test hybrid caching
    test_domains = ["google.com", "github.com", "example.com"]
    test_queries = [
        {"cve_id": "CVE-2021-44228", "product": "log4j"},
        {"product": "apache", "version": "2.4.49"}
    ]
    
    # Test DNS caching
    for domain in test_domains:
        try:
            # Memory cache
            ips_memory = await dns_memory_cache.resolve(domain)
            logger.info(f"Memory DNS cache: {domain} -> {ips_memory}")
            
            # Disk cache
            ips_disk = await dns_disk_cache.resolve(domain)
            logger.info(f"Disk DNS cache: {domain} -> {ips_disk}")
            
        except Exception as e:
            logger.error(f"DNS resolution failed: {e}")
    
    # Test vulnerability caching
    for query in test_queries:
        try:
            # Memory cache
            result_memory = await vuln_memory_cache.query_vulnerability_db(query)
            logger.info(f"Memory vuln cache: {query} -> {result_memory}")
            
            # Disk cache
            result_disk = await vuln_disk_cache.query_vulnerability_db(query)
            logger.info(f"Disk vuln cache: {query} -> {result_disk}")
            
        except Exception as e:
            logger.error(f"Vulnerability query failed: {e}")
    
    # Print statistics
    logger.info("Memory DNS cache stats:", dns_memory_cache.get_stats())
    logger.info("Disk DNS cache stats:", dns_disk_cache.get_stats())
    logger.info("Memory vuln cache stats:", vuln_memory_cache.get_stats())
    logger.info("Disk vuln cache stats:", vuln_disk_cache.get_stats())
    
    # Cleanup
    await dns_memory_cache.cleanup()
    await dns_disk_cache.cleanup()
    await vuln_memory_cache.cleanup()
    await vuln_disk_cache.cleanup()


if __name__ == "__main__":
    # Run demonstrations
    async def main():
        
    """main function."""
try:
            await demonstrate_lazy_loading()
            await demonstrate_dns_caching()
            await demonstrate_vulnerability_caching()
            await demonstrate_hybrid_caching()
        except KeyboardInterrupt:
            logger.info("Demonstration interrupted by user")
        except Exception as e:
            logger.error(f"Demonstration error: {e}")
    
    asyncio.run(main()) 