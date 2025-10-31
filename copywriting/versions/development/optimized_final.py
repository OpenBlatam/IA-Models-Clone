from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import json
import time
import logging
import hashlib
import os
import threading
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import uuid
            import orjson
            import ujson
            import blake3
            import xxhash
            import lz4.frame
            import zstandard as zstd
            import gzip
            import redis
                import uvloop
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OPTIMIZED FINAL - Sistema Completamente Optimizado
==================================================

Versión final optimizada con todas las mejoras de producción:
- Performance ultra-optimizado
- Cache inteligente multi-nivel
- Circuit breaker para tolerancia a fallos
- Monitoreo avanzado
- Optimizaciones de memoria
"""


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# ENUMS & CONFIGURATION
# ============================================================================

class PerformanceTier(Enum):
    """Performance tiers"""
    ULTRA_MAXIMUM = ("ULTRA MAXIMUM", 95.0)
    MAXIMUM = ("MAXIMUM", 85.0)
    ULTRA = ("ULTRA", 70.0)
    OPTIMIZED = ("OPTIMIZED", 50.0)
    ENHANCED = ("ENHANCED", 30.0)
    STANDARD = ("STANDARD", 0.0)

# ============================================================================
# CIRCUIT BREAKER
# ============================================================================

class CircuitBreaker:
    """Circuit breaker for fault tolerance"""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        
    """__init__ function."""
self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"
        self._lock = threading.Lock()
    
    def __call__(self, func) -> Any:
        async def wrapper(*args, **kwargs) -> Any:
            with self._lock:
                if self.state == "OPEN":
                    if time.time() - self.last_failure_time < self.timeout:
                        raise Exception("Circuit breaker is OPEN")
                    else:
                        self.state = "HALF_OPEN"
                
                try:
                    result = await func(*args, **kwargs)
                    if self.state == "HALF_OPEN":
                        self.state = "CLOSED"
                        self.failure_count = 0
                    return result
                except Exception as e:
                    self.failure_count += 1
                    self.last_failure_time = time.time()
                    
                    if self.failure_count >= self.failure_threshold:
                        self.state = "OPEN"
                        logger.error(f"Circuit breaker opened: {func.__name__}")
                    raise e
        return wrapper

# ============================================================================
# OPTIMIZATION ENGINE
# ============================================================================

class OptimizedEngine:
    """Optimized engine with auto-detection"""
    
    def __init__(self) -> Any:
        self.libraries = self._scan_libraries()
        self.json_handler = self._setup_json()
        self.hash_handler = self._setup_hash()
        self.compression_handler = self._setup_compression()
        self.cache_handler = self._setup_redis()
        
        self.optimization_score = self._calculate_score()
        self.performance_tier = self._determine_tier()
        
        logger.info(f"OptimizedEngine: {self.optimization_score:.1f}/100 - {self.performance_tier.value[0]}")
    
    def _scan_libraries(self) -> Dict[str, bool]:
        """Scan available optimization libraries"""
        libs = [
            "orjson", "msgspec", "ujson", "blake3", "xxhash", "mmh3",
            "lz4", "zstandard", "blosc2", "numba", "polars", "duckdb",
            "redis", "uvloop", "rapidfuzz", "psutil"
        ]
        
        available = {}
        for lib in libs:
            try:
                __import__(lib)
                available[lib] = True
            except ImportError:
                available[lib] = False
        
        count = sum(available.values())
        logger.info(f"Libraries available: {count}/{len(libs)}")
        return available
    
    def _setup_json(self) -> Dict[str, Any]:
        """Setup optimized JSON handler"""
        if self.libraries.get("orjson"):
            return {
                "dumps": lambda x: orjson.dumps(x).decode(),
                "loads": orjson.loads,
                "name": "orjson",
                "speed": 5.0
            }
        elif self.libraries.get("ujson"):
            return {
                "dumps": ujson.dumps,
                "loads": ujson.loads,
                "name": "ujson",
                "speed": 3.0
            }
        else:
            return {
                "dumps": json.dumps,
                "loads": json.loads,
                "name": "json",
                "speed": 1.0
            }
    
    def _setup_hash(self) -> Dict[str, Any]:
        """Setup optimized hash handler"""
        if self.libraries.get("blake3"):
            return {
                "hash": lambda x: blake3.blake3(x.encode()).hexdigest()[:16],
                "name": "blake3",
                "speed": 8.0
            }
        elif self.libraries.get("xxhash"):
            return {
                "hash": lambda x: xxhash.xxh64(x.encode()).hexdigest()[:16],
                "name": "xxhash",
                "speed": 6.0
            }
        else:
            return {
                "hash": lambda x: hashlib.sha256(x.encode()).hexdigest()[:16],
                "name": "sha256",
                "speed": 1.0
            }
    
    def _setup_compression(self) -> Dict[str, Any]:
        """Setup optimized compression handler"""
        if self.libraries.get("lz4"):
            return {
                "compress": lz4.frame.compress,
                "decompress": lz4.frame.decompress,
                "name": "lz4",
                "speed": 10.0
            }
        elif self.libraries.get("zstandard"):
            comp = zstd.ZstdCompressor(level=1)
            decomp = zstd.ZstdDecompressor()
            return {
                "compress": comp.compress,
                "decompress": decomp.decompress,
                "name": "zstandard",
                "speed": 5.0
            }
        else:
            return {
                "compress": gzip.compress,
                "decompress": gzip.decompress,
                "name": "gzip",
                "speed": 1.0
            }
    
    def _setup_redis(self) -> Optional[Any]:
        """Setup Redis cache"""
        if not self.libraries.get("redis"):
            return None
        
        try:
            client = redis.from_url(
                os.getenv('REDIS_URL', 'redis://localhost:6379/0'),
                decode_responses=True,
                socket_timeout=5
            )
            client.ping()
            logger.info("Redis connected")
            return client
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            return None
    
    def _calculate_score(self) -> float:
        """Calculate optimization score"""
        score = 0.0
        
        # Base optimizations
        score += self.json_handler["speed"] * 5
        score += self.hash_handler["speed"] * 3
        score += self.compression_handler["speed"] * 2
        
        # Library bonuses
        bonuses = {
            "polars": 15, "duckdb": 10, "numba": 12, "uvloop": 8,
            "rapidfuzz": 5, "psutil": 3
        }
        
        for lib, bonus in bonuses.items():
            if self.libraries.get(lib):
                score += bonus
        
        if self.cache_handler:
            score += 10
        
        return min(score, 100.0)
    
    def _determine_tier(self) -> PerformanceTier:
        """Determine performance tier"""
        for tier in PerformanceTier:
            if self.optimization_score >= tier.value[1]:
                return tier
        return PerformanceTier.STANDARD

# ============================================================================
# INTELLIGENT CACHE MANAGER
# ============================================================================

class IntelligentCacheManager:
    """Intelligent multi-level cache with optimization"""
    
    def __init__(self, engine: OptimizedEngine):
        
    """__init__ function."""
self.engine = engine
        
        # Configuration
        self.memory_size = 3000
        self.ttl = 7200
        self.compression_threshold = 512
        
        # Storage levels
        self.memory_cache: Dict[str, Any] = {}
        self.compressed_cache: Dict[str, bytes] = {}
        self.timestamps: Dict[str, float] = {}
        self.access_counts: Dict[str, int] = {}
        self.priorities: Dict[str, int] = {}
        
        # External cache
        self.redis = engine.cache_handler
        
        # Metrics
        self.metrics = {
            "memory_hits": 0, "compressed_hits": 0, "redis_hits": 0,
            "misses": 0, "sets": 0, "evictions": 0, "errors": 0,
            "compression_savings": 0
        }
        
        # Circuit breaker
        self.circuit_breaker = CircuitBreaker(failure_threshold=3, timeout=30)
        
        # Background cleanup
        self._start_cleanup()
        
        logger.info("IntelligentCacheManager initialized")
    
    def _start_cleanup(self) -> Any:
        """Start background cleanup task"""
        async def cleanup():
            
    """cleanup function."""
while True:
                try:
                    await asyncio.sleep(300)  # Every 5 minutes
                    await self._cleanup_expired()
                except Exception as e:
                    logger.error(f"Cleanup error: {e}")
        
        asyncio.create_task(cleanup())
    
    async def _cleanup_expired(self) -> Any:
        """Clean expired entries"""
        current_time = time.time()
        expired = []
        
        for key, timestamp in self.timestamps.items():
            if current_time - timestamp > self.ttl:
                expired.append(key)
        
        for key in expired:
            self._evict_key(key)
            self.metrics["evictions"] += 1
        
        if expired:
            logger.info(f"Cleaned {len(expired)} expired entries")
    
    @CircuitBreaker(failure_threshold=3, timeout=30)
    async def get(self, key: str, priority: int = 1) -> Optional[Any]:
        """Intelligent cache get with priority"""
        cache_key = self._generate_key(key)
        
        try:
            # L1: Memory cache
            if cache_key in self.memory_cache:
                if time.time() - self.timestamps.get(cache_key, 0) < self.ttl:
                    self._update_access(cache_key, priority)
                    self.metrics["memory_hits"] += 1
                    return self.memory_cache[cache_key]
                else:
                    self._evict_from_memory(cache_key)
            
            # L2: Compressed cache
            if cache_key in self.compressed_cache:
                try:
                    compressed = self.compressed_cache[cache_key]
                    decompressed = self.engine.compression_handler["decompress"](compressed)
                    value = self.engine.json_handler["loads"](decompressed.decode())
                    
                    # Promote to memory if high priority
                    if priority >= 3:
                        await self._promote_to_memory(cache_key, value, priority)
                    
                    self.metrics["compressed_hits"] += 1
                    return value
                except Exception:
                    del self.compressed_cache[cache_key]
            
            # L3: Redis cache
            if self.redis:
                try:
                    data = self.redis.get(f"opt:{cache_key}")
                    if data:
                        value = self.engine.json_handler["loads"](data)
                        await self.set(key, value, priority, skip_redis=True)
                        self.metrics["redis_hits"] += 1
                        return value
                except Exception:
                    pass
            
            self.metrics["misses"] += 1
            return None
            
        except Exception as e:
            self.metrics["errors"] += 1
            logger.error(f"Cache get error: {e}")
            return None
    
    async def set(self, key: str, value: Any, priority: int = 1, skip_redis: bool = False) -> bool:
        """Intelligent cache set with optimization"""
        cache_key = self._generate_key(key)
        
        try:
            # Calculate data size
            json_data = self.engine.json_handler["dumps"](value).encode()
            data_size = len(json_data)
            
            # Smart placement decision
            if data_size < self.compression_threshold or priority >= 4:
                # Small or high-priority items go to memory
                await self._store_in_memory(cache_key, value, priority)
            else:
                # Large items get compressed
                await self._store_compressed(cache_key, value, json_data, priority)
            
            # Store in Redis asynchronously
            if self.redis and not skip_redis:
                asyncio.create_task(self._store_in_redis(cache_key, value))
            
            self.metrics["sets"] += 1
            return True
            
        except Exception as e:
            self.metrics["errors"] += 1
            logger.error(f"Cache set error: {e}")
            return False
    
    async def _store_in_memory(self, cache_key: str, value: Any, priority: int):
        """Store in memory with intelligent eviction"""
        # Evict if necessary
        while len(self.memory_cache) >= self.memory_size:
            victim = self._select_eviction_victim()
            if victim:
                self._evict_from_memory(victim)
                self.metrics["evictions"] += 1
            else:
                break
        
        self.memory_cache[cache_key] = value
        self.timestamps[cache_key] = time.time()
        self.priorities[cache_key] = priority
        self.access_counts[cache_key] = 1
    
    async def _store_compressed(self, cache_key: str, value: Any, json_data: bytes, priority: int):
        """Store in compressed cache"""
        try:
            compressed = self.engine.compression_handler["compress"](json_data)
            compression_ratio = len(compressed) / len(json_data)
            
            if compression_ratio < 0.9:  # Only if beneficial
                self.compressed_cache[cache_key] = compressed
                self.timestamps[cache_key] = time.time()
                self.priorities[cache_key] = priority
                self.access_counts[cache_key] = 1
                
                savings = len(json_data) - len(compressed)
                self.metrics["compression_savings"] += savings
            else:
                # Store uncompressed in memory
                await self._store_in_memory(cache_key, value, priority)
                
        except Exception:
            await self._store_in_memory(cache_key, value, priority)
    
    async def _store_in_redis(self, cache_key: str, value: Any):
        """Store in Redis asynchronously"""
        try:
            data = self.engine.json_handler["dumps"](value)
            self.redis.setex(f"opt:{cache_key}", self.ttl, data)
        except Exception:
            pass
    
    async def _promote_to_memory(self, cache_key: str, value: Any, priority: int):
        """Promote to memory cache"""
        if len(self.memory_cache) < self.memory_size:
            self.memory_cache[cache_key] = value
            self.timestamps[cache_key] = time.time()
            self.priorities[cache_key] = priority
            self._update_access(cache_key, priority)
    
    def _select_eviction_victim(self) -> Optional[str]:
        """Select victim using LFU+LRU+Priority"""
        if not self.memory_cache:
            return None
        
        candidates = []
        current_time = time.time()
        
        for key in self.memory_cache.keys():
            access_count = self.access_counts.get(key, 1)
            last_access = self.timestamps.get(key, 0)
            priority = self.priorities.get(key, 1)
            
            # Composite score (lower is better victim)
            score = (1.0 / access_count) + (current_time - last_access) * 0.01 + (1.0 / priority)
            candidates.append((key, score))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0] if candidates else None
    
    def _update_access(self, cache_key: str, priority: int):
        """Update access statistics"""
        self.access_counts[cache_key] = self.access_counts.get(cache_key, 0) + 1
        self.timestamps[cache_key] = time.time()
        self.priorities[cache_key] = max(self.priorities.get(cache_key, 1), priority)
    
    def _evict_key(self, cache_key: str):
        """Evict from all caches"""
        self._evict_from_memory(cache_key)
        self.compressed_cache.pop(cache_key, None)
    
    def _evict_from_memory(self, cache_key: str):
        """Evict from memory cache"""
        self.memory_cache.pop(cache_key, None)
        self.timestamps.pop(cache_key, None)
        self.access_counts.pop(cache_key, None)
        self.priorities.pop(cache_key, None)
    
    def _generate_key(self, key: str) -> str:
        """Generate optimized cache key"""
        return self.engine.hash_handler["hash"](key)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics"""
        total_hits = self.metrics["memory_hits"] + self.metrics["compressed_hits"] + self.metrics["redis_hits"]
        total_requests = total_hits + self.metrics["misses"]
        
        hit_rate = (total_hits / max(total_requests, 1)) * 100
        
        return {
            "hit_rate_percent": hit_rate,
            "memory_hit_rate": (self.metrics["memory_hits"] / max(total_requests, 1)) * 100,
            "compressed_hit_rate": (self.metrics["compressed_hits"] / max(total_requests, 1)) * 100,
            "redis_hit_rate": (self.metrics["redis_hits"] / max(total_requests, 1)) * 100,
            "total_requests": total_requests,
            "memory_cache_size": len(self.memory_cache),
            "compressed_cache_size": len(self.compressed_cache),
            "compression_savings_bytes": self.metrics["compression_savings"],
            **self.metrics
        }

# ============================================================================
# REQUEST MODEL
# ============================================================================

@dataclass
class OptimizedRequest:
    """Optimized request model"""
    prompt: str
    tone: str = "professional"
    language: str = "es"
    use_case: str = "general"
    target_length: Optional[int] = None
    keywords: List[str] = field(default_factory=list)
    use_cache: bool = True
    priority: int = 1
    client_id: Optional[str] = None
    request_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def __post_init__(self) -> Any:
        if not self.prompt or len(self.prompt.strip()) == 0:
            raise ValueError("Prompt cannot be empty")
        
        # Optimize for performance
        if len(self.prompt) > 500:
            self.prompt = self.prompt[:500]
        
        if len(self.keywords) > 5:
            self.keywords = self.keywords[:5]
    
    def to_cache_key(self) -> str:
        """Generate cache key"""
        components = [
            self.prompt[:100],
            self.tone, self.language, self.use_case,
            str(self.target_length) if self.target_length else "",
            "|".join(sorted(self.keywords)) if self.keywords else ""
        ]
        return "|".join(c for c in components if c)

# ============================================================================
# OPTIMIZED COPYWRITING SERVICE
# ============================================================================

class OptimizedCopywritingService:
    """Final optimized copywriting service"""
    
    def __init__(self) -> Any:
        # Initialize optimized components
        self.engine = OptimizedEngine()
        self.cache_manager = IntelligentCacheManager(self.engine)
        
        # Performance tracking
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_time": 0.0,
            "start_time": time.time()
        }
        
        # Content templates
        self.templates = {
            "professional": "Como experto en {use_case}, presento {prompt}. Solución profesional optimizada para máximos resultados.",
            "casual": "¡Hola! Te cuento sobre {prompt} para {use_case}. Es algo genial que te va a encantar.",
            "urgent": "⚡ ¡OPORTUNIDAD ÚNICA! {prompt} - Solución para {use_case} disponible por tiempo limitado.",
            "creative": "¡Imagina las posibilidades! {prompt} revoluciona {use_case}. Innovación que redefine límites.",
            "technical": "Análisis técnico: {prompt} optimiza {use_case} con métricas avanzadas y resultados medibles.",
            "friendly": "¡Hola amigo! {prompt} es perfecto para {use_case}. Una propuesta que cambiará tu perspectiva."
        }
        
        # Setup uvloop if available
        if self.engine.libraries.get("uvloop"):
            try:
                uvloop.install()
                logger.info("uvloop activated")
            except Exception:
                pass
        
        logger.info("OptimizedCopywritingService initialized")
        self._show_status()
    
    async def generate_copy(self, request: OptimizedRequest) -> Dict[str, Any]:
        """Generate optimized copy"""
        start_time = time.time()
        
        try:
            self.metrics["total_requests"] += 1
            
            # Check cache
            cache_key = request.to_cache_key()
            if request.use_cache:
                cached_result = await self.cache_manager.get(cache_key, request.priority)
                if cached_result:
                    response_time = (time.time() - start_time) * 1000
                    self._record_success(response_time)
                    
                    return {
                        "content": cached_result["content"],
                        "request_id": request.request_id,
                        "response_time_ms": response_time,
                        "cache_hit": True,
                        "optimization_score": self.engine.optimization_score,
                        "performance_tier": self.engine.performance_tier.value[0],
                        "word_count": cached_result["word_count"],
                        "character_count": cached_result["character_count"]
                    }
            
            # Generate content
            content = await self._generate_content(request)
            response_time = (time.time() - start_time) * 1000
            
            # Create result
            result = {
                "content": content,
                "word_count": len(content.split()),
                "character_count": len(content)
            }
            
            # Cache result
            if request.use_cache:
                await self.cache_manager.set(cache_key, result, request.priority)
            
            self._record_success(response_time)
            
            return {
                "content": content,
                "request_id": request.request_id,
                "response_time_ms": response_time,
                "cache_hit": False,
                "optimization_score": self.engine.optimization_score,
                "performance_tier": self.engine.performance_tier.value[0],
                "word_count": result["word_count"],
                "character_count": result["character_count"]
            }
            
        except Exception as e:
            self.metrics["failed_requests"] += 1
            logger.error(f"Generation failed: {e}")
            raise
    
    async def _generate_content(self, request: OptimizedRequest) -> str:
        """Generate optimized content"""
        # Get template
        template = self.templates.get(request.tone, self.templates["professional"f"])
        
        # Generate content
        content = template"
        
        # Add keywords
        if request.keywords:
            content += f" Palabras clave: {', '.join(request.keywords)}."
        
        # Adjust length if needed
        if request.target_length:
            words = content.split()
            if len(words) < request.target_length:
                content += " Esta propuesta integral garantiza resultados excepcionales y sostenibles."
        
        # Minimal processing delay
        await asyncio.sleep(0.002)
        
        return content
    
    def _record_success(self, response_time: float):
        """Record successful request"""
        self.metrics["successful_requests"] += 1
        self.metrics["total_time"] += response_time
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check"""
        try:
            # Performance test
            test_request = OptimizedRequest(
                prompt="Health check test",
                tone="professional",
                use_cache=False
            )
            
            start_time = time.time()
            response = await self.generate_copy(test_request)
            test_time = (time.time() - start_time) * 1000
            
            # Calculate metrics
            avg_time = self.metrics["total_time"] / max(self.metrics["successful_requests"], 1)
            success_rate = (self.metrics["successful_requests"] / max(self.metrics["total_requests"], 1)) * 100
            uptime = time.time() - self.metrics["start_time"]
            
            return {
                "status": "healthy",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "performance": {
                    "optimization_score": self.engine.optimization_score,
                    "performance_tier": self.engine.performance_tier.value[0],
                    "test_response_time_ms": test_time,
                    "avg_response_time_ms": avg_time,
                    "success_rate_percent": success_rate,
                    "total_requests": self.metrics["total_requests"],
                    "uptime_seconds": uptime
                },
                "optimization": {
                    "json_handler": self.engine.json_handler["name"],
                    "hash_handler": self.engine.hash_handler["name"],
                    "compression_handler": self.engine.compression_handler["name"],
                    "redis_available": self.engine.cache_handler is not None,
                    "libraries_available": sum(self.engine.libraries.values())
                },
                "cache": self.cache_manager.get_metrics()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    def _show_status(self) -> Any:
        """Show service status"""
        print(f"\n{'='*70}")
        print("🚀 OPTIMIZED COPYWRITING SERVICE - FINAL VERSION")
        print(f"{'='*70}")
        print(f"📊 Optimization Score: {self.engine.optimization_score:.1f}/100")
        print(f"🏆 Performance Tier: {self.engine.performance_tier.value[0]}")
        print(f"\n⚡ Optimizations Active:")
        print(f"   🔥 JSON: {self.engine.json_handler['name']} ({self.engine.json_handler['speed']:.1f}x)")
        print(f"   🔥 Hash: {self.engine.hash_handler['name']} ({self.engine.hash_handler['speed']:.1f}x)")
        print(f"   🔥 Compression: {self.engine.compression_handler['name']} ({self.engine.compression_handler['speed']:.1f}x)")
        print(f"   🔥 Cache: Multi-level intelligent caching")
        print(f"   🔥 Redis: {'✅ Connected' if self.engine.cache_handler else '❌ Not Available'}")
        print(f"   🔥 Circuit Breaker: ✅ Fault tolerance enabled")
        print(f"{'='*70}")

# ============================================================================
# OPTIMIZED DEMO
# ============================================================================

async def optimized_demo():
    """Demo of final optimized system"""
    print("🚀 FINAL OPTIMIZATION DEMO")
    print("="*50)
    print("Sistema completamente optimizado y listo para producción")
    print("✅ Performance ultra-optimizado")
    print("✅ Cache inteligente multi-nivel")
    print("✅ Circuit breaker para tolerancia a fallos")
    print("✅ Monitoreo y métricas avanzadas")
    print("="*50)
    
    # Initialize service
    service = OptimizedCopywritingService()
    
    # Health check
    health = await service.health_check()
    print(f"\n🏥 System Status: {health['status'].upper()}")
    print(f"📊 Optimization Score: {health['performance']['optimization_score']:.1f}/100")
    print(f"🏆 Performance Tier: {health['performance']['performance_tier']}")
    print(f"⚡ Test Response: {health['performance']['test_response_time_ms']:.1f}ms")
    
    # Performance tests
    test_requests = [
        OptimizedRequest(
            prompt="Lanzamiento revolucionario IA",
            tone="professional",
            use_case="tech_launch",
            keywords=["IA", "revolucionario"],
            priority=5
        ),
        OptimizedRequest(
            prompt="Oferta especial limitada",
            tone="urgent",
            use_case="promotion",
            priority=4
        ),
        OptimizedRequest(
            prompt="Análisis técnico avanzado",
            tone="technical",
            use_case="analysis",
            priority=3
        )
    ]
    
    print(f"\n🔥 PERFORMANCE TESTING:")
    print("-" * 35)
    
    for i, request in enumerate(test_requests, 1):
        response = await service.generate_copy(request)
        print(f"\n{i}. {request.tone.upper()} (Priority: {request.priority})")
        print(f"   Content: {response['content'][:70]}...")
        print(f"   Time: {response['response_time_ms']:.1f}ms")
        print(f"   Cache: {'✅ HIT' if response['cache_hit'] else '❌ MISS'}")
        print(f"   Words: {response['word_count']}")
    
    # Cache effectiveness test
    print(f"\n🔄 CACHE EFFECTIVENESS:")
    cache_test = await service.generate_copy(test_requests[0])
    print(f"   Cached Request: {cache_test['response_time_ms']:.1f}ms")
    print(f"   Cache Hit: {'✅ YES' if cache_test['cache_hit'] else '❌ NO'}")
    
    # Final metrics
    final_health = await service.health_check()
    
    print(f"\n📊 FINAL METRICS:")
    print("-" * 25)
    print(f"   Optimization Score: {final_health['performance']['optimization_score']:.1f}/100")
    print(f"   Performance Tier: {final_health['performance']['performance_tier']}")
    print(f"   Average Response: {final_health['performance']['avg_response_time_ms']:.1f}ms")
    print(f"   Success Rate: {final_health['performance']['success_rate_percent']:.1f}%")
    print(f"   Cache Hit Rate: {final_health['cache']['hit_rate_percent']:.1f}%")
    print(f"   Total Requests: {final_health['performance']['total_requests']}")
    print(f"   Libraries Available: {final_health['optimization']['libraries_available']}")
    
    print(f"\n🎉 OPTIMIZATION COMPLETED!")
    print("🚀 Sistema optimizado al máximo")
    print("⚡ Rendimiento ultra-alto")
    print("🔧 Listo para producción enterprise")

async def main():
    """Main function"""
    await optimized_demo()

match __name__:
    case "__main__":
    asyncio.run(main()) 