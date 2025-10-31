from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import sys
import time
import psutil
import logging
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from functools import wraps
import asyncio
from concurrent.futures import ThreadPoolExecutor
import gc
            import orjson
            import msgspec
            import json
            import blake3
            import xxhash
            import hashlib
            import cramjam
            import lz4.frame
            import zstandard as zstd
            import gzip
                import uvloop
            from numba import jit
from typing import Any, List, Dict, Optional
"""
Optimization Manager
===================

Intelligent optimization system with automatic library detection,
performance monitoring, and graceful fallbacks.
"""


# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class OptimizationLibrary:
    """Information about an optimization library"""
    name: str
    available: bool
    version: Optional[str] = None
    performance_gain: float = 1.0
    category: str = "general"
    import_error: Optional[str] = None


@dataclass
class OptimizationProfile:
    """System optimization profile"""
    libraries: Dict[str, OptimizationLibrary] = field(default_factory=dict)
    total_score: float = 0.0
    performance_multiplier: float = 1.0
    recommendations: List[str] = field(default_factory=list)
    system_info: Dict[str, Any] = field(default_factory=dict)


class OptimizationManager:
    """Manages optimization libraries and performance monitoring"""
    
    def __init__(self) -> Any:
        self.profile = OptimizationProfile()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._performance_cache = {}
        self._jit_functions = {}
        self._setup_optimizations()
    
    def _setup_optimizations(self) -> Any:
        """Initialize optimization libraries and profile"""
        self._detect_libraries()
        self._calculate_profile()
        self._setup_system_monitoring()
    
    def _detect_libraries(self) -> Any:
        """Detect available optimization libraries"""
        libraries_to_check = {
            # Serialization
            "orjson": {"category": "serialization", "gain": 5.0},
            "msgspec": {"category": "serialization", "gain": 6.0},
            "simdjson": {"category": "serialization", "gain": 8.0},
            
            # Hashing
            "xxhash": {"category": "hashing", "gain": 4.0},
            "blake3": {"category": "hashing", "gain": 5.0},
            "mmh3": {"category": "hashing", "gain": 3.0},
            
            # Compression
            "lz4": {"category": "compression", "gain": 4.0},
            "blosc2": {"category": "compression", "gain": 6.0},
            "cramjam": {"category": "compression", "gain": 6.5},
            "zstandard": {"category": "compression", "gain": 5.0},
            
            # Data processing
            "polars": {"category": "data", "gain": 20.0},
            "pyarrow": {"category": "data", "gain": 8.0},
            "duckdb": {"category": "data", "gain": 12.0},
            
            # Event loop
            "uvloop": {"category": "async", "gain": 4.0},
            
            # JIT compilation
            "numba": {"category": "jit", "gain": 15.0},
            
            # Memory
            "pymalloc": {"category": "memory", "gain": 2.0},
            
            # Redis
            "hiredis": {"category": "redis", "gain": 3.0},
            
            # HTTP
            "httptools": {"category": "http", "gain": 3.5},
            "aiofiles": {"category": "io", "gain": 3.0},
        }
        
        for lib_name, lib_info in libraries_to_check.items():
            try:
                module = __import__(lib_name)
                version = getattr(module, "__version__", "unknown")
                
                self.profile.libraries[lib_name] = OptimizationLibrary(
                    name=lib_name,
                    available=True,
                    version=version,
                    performance_gain=lib_info["gain"],
                    category=lib_info["category"]
                )
                logger.info(f"✓ {lib_name} v{version} available (gain: {lib_info['gain']}x)")
                
            except ImportError as e:
                self.profile.libraries[lib_name] = OptimizationLibrary(
                    name=lib_name,
                    available=False,
                    performance_gain=lib_info["gain"],
                    category=lib_info["category"],
                    import_error=str(e)
                )
                logger.debug(f"✗ {lib_name} not available: {e}")
    
    def _calculate_profile(self) -> Any:
        """Calculate optimization profile and score"""
        available_libs = [lib for lib in self.profile.libraries.values() if lib.available]
        
        if not available_libs:
            self.profile.total_score = 1.0
            self.profile.performance_multiplier = 1.0
            return
        
        # Calculate weighted score
        total_gain = sum(lib.performance_gain for lib in available_libs)
        max_possible_gain = sum(lib.performance_gain for lib in self.profile.libraries.values())
        
        self.profile.total_score = (total_gain / max_possible_gain) * 100
        self.profile.performance_multiplier = min(total_gain / 10, 25.0)  # Cap at 25x
        
        # Generate recommendations
        missing_critical = [
            lib for lib in self.profile.libraries.values() 
            if not lib.available and lib.performance_gain >= 5.0
        ]
        
        if missing_critical:
            self.profile.recommendations.extend([
                f"Install {lib.name} for {lib.performance_gain}x {lib.category} performance gain"
                for lib in missing_critical[:5]  # Top 5 recommendations
            ])
    
    def _setup_system_monitoring(self) -> Any:
        """Setup system information monitoring"""
        try:
            self.profile.system_info = {
                "cpu_count": psutil.cpu_count(),
                "memory_total": psutil.virtual_memory().total,
                "memory_available": psutil.virtual_memory().available,
                "python_version": sys.version,
                "platform": sys.platform,
            }
        except Exception as e:
            logger.warning(f"Could not gather system info: {e}")
    
    def get_serializer(self) -> Optional[Dict[str, Any]]:
        """Get best available JSON serializer"""
        if self.profile.libraries.get("orjson", OptimizationLibrary("", False)).available:
            return {
                "dumps": lambda x: orjson.dumps(x).decode(),
                "loads": orjson.loads,
                "name": "orjson"
            }
        elif self.profile.libraries.get("msgspec", OptimizationLibrary("", False)).available:
            encoder = msgspec.json.Encoder()
            decoder = msgspec.json.Decoder()
            return {
                "dumps": lambda x: encoder.encode(x).decode(),
                "loads": decoder.decode,
                "name": "msgspec"
            }
        else:
            return {
                "dumps": json.dumps,
                "loads": json.loads,
                "name": "json"
            }
    
    def get_hasher(self) -> Optional[Dict[str, Any]]:
        """Get best available hash function"""
        if self.profile.libraries.get("blake3", OptimizationLibrary("", False)).available:
            return lambda data: blake3.blake3(data.encode() if isinstance(data, str) else data).hexdigest()
        elif self.profile.libraries.get("xxhash", OptimizationLibrary("", False)).available:
            return lambda data: xxhash.xxh64(data.encode() if isinstance(data, str) else data).hexdigest()
        else:
            return lambda data: hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()
    
    def get_compressor(self) -> Optional[Dict[str, Any]]:
        """Get best available compression"""
        if self.profile.libraries.get("cramjam", OptimizationLibrary("", False)).available:
            return {
                "compress": cramjam.lz4.compress,
                "decompress": cramjam.lz4.decompress,
                "name": "cramjam-lz4"
            }
        elif self.profile.libraries.get("lz4", OptimizationLibrary("", False)).available:
            return {
                "compress": lz4.frame.compress,
                "decompress": lz4.frame.decompress,
                "name": "lz4"
            }
        elif self.profile.libraries.get("zstandard", OptimizationLibrary("", False)).available:
            compressor = zstd.ZstdCompressor()
            decompressor = zstd.ZstdDecompressor()
            return {
                "compress": compressor.compress,
                "decompress": decompressor.decompress,
                "name": "zstandard"
            }
        else:
            return {
                "compress": gzip.compress,
                "decompress": gzip.decompress,
                "name": "gzip"
            }
    
    def setup_event_loop(self) -> Any:
        """Setup optimized event loop"""
        if self.profile.libraries.get("uvloop", OptimizationLibrary("", False)).available:
            try:
                asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
                logger.info("✓ Using uvloop for enhanced async performance")
                return True
            except Exception as e:
                logger.warning(f"Could not setup uvloop: {e}")
        return False
    
    def jit_compile(self, func: Callable, cache_key: Optional[str] = None) -> Callable:
        """JIT compile function if numba is available"""
        if not self.profile.libraries.get("numba", OptimizationLibrary("", False)).available:
            return func
        
        cache_key = cache_key or f"{func.__module__}.{func.__name__}"
        
        if cache_key in self._jit_functions:
            return self._jit_functions[cache_key]
        
        try:
            jit_func = jit(nopython=True, cache=True)(func)
            self._jit_functions[cache_key] = jit_func
            logger.debug(f"✓ JIT compiled {cache_key}")
            return jit_func
        except Exception as e:
            logger.warning(f"Could not JIT compile {cache_key}: {e}")
            return func
    
    def optimize_memory(self) -> Any:
        """Optimize memory usage"""
        # Force garbage collection
        collected = gc.collect()
        
        # Get memory info
        process = psutil.Process()
        memory_info = process.memory_info()
        
        logger.debug(f"Memory optimization: collected {collected} objects, "
                    f"RSS: {memory_info.rss / 1024 / 1024:.1f}MB")
        
        return {
            "collected_objects": collected,
            "memory_rss_mb": memory_info.rss / 1024 / 1024,
            "memory_vms_mb": memory_info.vms / 1024 / 1024
        }
    
    def performance_monitor(self, func_name: str = None):
        """Decorator for performance monitoring"""
        def decorator(func) -> Any:
            @wraps(func)
            async def async_wrapper(*args, **kwargs) -> Any:
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    success = True
                    error = None
                except Exception as e:
                    result = None
                    success = False
                    error = str(e)
                    raise
                finally:
                    execution_time = time.time() - start_time
                    self._record_performance(
                        func_name or func.__name__,
                        execution_time,
                        success,
                        error
                    )
                return result
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs) -> Any:
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    success = True
                    error = None
                except Exception as e:
                    result = None
                    success = False
                    error = str(e)
                    raise
                finally:
                    execution_time = time.time() - start_time
                    self._record_performance(
                        func_name or func.__name__,
                        execution_time,
                        success,
                        error
                    )
                return result
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator
    
    def _record_performance(self, func_name: str, execution_time: float, 
                          success: bool, error: Optional[str]):
        """Record performance metrics"""
        if func_name not in self._performance_cache:
            self._performance_cache[func_name] = {
                "total_calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "total_time": 0.0,
                "average_time": 0.0,
                "min_time": float('inf'),
                "max_time": 0.0,
                "recent_errors": []
            }
        
        stats = self._performance_cache[func_name]
        stats["total_calls"] += 1
        stats["total_time"] += execution_time
        stats["average_time"] = stats["total_time"] / stats["total_calls"]
        stats["min_time"] = min(stats["min_time"], execution_time)
        stats["max_time"] = max(stats["max_time"], execution_time)
        
        if success:
            stats["successful_calls"] += 1
        else:
            stats["failed_calls"] += 1
            stats["recent_errors"].append({
                "timestamp": time.time(),
                "error": error
            })
            # Keep only recent errors (last 10)
            stats["recent_errors"] = stats["recent_errors"][-10:]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            "optimization_profile": {
                "total_score": self.profile.total_score,
                "performance_multiplier": self.profile.performance_multiplier,
                "available_libraries": len([lib for lib in self.profile.libraries.values() if lib.available]),
                "total_libraries": len(self.profile.libraries),
                "recommendations": self.profile.recommendations
            },
            "function_performance": self._performance_cache.copy(),
            "system_info": self.profile.system_info
        }
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report"""
        available_libs = {name: lib for name, lib in self.profile.libraries.items() if lib.available}
        missing_libs = {name: lib for name, lib in self.profile.libraries.items() if not lib.available}
        
        # Group by category
        categories = {}
        for lib in self.profile.libraries.values():
            if lib.category not in categories:
                categories[lib.category] = {"available": [], "missing": []}
            
            if lib.available:
                categories[lib.category]["available"].append(lib)
            else:
                categories[lib.category]["missing"].append(lib)
        
        return {
            "summary": {
                "optimization_score": self.profile.total_score,
                "performance_multiplier": self.profile.performance_multiplier,
                "available_count": len(available_libs),
                "missing_count": len(missing_libs),
                "total_count": len(self.profile.libraries)
            },
            "categories": categories,
            "recommendations": self.profile.recommendations,
            "system_info": self.profile.system_info,
            "performance_stats": self._performance_cache
        }
    
    def cleanup(self) -> Any:
        """Cleanup resources"""
        try:
            self.executor.shutdown(wait=True)
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")


# Global optimization manager instance
optimization_manager = OptimizationManager()


def get_optimization_manager() -> OptimizationManager:
    """Get the global optimization manager"""
    return optimization_manager 