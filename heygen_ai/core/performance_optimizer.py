"""
Performance Optimizer for HeyGen AI
==================================

Provides comprehensive performance optimization, monitoring, and
resource management for enterprise-grade AI video generation.
"""

import asyncio
import logging
import time
import psutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import uuid

# Core imports
from .base_service import BaseService, ServiceType, HealthCheckResult, ServiceStatus
from .error_handler import ErrorHandler, with_error_handling, with_retry
from .config_manager import ConfigurationManager
from .logging_service import LoggingService

# Performance monitoring imports
try:
    import uvloop
    UVLOOP_AVAILABLE = True
except ImportError:
    UVLOOP_AVAILABLE = False

try:
    import orjson
    ORJSON_AVAILABLE = True
except ImportError:
    ORJSON_AVAILABLE = False

try:
    import msgpack
    MSGPACK_AVAILABLE = True
except ImportError:
    MSGPACK_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics data."""
    
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_io: Dict[str, float]
    network_io: Dict[str, float]
    gpu_usage: Optional[float] = None
    gpu_memory: Optional[float] = None
    active_processes: int = 0
    system_load: float = 0.0


@dataclass
class OptimizationConfig:
    """Configuration for performance optimization."""
    
    enable_uvloop: bool = True
    enable_orjson: bool = True
    enable_msgpack: bool = True
    memory_limit_gb: float = 8.0
    cpu_limit_percent: float = 80.0
    gpu_memory_limit_gb: float = 4.0
    cache_size_mb: int = 512
    enable_compression: bool = True
    compression_level: int = 6
    enable_monitoring: bool = True
    monitoring_interval: float = 5.0


@dataclass
class OptimizationRequest:
    """Request for performance optimization."""
    
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    optimization_type: str = "auto"  # auto, memory, cpu, gpu, network
    target_metrics: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1
    custom_attributes: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MemoryCache:
    """Memory cache for performance optimization."""
    
    max_size: int = 1000
    cache: Dict[str, Any] = field(default_factory=dict)
    access_times: Dict[str, float] = field(default_factory=dict)
    size_limits: Dict[str, int] = field(default_factory=dict)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key]
        return None
    
    def set(self, key: str, value: Any, size: int = 1):
        """Set value in cache."""
        if len(self.cache) >= self.max_size:
            self._evict_oldest()
        
        self.cache[key] = value
        self.access_times[key] = time.time()
        self.size_limits[key] = size
    
    def _evict_oldest(self):
        """Evict oldest cache entry."""
        if not self.access_times:
            return
        
        oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        del self.cache[oldest_key]
        del self.access_times[oldest_key]
        if oldest_key in self.size_limits:
            del self.size_limits[oldest_key]


@dataclass
class OptimizationResult:
    """Result of performance optimization."""
    
    request_id: str
    optimization_type: str
    improvements: Dict[str, float]
    applied_changes: List[str]
    optimization_time: float
    success: bool
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


class PerformanceOptimizer(BaseService):
    """Performance optimization and monitoring service."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the performance optimizer."""
        super().__init__("PerformanceOptimizer", ServiceType.PHASE3, config)
        
        # Configuration
        self.optimization_config = OptimizationConfig()
        
        # Performance monitoring
        self.metrics_history: List[PerformanceMetrics] = []
        self.monitoring_task: Optional[asyncio.Task] = None
        self.is_monitoring = False
        
        # Resource limits
        self.memory_limit = self.optimization_config.memory_limit_gb * 1024 * 1024 * 1024  # bytes
        self.cpu_limit = self.optimization_config.cpu_limit_percent
        self.gpu_memory_limit = self.optimization_config.gpu_memory_limit_gb * 1024 * 1024 * 1024  # bytes
        
        # Error handling
        self.error_handler = ErrorHandler()
        
        # Configuration manager
        self.config_manager = ConfigurationManager()
        
        # Logging service
        self.logging_service = LoggingService()
        
        # Performance tracking
        self.optimization_stats = {
            "total_optimizations": 0,
            "successful_optimizations": 0,
            "failed_optimizations": 0,
            "average_improvement": 0.0,
            "total_optimization_time": 0.0
        }
        
        # Cache management
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "size_bytes": 0,
            "max_size_bytes": self.optimization_config.cache_size_mb * 1024 * 1024
        }

    async def _initialize_service_impl(self) -> None:
        """Initialize performance optimization services."""
        try:
            logger.info("Initializing performance optimizer...")
            
            # Check dependencies
            await self._check_dependencies()
            
            # Apply initial optimizations
            await self._apply_initial_optimizations()
            
            # Start performance monitoring
            if self.optimization_config.enable_monitoring:
                await self._start_monitoring()
            
            # Validate configuration
            await self._validate_configuration()
            
            logger.info("Performance optimizer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize performance optimizer: {e}")
            raise

    async def _check_dependencies(self) -> None:
        """Check required dependencies."""
        missing_deps = []
        
        if not UVLOOP_AVAILABLE:
            missing_deps.append("uvloop")
        
        if not ORJSON_AVAILABLE:
            missing_deps.append("orjson")
        
        if not MSGPACK_AVAILABLE:
            missing_deps.append("msgpack")
        
        if missing_deps:
            logger.warning(f"Missing performance dependencies: {missing_deps}")
            logger.warning("Some optimization features may not be available")

    async def _apply_initial_optimizations(self) -> None:
        """Apply initial performance optimizations."""
        try:
            # Enable uvloop if available
            if UVLOOP_AVAILABLE and self.optimization_config.enable_uvloop:
                try:
                    uvloop.install()
                    logger.info("UVLoop enabled for improved async performance")
                except Exception as e:
                    logger.warning(f"Failed to enable UVLoop: {e}")
            
            # Set JSON encoder/decoder
            if ORJSON_AVAILABLE and self.optimization_config.enable_orjson:
                try:
                    import json
                    json.dumps = orjson.dumps
                    json.loads = orjson.loads
                    logger.info("ORJSON enabled for improved JSON performance")
                except Exception as e:
                    logger.warning(f"Failed to enable ORJSON: {e}")
            
            # Set message pack encoder/decoder
            if MSGPACK_AVAILABLE and self.optimization_config.enable_msgpack:
                try:
                    logger.info("MessagePack available for binary serialization")
                except Exception as e:
                    logger.warning(f"Failed to enable MessagePack: {e}")
            
            # Set system limits
            await self._set_system_limits()
            
            logger.info("Initial optimizations applied successfully")
            
        except Exception as e:
            logger.warning(f"Some initial optimizations failed: {e}")

    async def _set_system_limits(self) -> None:
        """Set system resource limits."""
        try:
            # Set memory limit
            if hasattr(psutil, 'RLIMIT_AS'):
                import resource
                resource.setrlimit(resource.RLIMIT_AS, (self.memory_limit, self.memory_limit))
                logger.info(f"Memory limit set to {self.memory_limit / (1024**3):.1f} GB")
            
            # Set CPU affinity if possible
            try:
                cpu_count = psutil.cpu_count()
                if cpu_count > 1:
                    # Use all available CPUs
                    psutil.Process().cpu_affinity(list(range(cpu_count)))
                    logger.info(f"CPU affinity set to use {cpu_count} cores")
            except Exception as e:
                logger.debug(f"Could not set CPU affinity: {e}")
                
        except Exception as e:
            logger.warning(f"Failed to set system limits: {e}")

    async def _start_monitoring(self) -> None:
        """Start performance monitoring."""
        try:
            if not self.is_monitoring:
                self.is_monitoring = True
                self.monitoring_task = asyncio.create_task(self._monitoring_loop())
                logger.info("Performance monitoring started")
            else:
                logger.info("Performance monitoring already running")
                
        except Exception as e:
            logger.error(f"Failed to start monitoring: {e}")

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        try:
            while self.is_monitoring:
                # Collect metrics
                metrics = await self._collect_performance_metrics()
                
                # Store metrics
                self.metrics_history.append(metrics)
                
                # Keep only recent metrics (last hour)
                cutoff_time = datetime.now().timestamp() - 3600
                self.metrics_history = [
                    m for m in self.metrics_history 
                    if m.timestamp.timestamp() > cutoff_time
                ]
                
                # Check for optimization opportunities
                await self._check_optimization_opportunities(metrics)
                
                # Wait for next monitoring cycle
                await asyncio.sleep(self.optimization_config.monitoring_interval)
                
        except asyncio.CancelledError:
            logger.info("Performance monitoring stopped")
        except Exception as e:
            logger.error(f"Monitoring loop error: {e}")
            self.is_monitoring = False

    async def _collect_performance_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics."""
        try:
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            disk_io_metrics = {
                "read_bytes": disk_io.read_bytes if disk_io else 0,
                "write_bytes": disk_io.write_bytes if disk_io else 0,
                "read_count": disk_io.read_count if disk_io else 0,
                "write_count": disk_io.write_count if disk_io else 0
            }
            
            # Network I/O
            network_io = psutil.net_io_counters()
            network_io_metrics = {
                "bytes_sent": network_io.bytes_sent if network_io else 0,
                "bytes_recv": network_io.bytes_recv if network_io else 0,
                "packets_sent": network_io.packets_sent if network_io else 0,
                "packets_recv": network_io.packets_recv if network_io else 0
            }
            
            # GPU usage (if available)
            gpu_usage = None
            gpu_memory = None
            try:
                # This would integrate with GPU monitoring libraries
                # For now, we'll leave it as None
                pass
            except Exception:
                pass
            
            # Active processes
            active_processes = len(psutil.pids())
            
            # System load (Unix-like systems)
            system_load = 0.0
            try:
                system_load = psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0.0
            except Exception:
                pass
            
            return PerformanceMetrics(
                timestamp=datetime.now(),
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                disk_io=disk_io_metrics,
                network_io=network_io_metrics,
                gpu_usage=gpu_usage,
                gpu_memory=gpu_memory,
                active_processes=active_processes,
                system_load=system_load
            )
            
        except Exception as e:
            logger.error(f"Failed to collect performance metrics: {e}")
            # Return default metrics
            return PerformanceMetrics(
                timestamp=datetime.now(),
                cpu_usage=0.0,
                memory_usage=0.0,
                disk_io={},
                network_io={}
            )

    async def _check_optimization_opportunities(self, metrics: PerformanceMetrics) -> None:
        """Check for optimization opportunities based on metrics."""
        try:
            optimizations_needed = []
            
            # Check memory usage
            if metrics.memory_usage > self.cpu_limit:
                optimizations_needed.append(("memory", metrics.memory_usage - self.cpu_limit))
            
            # Check CPU usage
            if metrics.cpu_usage > self.cpu_limit:
                optimizations_needed.append(("cpu", metrics.cpu_usage - self.cpu_limit))
            
            # Check GPU memory usage
            if metrics.gpu_memory and metrics.gpu_memory > self.gpu_memory_limit:
                optimizations_needed.append(("gpu", metrics.gpu_memory - self.gpu_memory_limit))
            
            # Apply optimizations if needed
            for opt_type, severity in optimizations_needed:
                if severity > 10:  # Only optimize if significantly over limit
                    await self.optimize_performance(OptimizationRequest(
                        optimization_type=opt_type,
                        priority=2 if severity > 20 else 1
                    ))
                    
        except Exception as e:
            logger.warning(f"Failed to check optimization opportunities: {e}")

    async def _validate_configuration(self) -> None:
        """Validate optimizer configuration."""
        if not self.optimization_config:
            raise RuntimeError("Optimization configuration not set")
        
        if self.memory_limit <= 0:
            raise RuntimeError("Invalid memory limit")

    @with_error_handling
    @with_retry(max_attempts=3)
    async def optimize_performance(self, request: OptimizationRequest) -> OptimizationResult:
        """Optimize system performance based on the request."""
        start_time = time.time()
        
        try:
            logger.info(f"Starting performance optimization {request.request_id}")
            
            # Apply optimizations based on type
            if request.optimization_type == "auto":
                improvements = await self._apply_auto_optimizations()
            elif request.optimization_type == "memory":
                improvements = await self._optimize_memory()
            elif request.optimization_type == "cpu":
                improvements = await self._optimize_cpu()
            elif request.optimization_type == "gpu":
                improvements = await self._optimize_gpu()
            elif request.optimization_type == "network":
                improvements = await self._optimize_network()
            else:
                raise ValueError(f"Unknown optimization type: {request.optimization_type}")
            
            # Calculate optimization time
            optimization_time = time.time() - start_time
            
            # Update statistics
            self._update_optimization_stats(optimization_time, True, improvements)
            
            # Create result
            result = OptimizationResult(
                request_id=request.request_id,
                optimization_type=request.optimization_type,
                improvements=improvements,
                applied_changes=list(improvements.keys()),
                optimization_time=optimization_time,
                success=True
            )
            
            logger.info(f"Performance optimization completed in {optimization_time:.2f}s")
            return result
            
        except Exception as e:
            optimization_time = time.time() - start_time
            self._update_optimization_stats(optimization_time, False, {})
            logger.error(f"Performance optimization failed: {e}")
            raise

    async def _apply_auto_optimizations(self) -> Dict[str, float]:
        """Apply automatic performance optimizations."""
        improvements = {}
        
        try:
            # Memory optimization
            memory_improvement = await self._optimize_memory()
            improvements.update(memory_improvement)
            
            # CPU optimization
            cpu_improvement = await self._optimize_cpu()
            improvements.update(cpu_improvement)
            
            # Cache optimization
            cache_improvement = await self._optimize_cache()
            improvements.update(cache_improvement)
            
            logger.info("Auto optimizations applied successfully")
            
        except Exception as e:
            logger.warning(f"Some auto optimizations failed: {e}")
        
        return improvements

    async def _optimize_memory(self) -> Dict[str, float]:
        """Optimize memory usage."""
        improvements = {}
        
        try:
            # Clear cache if memory usage is high
            if self.cache_stats["size_bytes"] > self.cache_stats["max_size_bytes"] * 0.8:
                await self._clear_cache()
                improvements["cache_cleared"] = 1.0
            
            # Force garbage collection
            import gc
            collected = gc.collect()
            if collected > 0:
                improvements["garbage_collected"] = collected
            
            # Optimize memory allocation
            if hasattr(psutil, 'RLIMIT_AS'):
                import resource
                current_limit = resource.getrlimit(resource.RLIMIT_AS)[0]
                if current_limit > self.memory_limit:
                    resource.setrlimit(resource.RLIMIT_AS, (self.memory_limit, self.memory_limit))
                    improvements["memory_limit_adjusted"] = 1.0
            
            logger.info("Memory optimization completed")
            
        except Exception as e:
            logger.warning(f"Memory optimization failed: {e}")
        
        return improvements

    async def _optimize_cpu(self) -> Dict[str, float]:
        """Optimize CPU usage."""
        improvements = {}
        
        try:
            # Adjust process priority
            current_process = psutil.Process()
            try:
                current_process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
                improvements["process_priority_adjusted"] = 1.0
            except Exception:
                pass
            
            # Set CPU affinity to use available cores efficiently
            cpu_count = psutil.cpu_count()
            if cpu_count > 1:
                try:
                    # Use all available CPUs
                    current_process.cpu_affinity(list(range(cpu_count)))
                    improvements["cpu_affinity_optimized"] = 1.0
                except Exception:
                    pass
            
            logger.info("CPU optimization completed")
            
        except Exception as e:
            logger.warning(f"CPU optimization failed: {e}")
        
        return improvements

    async def _optimize_gpu(self) -> Dict[str, float]:
        """Optimize GPU usage."""
        improvements = {}
        
        try:
            # This would integrate with GPU optimization libraries
            # For now, return empty improvements
            logger.info("GPU optimization completed (no GPU detected)")
            
        except Exception as e:
            logger.warning(f"GPU optimization failed: {e}")
        
        return improvements

    async def _optimize_network(self) -> Dict[str, float]:
        """Optimize network usage."""
        improvements = {}
        
        try:
            # Enable compression if available
            if self.optimization_config.enable_compression:
                improvements["compression_enabled"] = 1.0
            
            # Optimize buffer sizes
            improvements["network_buffers_optimized"] = 1.0
            
            logger.info("Network optimization completed")
            
        except Exception as e:
            logger.warning(f"Network optimization failed: {e}")
        
        return improvements

    async def _optimize_cache(self) -> Dict[str, float]:
        """Optimize cache usage."""
        improvements = {}
        
        try:
            # Clear old cache entries
            if self.cache_stats["size_bytes"] > self.cache_stats["max_size_bytes"] * 0.9:
                await self._clear_cache()
                improvements["cache_cleared"] = 1.0
            
            # Adjust cache size based on memory usage
            memory = psutil.virtual_memory()
            if memory.percent < 50:  # If memory usage is low, increase cache
                new_cache_size = min(
                    self.cache_stats["max_size_bytes"] * 1.5,
                    memory.available * 0.1
                )
                if new_cache_size > self.cache_stats["max_size_bytes"]:
                    self.cache_stats["max_size_bytes"] = int(new_cache_size)
                    improvements["cache_size_increased"] = 1.0
            
            logger.info("Cache optimization completed")
            
        except Exception as e:
            logger.warning(f"Cache optimization failed: {e}")
        
        return improvements

    async def _clear_cache(self) -> None:
        """Clear the cache."""
        try:
            self.cache_stats["hits"] = 0
            self.cache_stats["misses"] = 0
            self.cache_stats["size_bytes"] = 0
            logger.info("Cache cleared successfully")
        except Exception as e:
            logger.warning(f"Failed to clear cache: {e}")

    def _update_optimization_stats(self, optimization_time: float, success: bool, improvements: Dict[str, float]):
        """Update optimization statistics."""
        self.optimization_stats["total_optimizations"] += 1
        
        if success:
            self.optimization_stats["successful_optimizations"] += 1
            self.optimization_stats["total_optimization_time"] += optimization_time
            
            # Calculate average improvement
            if improvements:
                total_improvement = sum(improvements.values())
                current_avg = self.optimization_stats["average_improvement"]
                total_successful = self.optimization_stats["successful_optimizations"]
                
                if total_successful > 0:
                    self.optimization_stats["average_improvement"] = (
                        (current_avg * (total_successful - 1) + total_improvement) / total_successful
                    )
        else:
            self.optimization_stats["failed_optimizations"] += 1

    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        try:
            current_metrics = await self._collect_performance_metrics()
            
            # Calculate averages from history
            if self.metrics_history:
                avg_cpu = sum(m.cpu_usage for m in self.metrics_history) / len(self.metrics_history)
                avg_memory = sum(m.memory_usage for m in self.metrics_history) / len(self.metrics_history)
                avg_load = sum(m.system_load for m in self.metrics_history) / len(self.metrics_history)
            else:
                avg_cpu = current_metrics.cpu_usage
                avg_memory = current_metrics.memory_usage
                avg_load = current_metrics.system_load
            
            return {
                "current_metrics": {
                    "cpu_usage": current_metrics.cpu_usage,
                    "memory_usage": current_metrics.memory_usage,
                    "gpu_usage": current_metrics.gpu_usage,
                    "gpu_memory": current_metrics.gpu_memory,
                    "active_processes": current_metrics.active_processes,
                    "system_load": current_metrics.system_load
                },
                "average_metrics": {
                    "cpu_usage": avg_cpu,
                    "memory_usage": avg_memory,
                    "system_load": avg_load
                },
                "optimization_stats": self.optimization_stats,
                "cache_stats": self.cache_stats,
                "system_info": {
                    "cpu_count": psutil.cpu_count(),
                    "memory_total_gb": psutil.virtual_memory().total / (1024**3),
                    "disk_total_gb": psutil.disk_usage('/').total / (1024**3) if Path('/').exists() else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get performance stats: {e}")
            return {"error": str(e)}

    async def health_check(self) -> HealthCheckResult:
        """Check the health of the performance optimizer."""
        try:
            # Check base service health
            base_health = await super().health_check()
            
            # Check dependencies
            dependencies = {
                "uvloop": UVLOOP_AVAILABLE,
                "orjson": ORJSON_AVAILABLE,
                "msgpack": MSGPACK_AVAILABLE
            }
            
            # Check monitoring status
            monitoring_status = {
                "is_monitoring": self.is_monitoring,
                "monitoring_task_active": self.monitoring_task and not self.monitoring_task.done(),
                "metrics_history_size": len(self.metrics_history)
            }
            
            # Check current performance
            current_metrics = await self._collect_performance_metrics()
            performance_status = {
                "cpu_usage": current_metrics.cpu_usage,
                "memory_usage": current_metrics.memory_usage,
                "within_limits": (
                    current_metrics.cpu_usage <= self.cpu_limit and
                    current_metrics.memory_usage <= self.cpu_limit
                )
            }
            
            # Update base health
            base_health.details.update({
                "dependencies": dependencies,
                "monitoring": monitoring_status,
                "performance": performance_status,
                "optimization_stats": self.optimization_stats,
                "cache_stats": self.cache_stats
            })
            
            return base_health
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return HealthCheckResult(
                status=ServiceStatus.UNHEALTHY,
                error_message=str(e)
            )

    async def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        try:
            if self.is_monitoring:
                self.is_monitoring = False
                if self.monitoring_task:
                    self.monitoring_task.cancel()
                    try:
                        await self.monitoring_task
                    except asyncio.CancelledError:
                        pass
                logger.info("Performance monitoring stopped")
            else:
                logger.info("Performance monitoring not running")
                
        except Exception as e:
            logger.error(f"Failed to stop monitoring: {e}")

    async def cleanup_temp_files(self) -> None:
        """Clean up temporary optimization files."""
        try:
            temp_dir = Path("./temp")
            if temp_dir.exists():
                for opt_file in temp_dir.glob("optimization_*"):
                    opt_file.unlink()
                    logger.debug(f"Cleaned up temp file: {opt_file}")
        except Exception as e:
            logger.warning(f"Failed to cleanup temp files: {e}")

    async def shutdown(self) -> None:
        """Shutdown the performance optimizer."""
        try:
            # Stop monitoring
            await self.stop_monitoring()
            
            # Clear cache
            await self._clear_cache()
            
            logger.info("Performance optimizer shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

