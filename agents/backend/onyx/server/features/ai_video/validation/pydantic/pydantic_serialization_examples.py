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
import json
import pickle
import gzip
import base64
import hashlib
import logging
from typing import Any, Dict, List, Optional, Union, Type, TypeVar, Callable
from functools import wraps
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
import weakref
from collections import defaultdict
import statistics
import numpy as np
import torch
from PIL import Image
import io
from pydantic import BaseModel, Field, ConfigDict, computed_field, field_validator
from datetime import datetime, timedelta
from pathlib import Path
            import psutil
        import gc
from typing import Any, List, Dict, Optional
"""
🚀 PYDANTIC SERIALIZATION EXAMPLES - AI VIDEO SYSTEM
====================================================

Practical examples demonstrating optimized data serialization and deserialization
using Pydantic for the AI Video system.

Features demonstrated:
- Optimized model configurations
- Caching strategies
- Compression techniques
- Batch processing
- Performance monitoring
- Real-world use cases
"""



logger = logging.getLogger(__name__)

# Type variables
T = TypeVar('T', bound=BaseModel)

# ============================================================================
# 1. OPTIMIZED PYDANTIC MODELS
# ============================================================================

class VideoStatus(str, Enum):
    """Video processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class VideoQuality(str, Enum):
    """Video quality levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"

class OptimizedVideoModel(BaseModel):
    """Optimized video model for serialization."""
    
    model_config = ConfigDict(
        # Performance optimizations
        validate_assignment=False,
        extra="forbid",
        use_enum_values=True,
        populate_by_name=True,
        validate_default=False,
        
        # JSON optimizations
        json_encoders={
            datetime: lambda v: v.isoformat(),
            Path: str,
            np.ndarray: lambda v: v.tolist(),
            torch.Tensor: lambda v: v.cpu().numpy().tolist()
        },
        
        # Serialization optimizations
        str_strip_whitespace=True,
        from_attributes=True,
        arbitrary_types_allowed=True
    )
    
    # Core fields
    video_id: str = Field(..., min_length=1, max_length=50, description="Video identifier")
    title: str = Field(..., max_length=200, description="Video title")
    duration: float = Field(..., ge=0, le=3600, description="Duration in seconds")
    quality: VideoQuality = Field(default=VideoQuality.MEDIUM, description="Video quality")
    
    # Optional fields with defaults
    description: Optional[str] = Field(default=None, max_length=1000, description="Video description")
    tags: List[str] = Field(default_factory=list, description="Video tags")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    
    # Computed fields (not serialized by default)
    @computed_field
    @property
    def duration_minutes(self) -> float:
        """Duration in minutes."""
        return self.duration / 60
    
    @computed_field
    @property
    def is_long_video(self) -> bool:
        """Check if video is longer than 5 minutes."""
        return self.duration > 300
    
    @computed_field
    @property
    def file_size_mb(self) -> float:
        """Estimated file size in MB."""
        base_size = 10  # Base size in MB
        quality_multiplier = {"low": 0.5, "medium": 1.0, "high": 2.0, "ultra": 4.0}
        return base_size * quality_multiplier[self.quality] * (self.duration / 60)
    
    # Validators
    @field_validator('title')
    @classmethod
    def validate_title(cls, v: str) -> str:
        """Validate and sanitize title."""
        if not v or not v.strip():
            raise ValueError('Title cannot be empty')
        return v.strip()
    
    @field_validator('tags')
    @classmethod
    def validate_tags(cls, v: List[str]) -> List[str]:
        """Validate and sanitize tags."""
        return [tag.strip().lower() for tag in v if tag.strip()]

class VideoProcessingResult(BaseModel):
    """Result of video processing operation."""
    
    model_config = ConfigDict(
        validate_assignment=False,
        extra="forbid",
        use_enum_values=True,
        json_encoders={
            datetime: lambda v: v.isoformat(),
            Path: str
        }
    )
    
    video_id: str = Field(..., description="Video identifier")
    status: VideoStatus = Field(..., description="Processing status")
    processing_time: float = Field(..., ge=0, description="Processing time in seconds")
    
    # URLs and paths
    output_url: Optional[str] = Field(default=None, description="Output video URL")
    thumbnail_url: Optional[str] = Field(default=None, description="Thumbnail URL")
    preview_url: Optional[str] = Field(default=None, description="Preview URL")
    
    # Error information
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    error_code: Optional[str] = Field(default=None, description="Error code if failed")
    
    # Metadata
    file_size: Optional[int] = Field(default=None, ge=0, description="File size in bytes")
    resolution: Optional[str] = Field(default=None, description="Video resolution")
    format: Optional[str] = Field(default=None, description="Video format")
    
    # Timestamps
    started_at: datetime = Field(default_factory=datetime.now, description="Processing start time")
    completed_at: Optional[datetime] = Field(default=None, description="Processing completion time")
    
    @computed_field
    @property
    def is_completed(self) -> bool:
        """Check if processing is completed."""
        return self.status == VideoStatus.COMPLETED
    
    @computed_field
    @property
    def is_failed(self) -> bool:
        """Check if processing failed."""
        return self.status == VideoStatus.FAILED
    
    @computed_field
    @property
    def file_size_mb(self) -> Optional[float]:
        """File size in MB."""
        return self.file_size / (1024 * 1024) if self.file_size else None

class VideoBatchRequest(BaseModel):
    """Batch video processing request."""
    
    model_config = ConfigDict(
        validate_assignment=False,
        extra="forbid",
        use_enum_values=True
    )
    
    batch_id: str = Field(..., description="Batch identifier")
    videos: List[OptimizedVideoModel] = Field(..., min_length=1, max_length=100, description="Videos to process")
    priority: str = Field(default="normal", description="Processing priority")
    
    # Batch settings
    parallel_processing: bool = Field(default=True, description="Enable parallel processing")
    quality_override: Optional[VideoQuality] = Field(default=None, description="Override quality for all videos")
    
    # Metadata
    user_id: Optional[str] = Field(default=None, description="User identifier")
    project_id: Optional[str] = Field(default=None, description="Project identifier")
    
    @field_validator('videos')
    @classmethod
    def validate_batch_size(cls, v: List[OptimizedVideoModel]) -> List[OptimizedVideoModel]:
        """Validate batch size and apply quality override."""
        if len(v) > 100:
            raise ValueError("Batch size cannot exceed 100 videos")
        
        return v
    
    @computed_field
    @property
    def total_duration(self) -> float:
        """Total duration of all videos."""
        return sum(video.duration for video in self.videos)
    
    @computed_field
    @property
    def estimated_processing_time(self) -> float:
        """Estimated processing time in seconds."""
        base_time_per_minute = 30  # seconds per minute of video
        return self.total_duration / 60 * base_time_per_minute
    
    @computed_field
    @property
    def estimated_file_size_mb(self) -> float:
        """Estimated total file size in MB."""
        return sum(video.file_size_mb for video in self.videos)

# ============================================================================
# 2. SERIALIZATION CACHE IMPLEMENTATION
# ============================================================================

class SerializationCache:
    """Cache for serialized data to improve performance."""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        
    """__init__ function."""
self.max_size = max_size
        self.ttl = ttl
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, float] = {}
        self._lock = asyncio.Lock()
    
    def _generate_key(self, data: Any, options: Dict[str, Any]) -> str:
        """Generate cache key for data and options."""
        data_str = str(data)
        options_str = str(sorted(options.items()))
        content = f"{data_str}:{options_str}"
        return hashlib.md5(content.encode()).hexdigest()
    
    async def get(self, data: Any, options: Dict[str, Any]) -> Optional[bytes]:
        """Get serialized data from cache."""
        key = self._generate_key(data, options)
        
        async with self._lock:
            if key in self.cache:
                cache_entry = self.cache[key]
                
                # Check TTL
                if time.time() - cache_entry["timestamp"] > self.ttl:
                    del self.cache[key]
                    del self.access_times[key]
                    return None
                
                # Update access time
                self.access_times[key] = time.time()
                return cache_entry["data"]
        
        return None
    
    async def set(self, data: Any, options: Dict[str, Any], serialized_data: bytes):
        """Store serialized data in cache."""
        key = self._generate_key(data, options)
        
        async with self._lock:
            # Check if cache is full
            if len(self.cache) >= self.max_size:
                await self._evict_oldest()
            
            self.cache[key] = {
                "data": serialized_data,
                "timestamp": time.time(),
                "size": len(serialized_data)
            }
            self.access_times[key] = time.time()
    
    async def _evict_oldest(self) -> Any:
        """Evict oldest cache entries."""
        if not self.access_times:
            return
        
        # Find oldest entry
        oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        
        # Remove from cache
        del self.cache[oldest_key]
        del self.access_times[oldest_key]
    
    async def clear(self) -> Any:
        """Clear all cache entries."""
        async with self._lock:
            self.cache.clear()
            self.access_times.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_size = sum(entry["size"] for entry in self.cache.values())
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "total_size_bytes": total_size,
            "ttl": self.ttl
        }

# ============================================================================
# 3. OPTIMIZED SERIALIZER
# ============================================================================

class OptimizedSerializer:
    """Optimized serializer with caching and compression."""
    
    def __init__(self, enable_caching: bool = True, enable_compression: bool = True):
        
    """__init__ function."""
self.cache = SerializationCache() if enable_caching else None
        self.enable_compression = enable_compression
        self.stats = {
            "total_serializations": 0,
            "total_deserializations": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_serialization_time": 0.0,
            "total_deserialization_time": 0.0
        }
    
    async def serialize(self, data: BaseModel, options: Dict[str, Any] = None) -> bytes:
        """Serialize data with optimization."""
        options = options or {}
        start_time = time.time()
        
        try:
            # Check cache first
            if self.cache:
                cached_result = await self.cache.get(data, options)
                if cached_result is not None:
                    self.stats["cache_hits"] += 1
                    return cached_result
                
                self.stats["cache_misses"] += 1
            
            # Serialize data
            serialized = data.model_dump_json(**options).encode()
            
            # Compress if enabled and beneficial
            if self.enable_compression and len(serialized) > 1024:
                compressed = gzip.compress(serialized, compresslevel=6)
                
                # Add compression header
                header = {
                    "compressed": True,
                    "original_size": len(serialized),
                    "compressed_size": len(compressed)
                }
                
                final_data = json.dumps(header).encode() + b"\n" + compressed
                serialized = final_data
            
            # Cache result
            if self.cache:
                await self.cache.set(data, options, serialized)
            
            # Update statistics
            self.stats["total_serializations"] += 1
            self.stats["total_serialization_time"] += time.time() - start_time
            
            return serialized
            
        except Exception as e:
            logger.error(f"Serialization failed: {e}")
            raise
    
    async def deserialize(self, data: bytes, model_class: Type[T]) -> T:
        """Deserialize data with optimization."""
        start_time = time.time()
        
        try:
            # Check for compression header
            if data.startswith(b"{"):
                header_end = data.find(b"\n")
                if header_end != -1:
                    header_data = data[:header_end]
                    compressed_data = data[header_end + 1:]
                    
                    header = json.loads(header_data.decode())
                    
                    if header.get("compressed", False):
                        # Decompress data
                        decompressed = gzip.decompress(compressed_data)
                        data = decompressed
            
            # Deserialize data
            json_data = data.decode('utf-8')
            result = model_class.model_validate_json(json_data)
            
            # Update statistics
            self.stats["total_deserializations"] += 1
            self.stats["total_deserialization_time"] += time.time() - start_time
            
            return result
            
        except Exception as e:
            logger.error(f"Deserialization failed: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get serialization statistics."""
        return {
            "total_serializations": self.stats["total_serializations"],
            "total_deserializations": self.stats["total_deserializations"],
            "cache_hits": self.stats["cache_hits"],
            "cache_misses": self.stats["cache_misses"],
            "cache_hit_rate": (
                self.stats["cache_hits"] / (self.stats["cache_hits"] + self.stats["cache_misses"])
                if (self.stats["cache_hits"] + self.stats["cache_misses"]) > 0 else 0.0
            ),
            "avg_serialization_time": (
                self.stats["total_serialization_time"] / self.stats["total_serializations"]
                if self.stats["total_serializations"] > 0 else 0.0
            ),
            "avg_deserialization_time": (
                self.stats["total_deserialization_time"] / self.stats["total_deserializations"]
                if self.stats["total_deserializations"] > 0 else 0.0
            ),
            "cache_stats": self.cache.get_stats() if self.cache else None
        }

# ============================================================================
# 4. BATCH PROCESSING OPTIMIZER
# ============================================================================

class BatchSerializationOptimizer:
    """Optimizer for batch serialization operations."""
    
    def __init__(self, serializer: OptimizedSerializer):
        
    """__init__ function."""
self.serializer = serializer
        self.batch_stats = defaultdict(int)
    
    async def serialize_batch(
        self, 
        items: List[BaseModel], 
        options: Dict[str, Any] = None
    ) -> List[bytes]:
        """Serialize a batch of items efficiently."""
        options = options or {}
        start_time = time.time()
        
        # Use asyncio.gather for concurrent serialization
        tasks = [
            self.serializer.serialize(item, options) 
            for item in items
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        successful_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to serialize item {i}: {result}")
            else:
                successful_results.append(result)
        
        # Update batch statistics
        self.batch_stats["total_batches"] += 1
        self.batch_stats["total_items"] += len(items)
        self.batch_stats["successful_items"] += len(successful_results)
        self.batch_stats["batch_time"] += time.time() - start_time
        
        return successful_results
    
    async def deserialize_batch(
        self, 
        data_list: List[bytes], 
        model_class: Type[T]
    ) -> List[T]:
        """Deserialize a batch of data efficiently."""
        start_time = time.time()
        
        # Use asyncio.gather for concurrent deserialization
        tasks = [
            self.serializer.deserialize(data, model_class) 
            for data in data_list
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        successful_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to deserialize item {i}: {result}")
            else:
                successful_results.append(result)
        
        # Update batch statistics
        self.batch_stats["total_deserialize_batches"] += 1
        self.batch_stats["total_deserialize_items"] += len(data_list)
        self.batch_stats["successful_deserialize_items"] += len(successful_results)
        self.batch_stats["batch_deserialize_time"] += time.time() - start_time
        
        return successful_results
    
    def get_batch_stats(self) -> Dict[str, Any]:
        """Get batch serialization statistics."""
        return {
            "total_batches": self.batch_stats["total_batches"],
            "total_items": self.batch_stats["total_items"],
            "successful_items": self.batch_stats["successful_items"],
            "success_rate": (
                self.batch_stats["successful_items"] / self.batch_stats["total_items"]
                if self.batch_stats["total_items"] > 0 else 0.0
            ),
            "avg_batch_time": (
                self.batch_stats["batch_time"] / self.batch_stats["total_batches"]
                if self.batch_stats["total_batches"] > 0 else 0.0
            ),
            "total_deserialize_batches": self.batch_stats["total_deserialize_batches"],
            "total_deserialize_items": self.batch_stats["total_deserialize_items"],
            "successful_deserialize_items": self.batch_stats["successful_deserialize_items"],
            "deserialize_success_rate": (
                self.batch_stats["successful_deserialize_items"] / self.batch_stats["total_deserialize_items"]
                if self.batch_stats["total_deserialize_items"] > 0 else 0.0
            ),
            "avg_batch_deserialize_time": (
                self.batch_stats["batch_deserialize_time"] / self.batch_stats["total_deserialize_batches"]
                if self.batch_stats["total_deserialize_batches"] > 0 else 0.0
            )
        }

# ============================================================================
# 5. PERFORMANCE MONITORING
# ============================================================================

class SerializationPerformanceMonitor:
    """Monitor serialization performance."""
    
    def __init__(self) -> Any:
        self.performance_data = defaultdict(list)
        self.alerts = []
    
    @asynccontextmanager
    async def monitor_serialization(self, operation: str, data_size: int = 0):
        """Monitor serialization operation."""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            duration = end_time - start_time
            memory_delta = end_memory - start_memory
            
            self.performance_data[operation].append({
                "duration": duration,
                "memory_delta": memory_delta,
                "data_size": data_size,
                "timestamp": time.time()
            })
            
            # Keep only recent data
            if len(self.performance_data[operation]) > 1000:
                self.performance_data[operation] = self.performance_data[operation][-1000:]
            
            # Check for performance alerts
            if duration > 1.0:  # More than 1 second
                self.alerts.append({
                    "type": "slow_serialization",
                    "operation": operation,
                    "duration": duration,
                    "timestamp": time.time()
                })
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            return psutil.Process().memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        report = {
            "operations": {},
            "alerts": self.alerts[-10:],  # Last 10 alerts
            "timestamp": time.time()
        }
        
        for operation, data in self.performance_data.items():
            if data:
                durations = [item["duration"] for item in data]
                memory_deltas = [item["memory_delta"] for item in data]
                data_sizes = [item["data_size"] for item in data]
                
                report["operations"][operation] = {
                    "total_operations": len(data),
                    "avg_duration": statistics.mean(durations),
                    "max_duration": max(durations),
                    "min_duration": min(durations),
                    "avg_memory_delta": statistics.mean(memory_deltas),
                    "avg_data_size": statistics.mean(data_sizes) if data_sizes else 0,
                    "throughput": len(data) / sum(durations) if sum(durations) > 0 else 0
                }
        
        return report
    
    def clear_alerts(self) -> Any:
        """Clear performance alerts."""
        self.alerts.clear()

# ============================================================================
# 6. PRACTICAL EXAMPLES
# ============================================================================

async def example_1_basic_serialization():
    """Example 1: Basic serialization optimization."""
    print("=== Example 1: Basic Serialization Optimization ===")
    
    # Create optimized video model
    video = OptimizedVideoModel(
        video_id="video_123",
        title="Sample Video",
        duration=180.5,
        quality=VideoQuality.HIGH,
        tags=["sample", "demo", "test"],
        metadata={"quality": "high", "format": "mp4", "size": 1024*1024*50}
    )
    
    # Create serializer
    serializer = OptimizedSerializer(enable_caching=True, enable_compression=True)
    
    # Serialization options
    options = {
        "exclude_none": True,
        "use_enum_values": True,
        "separators": (',', ':')
    }
    
    # Serialize
    start_time = time.time()
    serialized = await serializer.serialize(video, options)
    serialization_time = time.time() - start_time
    
    print(f"Video: {video.title}")
    print(f"Serialization time: {serialization_time:.4f}s")
    print(f"Serialized size: {len(serialized)} bytes")
    print(f"Original size: {len(str(video))} bytes")
    
    # Deserialize
    start_time = time.time()
    deserialized = await serializer.deserialize(serialized, OptimizedVideoModel)
    deserialization_time = time.time() - start_time
    
    print(f"Deserialization time: {deserialization_time:.4f}s")
    print(f"Deserialized video ID: {deserialized.video_id}")
    print(f"Duration in minutes: {deserialized.duration_minutes}")
    print(f"File size estimate: {deserialized.file_size_mb:.2f} MB")
    
    # Get statistics
    stats = serializer.get_stats()
    print(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")
    print(f"Average serialization time: {stats['avg_serialization_time']:.4f}s")
    print()

async def example_2_batch_processing():
    """Example 2: Batch processing optimization."""
    print("=== Example 2: Batch Processing Optimization ===")
    
    # Create batch of videos
    video_batch = [
        OptimizedVideoModel(
            video_id=f"video_{i}",
            title=f"Batch Video {i}",
            duration=120.0 + i * 10,
            quality=VideoQuality.HIGH if i % 2 == 0 else VideoQuality.MEDIUM,
            tags=[f"batch_{i}", "processing"],
            metadata={"batch_index": i, "priority": "normal"}
        )
        for i in range(50)
    ]
    
    # Create batch optimizer
    serializer = OptimizedSerializer(enable_caching=True, enable_compression=True)
    batch_optimizer = BatchSerializationOptimizer(serializer)
    
    # Serialization options
    options = {
        "exclude_none": True,
        "use_enum_values": True,
        "separators": (',', ':')
    }
    
    # Serialize batch
    start_time = time.time()
    serialized_batch = await batch_optimizer.serialize_batch(video_batch, options)
    batch_time = time.time() - start_time
    
    print(f"Batch size: {len(video_batch)} videos")
    print(f"Batch serialization time: {batch_time:.4f}s")
    print(f"Average time per video: {batch_time / len(video_batch):.4f}s")
    print(f"Total serialized size: {sum(len(data) for data in serialized_batch)} bytes")
    
    # Deserialize batch
    start_time = time.time()
    deserialized_batch = await batch_optimizer.deserialize_batch(
        serialized_batch, OptimizedVideoModel
    )
    batch_deserialize_time = time.time() - start_time
    
    print(f"Batch deserialization time: {batch_deserialize_time:.4f}s")
    print(f"Average deserialize time per video: {batch_deserialize_time / len(deserialized_batch):.4f}s")
    
    # Get batch statistics
    batch_stats = batch_optimizer.get_batch_stats()
    print(f"Batch success rate: {batch_stats['success_rate']:.2%}")
    print(f"Average batch time: {batch_stats['avg_batch_time']:.4f}s")
    
    # Get serializer statistics
    serializer_stats = serializer.get_stats()
    print(f"Overall cache hit rate: {serializer_stats['cache_hit_rate']:.2%}")
    print()

async def example_3_performance_monitoring():
    """Example 3: Performance monitoring."""
    print("=== Example 3: Performance Monitoring ===")
    
    # Create performance monitor
    monitor = SerializationPerformanceMonitor()
    serializer = OptimizedSerializer(enable_caching=True, enable_compression=True)
    
    # Create test data
    test_videos = [
        OptimizedVideoModel(
            video_id=f"test_video_{i}",
            title=f"Test Video {i}",
            duration=60.0 + i * 5,
            quality=VideoQuality.HIGH,
            metadata={"test": True, "index": i}
        )
        for i in range(100)
    ]
    
    # Monitor serialization operations
    for i, video in enumerate(test_videos):
        async with monitor.monitor_serialization("video_serialization", len(str(video))):
            await serializer.serialize(video, {"exclude_none": True})
        
        # Monitor deserialization (simulate)
        if i % 10 == 0:  # Every 10th video
            async with monitor.monitor_serialization("video_deserialization", 1000):
                await asyncio.sleep(0.01)  # Simulate deserialization
    
    # Get performance report
    performance_report = monitor.get_performance_report()
    
    print("Performance Report:")
    for operation, stats in performance_report["operations"].items():
        print(f"  {operation}:")
        print(f"    Total operations: {stats['total_operations']}")
        print(f"    Average duration: {stats['avg_duration']:.4f}s")
        print(f"    Max duration: {stats['max_duration']:.4f}s")
        print(f"    Throughput: {stats['throughput']:.2f} ops/sec")
        print(f"    Average memory delta: {stats['avg_memory_delta']:.2f} MB")
    
    if performance_report["alerts"]:
        print(f"  Alerts: {len(performance_report['alerts'])}")
        for alert in performance_report["alerts"][-3:]:  # Last 3 alerts
            print(f"    - {alert['type']}: {alert['duration']:.4f}s")
    print()

async def example_4_video_processing_pipeline():
    """Example 4: Video processing pipeline with optimized serialization."""
    print("=== Example 4: Video Processing Pipeline ===")
    
    # Create video processing pipeline
    serializer = OptimizedSerializer(enable_caching=True, enable_compression=True)
    batch_optimizer = BatchSerializationOptimizer(serializer)
    monitor = SerializationPerformanceMonitor()
    
    # Create batch request
    batch_request = VideoBatchRequest(
        batch_id="batch_001",
        videos=[
            OptimizedVideoModel(
                video_id=f"pipeline_video_{i}",
                title=f"Pipeline Video {i}",
                duration=90.0 + i * 15,
                quality=VideoQuality.HIGH,
                tags=["pipeline", "processing"],
                metadata={"pipeline": True, "priority": "high"}
            )
            for i in range(20)
        ],
        priority="high",
        parallel_processing=True,
        user_id="user_123",
        project_id="project_456"
    )
    
    print(f"Batch ID: {batch_request.batch_id}")
    print(f"Total videos: {len(batch_request.videos)}")
    print(f"Total duration: {batch_request.total_duration:.2f} seconds")
    print(f"Estimated processing time: {batch_request.estimated_processing_time:.2f} seconds")
    print(f"Estimated file size: {batch_request.estimated_file_size_mb:.2f} MB")
    
    # Process batch request
    async with monitor.monitor_serialization("batch_request_serialization", len(str(batch_request))):
        serialized_request = await serializer.serialize(batch_request)
    
    print(f"Serialized request size: {len(serialized_request)} bytes")
    
    # Simulate processing results
    processing_results = []
    for video in batch_request.videos:
        result = VideoProcessingResult(
            video_id=video.video_id,
            status=VideoStatus.COMPLETED,
            processing_time=video.duration / 60 * 30,  # 30 seconds per minute
            output_url=f"/output/{video.video_id}.mp4",
            thumbnail_url=f"/thumbnails/{video.video_id}.jpg",
            file_size=int(video.file_size_mb * 1024 * 1024),
            resolution="1920x1080",
            format="mp4",
            completed_at=datetime.now()
        )
        processing_results.append(result)
    
    # Serialize results
    async with monitor.monitor_serialization("results_serialization", len(str(processing_results))):
        serialized_results = await batch_optimizer.serialize_batch(
            processing_results,
            {"exclude_none": True, "use_enum_values": True}
        )
    
    print(f"Serialized results size: {sum(len(r) for r in serialized_results)} bytes")
    
    # Get performance metrics
    performance_report = monitor.get_performance_report()
    batch_stats = batch_optimizer.get_batch_stats()
    serializer_stats = serializer.get_stats()
    
    print(f"Pipeline performance:")
    print(f"  Cache hit rate: {serializer_stats['cache_hit_rate']:.2%}")
    print(f"  Batch success rate: {batch_stats['success_rate']:.2%}")
    print(f"  Average serialization time: {serializer_stats['avg_serialization_time']:.4f}s")
    print()

async def example_5_memory_efficient_processing():
    """Example 5: Memory-efficient processing for large datasets."""
    print("=== Example 5: Memory-Efficient Processing ===")
    
    # Create large dataset
    large_dataset = [
        OptimizedVideoModel(
            video_id=f"large_video_{i}",
            title=f"Large Dataset Video {i}",
            duration=300.0 + i * 10,  # 5+ minutes each
            quality=VideoQuality.ULTRA,
            tags=["large_dataset", "memory_test"],
            metadata={
                "large": True,
                "index": i,
                "complex_metadata": {
                    "analysis_results": [j for j in range(100)],
                    "processing_steps": [f"step_{k}" for k in range(50)]
                }
            }
        )
        for i in range(1000)  # 1000 videos
    ]
    
    print(f"Large dataset: {len(large_dataset)} videos")
    print(f"Total estimated size: {sum(v.file_size_mb for v in large_dataset):.2f} MB")
    
    # Process in chunks to manage memory
    chunk_size = 50
    serializer = OptimizedSerializer(enable_caching=False, enable_compression=True)  # Disable cache for memory efficiency
    monitor = SerializationPerformanceMonitor()
    
    total_serialized_size = 0
    total_processing_time = 0
    
    for i in range(0, len(large_dataset), chunk_size):
        chunk = large_dataset[i:i + chunk_size]
        
        # Process chunk
        async with monitor.monitor_serialization(f"chunk_{i//chunk_size}", len(str(chunk))):
            start_time = time.time()
            
            # Serialize chunk
            serialized_chunk = await batch_optimizer.serialize_batch(
                chunk,
                {"exclude_none": True, "use_enum_values": True}
            )
            
            chunk_time = time.time() - start_time
            chunk_size_bytes = sum(len(data) for data in serialized_chunk)
            
            total_processing_time += chunk_time
            total_serialized_size += chunk_size_bytes
            
            print(f"  Chunk {i//chunk_size + 1}: {len(chunk)} videos, "
                  f"{chunk_time:.4f}s, {chunk_size_bytes/1024:.2f} KB")
        
        # Force garbage collection
        gc.collect()
    
    print(f"Total processing time: {total_processing_time:.4f}s")
    print(f"Total serialized size: {total_serialized_size/1024/1024:.2f} MB")
    print(f"Average time per video: {total_processing_time / len(large_dataset):.4f}s")
    print(f"Average size per video: {total_serialized_size / len(large_dataset):.0f} bytes")
    
    # Get performance report
    performance_report = monitor.get_performance_report()
    print(f"Memory-efficient processing completed successfully!")
    print()

# ============================================================================
# 7. MAIN EXECUTION
# ============================================================================

async def run_all_examples():
    """Run all serialization examples."""
    print("🚀 PYDANTIC SERIALIZATION OPTIMIZATION EXAMPLES")
    print("=" * 60)
    print()
    
    try:
        await example_1_basic_serialization()
        await example_2_batch_processing()
        await example_3_performance_monitoring()
        await example_4_video_processing_pipeline()
        await example_5_memory_efficient_processing()
        
        print("✅ All examples completed successfully!")
        print()
        print("Key Benefits Demonstrated:")
        print("  • Optimized serialization with caching")
        print("  • Compression for network efficiency")
        print("  • Batch processing for throughput")
        print("  • Performance monitoring and alerts")
        print("  • Memory-efficient large dataset handling")
        print("  • Type safety with Pydantic models")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        logger.error(f"Example execution failed: {e}")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run examples
    asyncio.run(run_all_examples()) 