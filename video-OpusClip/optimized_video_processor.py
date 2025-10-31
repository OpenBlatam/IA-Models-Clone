"""
Optimized Video Processor for Video-OpusClip

High-performance video processing with async support, memory optimization,
and intelligent resource management.
"""

import asyncio
import time
import hashlib
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import structlog

# Optional high-performance imports
try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except ImportError:
    cv2 = None
    np = None
    OPENCV_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

try:
    from moviepy.editor import VideoFileClip, AudioFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    VideoFileClip = None
    AudioFileClip = None
    MOVIEPY_AVAILABLE = False

from .optimized_config import get_config
from .optimized_cache import get_cache_manager, cached
from .models.video_models import VideoClipRequest, VideoClip, VideoClipResponse
from .utils.parallel_utils import HybridParallelProcessor, ParallelConfig

logger = structlog.get_logger()

# =============================================================================
# OPTIMIZED VIDEO PROCESSOR CONFIGURATION
# =============================================================================

@dataclass
class OptimizedVideoProcessorConfig:
    """Configuration for optimized video processing."""
    
    # Processing Settings
    max_concurrent_videos: int = 4
    max_video_duration: float = 600.0  # 10 minutes
    target_fps: int = 30
    target_resolution: Tuple[int, int] = (1920, 1080)
    
    # Quality Settings
    enable_hardware_acceleration: bool = True
    codec: str = "h264"
    bitrate: str = "2M"
    quality_preset: str = "fast"  # fast, balanced, quality
    
    # Memory Management
    enable_memory_pooling: bool = True
    max_memory_usage_mb: int = 4096
    enable_garbage_collection: bool = True
    gc_threshold: int = 100
    
    # Caching
    enable_processing_cache: bool = True
    cache_processed_clips: bool = True
    cache_analysis_results: bool = True
    
    # Performance
    enable_parallel_processing: bool = True
    enable_async_processing: bool = True
    enable_batch_processing: bool = True
    optimal_batch_size: int = 8

# =============================================================================
# MEMORY MANAGER
# =============================================================================

class MemoryManager:
    """Intelligent memory management for video processing."""
    
    def __init__(self, max_memory_mb: int = 4096):
        self.max_memory_mb = max_memory_mb
        self.current_usage = 0
        self.memory_pool = {}
        self.gc_counter = 0
        
    def check_memory_available(self, required_mb: int) -> bool:
        """Check if required memory is available."""
        try:
            import psutil
            available_memory = psutil.virtual_memory().available // (1024 * 1024)
            return available_memory >= required_mb
        except ImportError:
            # Fallback: assume memory is available
            return True
    
    def allocate_memory(self, key: str, size_mb: int) -> bool:
        """Allocate memory for processing."""
        if not self.check_memory_available(size_mb):
            logger.warning("Insufficient memory", required=size_mb, available=self.get_available_memory())
            return False
        
        self.memory_pool[key] = size_mb
        self.current_usage += size_mb
        logger.debug("Memory allocated", key=key, size=size_mb, total=self.current_usage)
        return True
    
    def release_memory(self, key: str) -> bool:
        """Release allocated memory."""
        if key in self.memory_pool:
            size = self.memory_pool.pop(key)
            self.current_usage -= size
            logger.debug("Memory released", key=key, size=size, total=self.current_usage)
            return True
        return False
    
    def get_available_memory(self) -> int:
        """Get available system memory in MB."""
        try:
            import psutil
            return psutil.virtual_memory().available // (1024 * 1024)
        except ImportError:
            return 8192  # Default to 8GB
    
    def should_garbage_collect(self) -> bool:
        """Check if garbage collection should be triggered."""
        self.gc_counter += 1
        return self.gc_counter >= 100
    
    def force_garbage_collection(self):
        """Force garbage collection."""
        import gc
        gc.collect()
        self.gc_counter = 0
        logger.debug("Garbage collection performed")

# =============================================================================
# OPTIMIZED VIDEO PROCESSOR
# =============================================================================

class OptimizedVideoProcessor:
    """High-performance video processor with async support and memory optimization."""
    
    def __init__(self, config: Optional[OptimizedVideoProcessorConfig] = None):
        self.config = config or OptimizedVideoProcessorConfig()
        self.memory_manager = MemoryManager(self.config.max_memory_usage_mb)
        self.parallel_processor = HybridParallelProcessor(ParallelConfig())
        self.cache_manager = get_cache_manager()
        self.config_instance = get_config()
        
        # Initialize processing pools
        self._initialize_pools()
        
        # Performance tracking
        self.processing_stats = {
            "total_processed": 0,
            "total_processing_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "memory_allocations": 0,
            "memory_releases": 0
        }
    
    def _initialize_pools(self):
        """Initialize processing pools."""
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.config.max_concurrent_videos,
            thread_name_prefix="video_processor"
        )
        
        if self.config.enable_parallel_processing:
            self.process_pool = ProcessPoolExecutor(
                max_workers=self.config.max_concurrent_videos
            )
        else:
            self.process_pool = None
    
    async def process_video(self, request: VideoClipRequest) -> VideoClipResponse:
        """Process a single video with optimization."""
        start_time = time.perf_counter()
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(request)
            cached_result = await self.cache_manager.get_video_analysis(
                request.youtube_url,
                request.language,
                request.target_platform
            )
            
            if cached_result and self.config.enable_processing_cache:
                self.processing_stats["cache_hits"] += 1
                logger.info("Cache hit for video processing", url=request.youtube_url)
                return VideoClipResponse(**cached_result)
            
            self.processing_stats["cache_misses"] += 1
            
            # Allocate memory for processing
            estimated_memory = self._estimate_memory_usage(request)
            if not self.memory_manager.allocate_memory(cache_key, estimated_memory):
                raise MemoryError(f"Insufficient memory for processing: {estimated_memory}MB required")
            
            # Process video
            result = await self._process_video_async(request)
            
            # Cache result
            if self.config.cache_processed_clips:
                await self.cache_manager.set_video_analysis(
                    request.youtube_url,
                    request.language,
                    request.target_platform,
                    result.dict()
                )
            
            # Update stats
            processing_time = time.perf_counter() - start_time
            self.processing_stats["total_processed"] += 1
            self.processing_stats["total_processing_time"] += processing_time
            
            logger.info(
                "Video processed successfully",
                url=request.youtube_url,
                processing_time=processing_time,
                memory_used=estimated_memory
            )
            
            return result
            
        except Exception as e:
            logger.error("Video processing failed", url=request.youtube_url, error=str(e))
            raise
        finally:
            # Release memory
            self.memory_manager.release_memory(cache_key)
            
            # Garbage collection if needed
            if self.memory_manager.should_garbage_collect():
                self.memory_manager.force_garbage_collection()
    
    async def process_video_batch(
        self,
        requests: List[VideoClipRequest],
        max_concurrent: Optional[int] = None
    ) -> List[VideoClipResponse]:
        """Process multiple videos in parallel with optimization."""
        if not requests:
            return []
        
        max_concurrent = max_concurrent or self.config.max_concurrent_videos
        
        # Group requests by estimated memory usage
        grouped_requests = self._group_requests_by_memory(requests)
        
        results = []
        for group in grouped_requests:
            # Process group with controlled concurrency
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def process_with_semaphore(req: VideoClipRequest) -> VideoClipResponse:
                async with semaphore:
                    return await self.process_video(req)
            
            group_results = await asyncio.gather(
                *[process_with_semaphore(req) for req in group],
                return_exceptions=True
            )
            
            # Handle exceptions
            for i, result in enumerate(group_results):
                if isinstance(result, Exception):
                    logger.error("Batch processing error", request=group[i].youtube_url, error=str(result))
                    # Create fallback response
                    fallback_response = self._create_fallback_response(group[i])
                    results.append(fallback_response)
                else:
                    results.append(result)
        
        return results
    
    async def _process_video_async(self, request: VideoClipRequest) -> VideoClipResponse:
        """Async video processing implementation."""
        # This is a placeholder for the actual video processing logic
        # In a real implementation, this would:
        # 1. Download video from YouTube
        # 2. Extract audio and video streams
        # 3. Analyze content for optimal clip points
        # 4. Generate captions and emojis
        # 5. Create optimized clips
        
        # Simulate processing time
        await asyncio.sleep(0.1)
        
        # Generate optimized clips
        clips = await self._generate_optimized_clips(request)
        
        return VideoClipResponse(
            youtube_url=request.youtube_url,
            clips=clips,
            logo_path=request.logo_path,
            language=request.language
        )
    
    async def _generate_optimized_clips(self, request: VideoClipRequest) -> List[VideoClip]:
        """Generate optimized video clips with AI analysis."""
        # This would integrate with the LangChain processor for intelligent clip generation
        # For now, return optimized dummy clips
        
        clips = []
        clip_length = min(request.max_clip_length, 30.0)  # Optimize for short-form
        
        # Generate clips based on optimal timing
        optimal_timings = self._calculate_optimal_timings(request)
        
        for i, timing in enumerate(optimal_timings):
            clip = VideoClip(
                start=timing["start"],
                end=timing["end"],
                caption=self._generate_optimized_caption(i, request),
                emojis=self._generate_optimized_emojis(i, request)
            )
            clips.append(clip)
        
        return clips
    
    def _calculate_optimal_timings(self, request: VideoClipRequest) -> List[Dict[str, float]]:
        """Calculate optimal clip timings for maximum engagement."""
        # This would use AI analysis to find the best moments
        # For now, use a simple algorithm
        
        timings = []
        current_time = 0.0
        clip_length = min(request.max_clip_length, 30.0)
        
        while current_time < 300.0:  # Max 5 minutes
            end_time = min(current_time + clip_length, 300.0)
            
            timings.append({
                "start": current_time,
                "end": end_time
            })
            
            current_time = end_time
            
            # Add some overlap for better transitions
            current_time -= 2.0
        
        return timings
    
    def _generate_optimized_caption(self, clip_index: int, request: VideoClipRequest) -> str:
        """Generate optimized caption for clip."""
        # This would use AI to generate engaging captions
        captions = [
            "🔥 This moment is absolutely incredible!",
            "💡 You won't believe what happens next!",
            "🎯 The key insight you need to know!",
            "🚀 This changes everything!",
            "✨ Pure magic happening here!"
        ]
        return captions[clip_index % len(captions)]
    
    def _generate_optimized_emojis(self, clip_index: int, request: VideoClipRequest) -> List[str]:
        """Generate optimized emojis for clip."""
        # This would use AI to select the most engaging emojis
        emoji_sets = [
            ["🔥", "💯", "👏"],
            ["💡", "✨", "🎯"],
            ["🚀", "⚡", "💪"],
            ["🎉", "🎊", "🏆"],
            ["💎", "🌟", "💫"]
        ]
        return emoji_sets[clip_index % len(emoji_sets)]
    
    def _generate_cache_key(self, request: VideoClipRequest) -> str:
        """Generate cache key for video request."""
        key_data = f"{request.youtube_url}:{request.language}:{request.target_platform}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _estimate_memory_usage(self, request: VideoClipRequest) -> int:
        """Estimate memory usage for video processing."""
        # Rough estimation based on video duration and quality
        base_memory = 512  # Base memory in MB
        duration_factor = min(request.max_clip_length / 60.0, 5.0)  # Max 5x for long videos
        quality_factor = 1.5 if request.target_platform in ["tiktok", "instagram"] else 1.0
        
        estimated_memory = int(base_memory * duration_factor * quality_factor)
        return min(estimated_memory, self.config.max_memory_usage_mb)
    
    def _group_requests_by_memory(self, requests: List[VideoClipRequest]) -> List[List[VideoClipRequest]]:
        """Group requests by memory usage for optimal processing."""
        groups = []
        current_group = []
        current_memory = 0
        
        for request in requests:
            estimated_memory = self._estimate_memory_usage(request)
            
            if current_memory + estimated_memory > self.config.max_memory_usage_mb:
                if current_group:
                    groups.append(current_group)
                    current_group = [request]
                    current_memory = estimated_memory
                else:
                    # Single request exceeds memory limit, process alone
                    groups.append([request])
            else:
                current_group.append(request)
                current_memory += estimated_memory
        
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def _create_fallback_response(self, request: VideoClipRequest) -> VideoClipResponse:
        """Create fallback response when processing fails."""
        fallback_clip = VideoClip(
            start=0.0,
            end=min(request.max_clip_length, 30.0),
            caption="Video processing temporarily unavailable",
            emojis=["⚠️", "🔄"]
        )
        
        return VideoClipResponse(
            youtube_url=request.youtube_url,
            clips=[fallback_clip],
            logo_path=request.logo_path,
            language=request.language
        )
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        stats = self.processing_stats.copy()
        
        if stats["total_processed"] > 0:
            stats["average_processing_time"] = stats["total_processing_time"] / stats["total_processed"]
            stats["cache_hit_rate"] = stats["cache_hits"] / (stats["cache_hits"] + stats["cache_misses"])
        else:
            stats["average_processing_time"] = 0.0
            stats["cache_hit_rate"] = 0.0
        
        stats["available_memory_mb"] = self.memory_manager.get_available_memory()
        stats["memory_usage_mb"] = self.memory_manager.current_usage
        
        return stats
    
    def cleanup(self):
        """Cleanup resources."""
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
        
        if self.process_pool:
            self.process_pool.shutdown(wait=True)
        
        self.memory_manager.force_garbage_collection()
        logger.info("Video processor cleanup completed")

# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_optimized_video_processor(
    config: Optional[OptimizedVideoProcessorConfig] = None
) -> OptimizedVideoProcessor:
    """Create an optimized video processor instance."""
    return OptimizedVideoProcessor(config)

def create_high_performance_processor() -> OptimizedVideoProcessor:
    """Create a high-performance video processor with optimized settings."""
    config = OptimizedVideoProcessorConfig(
        max_concurrent_videos=8,
        enable_hardware_acceleration=True,
        enable_parallel_processing=True,
        enable_async_processing=True,
        enable_batch_processing=True,
        optimal_batch_size=16,
        max_memory_usage_mb=8192
    )
    return OptimizedVideoProcessor(config)

def create_memory_efficient_processor() -> OptimizedVideoProcessor:
    """Create a memory-efficient video processor."""
    config = OptimizedVideoProcessorConfig(
        max_concurrent_videos=2,
        enable_hardware_acceleration=False,
        enable_parallel_processing=False,
        enable_async_processing=True,
        enable_batch_processing=True,
        optimal_batch_size=4,
        max_memory_usage_mb=2048,
        enable_memory_pooling=True,
        enable_garbage_collection=True
    )
    return OptimizedVideoProcessor(config) 