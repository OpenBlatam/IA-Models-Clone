"""
Improved Batch Processor

Enhanced batch processing with:
- Async operations and performance optimization
- Comprehensive error handling with early returns
- Resource management and monitoring
- Parallel processing capabilities
- Progress tracking and reporting
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Union
import asyncio
import time
import structlog
from dataclasses import dataclass
from enum import Enum

from ..models.improved_models import (
    VideoClipRequest,
    VideoClipResponse,
    VideoClipBatchRequest,
    VideoClipBatchResponse
)
from ..error_handling import (
    VideoProcessingError,
    ValidationError,
    ResourceError,
    handle_processing_errors
)
from ..monitoring import monitor_performance

logger = structlog.get_logger("batch_processor")

# =============================================================================
# BATCH PROCESSOR CONFIGURATION
# =============================================================================

@dataclass
class BatchProcessorConfig:
    """Configuration for batch video processor."""
    max_workers: int = 8
    batch_size: int = 10
    enable_parallel_processing: bool = True
    timeout: float = 300.0
    retry_attempts: int = 3
    retry_delay: float = 1.0
    progress_reporting_interval: float = 5.0
    memory_limit_mb: int = 1024
    enable_progress_tracking: bool = True

# =============================================================================
# BATCH PROCESSING MODELS
# =============================================================================

@dataclass
class BatchProgress:
    """Batch processing progress information."""
    batch_id: str
    total_items: int
    processed_items: int
    failed_items: int
    current_item: int
    start_time: float
    estimated_completion: Optional[float] = None
    progress_percentage: float = 0.0
    items_per_second: float = 0.0

@dataclass
class BatchResult:
    """Result of batch processing operation."""
    batch_id: str
    success: bool
    total_items: int
    successful_items: int
    failed_items: int
    processing_time: float
    results: List[VideoClipResponse]
    errors: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]

# =============================================================================
# BATCH PROCESSOR
# =============================================================================

class BatchVideoProcessor:
    """Enhanced batch video processor with parallel processing capabilities."""
    
    def __init__(self, config: BatchProcessorConfig):
        self.config = config
        self._processing_batches: Dict[str, BatchProgress] = {}
        self._stats = {
            'batches_processed': 0,
            'total_items_processed': 0,
            'total_items_failed': 0,
            'total_processing_time': 0.0,
            'average_batch_time': 0.0,
            'average_items_per_second': 0.0
        }
        self._semaphore: Optional[asyncio.Semaphore] = None
    
    async def initialize(self) -> None:
        """Initialize batch processor."""
        self._semaphore = asyncio.Semaphore(self.config.max_workers)
        logger.info("Batch video processor initialized", max_workers=self.config.max_workers)
    
    async def close(self) -> None:
        """Close batch processor."""
        # Wait for all active batches to complete
        if self._processing_batches:
            logger.info("Waiting for active batches to complete", active_batches=len(self._processing_batches))
            await asyncio.sleep(1.0)  # Give time for completion
        
        self._processing_batches.clear()
        logger.info("Batch video processor closed")
    
    @monitor_performance("batch_processing")
    async def process_batch_async(self, requests: List[VideoClipRequest]) -> List[VideoClipResponse]:
        """Process batch of video requests with comprehensive error handling."""
        # Early return for empty batch
        if not requests:
            return []
        
        # Early return for batch size limit
        if len(requests) > 100:
            raise ValidationError("Batch size cannot exceed 100 requests")
        
        # Early return for system resource check
        if not await self._check_system_resources():
            raise ResourceError("Insufficient system resources for batch processing")
        
        # Happy path: Process batch
        batch_id = f"batch_{int(time.time())}_{hash(str(requests)) % 10000}"
        start_time = time.perf_counter()
        
        try:
            # Initialize batch progress tracking
            progress = BatchProgress(
                batch_id=batch_id,
                total_items=len(requests),
                processed_items=0,
                failed_items=0,
                current_item=0,
                start_time=start_time
            )
            self._processing_batches[batch_id] = progress
            
            # Process items in parallel with semaphore control
            results = await self._process_items_parallel(requests, progress)
            
            # Calculate final statistics
            processing_time = time.perf_counter() - start_time
            successful_count = len([r for r in results if r.success])
            failed_count = len(results) - successful_count
            
            # Update batch statistics
            self._update_batch_stats(len(requests), processing_time)
            
            # Clean up progress tracking
            if batch_id in self._processing_batches:
                del self._processing_batches[batch_id]
            
            logger.info(
                "Batch processing completed",
                batch_id=batch_id,
                total_items=len(requests),
                successful_count=successful_count,
                failed_count=failed_count,
                processing_time=processing_time
            )
            
            return results
            
        except Exception as e:
            processing_time = time.perf_counter() - start_time
            
            # Clean up progress tracking
            if batch_id in self._processing_batches:
                del self._processing_batches[batch_id]
            
            logger.error(
                "Batch processing failed",
                batch_id=batch_id,
                error=str(e),
                processing_time=processing_time
            )
            
            raise VideoProcessingError(f"Batch processing failed: {str(e)}")
    
    async def _process_items_parallel(
        self,
        requests: List[VideoClipRequest],
        progress: BatchProgress
    ) -> List[VideoClipResponse]:
        """Process items in parallel with progress tracking."""
        results = []
        
        # Create tasks for parallel processing
        tasks = []
        for i, request in enumerate(requests):
            task = asyncio.create_task(
                self._process_single_item_with_semaphore(request, i, progress)
            )
            tasks.append(task)
        
        # Wait for all tasks to complete
        task_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        for i, result in enumerate(task_results):
            if isinstance(result, Exception):
                logger.error(
                    "Item processing failed",
                    index=i,
                    error=str(result),
                    youtube_url=requests[i].youtube_url
                )
                
                # Create error response
                error_response = VideoClipResponse(
                    success=False,
                    youtube_url=requests[i].youtube_url,
                    error=str(result),
                    processing_time=0.0
                )
                results.append(error_response)
            else:
                results.append(result)
        
        return results
    
    async def _process_single_item_with_semaphore(
        self,
        request: VideoClipRequest,
        index: int,
        progress: BatchProgress
    ) -> VideoClipResponse:
        """Process single item with semaphore control and progress tracking."""
        async with self._semaphore:
            try:
                # Update progress
                progress.current_item = index + 1
                progress.processed_items += 1
                progress.progress_percentage = (progress.processed_items / progress.total_items) * 100
                
                # Calculate processing rate
                elapsed_time = time.perf_counter() - progress.start_time
                if elapsed_time > 0:
                    progress.items_per_second = progress.processed_items / elapsed_time
                
                # Estimate completion time
                if progress.items_per_second > 0:
                    remaining_items = progress.total_items - progress.processed_items
                    progress.estimated_completion = time.perf_counter() + (remaining_items / progress.items_per_second)
                
                # Simulate video processing (replace with actual processing logic)
                await self._simulate_video_processing(request)
                
                # Create successful response
                response = VideoClipResponse(
                    success=True,
                    youtube_url=request.youtube_url,
                    title=f"Processed Video {index + 1}",
                    description=f"Video processed from {request.youtube_url}",
                    duration=request.max_clip_length,
                    language=request.language,
                    quality=request.quality,
                    format=request.format,
                    file_path=f"/processed/batch_{progress.batch_id}_item_{index + 1}.{request.format}",
                    file_size=1024 * 1024,  # 1MB placeholder
                    resolution="1920x1080",
                    fps=30.0,
                    bitrate=2000000,  # 2Mbps
                    processing_time=0.5,  # Simulated processing time
                    metadata={
                        "batch_id": progress.batch_id,
                        "item_index": index,
                        "processing_config": {
                            "quality": request.quality,
                            "format": request.format,
                            "max_length": request.max_clip_length
                        }
                    }
                )
                
                return response
                
            except Exception as e:
                # Update failed count
                progress.failed_items += 1
                
                logger.error(
                    "Single item processing failed",
                    index=index,
                    error=str(e),
                    youtube_url=request.youtube_url
                )
                
                # Create error response
                error_response = VideoClipResponse(
                    success=False,
                    youtube_url=request.youtube_url,
                    error=str(e),
                    processing_time=0.0
                )
                
                return error_response
    
    async def _simulate_video_processing(self, request: VideoClipRequest) -> None:
        """Simulate video processing (replace with actual processing logic)."""
        # Simulate processing time based on video length and quality
        base_time = 1.0  # Base processing time
        quality_multiplier = {
            "low": 0.5,
            "medium": 1.0,
            "high": 1.5,
            "ultra": 2.0
        }
        
        processing_time = base_time * quality_multiplier.get(request.quality, 1.0)
        processing_time *= (request.max_clip_length / 60.0)  # Scale by duration
        
        # Add some randomness
        processing_time *= (0.8 + (hash(request.youtube_url) % 40) / 100.0)
        
        await asyncio.sleep(min(processing_time, 5.0))  # Cap at 5 seconds for demo
    
    async def _check_system_resources(self) -> bool:
        """Check if system has sufficient resources for batch processing."""
        try:
            import psutil
            
            # Check memory usage
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                logger.warning("High memory usage detected", memory_percent=memory.percent)
                return False
            
            # Check available memory
            available_memory_mb = memory.available / (1024 * 1024)
            if available_memory_mb < self.config.memory_limit_mb:
                logger.warning("Insufficient available memory", available_mb=available_memory_mb)
                return False
            
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            if cpu_percent > 95:
                logger.warning("High CPU usage detected", cpu_percent=cpu_percent)
                return False
            
            return True
            
        except ImportError:
            logger.warning("psutil not available, skipping resource check")
            return True
        except Exception as e:
            logger.warning("Resource check failed", error=str(e))
            return True
    
    def _update_batch_stats(self, batch_size: int, processing_time: float) -> None:
        """Update batch processing statistics."""
        self._stats['batches_processed'] += 1
        self._stats['total_items_processed'] += batch_size
        self._stats['total_processing_time'] += processing_time
        
        if self._stats['batches_processed'] > 0:
            self._stats['average_batch_time'] = (
                self._stats['total_processing_time'] / self._stats['batches_processed']
            )
            
            if self._stats['total_processing_time'] > 0:
                self._stats['average_items_per_second'] = (
                    self._stats['total_items_processed'] / self._stats['total_processing_time']
                )
    
    def get_batch_progress(self, batch_id: str) -> Optional[BatchProgress]:
        """Get progress information for a specific batch."""
        return self._processing_batches.get(batch_id)
    
    def get_all_batch_progress(self) -> Dict[str, BatchProgress]:
        """Get progress information for all active batches."""
        return self._processing_batches.copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get batch processing statistics."""
        return {
            **self._stats,
            'active_batches': len(self._processing_batches),
            'config': {
                'max_workers': self.config.max_workers,
                'batch_size': self.config.batch_size,
                'enable_parallel_processing': self.config.enable_parallel_processing,
                'timeout': self.config.timeout,
                'memory_limit_mb': self.config.memory_limit_mb
            }
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get processor health status."""
        total_items = self._stats['total_items_processed'] + self._stats['total_items_failed']
        success_rate = (
            self._stats['total_items_processed'] / total_items * 100
            if total_items > 0 else 0
        )
        
        return {
            'healthy': len(self._processing_batches) < 10,  # Not too many active batches
            'active_batches': len(self._processing_batches),
            'total_batches_processed': self._stats['batches_processed'],
            'total_items_processed': self._stats['total_items_processed'],
            'success_rate': success_rate,
            'average_batch_time': self._stats['average_batch_time'],
            'average_items_per_second': self._stats['average_items_per_second']
        }

# =============================================================================
# BATCH PROGRESS TRACKER
# =============================================================================

class BatchProgressTracker:
    """Track and report batch processing progress."""
    
    def __init__(self, batch_processor: BatchVideoProcessor):
        self.batch_processor = batch_processor
        self._tracking_task: Optional[asyncio.Task] = None
        self._tracking_enabled = False
    
    async def start_tracking(self) -> None:
        """Start progress tracking."""
        if self._tracking_enabled:
            return
        
        self._tracking_enabled = True
        self._tracking_task = asyncio.create_task(self._tracking_loop())
        logger.info("Batch progress tracking started")
    
    async def stop_tracking(self) -> None:
        """Stop progress tracking."""
        self._tracking_enabled = False
        if self._tracking_task:
            self._tracking_task.cancel()
            try:
                await self._tracking_task
            except asyncio.CancelledError:
                pass
        logger.info("Batch progress tracking stopped")
    
    async def _tracking_loop(self) -> None:
        """Background progress tracking loop."""
        while self._tracking_enabled:
            try:
                await asyncio.sleep(5.0)  # Report every 5 seconds
                
                active_batches = self.batch_processor.get_all_batch_progress()
                if active_batches:
                    logger.info(
                        "Batch progress update",
                        active_batches=len(active_batches),
                        batches=[{
                            'batch_id': batch.batch_id,
                            'progress_percentage': batch.progress_percentage,
                            'items_per_second': batch.items_per_second
                        } for batch in active_batches.values()]
                    )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("Progress tracking error", error=str(e))

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'BatchProcessorConfig',
    'BatchProgress',
    'BatchResult',
    'BatchVideoProcessor',
    'BatchProgressTracker'
]






























