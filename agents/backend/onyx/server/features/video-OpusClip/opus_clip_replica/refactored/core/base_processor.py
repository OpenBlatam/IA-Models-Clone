"""
Base Processor for Refactored Opus Clip

Provides a standardized base class for all processors with:
- Async processing capabilities
- Error handling and logging
- Performance monitoring
- Configuration management
- Result caching
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
import asyncio
import time
import uuid
from datetime import datetime
import structlog
from dataclasses import dataclass
from enum import Enum

logger = structlog.get_logger("base_processor")

class ProcessorStatus(Enum):
    """Processor status enumeration."""
    IDLE = "idle"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class ProcessorResult:
    """Standardized processor result."""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time: float = 0.0
    metadata: Optional[Dict[str, Any]] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class ProcessorConfig:
    """Processor configuration."""
    max_retries: int = 3
    timeout_seconds: float = 300.0
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    enable_monitoring: bool = True
    log_level: str = "INFO"

class BaseProcessor(ABC):
    """
    Base class for all Opus Clip processors.
    
    Provides standardized async processing with:
    - Error handling and retries
    - Performance monitoring
    - Result caching
    - Configuration management
    - Logging
    """
    
    def __init__(self, config: Optional[ProcessorConfig] = None):
        """Initialize the processor with configuration."""
        self.config = config or ProcessorConfig()
        self.logger = structlog.get_logger(self.__class__.__name__)
        self.status = ProcessorStatus.IDLE
        self.processing_start_time: Optional[float] = None
        self.result_cache: Dict[str, ProcessorResult] = {}
        self.processing_tasks: Dict[str, asyncio.Task] = {}
        
        # Performance metrics
        self.total_processed = 0
        self.total_errors = 0
        self.total_processing_time = 0.0
        self.average_processing_time = 0.0
        
        self.logger.info(f"Initialized {self.__class__.__name__}")
    
    @abstractmethod
    async def _process_impl(self, input_data: Dict[str, Any]) -> ProcessorResult:
        """
        Abstract method to implement the actual processing logic.
        
        Args:
            input_data: Input data for processing
            
        Returns:
            ProcessorResult: Processing result
        """
        pass
    
    async def process(self, input_data: Dict[str, Any], 
                     job_id: Optional[str] = None) -> ProcessorResult:
        """
        Process input data with error handling and monitoring.
        
        Args:
            input_data: Input data for processing
            job_id: Optional job ID for tracking
            
        Returns:
            ProcessorResult: Processing result
        """
        job_id = job_id or str(uuid.uuid4())
        
        try:
            # Check cache first
            if self.config.enable_caching:
                cache_key = self._generate_cache_key(input_data)
                cached_result = self._get_cached_result(cache_key)
                if cached_result:
                    self.logger.info(f"Returning cached result for job {job_id}")
                    return cached_result
            
            # Check if already processing
            if job_id in self.processing_tasks:
                self.logger.warning(f"Job {job_id} already processing")
                return await self.processing_tasks[job_id]
            
            # Start processing
            self.status = ProcessorStatus.PROCESSING
            self.processing_start_time = time.time()
            
            self.logger.info(f"Starting processing for job {job_id}")
            
            # Create processing task
            task = asyncio.create_task(
                self._process_with_retry(input_data, job_id)
            )
            self.processing_tasks[job_id] = task
            
            # Wait for completion with timeout
            try:
                result = await asyncio.wait_for(
                    task, 
                    timeout=self.config.timeout_seconds
                )
            except asyncio.TimeoutError:
                self.logger.error(f"Processing timeout for job {job_id}")
                result = ProcessorResult(
                    success=False,
                    error=f"Processing timeout after {self.config.timeout_seconds}s"
                )
            
            # Update metrics
            self._update_metrics(result)
            
            # Cache result if successful
            if result.success and self.config.enable_caching:
                self._cache_result(cache_key, result)
            
            # Clean up
            if job_id in self.processing_tasks:
                del self.processing_tasks[job_id]
            
            self.status = ProcessorStatus.COMPLETED if result.success else ProcessorStatus.FAILED
            
            self.logger.info(f"Completed processing for job {job_id}: {result.success}")
            return result
            
        except Exception as e:
            self.logger.error(f"Processing failed for job {job_id}: {e}")
            self.status = ProcessorStatus.FAILED
            self.total_errors += 1
            
            result = ProcessorResult(
                success=False,
                error=str(e)
            )
            
            # Clean up
            if job_id in self.processing_tasks:
                del self.processing_tasks[job_id]
            
            return result
    
    async def _process_with_retry(self, input_data: Dict[str, Any], 
                                 job_id: str) -> ProcessorResult:
        """Process with retry logic."""
        last_error = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                if attempt > 0:
                    self.logger.info(f"Retry attempt {attempt} for job {job_id}")
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                
                result = await self._process_impl(input_data)
                
                if result.success:
                    return result
                else:
                    last_error = result.error
                    
            except Exception as e:
                last_error = str(e)
                self.logger.error(f"Attempt {attempt + 1} failed for job {job_id}: {e}")
        
        # All retries failed
        return ProcessorResult(
            success=False,
            error=f"All {self.config.max_retries + 1} attempts failed. Last error: {last_error}"
        )
    
    def _generate_cache_key(self, input_data: Dict[str, Any]) -> str:
        """Generate cache key from input data."""
        import hashlib
        import json
        
        # Sort keys for consistent hashing
        sorted_data = json.dumps(input_data, sort_keys=True)
        return hashlib.md5(sorted_data.encode()).hexdigest()
    
    def _get_cached_result(self, cache_key: str) -> Optional[ProcessorResult]:
        """Get cached result if valid."""
        if cache_key not in self.result_cache:
            return None
        
        result = self.result_cache[cache_key]
        
        # Check TTL
        age_seconds = (datetime.now() - result.timestamp).total_seconds()
        if age_seconds > self.config.cache_ttl_seconds:
            del self.result_cache[cache_key]
            return None
        
        return result
    
    def _cache_result(self, cache_key: str, result: ProcessorResult):
        """Cache processing result."""
        self.result_cache[cache_key] = result
        
        # Clean old cache entries
        if len(self.result_cache) > 1000:  # Limit cache size
            oldest_key = min(
                self.result_cache.keys(),
                key=lambda k: self.result_cache[k].timestamp
            )
            del self.result_cache[oldest_key]
    
    def _update_metrics(self, result: ProcessorResult):
        """Update performance metrics."""
        if result.processing_time > 0:
            self.total_processed += 1
            self.total_processing_time += result.processing_time
            self.average_processing_time = self.total_processing_time / self.total_processed
        
        if not result.success:
            self.total_errors += 1
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a processing job."""
        if job_id in self.processing_tasks:
            task = self.processing_tasks[job_id]
            task.cancel()
            del self.processing_tasks[job_id]
            self.logger.info(f"Cancelled job {job_id}")
            return True
        return False
    
    async def get_status(self) -> Dict[str, Any]:
        """Get processor status and metrics."""
        return {
            "status": self.status.value,
            "total_processed": self.total_processed,
            "total_errors": self.total_errors,
            "average_processing_time": self.average_processing_time,
            "active_jobs": len(self.processing_tasks),
            "cache_size": len(self.result_cache),
            "config": {
                "max_retries": self.config.max_retries,
                "timeout_seconds": self.config.timeout_seconds,
                "enable_caching": self.config.enable_caching,
                "cache_ttl_seconds": self.config.cache_ttl_seconds
            }
        }
    
    async def clear_cache(self):
        """Clear result cache."""
        self.result_cache.clear()
        self.logger.info("Cache cleared")
    
    async def shutdown(self):
        """Shutdown processor and cleanup resources."""
        # Cancel all active tasks
        for job_id, task in self.processing_tasks.items():
            task.cancel()
            self.logger.info(f"Cancelled job {job_id} during shutdown")
        
        self.processing_tasks.clear()
        self.status = ProcessorStatus.IDLE
        
        self.logger.info(f"Shutdown {self.__class__.__name__}")


