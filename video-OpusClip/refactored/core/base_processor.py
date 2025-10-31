"""
Base Processor Class

Abstract base class for all video processing components with common functionality,
error handling, and performance monitoring.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
import asyncio
import time
import structlog
from dataclasses import dataclass
from enum import Enum
import uuid
from pathlib import Path
import json

from ..error_handling import ErrorHandler, ProcessingError, ValidationError

logger = structlog.get_logger("base_processor")
error_handler = ErrorHandler()

class ProcessorStatus(Enum):
    """Status of a processor."""
    IDLE = "idle"
    INITIALIZING = "initializing"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class ProcessorConfig:
    """Base configuration for processors."""
    name: str
    version: str
    enabled: bool = True
    max_workers: int = 4
    timeout: float = 300.0  # 5 minutes
    retry_attempts: int = 3
    cache_enabled: bool = True
    cache_ttl: float = 3600.0  # 1 hour
    log_level: str = "INFO"

@dataclass
class ProcessingResult:
    """Base result for processing operations."""
    success: bool
    processor_name: str
    processing_time: float
    result_data: Dict[str, Any]
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None

class BaseProcessor(ABC):
    """Abstract base class for all video processors."""
    
    def __init__(self, config: ProcessorConfig):
        self.config = config
        self.status = ProcessorStatus.IDLE
        self.current_job_id: Optional[str] = None
        self.start_time: Optional[float] = None
        self.retry_count = 0
        self.cache = {} if config.cache_enabled else None
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup processor-specific logging."""
        self.logger = structlog.get_logger(f"processor.{self.config.name}")
    
    async def initialize(self) -> bool:
        """Initialize the processor."""
        try:
            self.status = ProcessorStatus.INITIALIZING
            self.logger.info(f"Initializing {self.config.name} processor")
            
            success = await self._initialize_impl()
            
            if success:
                self.status = ProcessorStatus.IDLE
                self.logger.info(f"{self.config.name} processor initialized successfully")
            else:
                self.status = ProcessorStatus.FAILED
                self.logger.error(f"{self.config.name} processor initialization failed")
            
            return success
            
        except Exception as e:
            self.status = ProcessorStatus.FAILED
            self.logger.error(f"{self.config.name} processor initialization error: {e}")
            return False
    
    @abstractmethod
    async def _initialize_impl(self) -> bool:
        """Implementation-specific initialization."""
        pass
    
    async def process(self, job_id: str, input_data: Dict[str, Any]) -> ProcessingResult:
        """Process input data with error handling and retry logic."""
        try:
            self.current_job_id = job_id
            self.status = ProcessorStatus.PROCESSING
            self.start_time = time.time()
            
            self.logger.info(f"Starting processing job {job_id} with {self.config.name}")
            
            # Check cache first
            if self.cache is not None:
                cache_key = self._generate_cache_key(input_data)
                cached_result = self._get_from_cache(cache_key)
                if cached_result:
                    self.logger.info(f"Using cached result for job {job_id}")
                    return cached_result
            
            # Process with retry logic
            result = await self._process_with_retry(input_data)
            
            # Cache result if enabled
            if self.cache is not None and result.success:
                self._store_in_cache(cache_key, result)
            
            self.status = ProcessorStatus.COMPLETED
            self.logger.info(f"Job {job_id} completed successfully in {result.processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            self.status = ProcessorStatus.FAILED
            self.logger.error(f"Job {job_id} failed: {e}")
            return ProcessingResult(
                success=False,
                processor_name=self.config.name,
                processing_time=time.time() - (self.start_time or time.time()),
                result_data={},
                error_message=str(e)
            )
        finally:
            self.current_job_id = None
            self.start_time = None
    
    async def _process_with_retry(self, input_data: Dict[str, Any]) -> ProcessingResult:
        """Process with retry logic."""
        last_error = None
        
        for attempt in range(self.config.retry_attempts):
            try:
                if attempt > 0:
                    self.logger.info(f"Retry attempt {attempt + 1} for job {self.current_job_id}")
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                
                result = await self._process_impl(input_data)
                return result
                
            except Exception as e:
                last_error = e
                self.logger.warning(f"Attempt {attempt + 1} failed: {e}")
                continue
        
        # All retries failed
        return ProcessingResult(
            success=False,
            processor_name=self.config.name,
            processing_time=time.time() - (self.start_time or time.time()),
            result_data={},
            error_message=f"All retry attempts failed. Last error: {last_error}"
        )
    
    @abstractmethod
    async def _process_impl(self, input_data: Dict[str, Any]) -> ProcessingResult:
        """Implementation-specific processing."""
        pass
    
    async def cancel(self) -> bool:
        """Cancel current processing job."""
        if self.status == ProcessorStatus.PROCESSING:
            self.status = ProcessorStatus.CANCELLED
            self.logger.info(f"Cancelled job {self.current_job_id}")
            return True
        return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get current processor status."""
        return {
            "name": self.config.name,
            "version": self.config.version,
            "status": self.status.value,
            "current_job_id": self.current_job_id,
            "uptime": time.time() - (self.start_time or time.time()) if self.start_time else 0,
            "retry_count": self.retry_count,
            "cache_enabled": self.cache is not None,
            "cache_size": len(self.cache) if self.cache else 0
        }
    
    def _generate_cache_key(self, input_data: Dict[str, Any]) -> str:
        """Generate cache key from input data."""
        # Create a hash of the input data for caching
        data_str = json.dumps(input_data, sort_keys=True)
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, data_str))
    
    def _get_from_cache(self, cache_key: str) -> Optional[ProcessingResult]:
        """Get result from cache."""
        if cache_key in self.cache:
            cached_data = self.cache[cache_key]
            # Check if cache entry is still valid
            if time.time() - cached_data["timestamp"] < self.config.cache_ttl:
                return cached_data["result"]
            else:
                # Remove expired cache entry
                del self.cache[cache_key]
        return None
    
    def _store_in_cache(self, cache_key: str, result: ProcessingResult):
        """Store result in cache."""
        self.cache[cache_key] = {
            "result": result,
            "timestamp": time.time()
        }
    
    async def cleanup(self):
        """Cleanup processor resources."""
        try:
            self.logger.info(f"Cleaning up {self.config.name} processor")
            await self._cleanup_impl()
            self.status = ProcessorStatus.IDLE
        except Exception as e:
            self.logger.error(f"Cleanup failed for {self.config.name}: {e}")
    
    async def _cleanup_impl(self):
        """Implementation-specific cleanup."""
        pass

class ProcessorManager:
    """Manages multiple processors with load balancing and monitoring."""
    
    def __init__(self):
        self.processors: Dict[str, BaseProcessor] = {}
        self.job_queue = asyncio.Queue()
        self.active_jobs: Dict[str, BaseProcessor] = {}
        self.completed_jobs: Dict[str, ProcessingResult] = {}
        self.logger = structlog.get_logger("processor_manager")
    
    def register_processor(self, processor: BaseProcessor):
        """Register a processor."""
        self.processors[processor.config.name] = processor
        self.logger.info(f"Registered processor: {processor.config.name}")
    
    async def initialize_all(self) -> bool:
        """Initialize all registered processors."""
        try:
            self.logger.info("Initializing all processors")
            
            init_tasks = [
                processor.initialize() 
                for processor in self.processors.values()
            ]
            
            results = await asyncio.gather(*init_tasks, return_exceptions=True)
            
            success_count = sum(1 for result in results if result is True)
            total_count = len(results)
            
            self.logger.info(f"Initialized {success_count}/{total_count} processors")
            return success_count == total_count
            
        except Exception as e:
            self.logger.error(f"Processor initialization failed: {e}")
            return False
    
    async def process_job(self, job_id: str, processor_name: str, input_data: Dict[str, Any]) -> ProcessingResult:
        """Process a job with a specific processor."""
        try:
            if processor_name not in self.processors:
                raise ValueError(f"Processor {processor_name} not found")
            
            processor = self.processors[processor_name]
            
            if not processor.config.enabled:
                raise ValueError(f"Processor {processor_name} is disabled")
            
            # Check if processor is available
            if processor.status != ProcessorStatus.IDLE:
                raise ValueError(f"Processor {processor_name} is busy")
            
            # Process the job
            result = await processor.process(job_id, input_data)
            
            # Store result
            self.completed_jobs[job_id] = result
            
            return result
            
        except Exception as e:
            self.logger.error(f"Job processing failed: {e}")
            return ProcessingResult(
                success=False,
                processor_name=processor_name,
                processing_time=0.0,
                result_data={},
                error_message=str(e)
            )
    
    async def process_pipeline(self, job_id: str, pipeline: List[Dict[str, Any]]) -> List[ProcessingResult]:
        """Process a pipeline of jobs."""
        try:
            self.logger.info(f"Processing pipeline for job {job_id}")
            
            results = []
            current_data = {}
            
            for step in pipeline:
                processor_name = step["processor"]
                input_data = step.get("input_data", current_data)
                
                result = await self.process_job(f"{job_id}_{step['step']}", processor_name, input_data)
                results.append(result)
                
                if not result.success:
                    self.logger.error(f"Pipeline step failed: {step}")
                    break
                
                # Pass result data to next step
                current_data.update(result.result_data)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Pipeline processing failed: {e}")
            return []
    
    def get_processor_status(self, processor_name: Optional[str] = None) -> Dict[str, Any]:
        """Get status of processors."""
        if processor_name:
            if processor_name in self.processors:
                return self.processors[processor_name].get_status()
            else:
                return {"error": f"Processor {processor_name} not found"}
        else:
            return {
                name: processor.get_status() 
                for name, processor in self.processors.items()
            }
    
    async def cleanup_all(self):
        """Cleanup all processors."""
        try:
            self.logger.info("Cleaning up all processors")
            
            cleanup_tasks = [
                processor.cleanup() 
                for processor in self.processors.values()
            ]
            
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")

# Export classes
__all__ = [
    "BaseProcessor", 
    "ProcessorConfig", 
    "ProcessingResult", 
    "ProcessorStatus",
    "ProcessorManager"
]


