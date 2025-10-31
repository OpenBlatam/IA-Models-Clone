"""
Batch Processing System
======================

Advanced batch processing system for optimized document generation.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
import numpy as np
from queue import Queue, Empty
import weakref

logger = logging.getLogger(__name__)

class BatchStatus(str, Enum):
    """Batch processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ProcessingStrategy(str, Enum):
    """Processing strategies."""
    SEQUENTIAL = "sequential"  # Process one by one
    PARALLEL = "parallel"      # Process in parallel
    PIPELINE = "pipeline"     # Process in pipeline
    ADAPTIVE = "adaptive"     # Adapt based on load

@dataclass
class BatchItem:
    """Batch processing item."""
    id: str
    data: Any
    priority: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BatchJob:
    """Batch processing job."""
    id: str
    items: List[BatchItem]
    status: BatchStatus = BatchStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    results: List[Any] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    progress: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BatchConfig:
    """Batch processing configuration."""
    max_batch_size: int = 100
    max_workers: int = 4
    processing_timeout: int = 300
    retry_attempts: int = 3
    retry_delay: float = 1.0
    strategy: ProcessingStrategy = ProcessingStrategy.ADAPTIVE
    enable_priority: bool = True
    enable_compression: bool = True
    enable_monitoring: bool = True

class BatchProcessor:
    """
    Advanced batch processing system.
    
    Features:
    - Multiple processing strategies
    - Priority-based processing
    - Adaptive load balancing
    - Progress tracking
    - Error handling and retry
    - Resource optimization
    """
    
    def __init__(self, config: Optional[BatchConfig] = None):
        self.config = config or BatchConfig()
        self.jobs = {}
        self.job_queue = Queue()
        self.priority_queue = defaultdict(list)
        self.workers = []
        self.thread_pool = ThreadPoolExecutor(max_workers=self.config.max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=self.config.max_workers)
        self.running = False
        self.stats = {
            'total_jobs': 0,
            'completed_jobs': 0,
            'failed_jobs': 0,
            'total_items': 0,
            'processed_items': 0,
            'processing_time': 0.0
        }
        self.lock = threading.Lock()
        
    async def initialize(self):
        """Initialize batch processor."""
        logger.info("Initializing Batch Processor...")
        
        try:
            # Start worker threads
            self.running = True
            for i in range(self.config.max_workers):
                worker = asyncio.create_task(self._worker(f"worker-{i}"))
                self.workers.append(worker)
            
            # Start monitoring
            if self.config.enable_monitoring:
                asyncio.create_task(self._monitor_processing())
            
            logger.info("Batch Processor initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Batch Processor: {str(e)}")
            raise
    
    async def _worker(self, worker_id: str):
        """Worker thread for processing batches."""
        while self.running:
            try:
                # Get next job
                job = await self._get_next_job()
                if job is None:
                    await asyncio.sleep(0.1)
                    continue
                
                # Process job
                await self._process_job(job, worker_id)
                
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {str(e)}")
                await asyncio.sleep(1)
    
    async def _get_next_job(self) -> Optional[BatchJob]:
        """Get next job to process."""
        try:
            # Try priority queue first
            if self.config.enable_priority:
                for priority in sorted(self.priority_queue.keys(), reverse=True):
                    if self.priority_queue[priority]:
                        job_id = self.priority_queue[priority].pop(0)
                        if job_id in self.jobs:
                            return self.jobs[job_id]
            
            # Try regular queue
            try:
                job_id = self.job_queue.get_nowait()
                if job_id in self.jobs:
                    return self.jobs[job_id]
            except Empty:
                pass
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting next job: {str(e)}")
            return None
    
    async def _process_job(self, job: BatchJob, worker_id: str):
        """Process a batch job."""
        try:
            logger.info(f"Worker {worker_id} processing job {job.id}")
            
            # Update job status
            job.status = BatchStatus.PROCESSING
            job.started_at = datetime.utcnow()
            
            # Process based on strategy
            if self.config.strategy == ProcessingStrategy.SEQUENTIAL:
                results = await self._process_sequential(job)
            elif self.config.strategy == ProcessingStrategy.PARALLEL:
                results = await self._process_parallel(job)
            elif self.config.strategy == ProcessingStrategy.PIPELINE:
                results = await self._process_pipeline(job)
            elif self.config.strategy == ProcessingStrategy.ADAPTIVE:
                results = await self._process_adaptive(job)
            else:
                results = await self._process_sequential(job)
            
            # Update job results
            job.results = results
            job.status = BatchStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            job.progress = 100.0
            
            # Update stats
            with self.lock:
                self.stats['completed_jobs'] += 1
                self.stats['processed_items'] += len(job.items)
            
            logger.info(f"Job {job.id} completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to process job {job.id}: {str(e)}")
            
            # Update job status
            job.status = BatchStatus.FAILED
            job.errors.append(str(e))
            
            # Update stats
            with self.lock:
                self.stats['failed_jobs'] += 1
    
    async def _process_sequential(self, job: BatchJob) -> List[Any]:
        """Process items sequentially."""
        results = []
        
        for i, item in enumerate(job.items):
            try:
                # Process item
                result = await self._process_item(item)
                results.append(result)
                
                # Update progress
                job.progress = (i + 1) / len(job.items) * 100
                
            except Exception as e:
                logger.error(f"Failed to process item {item.id}: {str(e)}")
                job.errors.append(f"Item {item.id}: {str(e)}")
                results.append(None)
        
        return results
    
    async def _process_parallel(self, job: BatchJob) -> List[Any]:
        """Process items in parallel."""
        try:
            # Create tasks for all items
            tasks = [self._process_item(item) for item in job.items]
            
            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to process item {job.items[i].id}: {str(result)}")
                    job.errors.append(f"Item {job.items[i].id}: {str(result)}")
                    processed_results.append(None)
                else:
                    processed_results.append(result)
            
            # Update progress
            job.progress = 100.0
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Failed to process job in parallel: {str(e)}")
            raise
    
    async def _process_pipeline(self, job: BatchJob) -> List[Any]:
        """Process items in pipeline."""
        results = []
        
        # Create processing pipeline
        input_queue = asyncio.Queue()
        output_queue = asyncio.Queue()
        
        # Add items to input queue
        for item in job.items:
            await input_queue.put(item)
        
        # Create pipeline stages
        stages = [
            self._pipeline_stage_1,
            self._pipeline_stage_2,
            self._pipeline_stage_3
        ]
        
        # Process through pipeline
        current_queue = input_queue
        for stage in stages:
            next_queue = asyncio.Queue()
            
            # Process all items through current stage
            tasks = []
            for _ in range(len(job.items)):
                task = asyncio.create_task(self._run_pipeline_stage(stage, current_queue, next_queue))
                tasks.append(task)
            
            await asyncio.gather(*tasks)
            current_queue = next_queue
        
        # Collect results
        while not current_queue.empty():
            result = await current_queue.get()
            results.append(result)
        
        # Update progress
        job.progress = 100.0
        
        return results
    
    async def _process_adaptive(self, job: BatchJob) -> List[Any]:
        """Process items with adaptive strategy."""
        # Analyze job characteristics
        job_size = len(job.items)
        item_complexity = self._analyze_item_complexity(job.items)
        system_load = self._get_system_load()
        
        # Choose strategy based on analysis
        if job_size < 10 and item_complexity < 0.5:
            # Small, simple job - use sequential
            return await self._process_sequential(job)
        elif job_size > 50 or system_load > 0.8:
            # Large job or high system load - use pipeline
            return await self._process_pipeline(job)
        else:
            # Medium job - use parallel
            return await self._process_parallel(job)
    
    async def _run_pipeline_stage(self, stage_func: Callable, input_queue: asyncio.Queue, output_queue: asyncio.Queue):
        """Run a pipeline stage."""
        try:
            item = await input_queue.get()
            result = await stage_func(item)
            await output_queue.put(result)
        except Exception as e:
            logger.error(f"Pipeline stage error: {str(e)}")
            await output_queue.put(None)
    
    async def _pipeline_stage_1(self, item: BatchItem) -> Any:
        """Pipeline stage 1."""
        # Simulate processing
        await asyncio.sleep(0.1)
        return f"stage1_{item.id}"
    
    async def _pipeline_stage_2(self, item: Any) -> Any:
        """Pipeline stage 2."""
        # Simulate processing
        await asyncio.sleep(0.1)
        return f"stage2_{item}"
    
    async def _pipeline_stage_3(self, item: Any) -> Any:
        """Pipeline stage 3."""
        # Simulate processing
        await asyncio.sleep(0.1)
        return f"stage3_{item}"
    
    def _analyze_item_complexity(self, items: List[BatchItem]) -> float:
        """Analyze item complexity."""
        # Simple heuristic based on data size
        total_size = sum(len(str(item.data)) for item in items)
        return min(total_size / (len(items) * 1000), 1.0)
    
    def _get_system_load(self) -> float:
        """Get current system load."""
        try:
            import psutil
            return psutil.cpu_percent() / 100.0
        except:
            return 0.5
    
    async def _process_item(self, item: BatchItem) -> Any:
        """Process individual item."""
        try:
            # Simulate processing
            await asyncio.sleep(0.1)
            
            # Return processed result
            return {
                'id': item.id,
                'processed_data': str(item.data),
                'processed_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to process item {item.id}: {str(e)}")
            raise
    
    async def _monitor_processing(self):
        """Monitor processing performance."""
        while self.running:
            try:
                await asyncio.sleep(10)  # Monitor every 10 seconds
                
                # Log processing stats
                with self.lock:
                    logger.info(f"Batch processing stats: {self.stats}")
                
            except Exception as e:
                logger.error(f"Error monitoring processing: {str(e)}")
    
    async def submit_job(self, 
                        items: List[Any], 
                        job_id: Optional[str] = None,
                        priority: int = 0,
                        metadata: Optional[Dict[str, Any]] = None) -> str:
        """Submit a batch job for processing."""
        try:
            # Generate job ID
            if job_id is None:
                job_id = f"job_{int(time.time())}_{len(self.jobs)}"
            
            # Create batch items
            batch_items = []
            for i, item_data in enumerate(items):
                item = BatchItem(
                    id=f"{job_id}_item_{i}",
                    data=item_data,
                    priority=priority,
                    metadata=metadata or {}
                )
                batch_items.append(item)
            
            # Create batch job
            job = BatchJob(
                id=job_id,
                items=batch_items,
                metadata=metadata or {}
            )
            
            # Add to job queue
            self.jobs[job_id] = job
            
            if self.config.enable_priority:
                self.priority_queue[priority].append(job_id)
            else:
                self.job_queue.put(job_id)
            
            # Update stats
            with self.lock:
                self.stats['total_jobs'] += 1
                self.stats['total_items'] += len(items)
            
            logger.info(f"Submitted batch job {job_id} with {len(items)} items")
            return job_id
            
        except Exception as e:
            logger.error(f"Failed to submit batch job: {str(e)}")
            raise
    
    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status."""
        if job_id not in self.jobs:
            return None
        
        job = self.jobs[job_id]
        
        return {
            'id': job.id,
            'status': job.status.value,
            'progress': job.progress,
            'created_at': job.created_at.isoformat(),
            'started_at': job.started_at.isoformat() if job.started_at else None,
            'completed_at': job.completed_at.isoformat() if job.completed_at else None,
            'total_items': len(job.items),
            'processed_items': len(job.results),
            'errors': job.errors,
            'metadata': job.metadata
        }
    
    async def get_job_results(self, job_id: str) -> Optional[List[Any]]:
        """Get job results."""
        if job_id not in self.jobs:
            return None
        
        job = self.jobs[job_id]
        
        if job.status != BatchStatus.COMPLETED:
            return None
        
        return job.results
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a job."""
        if job_id not in self.jobs:
            return False
        
        job = self.jobs[job_id]
        
        if job.status in [BatchStatus.COMPLETED, BatchStatus.FAILED]:
            return False
        
        job.status = BatchStatus.CANCELLED
        logger.info(f"Cancelled job {job_id}")
        return True
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        with self.lock:
            return {
                'total_jobs': self.stats['total_jobs'],
                'completed_jobs': self.stats['completed_jobs'],
                'failed_jobs': self.stats['failed_jobs'],
                'total_items': self.stats['total_items'],
                'processed_items': self.stats['processed_items'],
                'processing_time': self.stats['processing_time'],
                'active_jobs': len([job for job in self.jobs.values() if job.status == BatchStatus.PROCESSING]),
                'pending_jobs': len([job for job in self.jobs.values() if job.status == BatchStatus.PENDING]),
                'queue_size': self.job_queue.qsize(),
                'priority_queues': {str(k): len(v) for k, v in self.priority_queue.items()}
            }
    
    async def cleanup(self):
        """Cleanup batch processor."""
        try:
            self.running = False
            
            # Cancel all workers
            for worker in self.workers:
                worker.cancel()
            
            # Wait for workers to finish
            await asyncio.gather(*self.workers, return_exceptions=True)
            
            # Shutdown thread pools
            self.thread_pool.shutdown(wait=True)
            self.process_pool.shutdown(wait=True)
            
            logger.info("Batch Processor cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Failed to cleanup Batch Processor: {str(e)}")

# Global batch processor
batch_processor = BatchProcessor()

# Decorators for batch processing
def batch_process(batch_size: int = 100, strategy: ProcessingStrategy = ProcessingStrategy.ADAPTIVE):
    """Decorator to enable batch processing for a function."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # This would integrate with the batch processor
            # For now, just call the function normally
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator

def parallel_process(max_workers: int = 4):
    """Decorator to enable parallel processing for a function."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # This would integrate with parallel processing
            # For now, just call the function normally
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator











