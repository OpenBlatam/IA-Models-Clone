"""
Precomputation Engine
====================

Ultra-fast precomputation system for maximum performance.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
import numpy as np
from functools import wraps
import weakref
from collections import defaultdict, deque
import pickle
import json
import hashlib

logger = logging.getLogger(__name__)

class PrecomputationStrategy(str, Enum):
    """Precomputation strategies."""
    IMMEDIATE = "immediate"      # Compute immediately
    BACKGROUND = "background"    # Compute in background
    SCHEDULED = "scheduled"      # Compute on schedule
    PREDICTIVE = "predictive"    # Compute predictively

class PrecomputationPriority(str, Enum):
    """Precomputation priorities."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class PrecomputationTask:
    """Precomputation task definition."""
    id: str
    func: Callable
    args: Tuple = ()
    kwargs: Dict[str, Any] = field(default_factory=dict)
    strategy: PrecomputationStrategy = PrecomputationStrategy.BACKGROUND
    priority: PrecomputationPriority = PrecomputationPriority.NORMAL
    ttl: Optional[int] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    scheduled_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PrecomputationResult:
    """Precomputation result."""
    task_id: str
    result: Any
    computed_at: datetime
    computation_time: float
    size: int
    compressed: bool = False

@dataclass
class PrecomputationConfig:
    """Precomputation configuration."""
    max_tasks: int = 1000
    max_results: int = 10000
    max_workers: int = 8
    enable_compression: bool = True
    enable_caching: bool = True
    enable_monitoring: bool = True
    cleanup_interval: int = 3600
    ttl_default: int = 86400

class PrecomputationEngine:
    """
    Ultra-fast precomputation engine.
    
    Features:
    - Multiple computation strategies
    - Priority-based scheduling
    - Result caching and compression
    - Background processing
    - Predictive computation
    """
    
    def __init__(self, config: Optional[PrecomputationConfig] = None):
        self.config = config or PrecomputationConfig()
        self.tasks = {}
        self.results = {}
        self.task_queue = asyncio.PriorityQueue()
        self.background_workers = []
        self.running = False
        self.stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'cached_results': 0,
            'compressed_results': 0,
            'total_computation_time': 0.0,
            'average_computation_time': 0.0
        }
        self.lock = asyncio.Lock()
        
    async def initialize(self):
        """Initialize precomputation engine."""
        logger.info("Initializing Precomputation Engine...")
        
        try:
            self.running = True
            
            # Start background workers
            for i in range(self.config.max_workers):
                worker = asyncio.create_task(self._background_worker(f"worker-{i}"))
                self.background_workers.append(worker)
            
            # Start cleanup task
            if self.config.cleanup_interval > 0:
                asyncio.create_task(self._cleanup_worker())
            
            logger.info("Precomputation Engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Precomputation Engine: {str(e)}")
            raise
    
    async def _background_worker(self, worker_id: str):
        """Background computation worker."""
        while self.running:
            try:
                # Get task from queue
                priority, task = await asyncio.wait_for(
                    self.task_queue.get(), 
                    timeout=1.0
                )
                
                # Execute task
                await self._execute_task(task, worker_id)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Background worker {worker_id} error: {str(e)}")
    
    async def _execute_task(self, task: PrecomputationTask, worker_id: str):
        """Execute precomputation task."""
        try:
            logger.debug(f"Worker {worker_id} executing task: {task.id}")
            start_time = time.time()
            
            # Execute function
            if asyncio.iscoroutinefunction(task.func):
                result = await task.func(*task.args, **task.kwargs)
            else:
                result = task.func(*task.args, **task.kwargs)
            
            computation_time = time.time() - start_time
            
            # Create result
            precomputation_result = PrecomputationResult(
                task_id=task.id,
                result=result,
                computed_at=datetime.utcnow(),
                computation_time=computation_time,
                size=len(pickle.dumps(result))
            )
            
            # Compress if enabled
            if self.config.enable_compression and precomputation_result.size > 1024:
                compressed_result = await self._compress_result(precomputation_result)
                if compressed_result:
                    precomputation_result = compressed_result
                    self.stats['compressed_results'] += 1
            
            # Store result
            async with self.lock:
                self.results[task.id] = precomputation_result
                self.stats['completed_tasks'] += 1
                self.stats['total_computation_time'] += computation_time
                self.stats['average_computation_time'] = (
                    self.stats['total_computation_time'] / self.stats['completed_tasks']
                )
            
            logger.debug(f"Task {task.id} completed in {computation_time:.3f}s")
            
        except Exception as e:
            logger.error(f"Task {task.id} failed: {str(e)}")
            self.stats['failed_tasks'] += 1
    
    async def _compress_result(self, result: PrecomputationResult) -> Optional[PrecomputationResult]:
        """Compress precomputation result."""
        try:
            # Compress result data
            compressed_data = await self._compress_data(result.result)
            
            if compressed_data:
                result.result = compressed_data
                result.compressed = True
                result.size = len(compressed_data)
                return result
            
        except Exception as e:
            logger.error(f"Failed to compress result {result.task_id}: {str(e)}")
        
        return None
    
    async def _compress_data(self, data: Any) -> Optional[bytes]:
        """Compress data."""
        try:
            # Serialize data
            serialized = pickle.dumps(data)
            
            # Compress using gzip
            import gzip
            compressed = gzip.compress(serialized)
            
            # Return if compression is beneficial
            if len(compressed) < len(serialized):
                return compressed
            
        except Exception as e:
            logger.error(f"Failed to compress data: {str(e)}")
        
        return None
    
    async def _decompress_data(self, compressed_data: bytes) -> Any:
        """Decompress data."""
        try:
            # Decompress using gzip
            import gzip
            decompressed = gzip.decompress(compressed_data)
            
            # Deserialize data
            return pickle.loads(decompressed)
            
        except Exception as e:
            logger.error(f"Failed to decompress data: {str(e)}")
            return None
    
    async def _cleanup_worker(self):
        """Cleanup expired results."""
        while self.running:
            try:
                await asyncio.sleep(self.config.cleanup_interval)
                
                # Cleanup expired results
                await self._cleanup_expired_results()
                
            except Exception as e:
                logger.error(f"Cleanup worker error: {str(e)}")
    
    async def _cleanup_expired_results(self):
        """Cleanup expired results."""
        try:
            current_time = datetime.utcnow()
            expired_results = []
            
            async with self.lock:
                for task_id, result in self.results.items():
                    # Check TTL
                    if result.computed_at + timedelta(seconds=self.config.ttl_default) < current_time:
                        expired_results.append(task_id)
                
                # Remove expired results
                for task_id in expired_results:
                    del self.results[task_id]
                    self.stats['cached_results'] -= 1
            
            if expired_results:
                logger.info(f"Cleaned up {len(expired_results)} expired results")
                
        except Exception as e:
            logger.error(f"Failed to cleanup expired results: {str(e)}")
    
    async def schedule_task(self, 
                           task_id: str,
                           func: Callable,
                           *args,
                           strategy: PrecomputationStrategy = PrecomputationStrategy.BACKGROUND,
                           priority: PrecomputationPriority = PrecomputationPriority.NORMAL,
                           ttl: Optional[int] = None,
                           scheduled_at: Optional[datetime] = None,
                           **kwargs) -> str:
        """Schedule precomputation task."""
        try:
            # Check if task already exists
            if task_id in self.tasks:
                logger.warning(f"Task {task_id} already exists")
                return task_id
            
            # Create task
            task = PrecomputationTask(
                id=task_id,
                func=func,
                args=args,
                kwargs=kwargs,
                strategy=strategy,
                priority=priority,
                ttl=ttl or self.config.ttl_default,
                scheduled_at=scheduled_at
            )
            
            # Store task
            async with self.lock:
                self.tasks[task_id] = task
                self.stats['total_tasks'] += 1
            
            # Schedule based on strategy
            if strategy == PrecomputationStrategy.IMMEDIATE:
                # Execute immediately
                await self._execute_task(task, "immediate")
            elif strategy == PrecomputationStrategy.BACKGROUND:
                # Add to background queue
                priority_value = self._get_priority_value(priority)
                await self.task_queue.put((priority_value, task))
            elif strategy == PrecomputationStrategy.SCHEDULED:
                # Schedule for later execution
                if scheduled_at:
                    await self._schedule_task(task, scheduled_at)
            elif strategy == PrecomputationStrategy.PREDICTIVE:
                # Add to predictive queue
                priority_value = self._get_priority_value(priority)
                await self.task_queue.put((priority_value, task))
            
            logger.debug(f"Scheduled task: {task_id}")
            return task_id
            
        except Exception as e:
            logger.error(f"Failed to schedule task {task_id}: {str(e)}")
            raise
    
    def _get_priority_value(self, priority: PrecomputationPriority) -> int:
        """Get priority value for queue."""
        priority_values = {
            PrecomputationPriority.CRITICAL: 0,
            PrecomputationPriority.HIGH: 1,
            PrecomputationPriority.NORMAL: 2,
            PrecomputationPriority.LOW: 3
        }
        return priority_values.get(priority, 2)
    
    async def _schedule_task(self, task: PrecomputationTask, scheduled_at: datetime):
        """Schedule task for later execution."""
        try:
            # Calculate delay
            delay = (scheduled_at - datetime.utcnow()).total_seconds()
            
            if delay > 0:
                # Schedule task
                asyncio.create_task(self._delayed_execution(task, delay))
            else:
                # Execute immediately
                await self._execute_task(task, "scheduled")
                
        except Exception as e:
            logger.error(f"Failed to schedule task {task.id}: {str(e)}")
    
    async def _delayed_execution(self, task: PrecomputationTask, delay: float):
        """Execute task after delay."""
        try:
            await asyncio.sleep(delay)
            await self._execute_task(task, "scheduled")
        except Exception as e:
            logger.error(f"Delayed execution failed for task {task.id}: {str(e)}")
    
    async def get_result(self, task_id: str) -> Optional[Any]:
        """Get precomputation result."""
        try:
            async with self.lock:
                if task_id not in self.results:
                    return None
                
                result = self.results[task_id]
                
                # Decompress if needed
                if result.compressed:
                    decompressed_result = await self._decompress_data(result.result)
                    if decompressed_result is not None:
                        return decompressed_result
                
                return result.result
                
        except Exception as e:
            logger.error(f"Failed to get result for task {task_id}: {str(e)}")
            return None
    
    def is_result_available(self, task_id: str) -> bool:
        """Check if result is available."""
        return task_id in self.results
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status."""
        if task_id not in self.tasks:
            return None
        
        task = self.tasks[task_id]
        result_available = self.is_result_available(task_id)
        
        return {
            'task_id': task_id,
            'strategy': task.strategy.value,
            'priority': task.priority.value,
            'created_at': task.created_at.isoformat(),
            'scheduled_at': task.scheduled_at.isoformat() if task.scheduled_at else None,
            'result_available': result_available,
            'ttl': task.ttl
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get precomputation statistics."""
        return {
            'total_tasks': self.stats['total_tasks'],
            'completed_tasks': self.stats['completed_tasks'],
            'failed_tasks': self.stats['failed_tasks'],
            'cached_results': len(self.results),
            'compressed_results': self.stats['compressed_results'],
            'total_computation_time': self.stats['total_computation_time'],
            'average_computation_time': self.stats['average_computation_time'],
            'config': {
                'max_tasks': self.config.max_tasks,
                'max_results': self.config.max_results,
                'max_workers': self.config.max_workers,
                'compression_enabled': self.config.enable_compression,
                'caching_enabled': self.config.enable_caching,
                'cleanup_interval': self.config.cleanup_interval,
                'ttl_default': self.config.ttl_default
            }
        }
    
    async def cleanup(self):
        """Cleanup precomputation engine."""
        try:
            self.running = False
            
            # Cancel background workers
            for worker in self.background_workers:
                worker.cancel()
            
            await asyncio.gather(*self.background_workers, return_exceptions=True)
            
            # Clear data
            self.tasks.clear()
            self.results.clear()
            
            logger.info("Precomputation Engine cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Failed to cleanup Precomputation Engine: {str(e)}")

# Global precomputation engine
precomputation_engine = PrecomputationEngine()

# Decorators for precomputation
def precompute_task(task_id: str, 
                   strategy: PrecomputationStrategy = PrecomputationStrategy.BACKGROUND,
                   priority: PrecomputationPriority = PrecomputationPriority.NORMAL,
                   ttl: Optional[int] = None):
    """Decorator for precomputation tasks."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Check if result is already available
            result = await precomputation_engine.get_result(task_id)
            if result is not None:
                return result
            
            # Schedule task if not already scheduled
            if not precomputation_engine.is_result_available(task_id):
                await precomputation_engine.schedule_task(
                    task_id=task_id,
                    func=func,
                    *args,
                    strategy=strategy,
                    priority=priority,
                    ttl=ttl,
                    **kwargs
                )
            
            # Execute function
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator

def precompute_result(task_id: str, ttl: Optional[int] = None):
    """Decorator for precomputed results."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Check if result is available
            result = await precomputation_engine.get_result(task_id)
            if result is not None:
                return result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            
            # Store result
            await precomputation_engine.schedule_task(
                task_id=task_id,
                func=lambda: result,
                strategy=PrecomputationStrategy.IMMEDIATE,
                ttl=ttl
            )
            
            return result
        
        return wrapper
    return decorator











