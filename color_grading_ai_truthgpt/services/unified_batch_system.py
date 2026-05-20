"""
Unified Batch System for Color Grading AI
==========================================

Consolidates batch processing services:
- BatchProcessor (basic batch processing)
- AdvancedBatchOptimizer (advanced batch optimization)
- BatchOptimizer (batch optimization)

Features:
- Unified batch processing interface
- Multiple optimization strategies
- Progress tracking
- Error handling
- Resume capability
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .batch_processor import BatchProcessor, BatchJob, BatchItem
from .batch_optimizer_advanced import AdvancedBatchOptimizer, BatchStrategy
from .batch_optimizer import BatchOptimizer, BatchOptimization

logger = logging.getLogger(__name__)


class BatchMode(Enum):
    """Batch processing modes."""
    BASIC = "basic"  # Simple batch processing
    OPTIMIZED = "optimized"  # With optimization
    ADVANCED = "advanced"  # Advanced optimization strategies


@dataclass
class UnifiedBatchResult:
    """Unified batch result."""
    job_id: str
    success: bool
    total: int
    completed: int
    failed: int
    results: List[Any] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class UnifiedBatchSystem:
    """
    Unified batch processing system.
    
    Consolidates:
    - BatchProcessor: Basic batch processing
    - AdvancedBatchOptimizer: Advanced optimization
    - BatchOptimizer: Batch optimization
    
    Features:
    - Unified interface for batch processing
    - Multiple optimization strategies
    - Progress tracking
    - Error handling
    - Resume capability
    """
    
    def __init__(
        self,
        max_parallel: int = 3,
        default_mode: BatchMode = BatchMode.BASIC
    ):
        """
        Initialize unified batch system.
        
        Args:
            max_parallel: Maximum parallel processing
            default_mode: Default batch mode
        """
        self.max_parallel = max_parallel
        self.default_mode = default_mode
        
        # Initialize components
        self.batch_processor = BatchProcessor(max_parallel=max_parallel)
        self.advanced_optimizer = AdvancedBatchOptimizer()
        self.batch_optimizer = BatchOptimizer()
        
        logger.info(f"Initialized UnifiedBatchSystem (mode={default_mode.value})")
    
    async def process_batch(
        self,
        items: List[Dict[str, Any]],
        processor_func: Callable,
        mode: Optional[BatchMode] = None,
        optimization_strategy: Optional[BatchStrategy] = None,
        job_id: Optional[str] = None
    ) -> UnifiedBatchResult:
        """
        Process batch of items.
        
        Args:
            items: List of items to process
            processor_func: Processing function
            mode: Batch processing mode
            optimization_strategy: Optional optimization strategy
            job_id: Optional job ID
            
        Returns:
            Unified batch result
        """
        mode = mode or self.default_mode
        start_time = datetime.now()
        
        try:
            if mode == BatchMode.BASIC:
                # Use basic batch processor
                batch_job = await self.batch_processor.process_batch(
                    items=items,
                    processor_func=processor_func,
                    job_id=job_id
                )
                
                return UnifiedBatchResult(
                    job_id=batch_job.id,
                    success=batch_job.status == "completed",
                    total=batch_job.total,
                    completed=batch_job.completed,
                    failed=batch_job.failed,
                    results=[item.result for item in batch_job.items if item.result],
                    errors=[item.error for item in batch_job.items if item.error],
                    execution_time=(datetime.now() - start_time).total_seconds()
                )
            
            elif mode == BatchMode.OPTIMIZED:
                # Use batch optimizer
                optimization = self.batch_optimizer.optimize_batch(items)
                optimized_items = optimization.optimized_items
                
                batch_job = await self.batch_processor.process_batch(
                    items=optimized_items,
                    processor_func=processor_func,
                    job_id=job_id
                )
                
                return UnifiedBatchResult(
                    job_id=batch_job.id,
                    success=batch_job.status == "completed",
                    total=batch_job.total,
                    completed=batch_job.completed,
                    failed=batch_job.failed,
                    results=[item.result for item in batch_job.items if item.result],
                    errors=[item.error for item in batch_job.items if item.error],
                    execution_time=(datetime.now() - start_time).total_seconds()
                )
            
            elif mode == BatchMode.ADVANCED:
                # Use advanced batch optimizer
                strategy = optimization_strategy or BatchStrategy.SIZE_BASED
                batch_result = await self.advanced_optimizer.process_batch(
                    items=items,
                    processor_func=processor_func,
                    strategy=strategy
                )
                
                return UnifiedBatchResult(
                    job_id=batch_result.batch_id,
                    success=batch_result.success,
                    total=len(batch_result.items),
                    completed=sum(1 for item in batch_result.items if item.success),
                    failed=sum(1 for item in batch_result.items if not item.success),
                    results=[item.result for item in batch_result.items if item.success],
                    errors=[item.error for item in batch_result.items if not item.success],
                    execution_time=(datetime.now() - start_time).total_seconds()
                )
            
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            return UnifiedBatchResult(
                job_id=job_id or "unknown",
                success=False,
                total=len(items),
                completed=0,
                failed=len(items),
                errors=[str(e)],
                execution_time=(datetime.now() - start_time).total_seconds()
            )
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get batch job status.
        
        Args:
            job_id: Job ID
            
        Returns:
            Job status or None
        """
        job = self.batch_processor._jobs.get(job_id)
        if not job:
            return None
        
        return {
            "id": job.id,
            "status": job.status,
            "total": job.total,
            "completed": job.completed,
            "failed": job.failed,
            "progress": job.completed / job.total if job.total > 0 else 0.0,
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get batch processing statistics."""
        return {
            "mode": self.default_mode.value,
            "max_parallel": self.max_parallel,
            "active_jobs": len(self.batch_processor._jobs),
            "total_jobs": len(self.batch_processor._jobs),
        }


