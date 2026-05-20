"""
Processor Base
==============

Base classes for all processor types.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable, Awaitable, TypeVar, Generic
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


class ProcessingStatus(Enum):
    """Processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ProcessingConfig:
    """Processing configuration."""
    name: str
    enabled: bool = True
    timeout: Optional[float] = None
    retry_count: int = 0
    parallel: bool = False
    max_concurrent: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessingResult:
    """Processing result."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    duration: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class BaseProcessor(ABC, Generic[T, R]):
    """Base processor interface."""
    
    def __init__(self, config: ProcessingConfig):
        """
        Initialize processor.
        
        Args:
            config: Processing configuration
        """
        self.config = config
        self.stats = {
            "total_items": 0,
            "successful_items": 0,
            "failed_items": 0,
            "total_duration": 0.0
        }
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(config.max_concurrent) if config.parallel else None
    
    @abstractmethod
    async def process(self, item: T) -> ProcessingResult:
        """
        Process a single item.
        
        Args:
            item: Item to process
            
        Returns:
            Processing result
        """
        pass
    
    async def process_batch(self, items: List[T]) -> List[ProcessingResult]:
        """
        Process batch of items.
        
        Args:
            items: List of items to process
            
        Returns:
            List of processing results
        """
        if not self.config.enabled:
            return [
                ProcessingResult(
                    success=False,
                    error="Processor is disabled"
                )
                for _ in items
            ]
        
        if self.config.parallel and self._semaphore:
            async def process_with_semaphore(item: T) -> ProcessingResult:
                async with self._semaphore:
                    return await self.process(item)
            
            tasks = [process_with_semaphore(item) for item in items]
            results = await asyncio.gather(*tasks)
        else:
            results = []
            for item in items:
                result = await self.process(item)
                results.append(result)
        
        # Update stats
        async with self._lock:
            self.stats["total_items"] += len(items)
            self.stats["successful_items"] += sum(1 for r in results if r.success)
            self.stats["failed_items"] += sum(1 for r in results if not r.success)
            self.stats["total_duration"] += sum(r.duration for r in results)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics."""
        total = self.stats["total_items"]
        success_rate = (
            self.stats["successful_items"] / total
            if total > 0 else 0.0
        )
        avg_duration = (
            self.stats["total_duration"] / total
            if total > 0 else 0.0
        )
        
        return {
            "name": self.config.name,
            "enabled": self.config.enabled,
            "total_items": total,
            "successful_items": self.stats["successful_items"],
            "failed_items": self.stats["failed_items"],
            "success_rate": success_rate,
            "avg_duration": avg_duration,
            "parallel": self.config.parallel,
            "max_concurrent": self.config.max_concurrent
        }


class AsyncProcessor(BaseProcessor[T, R]):
    """Async processor implementation."""
    
    def __init__(
        self,
        config: ProcessingConfig,
        processor_func: Callable[[T], Awaitable[R]]
    ):
        """
        Initialize async processor.
        
        Args:
            config: Processing configuration
            processor_func: Processor function
        """
        super().__init__(config)
        self.processor_func = processor_func
    
    async def process(self, item: T) -> ProcessingResult:
        """Process item with processor function."""
        if not self.config.enabled:
            return ProcessingResult(
                success=False,
                error="Processor is disabled"
            )
        
        start = datetime.now()
        
        try:
            if self.config.timeout:
                result_data = await asyncio.wait_for(
                    self.processor_func(item),
                    timeout=self.config.timeout
                )
            else:
                result_data = await self.processor_func(item)
            
            duration = (datetime.now() - start).total_seconds()
            result = ProcessingResult(
                success=True,
                data=result_data,
                duration=duration
            )
            
            return result
            
        except asyncio.TimeoutError:
            duration = (datetime.now() - start).total_seconds()
            return ProcessingResult(
                success=False,
                error=f"Processing timeout after {self.config.timeout}s",
                duration=duration
            )
            
        except Exception as e:
            duration = (datetime.now() - start).total_seconds()
            return ProcessingResult(
                success=False,
                error=str(e),
                duration=duration
            )




