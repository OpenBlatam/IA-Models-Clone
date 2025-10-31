"""
Base Processor Implementation

Ultra-specialized base processor with advanced features for
data processing, transformation, and analysis.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, TypeVar, Generic, Callable, Union, Iterator, AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import weakref
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


class ProcessorType(Enum):
    """Processor type enumeration"""
    DATA = "data"
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    AI = "ai"
    VALIDATION = "validation"
    TRANSFORMATION = "transformation"
    AGGREGATION = "aggregation"
    FILTERING = "filtering"
    ENRICHMENT = "enrichment"


class ProcessingMode(Enum):
    """Processing mode enumeration"""
    BATCH = "batch"
    STREAM = "stream"
    REALTIME = "realtime"
    ASYNC = "async"


class ProcessorPriority(Enum):
    """Processor priority enumeration"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class ProcessorConfig:
    """Processor configuration"""
    name: str
    processor_type: ProcessorType
    processing_mode: ProcessingMode = ProcessingMode.BATCH
    priority: ProcessorPriority = ProcessorPriority.NORMAL
    enabled: bool = True
    timeout: Optional[float] = None
    retry_count: int = 0
    batch_size: int = 100
    max_concurrent: int = 10
    memory_limit: Optional[int] = None
    cpu_limit: Optional[float] = None
    custom_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessingContext:
    """Processing context"""
    processor_name: str
    processing_mode: ProcessingMode
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    input_size: int = 0
    output_size: int = 0
    memory_used: int = 0
    cpu_used: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseProcessor(ABC, Generic[T, R]):
    """Base processor with advanced features"""
    
    def __init__(self, config: ProcessorConfig):
        self.config = config
        self._enabled = config.enabled
        self._processing_count = 0
        self._success_count = 0
        self._error_count = 0
        self._total_duration = 0.0
        self._total_input_size = 0
        self._total_output_size = 0
        self._callbacks: List[Callable] = []
        self._error_handlers: List[Callable] = []
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(config.max_concurrent)
    
    @abstractmethod
    async def _process(self, data: T, context: ProcessingContext) -> R:
        """Process data (override in subclasses)"""
        pass
    
    async def process(self, data: T) -> R:
        """Process single data item"""
        if not self._enabled:
            raise RuntimeError(f"Processor '{self.config.name}' is disabled")
        
        context = ProcessingContext(
            processor_name=self.config.name,
            processing_mode=self.config.processing_mode,
            start_time=datetime.utcnow(),
            input_size=self._get_data_size(data)
        )
        
        async with self._semaphore:
            try:
                # Pre-process hook
                await self._pre_process(context)
                
                # Process data
                result = await asyncio.wait_for(
                    self._process(data, context),
                    timeout=self.config.timeout
                )
                
                context.output_size = self._get_data_size(result)
                
                # Post-process hook
                await self._post_process(context)
                
                return result
                
            except Exception as e:
                await self._handle_error(context, e)
                raise
    
    async def process_batch(self, data_list: List[T]) -> List[R]:
        """Process multiple data items"""
        if not self._enabled:
            raise RuntimeError(f"Processor '{self.config.name}' is disabled")
        
        # Process in batches
        results = []
        for i in range(0, len(data_list), self.config.batch_size):
            batch = data_list[i:i + self.config.batch_size]
            
            # Process batch concurrently
            tasks = [
                self.process(data)
                for data in batch
            ]
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            results.extend(batch_results)
        
        return results
    
    async def process_stream(self, data_stream: AsyncIterator[T]) -> AsyncIterator[R]:
        """Process data stream"""
        if not self._enabled:
            raise RuntimeError(f"Processor '{self.config.name}' is disabled")
        
        async for data in data_stream:
            try:
                result = await self.process(data)
                yield result
            except Exception as e:
                logger.error(f"Error processing stream data: {e}")
                # Continue processing or re-raise based on configuration
                raise
    
    def process_sync(self, data: T) -> R:
        """Process data synchronously"""
        if not self._enabled:
            raise RuntimeError(f"Processor '{self.config.name}' is disabled")
        
        # Run async process in sync context
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.process(data))
    
    def process_batch_sync(self, data_list: List[T]) -> List[R]:
        """Process multiple data items synchronously"""
        if not self._enabled:
            raise RuntimeError(f"Processor '{self.config.name}' is disabled")
        
        # Run async process_batch in sync context
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.process_batch(data_list))
    
    def _get_data_size(self, data: Any) -> int:
        """Get data size in bytes"""
        try:
            if hasattr(data, '__sizeof__'):
                return data.__sizeof__()
            elif isinstance(data, (str, bytes)):
                return len(data)
            elif isinstance(data, (list, tuple)):
                return sum(self._get_data_size(item) for item in data)
            elif isinstance(data, dict):
                return sum(
                    self._get_data_size(key) + self._get_data_size(value)
                    for key, value in data.items()
                )
            else:
                return 1  # Default size
        except Exception:
            return 1  # Default size
    
    async def _pre_process(self, context: ProcessingContext) -> None:
        """Pre-process hook (override in subclasses)"""
        pass
    
    async def _post_process(self, context: ProcessingContext) -> None:
        """Post-process hook (override in subclasses)"""
        context.end_time = datetime.utcnow()
        context.duration = (context.end_time - context.start_time).total_seconds()
        
        # Update metrics
        self._update_metrics(context)
        
        # Call callbacks
        for callback in self._callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(context)
                else:
                    callback(context)
            except Exception as e:
                logger.error(f"Error in callback: {e}")
    
    async def _handle_error(self, context: ProcessingContext, error: Exception) -> None:
        """Handle processing errors"""
        context.end_time = datetime.utcnow()
        context.duration = (context.end_time - context.start_time).total_seconds()
        
        # Update metrics
        self._update_metrics(context)
        
        # Call error handlers
        for handler in self._error_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(context, error)
                else:
                    handler(context, error)
            except Exception as e:
                logger.error(f"Error in error handler: {e}")
    
    def _update_metrics(self, context: ProcessingContext) -> None:
        """Update processor metrics"""
        self._processing_count += 1
        if context.end_time:
            self._total_duration += context.duration or 0
            self._total_input_size += context.input_size
            self._total_output_size += context.output_size
    
    def add_callback(self, callback: Callable) -> None:
        """Add callback for events"""
        self._callbacks.append(callback)
    
    def remove_callback(self, callback: Callable) -> None:
        """Remove callback"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def add_error_handler(self, handler: Callable) -> None:
        """Add error handler"""
        self._error_handlers.append(handler)
    
    def remove_error_handler(self, handler: Callable) -> None:
        """Remove error handler"""
        if handler in self._error_handlers:
            self._error_handlers.remove(handler)
    
    def enable(self) -> None:
        """Enable processor"""
        self._enabled = True
    
    def disable(self) -> None:
        """Disable processor"""
        self._enabled = False
    
    def is_enabled(self) -> bool:
        """Check if processor is enabled"""
        return self._enabled
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get processor metrics"""
        avg_duration = (
            self._total_duration / self._processing_count
            if self._processing_count > 0 else 0
        )
        
        compression_ratio = (
            self._total_output_size / self._total_input_size
            if self._total_input_size > 0 else 1
        )
        
        throughput = (
            self._total_input_size / self._total_duration
            if self._total_duration > 0 else 0
        )
        
        return {
            "name": self.config.name,
            "type": self.config.processor_type.value,
            "mode": self.config.processing_mode.value,
            "priority": self.config.priority.value,
            "enabled": self._enabled,
            "processing_count": self._processing_count,
            "success_count": self._success_count,
            "error_count": self._error_count,
            "total_duration": self._total_duration,
            "average_duration": avg_duration,
            "total_input_size": self._total_input_size,
            "total_output_size": self._total_output_size,
            "compression_ratio": compression_ratio,
            "throughput": throughput,
            "max_concurrent": self.config.max_concurrent,
            "current_concurrent": self.config.max_concurrent - self._semaphore._value
        }
    
    def reset_metrics(self) -> None:
        """Reset processor metrics"""
        self._processing_count = 0
        self._success_count = 0
        self._error_count = 0
        self._total_duration = 0.0
        self._total_input_size = 0
        self._total_output_size = 0
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.config.name}', enabled={self._enabled})"


class ProcessorChain:
    """Chain of processors with priority ordering"""
    
    def __init__(self):
        self._processors: List[BaseProcessor] = []
        self._lock = asyncio.Lock()
    
    def add_processor(self, processor: BaseProcessor) -> None:
        """Add processor to chain"""
        self._processors.append(processor)
        # Sort by priority (higher priority first)
        self._processors.sort(
            key=lambda p: p.config.priority.value,
            reverse=True
        )
    
    def remove_processor(self, name: str) -> None:
        """Remove processor from chain"""
        self._processors = [
            p for p in self._processors
            if p.config.name != name
        ]
    
    async def process(self, data: Any) -> Any:
        """Process data through processor chain"""
        result = data
        
        for processor in self._processors:
            if processor.is_enabled():
                try:
                    result = await processor.process(result)
                except Exception as e:
                    logger.error(f"Error in processor '{processor.config.name}': {e}")
                    # Continue to next processor or re-raise based on configuration
                    raise
        
        return result
    
    async def process_batch(self, data_list: List[Any]) -> List[Any]:
        """Process multiple data items through processor chain"""
        results = data_list
        
        for processor in self._processors:
            if processor.is_enabled():
                try:
                    results = await processor.process_batch(results)
                except Exception as e:
                    logger.error(f"Error in processor '{processor.config.name}': {e}")
                    raise
        
        return results
    
    async def process_stream(self, data_stream: AsyncIterator[Any]) -> AsyncIterator[Any]:
        """Process data stream through processor chain"""
        current_stream = data_stream
        
        for processor in self._processors:
            if processor.is_enabled():
                current_stream = processor.process_stream(current_stream)
        
        async for result in current_stream:
            yield result
    
    def get_processors(self) -> List[BaseProcessor]:
        """Get all processors"""
        return self._processors.copy()
    
    def get_processors_by_type(self, processor_type: ProcessorType) -> List[BaseProcessor]:
        """Get processors by type"""
        return [
            p for p in self._processors
            if p.config.processor_type == processor_type
        ]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics for all processors"""
        return {
            processor.config.name: processor.get_metrics()
            for processor in self._processors
        }


class ProcessorRegistry:
    """Registry for managing processors"""
    
    def __init__(self):
        self._processors: Dict[str, BaseProcessor] = {}
        self._chains: Dict[str, ProcessorChain] = {}
        self._lock = asyncio.Lock()
    
    async def register(self, processor: BaseProcessor) -> None:
        """Register processor"""
        async with self._lock:
            self._processors[processor.config.name] = processor
            logger.info(f"Registered processor: {processor.config.name}")
    
    async def unregister(self, name: str) -> None:
        """Unregister processor"""
        async with self._lock:
            if name in self._processors:
                del self._processors[name]
                logger.info(f"Unregistered processor: {name}")
    
    def get(self, name: str) -> Optional[BaseProcessor]:
        """Get processor by name"""
        return self._processors.get(name)
    
    def get_by_type(self, processor_type: ProcessorType) -> List[BaseProcessor]:
        """Get processors by type"""
        return [
            processor for processor in self._processors.values()
            if processor.config.processor_type == processor_type
        ]
    
    def create_chain(self, name: str, processor_names: List[str]) -> ProcessorChain:
        """Create processor chain"""
        chain = ProcessorChain()
        
        for processor_name in processor_names:
            processor = self.get(processor_name)
            if processor:
                chain.add_processor(processor)
        
        self._chains[name] = chain
        return chain
    
    def get_chain(self, name: str) -> Optional[ProcessorChain]:
        """Get processor chain"""
        return self._chains.get(name)
    
    def list_all(self) -> List[BaseProcessor]:
        """List all processors"""
        return list(self._processors.values())
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics for all processors"""
        return {
            name: processor.get_metrics()
            for name, processor in self._processors.items()
        }


# Global processor registry
processor_registry = ProcessorRegistry()


# Convenience functions
async def register_processor(processor: BaseProcessor):
    """Register processor"""
    await processor_registry.register(processor)


def get_processor(name: str) -> Optional[BaseProcessor]:
    """Get processor by name"""
    return processor_registry.get(name)


def create_processor_chain(name: str, processor_names: List[str]) -> ProcessorChain:
    """Create processor chain"""
    return processor_registry.create_chain(name, processor_names)


# Processor factory functions
def create_processor(processor_type: ProcessorType, name: str, **kwargs) -> BaseProcessor:
    """Create processor by type"""
    config = ProcessorConfig(
        name=name,
        processor_type=processor_type,
        **kwargs
    )
    
    # This would be implemented with specific processor classes
    # For now, return a placeholder
    raise NotImplementedError(f"Processor type {processor_type} not implemented yet")


# Common processor combinations
def data_processing_chain(name: str = "data_processing") -> ProcessorChain:
    """Create data processing processor chain"""
    return create_processor_chain(name, [
        "validation",
        "transformation",
        "enrichment",
        "aggregation"
    ])


def ai_processing_chain(name: str = "ai_processing") -> ProcessorChain:
    """Create AI processing processor chain"""
    return create_processor_chain(name, [
        "text_processing",
        "nlp_processing",
        "ai_inference",
        "post_processing"
    ])


def media_processing_chain(name: str = "media_processing") -> ProcessorChain:
    """Create media processing processor chain"""
    return create_processor_chain(name, [
        "image_processing",
        "audio_processing",
        "video_processing",
        "metadata_extraction"
    ])


def analytics_processing_chain(name: str = "analytics_processing") -> ProcessorChain:
    """Create analytics processing processor chain"""
    return create_processor_chain(name, [
        "data_validation",
        "statistics",
        "aggregation",
        "reporting"
    ])





















