"""
Base Compressor Implementation

Ultra-specialized base compressor with advanced features for
data compression, decompression, and optimization.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, TypeVar, Generic, Callable, Union, BinaryIO, TextIO
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import weakref
from contextlib import asynccontextmanager
import io

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


class CompressorType(Enum):
    """Compressor type enumeration"""
    GZIP = "gzip"
    BROTLI = "brotli"
    LZ4 = "lz4"
    ZSTD = "zstd"
    LZMA = "lzma"
    BZIP2 = "bzip2"
    DEFLATE = "deflate"
    SNAPPY = "snappy"
    CUSTOM = "custom"


class CompressionMode(Enum):
    """Compression mode enumeration"""
    FAST = "fast"
    BALANCED = "balanced"
    HIGH = "high"
    MAXIMUM = "maximum"
    LOSSLESS = "lossless"
    LOSSY = "lossy"


class CompressorPriority(Enum):
    """Compressor priority enumeration"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class CompressorConfig:
    """Compressor configuration"""
    name: str
    compressor_type: CompressorType
    compression_mode: CompressionMode = CompressionMode.BALANCED
    priority: CompressorPriority = CompressorPriority.NORMAL
    enabled: bool = True
    timeout: Optional[float] = None
    retry_count: int = 0
    compression_level: int = 6
    chunk_size: int = 8192
    buffer_size: int = 65536
    custom_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompressionContext:
    """Compression context"""
    compressor_name: str
    compression_mode: CompressionMode
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    input_size: int = 0
    output_size: int = 0
    compression_ratio: float = 1.0
    compression_speed: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseCompressor(ABC, Generic[T, R]):
    """Base compressor with advanced features"""
    
    def __init__(self, config: CompressorConfig):
        self.config = config
        self._enabled = config.enabled
        self._compression_count = 0
        self._decompression_count = 0
        self._success_count = 0
        self._error_count = 0
        self._total_duration = 0.0
        self._total_input_size = 0
        self._total_output_size = 0
        self._callbacks: List[Callable] = []
        self._error_handlers: List[Callable] = []
        self._lock = asyncio.Lock()
    
    @abstractmethod
    async def _compress(self, data: T, context: CompressionContext) -> R:
        """Compress data (override in subclasses)"""
        pass
    
    @abstractmethod
    async def _decompress(self, data: R, context: CompressionContext) -> T:
        """Decompress data (override in subclasses)"""
        pass
    
    async def compress(self, data: T) -> R:
        """Compress data with context"""
        if not self._enabled:
            raise RuntimeError(f"Compressor '{self.config.name}' is disabled")
        
        context = CompressionContext(
            compressor_name=self.config.name,
            compression_mode=self.config.compression_mode,
            start_time=datetime.utcnow(),
            input_size=self._get_data_size(data)
        )
        
        try:
            # Pre-compression hook
            await self._pre_compress(context)
            
            # Compress data
            result = await asyncio.wait_for(
                self._compress(data, context),
                timeout=self.config.timeout
            )
            
            context.output_size = self._get_data_size(result)
            context.compression_ratio = (
                context.input_size / context.output_size
                if context.output_size > 0 else 1
            )
            
            # Post-compression hook
            await self._post_compress(context)
            
            return result
            
        except Exception as e:
            await self._handle_error(context, e)
            raise
    
    async def decompress(self, data: R) -> T:
        """Decompress data with context"""
        if not self._enabled:
            raise RuntimeError(f"Compressor '{self.config.name}' is disabled")
        
        context = CompressionContext(
            compressor_name=self.config.name,
            compression_mode=self.config.compression_mode,
            start_time=datetime.utcnow(),
            input_size=self._get_data_size(data)
        )
        
        try:
            # Pre-decompression hook
            await self._pre_decompress(context)
            
            # Decompress data
            result = await asyncio.wait_for(
                self._decompress(data, context),
                timeout=self.config.timeout
            )
            
            context.output_size = self._get_data_size(result)
            
            # Post-decompression hook
            await self._post_decompress(context)
            
            return result
            
        except Exception as e:
            await self._handle_error(context, e)
            raise
    
    async def compress_to_file(self, data: T, file_path: str) -> None:
        """Compress data to file"""
        if not self._enabled:
            raise RuntimeError(f"Compressor '{self.config.name}' is disabled")
        
        compressed_data = await self.compress(data)
        
        # Write to file
        if isinstance(compressed_data, (str, bytes)):
            mode = 'wb' if isinstance(compressed_data, bytes) else 'w'
            encoding = None if isinstance(compressed_data, bytes) else 'utf-8'
            
            with open(file_path, mode, encoding=encoding) as f:
                f.write(compressed_data)
        else:
            # For other types, convert to string first
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(str(compressed_data))
    
    async def decompress_from_file(self, file_path: str) -> T:
        """Decompress data from file"""
        if not self._enabled:
            raise RuntimeError(f"Compressor '{self.config.name}' is disabled")
        
        # Read from file
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = f.read()
        
        return await self.decompress(data)
    
    async def compress_to_stream(self, data: T, stream: Union[BinaryIO, TextIO]) -> None:
        """Compress data to stream"""
        if not self._enabled:
            raise RuntimeError(f"Compressor '{self.config.name}' is disabled")
        
        compressed_data = await self.compress(data)
        
        if isinstance(stream, BinaryIO):
            if isinstance(compressed_data, str):
                compressed_data = compressed_data.encode('utf-8')
            stream.write(compressed_data)
        else:
            if isinstance(compressed_data, bytes):
                compressed_data = compressed_data.decode('utf-8')
            stream.write(compressed_data)
    
    async def decompress_from_stream(self, stream: Union[BinaryIO, TextIO]) -> T:
        """Decompress data from stream"""
        if not self._enabled:
            raise RuntimeError(f"Compressor '{self.config.name}' is disabled")
        
        data = stream.read()
        
        if isinstance(stream, BinaryIO) and isinstance(data, bytes):
            # Try to decode as text first
            try:
                data = data.decode('utf-8')
            except UnicodeDecodeError:
                # Keep as bytes if decoding fails
                pass
        
        return await self.decompress(data)
    
    def compress_sync(self, data: T) -> R:
        """Compress data synchronously"""
        if not self._enabled:
            raise RuntimeError(f"Compressor '{self.config.name}' is disabled")
        
        # Run async compress in sync context
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.compress(data))
    
    def decompress_sync(self, data: R) -> T:
        """Decompress data synchronously"""
        if not self._enabled:
            raise RuntimeError(f"Compressor '{self.config.name}' is disabled")
        
        # Run async decompress in sync context
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.decompress(data))
    
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
    
    async def _pre_compress(self, context: CompressionContext) -> None:
        """Pre-compression hook (override in subclasses)"""
        pass
    
    async def _post_compress(self, context: CompressionContext) -> None:
        """Post-compression hook (override in subclasses)"""
        context.end_time = datetime.utcnow()
        context.duration = (context.end_time - context.start_time).total_seconds()
        
        # Calculate compression speed
        if context.duration > 0:
            context.compression_speed = context.input_size / context.duration
        
        # Update metrics
        self._update_metrics(True, context)
        
        # Call callbacks
        for callback in self._callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(context, "compress")
                else:
                    callback(context, "compress")
            except Exception as e:
                logger.error(f"Error in callback: {e}")
    
    async def _pre_decompress(self, context: CompressionContext) -> None:
        """Pre-decompression hook (override in subclasses)"""
        pass
    
    async def _post_decompress(self, context: CompressionContext) -> None:
        """Post-decompression hook (override in subclasses)"""
        context.end_time = datetime.utcnow()
        context.duration = (context.end_time - context.start_time).total_seconds()
        
        # Calculate decompression speed
        if context.duration > 0:
            context.compression_speed = context.input_size / context.duration
        
        # Update metrics
        self._update_metrics(True, context)
        
        # Call callbacks
        for callback in self._callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(context, "decompress")
                else:
                    callback(context, "decompress")
            except Exception as e:
                logger.error(f"Error in callback: {e}")
    
    async def _handle_error(self, context: CompressionContext, error: Exception) -> None:
        """Handle compression errors"""
        context.end_time = datetime.utcnow()
        context.duration = (context.end_time - context.start_time).total_seconds()
        
        # Update metrics
        self._update_metrics(False, context)
        
        # Call error handlers
        for handler in self._error_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(context, error)
                else:
                    handler(context, error)
            except Exception as e:
                logger.error(f"Error in error handler: {e}")
    
    def _update_metrics(self, success: bool, context: CompressionContext) -> None:
        """Update compressor metrics"""
        if success:
            self._success_count += 1
        else:
            self._error_count += 1
        
        if context.duration:
            self._total_duration += context.duration
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
        """Enable compressor"""
        self._enabled = True
    
    def disable(self) -> None:
        """Disable compressor"""
        self._enabled = False
    
    def is_enabled(self) -> bool:
        """Check if compressor is enabled"""
        return self._enabled
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get compressor metrics"""
        avg_duration = (
            self._total_duration / (self._compression_count + self._decompression_count)
            if (self._compression_count + self._decompression_count) > 0 else 0
        )
        
        compression_ratio = (
            self._total_input_size / self._total_output_size
            if self._total_output_size > 0 else 1
        )
        
        throughput = (
            self._total_input_size / self._total_duration
            if self._total_duration > 0 else 0
        )
        
        return {
            "name": self.config.name,
            "type": self.config.compressor_type.value,
            "mode": self.config.compression_mode.value,
            "priority": self.config.priority.value,
            "enabled": self._enabled,
            "compression_count": self._compression_count,
            "decompression_count": self._decompression_count,
            "success_count": self._success_count,
            "error_count": self._error_count,
            "total_duration": self._total_duration,
            "average_duration": avg_duration,
            "total_input_size": self._total_input_size,
            "total_output_size": self._total_output_size,
            "compression_ratio": compression_ratio,
            "throughput": throughput
        }
    
    def reset_metrics(self) -> None:
        """Reset compressor metrics"""
        self._compression_count = 0
        self._decompression_count = 0
        self._success_count = 0
        self._error_count = 0
        self._total_duration = 0.0
        self._total_input_size = 0
        self._total_output_size = 0
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.config.name}', enabled={self._enabled})"


class CompressorChain:
    """Chain of compressors with priority ordering"""
    
    def __init__(self):
        self._compressors: List[BaseCompressor] = []
        self._lock = asyncio.Lock()
    
    def add_compressor(self, compressor: BaseCompressor) -> None:
        """Add compressor to chain"""
        self._compressors.append(compressor)
        # Sort by priority (higher priority first)
        self._compressors.sort(
            key=lambda c: c.config.priority.value,
            reverse=True
        )
    
    def remove_compressor(self, name: str) -> None:
        """Remove compressor from chain"""
        self._compressors = [
            c for c in self._compressors
            if c.config.name != name
        ]
    
    async def compress(self, data: Any) -> Any:
        """Compress data through compressor chain"""
        result = data
        
        for compressor in self._compressors:
            if compressor.is_enabled():
                try:
                    result = await compressor.compress(result)
                except Exception as e:
                    logger.error(f"Error in compressor '{compressor.config.name}': {e}")
                    # Continue to next compressor or re-raise based on configuration
                    raise
        
        return result
    
    async def decompress(self, data: Any) -> Any:
        """Decompress data through compressor chain"""
        result = data
        
        # Reverse order for decompression
        for compressor in reversed(self._compressors):
            if compressor.is_enabled():
                try:
                    result = await compressor.decompress(result)
                except Exception as e:
                    logger.error(f"Error in compressor '{compressor.config.name}': {e}")
                    # Continue to next compressor or re-raise based on configuration
                    raise
        
        return result
    
    def get_compressors(self) -> List[BaseCompressor]:
        """Get all compressors"""
        return self._compressors.copy()
    
    def get_compressors_by_type(self, compressor_type: CompressorType) -> List[BaseCompressor]:
        """Get compressors by type"""
        return [
            c for c in self._compressors
            if c.config.compressor_type == compressor_type
        ]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics for all compressors"""
        return {
            compressor.config.name: compressor.get_metrics()
            for compressor in self._compressors
        }


class CompressorRegistry:
    """Registry for managing compressors"""
    
    def __init__(self):
        self._compressors: Dict[str, BaseCompressor] = {}
        self._chains: Dict[str, CompressorChain] = {}
        self._lock = asyncio.Lock()
    
    async def register(self, compressor: BaseCompressor) -> None:
        """Register compressor"""
        async with self._lock:
            self._compressors[compressor.config.name] = compressor
            logger.info(f"Registered compressor: {compressor.config.name}")
    
    async def unregister(self, name: str) -> None:
        """Unregister compressor"""
        async with self._lock:
            if name in self._compressors:
                del self._compressors[name]
                logger.info(f"Unregistered compressor: {name}")
    
    def get(self, name: str) -> Optional[BaseCompressor]:
        """Get compressor by name"""
        return self._compressors.get(name)
    
    def get_by_type(self, compressor_type: CompressorType) -> List[BaseCompressor]:
        """Get compressors by type"""
        return [
            compressor for compressor in self._compressors.values()
            if compressor.config.compressor_type == compressor_type
        ]
    
    def create_chain(self, name: str, compressor_names: List[str]) -> CompressorChain:
        """Create compressor chain"""
        chain = CompressorChain()
        
        for compressor_name in compressor_names:
            compressor = self.get(compressor_name)
            if compressor:
                chain.add_compressor(compressor)
        
        self._chains[name] = chain
        return chain
    
    def get_chain(self, name: str) -> Optional[CompressorChain]:
        """Get compressor chain"""
        return self._chains.get(name)
    
    def list_all(self) -> List[BaseCompressor]:
        """List all compressors"""
        return list(self._compressors.values())
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics for all compressors"""
        return {
            name: compressor.get_metrics()
            for name, compressor in self._compressors.items()
        }


# Global compressor registry
compressor_registry = CompressorRegistry()


# Convenience functions
async def register_compressor(compressor: BaseCompressor):
    """Register compressor"""
    await compressor_registry.register(compressor)


def get_compressor(name: str) -> Optional[BaseCompressor]:
    """Get compressor by name"""
    return compressor_registry.get(name)


def create_compressor_chain(name: str, compressor_names: List[str]) -> CompressorChain:
    """Create compressor chain"""
    return compressor_registry.create_chain(name, compressor_names)


# Compressor factory functions
def create_compressor(compressor_type: CompressorType, name: str, **kwargs) -> BaseCompressor:
    """Create compressor by type"""
    config = CompressorConfig(
        name=name,
        compressor_type=compressor_type,
        **kwargs
    )
    
    # This would be implemented with specific compressor classes
    # For now, return a placeholder
    raise NotImplementedError(f"Compressor type {compressor_type} not implemented yet")


# Common compressor combinations
def fast_compression_chain(name: str = "fast_compression") -> CompressorChain:
    """Create fast compression compressor chain"""
    return create_compressor_chain(name, [
        "lz4_compressor",
        "snappy_compressor"
    ])


def high_compression_chain(name: str = "high_compression") -> CompressorChain:
    """Create high compression compressor chain"""
    return create_compressor_chain(name, [
        "zstd_compressor",
        "brotli_compressor",
        "lzma_compressor"
    ])


def balanced_compression_chain(name: str = "balanced_compression") -> CompressorChain:
    """Create balanced compression compressor chain"""
    return create_compressor_chain(name, [
        "gzip_compressor",
        "bzip2_compressor"
    ])


def web_compression_chain(name: str = "web_compression") -> CompressorChain:
    """Create web compression compressor chain"""
    return create_compressor_chain(name, [
        "gzip_compressor",
        "brotli_compressor",
        "deflate_compressor"
    ])





















