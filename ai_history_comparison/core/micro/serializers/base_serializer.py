"""
Base Serializer Implementation

Ultra-specialized base serializer with advanced features for
data serialization, deserialization, and format conversion.
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


class SerializerType(Enum):
    """Serializer type enumeration"""
    JSON = "json"
    XML = "xml"
    YAML = "yaml"
    CSV = "csv"
    BINARY = "binary"
    PROTOBUF = "protobuf"
    AVRO = "avro"
    PARQUET = "parquet"
    HDF5 = "hdf5"
    CUSTOM = "custom"


class SerializationMode(Enum):
    """Serialization mode enumeration"""
    COMPACT = "compact"
    PRETTY = "pretty"
    COMPRESSED = "compressed"
    CHUNKED = "chunked"
    STREAMING = "streaming"


class SerializerPriority(Enum):
    """Serializer priority enumeration"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class SerializerConfig:
    """Serializer configuration"""
    name: str
    serializer_type: SerializerType
    serialization_mode: SerializationMode = SerializationMode.COMPACT
    priority: SerializerPriority = SerializerPriority.NORMAL
    enabled: bool = True
    timeout: Optional[float] = None
    retry_count: int = 0
    compression_level: int = 6
    encoding: str = "utf-8"
    custom_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SerializationContext:
    """Serialization context"""
    serializer_name: str
    serialization_mode: SerializationMode
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    input_size: int = 0
    output_size: int = 0
    compression_ratio: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseSerializer(ABC, Generic[T, R]):
    """Base serializer with advanced features"""
    
    def __init__(self, config: SerializerConfig):
        self.config = config
        self._enabled = config.enabled
        self._serialization_count = 0
        self._deserialization_count = 0
        self._success_count = 0
        self._error_count = 0
        self._total_duration = 0.0
        self._total_input_size = 0
        self._total_output_size = 0
        self._callbacks: List[Callable] = []
        self._error_handlers: List[Callable] = []
        self._lock = asyncio.Lock()
    
    @abstractmethod
    async def _serialize(self, data: T, context: SerializationContext) -> R:
        """Serialize data (override in subclasses)"""
        pass
    
    @abstractmethod
    async def _deserialize(self, data: R, context: SerializationContext) -> T:
        """Deserialize data (override in subclasses)"""
        pass
    
    async def serialize(self, data: T) -> R:
        """Serialize data with context"""
        if not self._enabled:
            raise RuntimeError(f"Serializer '{self.config.name}' is disabled")
        
        context = SerializationContext(
            serializer_name=self.config.name,
            serialization_mode=self.config.serialization_mode,
            start_time=datetime.utcnow(),
            input_size=self._get_data_size(data)
        )
        
        try:
            # Pre-serialization hook
            await self._pre_serialize(context)
            
            # Serialize data
            result = await asyncio.wait_for(
                self._serialize(data, context),
                timeout=self.config.timeout
            )
            
            context.output_size = self._get_data_size(result)
            context.compression_ratio = (
                context.input_size / context.output_size
                if context.output_size > 0 else 1
            )
            
            # Post-serialization hook
            await self._post_serialize(context)
            
            return result
            
        except Exception as e:
            await self._handle_error(context, e)
            raise
    
    async def deserialize(self, data: R) -> T:
        """Deserialize data with context"""
        if not self._enabled:
            raise RuntimeError(f"Serializer '{self.config.name}' is disabled")
        
        context = SerializationContext(
            serializer_name=self.config.name,
            serialization_mode=self.config.serialization_mode,
            start_time=datetime.utcnow(),
            input_size=self._get_data_size(data)
        )
        
        try:
            # Pre-deserialization hook
            await self._pre_deserialize(context)
            
            # Deserialize data
            result = await asyncio.wait_for(
                self._deserialize(data, context),
                timeout=self.config.timeout
            )
            
            context.output_size = self._get_data_size(result)
            
            # Post-deserialization hook
            await self._post_deserialize(context)
            
            return result
            
        except Exception as e:
            await self._handle_error(context, e)
            raise
    
    async def serialize_to_file(self, data: T, file_path: str) -> None:
        """Serialize data to file"""
        if not self._enabled:
            raise RuntimeError(f"Serializer '{self.config.name}' is disabled")
        
        serialized_data = await self.serialize(data)
        
        # Write to file
        if isinstance(serialized_data, (str, bytes)):
            mode = 'wb' if isinstance(serialized_data, bytes) else 'w'
            encoding = None if isinstance(serialized_data, bytes) else self.config.encoding
            
            with open(file_path, mode, encoding=encoding) as f:
                f.write(serialized_data)
        else:
            # For other types, convert to string first
            with open(file_path, 'w', encoding=self.config.encoding) as f:
                f.write(str(serialized_data))
    
    async def deserialize_from_file(self, file_path: str) -> T:
        """Deserialize data from file"""
        if not self._enabled:
            raise RuntimeError(f"Serializer '{self.config.name}' is disabled")
        
        # Read from file
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding=self.config.encoding) as f:
                data = f.read()
        
        return await self.deserialize(data)
    
    async def serialize_to_stream(self, data: T, stream: Union[BinaryIO, TextIO]) -> None:
        """Serialize data to stream"""
        if not self._enabled:
            raise RuntimeError(f"Serializer '{self.config.name}' is disabled")
        
        serialized_data = await self.serialize(data)
        
        if isinstance(stream, BinaryIO):
            if isinstance(serialized_data, str):
                serialized_data = serialized_data.encode(self.config.encoding)
            stream.write(serialized_data)
        else:
            if isinstance(serialized_data, bytes):
                serialized_data = serialized_data.decode(self.config.encoding)
            stream.write(serialized_data)
    
    async def deserialize_from_stream(self, stream: Union[BinaryIO, TextIO]) -> T:
        """Deserialize data from stream"""
        if not self._enabled:
            raise RuntimeError(f"Serializer '{self.config.name}' is disabled")
        
        data = stream.read()
        
        if isinstance(stream, BinaryIO) and isinstance(data, bytes):
            # Try to decode as text first
            try:
                data = data.decode(self.config.encoding)
            except UnicodeDecodeError:
                # Keep as bytes if decoding fails
                pass
        
        return await self.deserialize(data)
    
    def serialize_sync(self, data: T) -> R:
        """Serialize data synchronously"""
        if not self._enabled:
            raise RuntimeError(f"Serializer '{self.config.name}' is disabled")
        
        # Run async serialize in sync context
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.serialize(data))
    
    def deserialize_sync(self, data: R) -> T:
        """Deserialize data synchronously"""
        if not self._enabled:
            raise RuntimeError(f"Serializer '{self.config.name}' is disabled")
        
        # Run async deserialize in sync context
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.deserialize(data))
    
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
    
    async def _pre_serialize(self, context: SerializationContext) -> None:
        """Pre-serialization hook (override in subclasses)"""
        pass
    
    async def _post_serialize(self, context: SerializationContext) -> None:
        """Post-serialization hook (override in subclasses)"""
        context.end_time = datetime.utcnow()
        context.duration = (context.end_time - context.start_time).total_seconds()
        
        # Update metrics
        self._update_metrics(True, context)
        
        # Call callbacks
        for callback in self._callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(context, "serialize")
                else:
                    callback(context, "serialize")
            except Exception as e:
                logger.error(f"Error in callback: {e}")
    
    async def _pre_deserialize(self, context: SerializationContext) -> None:
        """Pre-deserialization hook (override in subclasses)"""
        pass
    
    async def _post_deserialize(self, context: SerializationContext) -> None:
        """Post-deserialization hook (override in subclasses)"""
        context.end_time = datetime.utcnow()
        context.duration = (context.end_time - context.start_time).total_seconds()
        
        # Update metrics
        self._update_metrics(True, context)
        
        # Call callbacks
        for callback in self._callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(context, "deserialize")
                else:
                    callback(context, "deserialize")
            except Exception as e:
                logger.error(f"Error in callback: {e}")
    
    async def _handle_error(self, context: SerializationContext, error: Exception) -> None:
        """Handle serialization errors"""
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
    
    def _update_metrics(self, success: bool, context: SerializationContext) -> None:
        """Update serializer metrics"""
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
        """Enable serializer"""
        self._enabled = True
    
    def disable(self) -> None:
        """Disable serializer"""
        self._enabled = False
    
    def is_enabled(self) -> bool:
        """Check if serializer is enabled"""
        return self._enabled
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get serializer metrics"""
        avg_duration = (
            self._total_duration / (self._serialization_count + self._deserialization_count)
            if (self._serialization_count + self._deserialization_count) > 0 else 0
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
            "type": self.config.serializer_type.value,
            "mode": self.config.serialization_mode.value,
            "priority": self.config.priority.value,
            "enabled": self._enabled,
            "serialization_count": self._serialization_count,
            "deserialization_count": self._deserialization_count,
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
        """Reset serializer metrics"""
        self._serialization_count = 0
        self._deserialization_count = 0
        self._success_count = 0
        self._error_count = 0
        self._total_duration = 0.0
        self._total_input_size = 0
        self._total_output_size = 0
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.config.name}', enabled={self._enabled})"


class SerializerChain:
    """Chain of serializers with priority ordering"""
    
    def __init__(self):
        self._serializers: List[BaseSerializer] = []
        self._lock = asyncio.Lock()
    
    def add_serializer(self, serializer: BaseSerializer) -> None:
        """Add serializer to chain"""
        self._serializers.append(serializer)
        # Sort by priority (higher priority first)
        self._serializers.sort(
            key=lambda s: s.config.priority.value,
            reverse=True
        )
    
    def remove_serializer(self, name: str) -> None:
        """Remove serializer from chain"""
        self._serializers = [
            s for s in self._serializers
            if s.config.name != name
        ]
    
    async def serialize(self, data: Any) -> Any:
        """Serialize data through serializer chain"""
        result = data
        
        for serializer in self._serializers:
            if serializer.is_enabled():
                try:
                    result = await serializer.serialize(result)
                except Exception as e:
                    logger.error(f"Error in serializer '{serializer.config.name}': {e}")
                    # Continue to next serializer or re-raise based on configuration
                    raise
        
        return result
    
    async def deserialize(self, data: Any) -> Any:
        """Deserialize data through serializer chain"""
        result = data
        
        # Reverse order for deserialization
        for serializer in reversed(self._serializers):
            if serializer.is_enabled():
                try:
                    result = await serializer.deserialize(result)
                except Exception as e:
                    logger.error(f"Error in serializer '{serializer.config.name}': {e}")
                    # Continue to next serializer or re-raise based on configuration
                    raise
        
        return result
    
    def get_serializers(self) -> List[BaseSerializer]:
        """Get all serializers"""
        return self._serializers.copy()
    
    def get_serializers_by_type(self, serializer_type: SerializerType) -> List[BaseSerializer]:
        """Get serializers by type"""
        return [
            s for s in self._serializers
            if s.config.serializer_type == serializer_type
        ]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics for all serializers"""
        return {
            serializer.config.name: serializer.get_metrics()
            for serializer in self._serializers
        }


class SerializerRegistry:
    """Registry for managing serializers"""
    
    def __init__(self):
        self._serializers: Dict[str, BaseSerializer] = {}
        self._chains: Dict[str, SerializerChain] = {}
        self._lock = asyncio.Lock()
    
    async def register(self, serializer: BaseSerializer) -> None:
        """Register serializer"""
        async with self._lock:
            self._serializers[serializer.config.name] = serializer
            logger.info(f"Registered serializer: {serializer.config.name}")
    
    async def unregister(self, name: str) -> None:
        """Unregister serializer"""
        async with self._lock:
            if name in self._serializers:
                del self._serializers[name]
                logger.info(f"Unregistered serializer: {name}")
    
    def get(self, name: str) -> Optional[BaseSerializer]:
        """Get serializer by name"""
        return self._serializers.get(name)
    
    def get_by_type(self, serializer_type: SerializerType) -> List[BaseSerializer]:
        """Get serializers by type"""
        return [
            serializer for serializer in self._serializers.values()
            if serializer.config.serializer_type == serializer_type
        ]
    
    def create_chain(self, name: str, serializer_names: List[str]) -> SerializerChain:
        """Create serializer chain"""
        chain = SerializerChain()
        
        for serializer_name in serializer_names:
            serializer = self.get(serializer_name)
            if serializer:
                chain.add_serializer(serializer)
        
        self._chains[name] = chain
        return chain
    
    def get_chain(self, name: str) -> Optional[SerializerChain]:
        """Get serializer chain"""
        return self._chains.get(name)
    
    def list_all(self) -> List[BaseSerializer]:
        """List all serializers"""
        return list(self._serializers.values())
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics for all serializers"""
        return {
            name: serializer.get_metrics()
            for name, serializer in self._serializers.items()
        }


# Global serializer registry
serializer_registry = SerializerRegistry()


# Convenience functions
async def register_serializer(serializer: BaseSerializer):
    """Register serializer"""
    await serializer_registry.register(serializer)


def get_serializer(name: str) -> Optional[BaseSerializer]:
    """Get serializer by name"""
    return serializer_registry.get(name)


def create_serializer_chain(name: str, serializer_names: List[str]) -> SerializerChain:
    """Create serializer chain"""
    return serializer_registry.create_chain(name, serializer_names)


# Serializer factory functions
def create_serializer(serializer_type: SerializerType, name: str, **kwargs) -> BaseSerializer:
    """Create serializer by type"""
    config = SerializerConfig(
        name=name,
        serializer_type=serializer_type,
        **kwargs
    )
    
    # This would be implemented with specific serializer classes
    # For now, return a placeholder
    raise NotImplementedError(f"Serializer type {serializer_type} not implemented yet")


# Common serializer combinations
def json_serialization_chain(name: str = "json_serialization") -> SerializerChain:
    """Create JSON serialization serializer chain"""
    return create_serializer_chain(name, [
        "json_serializer",
        "compression_serializer"
    ])


def binary_serialization_chain(name: str = "binary_serialization") -> SerializerChain:
    """Create binary serialization serializer chain"""
    return create_serializer_chain(name, [
        "binary_serializer",
        "compression_serializer",
        "encryption_serializer"
    ])


def data_serialization_chain(name: str = "data_serialization") -> SerializerChain:
    """Create data serialization serializer chain"""
    return create_serializer_chain(name, [
        "format_serializer",
        "compression_serializer",
        "validation_serializer"
    ])


def api_serialization_chain(name: str = "api_serialization") -> SerializerChain:
    """Create API serialization serializer chain"""
    return create_serializer_chain(name, [
        "json_serializer",
        "compression_serializer",
        "security_serializer"
    ])





















