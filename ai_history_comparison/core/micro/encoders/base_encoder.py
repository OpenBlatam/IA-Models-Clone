"""
Base Encoder Implementation

Ultra-specialized base encoder with advanced features for
data encoding, compression, and format conversion.
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


class EncoderType(Enum):
    """Encoder type enumeration"""
    TEXT = "text"
    BINARY = "binary"
    COMPRESSION = "compression"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    AI = "ai"
    CRYPTO = "crypto"
    FORMAT = "format"
    CUSTOM = "custom"


class EncodingMode(Enum):
    """Encoding mode enumeration"""
    FAST = "fast"
    BALANCED = "balanced"
    HIGH_QUALITY = "high_quality"
    LOSSLESS = "lossless"
    LOSSY = "lossy"


class EncoderPriority(Enum):
    """Encoder priority enumeration"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class EncoderConfig:
    """Encoder configuration"""
    name: str
    encoder_type: EncoderType
    encoding_mode: EncodingMode = EncodingMode.BALANCED
    priority: EncoderPriority = EncoderPriority.NORMAL
    enabled: bool = True
    timeout: Optional[float] = None
    retry_count: int = 0
    quality: float = 0.8
    compression_level: int = 6
    encoding: str = "utf-8"
    custom_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EncodingContext:
    """Encoding context"""
    encoder_name: str
    encoding_mode: EncodingMode
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    input_size: int = 0
    output_size: int = 0
    compression_ratio: float = 1.0
    quality_score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseEncoder(ABC, Generic[T, R]):
    """Base encoder with advanced features"""
    
    def __init__(self, config: EncoderConfig):
        self.config = config
        self._enabled = config.enabled
        self._encoding_count = 0
        self._decoding_count = 0
        self._success_count = 0
        self._error_count = 0
        self._total_duration = 0.0
        self._total_input_size = 0
        self._total_output_size = 0
        self._callbacks: List[Callable] = []
        self._error_handlers: List[Callable] = []
        self._lock = asyncio.Lock()
    
    @abstractmethod
    async def _encode(self, data: T, context: EncodingContext) -> R:
        """Encode data (override in subclasses)"""
        pass
    
    @abstractmethod
    async def _decode(self, data: R, context: EncodingContext) -> T:
        """Decode data (override in subclasses)"""
        pass
    
    async def encode(self, data: T) -> R:
        """Encode data with context"""
        if not self._enabled:
            raise RuntimeError(f"Encoder '{self.config.name}' is disabled")
        
        context = EncodingContext(
            encoder_name=self.config.name,
            encoding_mode=self.config.encoding_mode,
            start_time=datetime.utcnow(),
            input_size=self._get_data_size(data)
        )
        
        try:
            # Pre-encoding hook
            await self._pre_encode(context)
            
            # Encode data
            result = await asyncio.wait_for(
                self._encode(data, context),
                timeout=self.config.timeout
            )
            
            context.output_size = self._get_data_size(result)
            context.compression_ratio = (
                context.input_size / context.output_size
                if context.output_size > 0 else 1
            )
            
            # Post-encoding hook
            await self._post_encode(context)
            
            return result
            
        except Exception as e:
            await self._handle_error(context, e)
            raise
    
    async def decode(self, data: R) -> T:
        """Decode data with context"""
        if not self._enabled:
            raise RuntimeError(f"Encoder '{self.config.name}' is disabled")
        
        context = EncodingContext(
            encoder_name=self.config.name,
            encoding_mode=self.config.encoding_mode,
            start_time=datetime.utcnow(),
            input_size=self._get_data_size(data)
        )
        
        try:
            # Pre-decoding hook
            await self._pre_decode(context)
            
            # Decode data
            result = await asyncio.wait_for(
                self._decode(data, context),
                timeout=self.config.timeout
            )
            
            context.output_size = self._get_data_size(result)
            
            # Post-decoding hook
            await self._post_decode(context)
            
            return result
            
        except Exception as e:
            await self._handle_error(context, e)
            raise
    
    async def encode_to_file(self, data: T, file_path: str) -> None:
        """Encode data to file"""
        if not self._enabled:
            raise RuntimeError(f"Encoder '{self.config.name}' is disabled")
        
        encoded_data = await self.encode(data)
        
        # Write to file
        if isinstance(encoded_data, (str, bytes)):
            mode = 'wb' if isinstance(encoded_data, bytes) else 'w'
            encoding = None if isinstance(encoded_data, bytes) else self.config.encoding
            
            with open(file_path, mode, encoding=encoding) as f:
                f.write(encoded_data)
        else:
            # For other types, convert to string first
            with open(file_path, 'w', encoding=self.config.encoding) as f:
                f.write(str(encoded_data))
    
    async def decode_from_file(self, file_path: str) -> T:
        """Decode data from file"""
        if not self._enabled:
            raise RuntimeError(f"Encoder '{self.config.name}' is disabled")
        
        # Read from file
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding=self.config.encoding) as f:
                data = f.read()
        
        return await self.decode(data)
    
    async def encode_to_stream(self, data: T, stream: Union[BinaryIO, TextIO]) -> None:
        """Encode data to stream"""
        if not self._enabled:
            raise RuntimeError(f"Encoder '{self.config.name}' is disabled")
        
        encoded_data = await self.encode(data)
        
        if isinstance(stream, BinaryIO):
            if isinstance(encoded_data, str):
                encoded_data = encoded_data.encode(self.config.encoding)
            stream.write(encoded_data)
        else:
            if isinstance(encoded_data, bytes):
                encoded_data = encoded_data.decode(self.config.encoding)
            stream.write(encoded_data)
    
    async def decode_from_stream(self, stream: Union[BinaryIO, TextIO]) -> T:
        """Decode data from stream"""
        if not self._enabled:
            raise RuntimeError(f"Encoder '{self.config.name}' is disabled")
        
        data = stream.read()
        
        if isinstance(stream, BinaryIO) and isinstance(data, bytes):
            # Try to decode as text first
            try:
                data = data.decode(self.config.encoding)
            except UnicodeDecodeError:
                # Keep as bytes if decoding fails
                pass
        
        return await self.decode(data)
    
    def encode_sync(self, data: T) -> R:
        """Encode data synchronously"""
        if not self._enabled:
            raise RuntimeError(f"Encoder '{self.config.name}' is disabled")
        
        # Run async encode in sync context
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.encode(data))
    
    def decode_sync(self, data: R) -> T:
        """Decode data synchronously"""
        if not self._enabled:
            raise RuntimeError(f"Encoder '{self.config.name}' is disabled")
        
        # Run async decode in sync context
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.decode(data))
    
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
    
    async def _pre_encode(self, context: EncodingContext) -> None:
        """Pre-encoding hook (override in subclasses)"""
        pass
    
    async def _post_encode(self, context: EncodingContext) -> None:
        """Post-encoding hook (override in subclasses)"""
        context.end_time = datetime.utcnow()
        context.duration = (context.end_time - context.start_time).total_seconds()
        
        # Update metrics
        self._update_metrics(True, context)
        
        # Call callbacks
        for callback in self._callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(context, "encode")
                else:
                    callback(context, "encode")
            except Exception as e:
                logger.error(f"Error in callback: {e}")
    
    async def _pre_decode(self, context: EncodingContext) -> None:
        """Pre-decoding hook (override in subclasses)"""
        pass
    
    async def _post_decode(self, context: EncodingContext) -> None:
        """Post-decoding hook (override in subclasses)"""
        context.end_time = datetime.utcnow()
        context.duration = (context.end_time - context.start_time).total_seconds()
        
        # Update metrics
        self._update_metrics(True, context)
        
        # Call callbacks
        for callback in self._callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(context, "decode")
                else:
                    callback(context, "decode")
            except Exception as e:
                logger.error(f"Error in callback: {e}")
    
    async def _handle_error(self, context: EncodingContext, error: Exception) -> None:
        """Handle encoding errors"""
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
    
    def _update_metrics(self, success: bool, context: EncodingContext) -> None:
        """Update encoder metrics"""
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
        """Enable encoder"""
        self._enabled = True
    
    def disable(self) -> None:
        """Disable encoder"""
        self._enabled = False
    
    def is_enabled(self) -> bool:
        """Check if encoder is enabled"""
        return self._enabled
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get encoder metrics"""
        avg_duration = (
            self._total_duration / (self._encoding_count + self._decoding_count)
            if (self._encoding_count + self._decoding_count) > 0 else 0
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
            "type": self.config.encoder_type.value,
            "mode": self.config.encoding_mode.value,
            "priority": self.config.priority.value,
            "enabled": self._enabled,
            "encoding_count": self._encoding_count,
            "decoding_count": self._decoding_count,
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
        """Reset encoder metrics"""
        self._encoding_count = 0
        self._decoding_count = 0
        self._success_count = 0
        self._error_count = 0
        self._total_duration = 0.0
        self._total_input_size = 0
        self._total_output_size = 0
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.config.name}', enabled={self._enabled})"


class EncoderChain:
    """Chain of encoders with priority ordering"""
    
    def __init__(self):
        self._encoders: List[BaseEncoder] = []
        self._lock = asyncio.Lock()
    
    def add_encoder(self, encoder: BaseEncoder) -> None:
        """Add encoder to chain"""
        self._encoders.append(encoder)
        # Sort by priority (higher priority first)
        self._encoders.sort(
            key=lambda e: e.config.priority.value,
            reverse=True
        )
    
    def remove_encoder(self, name: str) -> None:
        """Remove encoder from chain"""
        self._encoders = [
            e for e in self._encoders
            if e.config.name != name
        ]
    
    async def encode(self, data: Any) -> Any:
        """Encode data through encoder chain"""
        result = data
        
        for encoder in self._encoders:
            if encoder.is_enabled():
                try:
                    result = await encoder.encode(result)
                except Exception as e:
                    logger.error(f"Error in encoder '{encoder.config.name}': {e}")
                    # Continue to next encoder or re-raise based on configuration
                    raise
        
        return result
    
    async def decode(self, data: Any) -> Any:
        """Decode data through encoder chain"""
        result = data
        
        # Reverse order for decoding
        for encoder in reversed(self._encoders):
            if encoder.is_enabled():
                try:
                    result = await encoder.decode(result)
                except Exception as e:
                    logger.error(f"Error in encoder '{encoder.config.name}': {e}")
                    # Continue to next encoder or re-raise based on configuration
                    raise
        
        return result
    
    def get_encoders(self) -> List[BaseEncoder]:
        """Get all encoders"""
        return self._encoders.copy()
    
    def get_encoders_by_type(self, encoder_type: EncoderType) -> List[BaseEncoder]:
        """Get encoders by type"""
        return [
            e for e in self._encoders
            if e.config.encoder_type == encoder_type
        ]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics for all encoders"""
        return {
            encoder.config.name: encoder.get_metrics()
            for encoder in self._encoders
        }


class EncoderRegistry:
    """Registry for managing encoders"""
    
    def __init__(self):
        self._encoders: Dict[str, BaseEncoder] = {}
        self._chains: Dict[str, EncoderChain] = {}
        self._lock = asyncio.Lock()
    
    async def register(self, encoder: BaseEncoder) -> None:
        """Register encoder"""
        async with self._lock:
            self._encoders[encoder.config.name] = encoder
            logger.info(f"Registered encoder: {encoder.config.name}")
    
    async def unregister(self, name: str) -> None:
        """Unregister encoder"""
        async with self._lock:
            if name in self._encoders:
                del self._encoders[name]
                logger.info(f"Unregistered encoder: {name}")
    
    def get(self, name: str) -> Optional[BaseEncoder]:
        """Get encoder by name"""
        return self._encoders.get(name)
    
    def get_by_type(self, encoder_type: EncoderType) -> List[BaseEncoder]:
        """Get encoders by type"""
        return [
            encoder for encoder in self._encoders.values()
            if encoder.config.encoder_type == encoder_type
        ]
    
    def create_chain(self, name: str, encoder_names: List[str]) -> EncoderChain:
        """Create encoder chain"""
        chain = EncoderChain()
        
        for encoder_name in encoder_names:
            encoder = self.get(encoder_name)
            if encoder:
                chain.add_encoder(encoder)
        
        self._chains[name] = chain
        return chain
    
    def get_chain(self, name: str) -> Optional[EncoderChain]:
        """Get encoder chain"""
        return self._chains.get(name)
    
    def list_all(self) -> List[BaseEncoder]:
        """List all encoders"""
        return list(self._encoders.values())
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics for all encoders"""
        return {
            name: encoder.get_metrics()
            for name, encoder in self._encoders.items()
        }


# Global encoder registry
encoder_registry = EncoderRegistry()


# Convenience functions
async def register_encoder(encoder: BaseEncoder):
    """Register encoder"""
    await encoder_registry.register(encoder)


def get_encoder(name: str) -> Optional[BaseEncoder]:
    """Get encoder by name"""
    return encoder_registry.get(name)


def create_encoder_chain(name: str, encoder_names: List[str]) -> EncoderChain:
    """Create encoder chain"""
    return encoder_registry.create_chain(name, encoder_names)


# Encoder factory functions
def create_encoder(encoder_type: EncoderType, name: str, **kwargs) -> BaseEncoder:
    """Create encoder by type"""
    config = EncoderConfig(
        name=name,
        encoder_type=encoder_type,
        **kwargs
    )
    
    # This would be implemented with specific encoder classes
    # For now, return a placeholder
    raise NotImplementedError(f"Encoder type {encoder_type} not implemented yet")


# Common encoder combinations
def text_encoding_chain(name: str = "text_encoding") -> EncoderChain:
    """Create text encoding encoder chain"""
    return create_encoder_chain(name, [
        "text_encoder",
        "compression_encoder"
    ])


def binary_encoding_chain(name: str = "binary_encoding") -> EncoderChain:
    """Create binary encoding encoder chain"""
    return create_encoder_chain(name, [
        "binary_encoder",
        "compression_encoder",
        "crypto_encoder"
    ])


def media_encoding_chain(name: str = "media_encoding") -> EncoderChain:
    """Create media encoding encoder chain"""
    return create_encoder_chain(name, [
        "image_encoder",
        "audio_encoder",
        "video_encoder",
        "compression_encoder"
    ])


def ai_encoding_chain(name: str = "ai_encoding") -> EncoderChain:
    """Create AI encoding encoder chain"""
    return create_encoder_chain(name, [
        "ai_encoder",
        "compression_encoder",
        "format_encoder"
    ])





















