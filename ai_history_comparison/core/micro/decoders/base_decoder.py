"""
Base Decoder Implementation

Ultra-specialized base decoder with advanced features for
data decoding, decompression, and format conversion.
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


class DecoderType(Enum):
    """Decoder type enumeration"""
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


class DecodingMode(Enum):
    """Decoding mode enumeration"""
    FAST = "fast"
    BALANCED = "balanced"
    HIGH_QUALITY = "high_quality"
    LOSSLESS = "lossless"
    LOSSY = "lossy"


class DecoderPriority(Enum):
    """Decoder priority enumeration"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class DecoderConfig:
    """Decoder configuration"""
    name: str
    decoder_type: DecoderType
    decoding_mode: DecodingMode = DecodingMode.BALANCED
    priority: DecoderPriority = DecoderPriority.NORMAL
    enabled: bool = True
    timeout: Optional[float] = None
    retry_count: int = 0
    quality: float = 0.8
    decompression_level: int = 6
    encoding: str = "utf-8"
    custom_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecodingContext:
    """Decoding context"""
    decoder_name: str
    decoding_mode: DecodingMode
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    input_size: int = 0
    output_size: int = 0
    decompression_ratio: float = 1.0
    quality_score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseDecoder(ABC, Generic[T, R]):
    """Base decoder with advanced features"""
    
    def __init__(self, config: DecoderConfig):
        self.config = config
        self._enabled = config.enabled
        self._decoding_count = 0
        self._encoding_count = 0
        self._success_count = 0
        self._error_count = 0
        self._total_duration = 0.0
        self._total_input_size = 0
        self._total_output_size = 0
        self._callbacks: List[Callable] = []
        self._error_handlers: List[Callable] = []
        self._lock = asyncio.Lock()
    
    @abstractmethod
    async def _decode(self, data: T, context: DecodingContext) -> R:
        """Decode data (override in subclasses)"""
        pass
    
    @abstractmethod
    async def _encode(self, data: R, context: DecodingContext) -> T:
        """Encode data (override in subclasses)"""
        pass
    
    async def decode(self, data: T) -> R:
        """Decode data with context"""
        if not self._enabled:
            raise RuntimeError(f"Decoder '{self.config.name}' is disabled")
        
        context = DecodingContext(
            decoder_name=self.config.name,
            decoding_mode=self.config.decoding_mode,
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
            context.decompression_ratio = (
                context.output_size / context.input_size
                if context.input_size > 0 else 1
            )
            
            # Post-decoding hook
            await self._post_decode(context)
            
            return result
            
        except Exception as e:
            await self._handle_error(context, e)
            raise
    
    async def encode(self, data: R) -> T:
        """Encode data with context"""
        if not self._enabled:
            raise RuntimeError(f"Decoder '{self.config.name}' is disabled")
        
        context = DecodingContext(
            decoder_name=self.config.name,
            decoding_mode=self.config.decoding_mode,
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
            
            # Post-encoding hook
            await self._post_encode(context)
            
            return result
            
        except Exception as e:
            await self._handle_error(context, e)
            raise
    
    async def decode_from_file(self, file_path: str) -> R:
        """Decode data from file"""
        if not self._enabled:
            raise RuntimeError(f"Decoder '{self.config.name}' is disabled")
        
        # Read from file
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding=self.config.encoding) as f:
                data = f.read()
        
        return await self.decode(data)
    
    async def encode_to_file(self, data: R, file_path: str) -> None:
        """Encode data to file"""
        if not self._enabled:
            raise RuntimeError(f"Decoder '{self.config.name}' is disabled")
        
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
    
    async def decode_from_stream(self, stream: Union[BinaryIO, TextIO]) -> R:
        """Decode data from stream"""
        if not self._enabled:
            raise RuntimeError(f"Decoder '{self.config.name}' is disabled")
        
        data = stream.read()
        
        if isinstance(stream, BinaryIO) and isinstance(data, bytes):
            # Try to decode as text first
            try:
                data = data.decode(self.config.encoding)
            except UnicodeDecodeError:
                # Keep as bytes if decoding fails
                pass
        
        return await self.decode(data)
    
    async def encode_to_stream(self, data: R, stream: Union[BinaryIO, TextIO]) -> None:
        """Encode data to stream"""
        if not self._enabled:
            raise RuntimeError(f"Decoder '{self.config.name}' is disabled")
        
        encoded_data = await self.encode(data)
        
        if isinstance(stream, BinaryIO):
            if isinstance(encoded_data, str):
                encoded_data = encoded_data.encode(self.config.encoding)
            stream.write(encoded_data)
        else:
            if isinstance(encoded_data, bytes):
                encoded_data = encoded_data.decode(self.config.encoding)
            stream.write(encoded_data)
    
    def decode_sync(self, data: T) -> R:
        """Decode data synchronously"""
        if not self._enabled:
            raise RuntimeError(f"Decoder '{self.config.name}' is disabled")
        
        # Run async decode in sync context
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.decode(data))
    
    def encode_sync(self, data: R) -> T:
        """Encode data synchronously"""
        if not self._enabled:
            raise RuntimeError(f"Decoder '{self.config.name}' is disabled")
        
        # Run async encode in sync context
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.encode(data))
    
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
    
    async def _pre_decode(self, context: DecodingContext) -> None:
        """Pre-decoding hook (override in subclasses)"""
        pass
    
    async def _post_decode(self, context: DecodingContext) -> None:
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
    
    async def _pre_encode(self, context: DecodingContext) -> None:
        """Pre-encoding hook (override in subclasses)"""
        pass
    
    async def _post_encode(self, context: DecodingContext) -> None:
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
    
    async def _handle_error(self, context: DecodingContext, error: Exception) -> None:
        """Handle decoding errors"""
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
    
    def _update_metrics(self, success: bool, context: DecodingContext) -> None:
        """Update decoder metrics"""
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
        """Enable decoder"""
        self._enabled = True
    
    def disable(self) -> None:
        """Disable decoder"""
        self._enabled = False
    
    def is_enabled(self) -> bool:
        """Check if decoder is enabled"""
        return self._enabled
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get decoder metrics"""
        avg_duration = (
            self._total_duration / (self._decoding_count + self._encoding_count)
            if (self._decoding_count + self._encoding_count) > 0 else 0
        )
        
        decompression_ratio = (
            self._total_output_size / self._total_input_size
            if self._total_input_size > 0 else 1
        )
        
        throughput = (
            self._total_input_size / self._total_duration
            if self._total_duration > 0 else 0
        )
        
        return {
            "name": self.config.name,
            "type": self.config.decoder_type.value,
            "mode": self.config.decoding_mode.value,
            "priority": self.config.priority.value,
            "enabled": self._enabled,
            "decoding_count": self._decoding_count,
            "encoding_count": self._encoding_count,
            "success_count": self._success_count,
            "error_count": self._error_count,
            "total_duration": self._total_duration,
            "average_duration": avg_duration,
            "total_input_size": self._total_input_size,
            "total_output_size": self._total_output_size,
            "decompression_ratio": decompression_ratio,
            "throughput": throughput
        }
    
    def reset_metrics(self) -> None:
        """Reset decoder metrics"""
        self._decoding_count = 0
        self._encoding_count = 0
        self._success_count = 0
        self._error_count = 0
        self._total_duration = 0.0
        self._total_input_size = 0
        self._total_output_size = 0
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.config.name}', enabled={self._enabled})"


class DecoderChain:
    """Chain of decoders with priority ordering"""
    
    def __init__(self):
        self._decoders: List[BaseDecoder] = []
        self._lock = asyncio.Lock()
    
    def add_decoder(self, decoder: BaseDecoder) -> None:
        """Add decoder to chain"""
        self._decoders.append(decoder)
        # Sort by priority (higher priority first)
        self._decoders.sort(
            key=lambda d: d.config.priority.value,
            reverse=True
        )
    
    def remove_decoder(self, name: str) -> None:
        """Remove decoder from chain"""
        self._decoders = [
            d for d in self._decoders
            if d.config.name != name
        ]
    
    async def decode(self, data: Any) -> Any:
        """Decode data through decoder chain"""
        result = data
        
        for decoder in self._decoders:
            if decoder.is_enabled():
                try:
                    result = await decoder.decode(result)
                except Exception as e:
                    logger.error(f"Error in decoder '{decoder.config.name}': {e}")
                    # Continue to next decoder or re-raise based on configuration
                    raise
        
        return result
    
    async def encode(self, data: Any) -> Any:
        """Encode data through decoder chain"""
        result = data
        
        # Reverse order for encoding
        for decoder in reversed(self._decoders):
            if decoder.is_enabled():
                try:
                    result = await decoder.encode(result)
                except Exception as e:
                    logger.error(f"Error in decoder '{decoder.config.name}': {e}")
                    # Continue to next decoder or re-raise based on configuration
                    raise
        
        return result
    
    def get_decoders(self) -> List[BaseDecoder]:
        """Get all decoders"""
        return self._decoders.copy()
    
    def get_decoders_by_type(self, decoder_type: DecoderType) -> List[BaseDecoder]:
        """Get decoders by type"""
        return [
            d for d in self._decoders
            if d.config.decoder_type == decoder_type
        ]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics for all decoders"""
        return {
            decoder.config.name: decoder.get_metrics()
            for decoder in self._decoders
        }


class DecoderRegistry:
    """Registry for managing decoders"""
    
    def __init__(self):
        self._decoders: Dict[str, BaseDecoder] = {}
        self._chains: Dict[str, DecoderChain] = {}
        self._lock = asyncio.Lock()
    
    async def register(self, decoder: BaseDecoder) -> None:
        """Register decoder"""
        async with self._lock:
            self._decoders[decoder.config.name] = decoder
            logger.info(f"Registered decoder: {decoder.config.name}")
    
    async def unregister(self, name: str) -> None:
        """Unregister decoder"""
        async with self._lock:
            if name in self._decoders:
                del self._decoders[name]
                logger.info(f"Unregistered decoder: {name}")
    
    def get(self, name: str) -> Optional[BaseDecoder]:
        """Get decoder by name"""
        return self._decoders.get(name)
    
    def get_by_type(self, decoder_type: DecoderType) -> List[BaseDecoder]:
        """Get decoders by type"""
        return [
            decoder for decoder in self._decoders.values()
            if decoder.config.decoder_type == decoder_type
        ]
    
    def create_chain(self, name: str, decoder_names: List[str]) -> DecoderChain:
        """Create decoder chain"""
        chain = DecoderChain()
        
        for decoder_name in decoder_names:
            decoder = self.get(decoder_name)
            if decoder:
                chain.add_decoder(decoder)
        
        self._chains[name] = chain
        return chain
    
    def get_chain(self, name: str) -> Optional[DecoderChain]:
        """Get decoder chain"""
        return self._chains.get(name)
    
    def list_all(self) -> List[BaseDecoder]:
        """List all decoders"""
        return list(self._decoders.values())
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics for all decoders"""
        return {
            name: decoder.get_metrics()
            for name, decoder in self._decoders.items()
        }


# Global decoder registry
decoder_registry = DecoderRegistry()


# Convenience functions
async def register_decoder(decoder: BaseDecoder):
    """Register decoder"""
    await decoder_registry.register(decoder)


def get_decoder(name: str) -> Optional[BaseDecoder]:
    """Get decoder by name"""
    return decoder_registry.get(name)


def create_decoder_chain(name: str, decoder_names: List[str]) -> DecoderChain:
    """Create decoder chain"""
    return decoder_registry.create_chain(name, decoder_names)


# Decoder factory functions
def create_decoder(decoder_type: DecoderType, name: str, **kwargs) -> BaseDecoder:
    """Create decoder by type"""
    config = DecoderConfig(
        name=name,
        decoder_type=decoder_type,
        **kwargs
    )
    
    # This would be implemented with specific decoder classes
    # For now, return a placeholder
    raise NotImplementedError(f"Decoder type {decoder_type} not implemented yet")


# Common decoder combinations
def text_decoding_chain(name: str = "text_decoding") -> DecoderChain:
    """Create text decoding decoder chain"""
    return create_decoder_chain(name, [
        "text_decoder",
        "compression_decoder"
    ])


def binary_decoding_chain(name: str = "binary_decoding") -> DecoderChain:
    """Create binary decoding decoder chain"""
    return create_decoder_chain(name, [
        "binary_decoder",
        "compression_decoder",
        "crypto_decoder"
    ])


def media_decoding_chain(name: str = "media_decoding") -> DecoderChain:
    """Create media decoding decoder chain"""
    return create_decoder_chain(name, [
        "image_decoder",
        "audio_decoder",
        "video_decoder",
        "compression_decoder"
    ])


def ai_decoding_chain(name: str = "ai_decoding") -> DecoderChain:
    """Create AI decoding decoder chain"""
    return create_decoder_chain(name, [
        "ai_decoder",
        "compression_decoder",
        "format_decoder"
    ])





















