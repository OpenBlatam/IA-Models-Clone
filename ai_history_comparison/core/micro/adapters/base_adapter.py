"""
Base Adapter Implementation

Ultra-specialized base adapter with advanced features for
interface transformation and protocol adaptation.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, TypeVar, Generic, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import weakref
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


class AdapterType(Enum):
    """Adapter type enumeration"""
    DATA = "data"
    API = "api"
    PROTOCOL = "protocol"
    FORMAT = "format"
    SERVICE = "service"
    DATABASE = "database"
    CACHE = "cache"
    QUEUE = "queue"
    FILE = "file"
    AI = "ai"


class AdapterState(Enum):
    """Adapter state enumeration"""
    CREATED = "created"
    INITIALIZING = "initializing"
    INITIALIZED = "initialized"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTING = "disconnecting"
    DISCONNECTED = "disconnected"
    ERROR = "error"


@dataclass
class AdapterConfig:
    """Adapter configuration"""
    name: str
    adapter_type: AdapterType
    enabled: bool = True
    timeout: float = 30.0
    retry_count: int = 3
    retry_delay: float = 1.0
    max_connections: int = 10
    connection_pool_size: int = 5
    health_check_interval: float = 60.0
    metrics_enabled: bool = True
    custom_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AdapterMetrics:
    """Adapter performance metrics"""
    requests_total: int = 0
    requests_successful: int = 0
    requests_failed: int = 0
    response_time_avg: float = 0.0
    response_time_min: float = float('inf')
    response_time_max: float = 0.0
    last_request_time: Optional[datetime] = None
    last_error_time: Optional[datetime] = None
    last_error_message: Optional[str] = None
    connection_count: int = 0
    active_connections: int = 0


class BaseAdapter(ABC, Generic[T, R]):
    """Base adapter with advanced features"""
    
    def __init__(self, config: AdapterConfig):
        self.config = config
        self.state = AdapterState.CREATED
        self.metrics = AdapterMetrics()
        self._connections: List[Any] = []
        self._connection_pool: List[Any] = []
        self._lock = asyncio.Lock()
        self._health_check_task: Optional[asyncio.Task] = None
        self._running = False
        self._callbacks: List[Callable] = []
        self._error_handlers: List[Callable] = []
    
    @abstractmethod
    async def _initialize(self) -> None:
        """Initialize adapter (override in subclasses)"""
        pass
    
    @abstractmethod
    async def _connect(self) -> Any:
        """Connect to target system (override in subclasses)"""
        pass
    
    @abstractmethod
    async def _disconnect(self, connection: Any) -> None:
        """Disconnect from target system (override in subclasses)"""
        pass
    
    @abstractmethod
    async def _adapt(self, data: T, connection: Any) -> R:
        """Adapt data (override in subclasses)"""
        pass
    
    @abstractmethod
    async def _validate_connection(self, connection: Any) -> bool:
        """Validate connection health (override in subclasses)"""
        pass
    
    async def initialize(self) -> None:
        """Initialize adapter"""
        if self.state != AdapterState.CREATED:
            return
        
        self.state = AdapterState.INITIALIZING
        try:
            await self._initialize()
            self.state = AdapterState.INITIALIZED
            logger.info(f"Adapter '{self.config.name}' initialized")
        except Exception as e:
            self.state = AdapterState.ERROR
            logger.error(f"Failed to initialize adapter '{self.config.name}': {e}")
            raise
    
    async def connect(self) -> Any:
        """Connect to target system"""
        if self.state not in [AdapterState.INITIALIZED, AdapterState.DISCONNECTED]:
            raise RuntimeError(f"Adapter '{self.config.name}' not ready for connection")
        
        self.state = AdapterState.CONNECTING
        try:
            connection = await self._connect()
            self._connections.append(connection)
            self.metrics.connection_count += 1
            self.metrics.active_connections = len(self._connections)
            self.state = AdapterState.CONNECTED
            logger.info(f"Adapter '{self.config.name}' connected")
            return connection
        except Exception as e:
            self.state = AdapterState.ERROR
            logger.error(f"Failed to connect adapter '{self.config.name}': {e}")
            raise
    
    async def disconnect(self, connection: Any) -> None:
        """Disconnect from target system"""
        if connection not in self._connections:
            return
        
        self.state = AdapterState.DISCONNECTING
        try:
            await self._disconnect(connection)
            self._connections.remove(connection)
            self.metrics.active_connections = len(self._connections)
            self.state = AdapterState.DISCONNECTED
            logger.info(f"Adapter '{self.config.name}' disconnected")
        except Exception as e:
            self.state = AdapterState.ERROR
            logger.error(f"Failed to disconnect adapter '{self.config.name}': {e}")
            raise
    
    async def adapt(self, data: T) -> R:
        """Adapt data using available connection"""
        if not self._connections:
            await self.connect()
        
        connection = self._connections[0]  # Use first available connection
        start_time = datetime.utcnow()
        
        try:
            result = await asyncio.wait_for(
                self._adapt(data, connection),
                timeout=self.config.timeout
            )
            
            # Update metrics
            self._update_metrics(True, start_time)
            return result
            
        except Exception as e:
            self._update_metrics(False, start_time, str(e))
            await self._handle_error(e)
            raise
    
    async def adapt_batch(self, data_list: List[T]) -> List[R]:
        """Adapt multiple data items"""
        if not self._connections:
            await self.connect()
        
        # Use connection pool for batch processing
        tasks = []
        for i, data in enumerate(data_list):
            connection = self._connections[i % len(self._connections)]
            task = asyncio.create_task(self._adapt(data, connection))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Separate successful results from exceptions
        successful_results = []
        errors = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                errors.append((i, result))
            else:
                successful_results.append(result)
        
        if errors:
            logger.warning(f"Batch adaptation completed with {len(errors)} errors")
        
        return successful_results
    
    async def start(self) -> None:
        """Start adapter"""
        if not self._running:
            self._running = True
            await self.initialize()
            await self.connect()
            
            # Start health check
            if self.config.health_check_interval > 0:
                self._health_check_task = asyncio.create_task(self._health_check_loop())
            
            logger.info(f"Adapter '{self.config.name}' started")
    
    async def stop(self) -> None:
        """Stop adapter"""
        self._running = False
        
        # Stop health check
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        # Disconnect all connections
        for connection in self._connections.copy():
            await self.disconnect(connection)
        
        logger.info(f"Adapter '{self.config.name}' stopped")
    
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
    
    async def _handle_error(self, error: Exception) -> None:
        """Handle errors"""
        self.metrics.last_error_time = datetime.utcnow()
        self.metrics.last_error_message = str(error)
        
        # Call error handlers
        for handler in self._error_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(error)
                else:
                    handler(error)
            except Exception as e:
                logger.error(f"Error in error handler: {e}")
    
    def _update_metrics(self, success: bool, start_time: datetime, error_message: str = None) -> None:
        """Update performance metrics"""
        response_time = (datetime.utcnow() - start_time).total_seconds()
        
        self.metrics.requests_total += 1
        if success:
            self.metrics.requests_successful += 1
        else:
            self.metrics.requests_failed += 1
            if error_message:
                self.metrics.last_error_message = error_message
        
        # Update response time statistics
        if self.metrics.response_time_avg == 0:
            self.metrics.response_time_avg = response_time
        else:
            self.metrics.response_time_avg = (
                (self.metrics.response_time_avg * (self.metrics.requests_total - 1) + response_time) /
                self.metrics.requests_total
            )
        
        self.metrics.response_time_min = min(self.metrics.response_time_min, response_time)
        self.metrics.response_time_max = max(self.metrics.response_time_max, response_time)
        self.metrics.last_request_time = datetime.utcnow()
    
    async def _health_check_loop(self) -> None:
        """Health check loop"""
        while self._running:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                
                # Check all connections
                for connection in self._connections.copy():
                    if not await self._validate_connection(connection):
                        logger.warning(f"Unhealthy connection detected in adapter '{self.config.name}'")
                        await self.disconnect(connection)
                
                # Ensure minimum connections
                if len(self._connections) < self.config.connection_pool_size:
                    await self.connect()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error in adapter '{self.config.name}': {e}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get adapter health status"""
        return {
            "name": self.config.name,
            "type": self.config.adapter_type.value,
            "state": self.state.value,
            "enabled": self.config.enabled,
            "connections": {
                "total": self.metrics.connection_count,
                "active": self.metrics.active_connections,
                "max": self.config.max_connections
            },
            "metrics": {
                "requests_total": self.metrics.requests_total,
                "requests_successful": self.metrics.requests_successful,
                "requests_failed": self.metrics.requests_failed,
                "success_rate": (
                    self.metrics.requests_successful / self.metrics.requests_total
                    if self.metrics.requests_total > 0 else 0
                ),
                "response_time_avg": self.metrics.response_time_avg,
                "response_time_min": self.metrics.response_time_min,
                "response_time_max": self.metrics.response_time_max,
                "last_request_time": self.metrics.last_request_time.isoformat() if self.metrics.last_request_time else None,
                "last_error_time": self.metrics.last_error_time.isoformat() if self.metrics.last_error_time else None,
                "last_error_message": self.metrics.last_error_message
            }
        }
    
    @asynccontextmanager
    async def connection_context(self):
        """Context manager for connection"""
        connection = await self.connect()
        try:
            yield connection
        finally:
            await self.disconnect(connection)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.config.name}', state='{self.state.value}')"


class AdapterRegistry:
    """Registry for managing adapters"""
    
    def __init__(self):
        self._adapters: Dict[str, BaseAdapter] = {}
        self._lock = asyncio.Lock()
    
    async def register(self, adapter: BaseAdapter) -> None:
        """Register adapter"""
        async with self._lock:
            self._adapters[adapter.config.name] = adapter
            logger.info(f"Registered adapter: {adapter.config.name}")
    
    async def unregister(self, name: str) -> None:
        """Unregister adapter"""
        async with self._lock:
            if name in self._adapters:
                adapter = self._adapters[name]
                await adapter.stop()
                del self._adapters[name]
                logger.info(f"Unregistered adapter: {name}")
    
    def get(self, name: str) -> Optional[BaseAdapter]:
        """Get adapter by name"""
        return self._adapters.get(name)
    
    def get_by_type(self, adapter_type: AdapterType) -> List[BaseAdapter]:
        """Get adapters by type"""
        return [
            adapter for adapter in self._adapters.values()
            if adapter.config.adapter_type == adapter_type
        ]
    
    def list_all(self) -> List[BaseAdapter]:
        """List all adapters"""
        return list(self._adapters.values())
    
    async def start_all(self) -> None:
        """Start all adapters"""
        for adapter in self._adapters.values():
            try:
                await adapter.start()
            except Exception as e:
                logger.error(f"Failed to start adapter '{adapter.config.name}': {e}")
    
    async def stop_all(self) -> None:
        """Stop all adapters"""
        for adapter in self._adapters.values():
            try:
                await adapter.stop()
            except Exception as e:
                logger.error(f"Failed to stop adapter '{adapter.config.name}': {e}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all adapters"""
        return {
            name: adapter.get_health_status()
            for name, adapter in self._adapters.items()
        }


# Global adapter registry
adapter_registry = AdapterRegistry()


# Convenience functions
async def register_adapter(adapter: BaseAdapter):
    """Register adapter"""
    await adapter_registry.register(adapter)


def get_adapter(name: str) -> Optional[BaseAdapter]:
    """Get adapter by name"""
    return adapter_registry.get(name)


async def start_all_adapters():
    """Start all adapters"""
    await adapter_registry.start_all()


async def stop_all_adapters():
    """Stop all adapters"""
    await adapter_registry.stop_all()





















