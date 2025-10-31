from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any, Callable, Union, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
import time
import traceback
from fastapi import FastAPI
import structlog
from typing import Any, List, Dict, Optional
"""
Lifespan Manager - FastAPI Application Lifecycle Management
Provides a clean way to manage startup and shutdown events using context managers.
"""



logger = structlog.get_logger(__name__)

class EventPriority(Enum):
    """Priority levels for startup/shutdown events."""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    OPTIONAL = 4

@dataclass
class LifecycleEvent:
    """Represents a lifecycle event (startup or shutdown)."""
    name: str
    handler: Callable
    priority: EventPriority = EventPriority.NORMAL
    timeout: Optional[float] = None
    retry_count: int = 0
    retry_delay: float = 1.0
    required: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

class LifespanManager:
    """
    Manages FastAPI application lifecycle events using context managers.
    
    This class provides a clean alternative to @app.on_event("startup") and 
    @app.on_event("shutdown") decorators, offering better error handling,
    prioritization, and resource management.
    """
    
    def __init__(self, app: Optional[FastAPI] = None):
        
    """__init__ function."""
self.app = app
        self.startup_events: List[LifecycleEvent] = []
        self.shutdown_events: List[LifecycleEvent] = []
        self._startup_completed = False
        self._shutdown_completed = False
        
    def add_startup_event(
        self,
        handler: Callable,
        name: Optional[str] = None,
        priority: EventPriority = EventPriority.NORMAL,
        timeout: Optional[float] = None,
        retry_count: int = 0,
        retry_delay: float = 1.0,
        required: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'LifespanManager':
        """
        Add a startup event handler.
        
        Args:
            handler: The event handler function (can be sync or async)
            name: Optional name for the event (for logging)
            priority: Event priority (higher priority events run first)
            timeout: Timeout in seconds for the event
            retry_count: Number of retries on failure
            retry_delay: Delay between retries in seconds
            required: Whether the event is required for startup
            metadata: Additional metadata for the event
            
        Returns:
            Self for method chaining
        """
        event = LifecycleEvent(
            name=name or handler.__name__,
            handler=handler,
            priority=priority,
            timeout=timeout,
            retry_count=retry_count,
            retry_delay=retry_delay,
            required=required,
            metadata=metadata or {}
        )
        self.startup_events.append(event)
        return self
    
    def add_shutdown_event(
        self,
        handler: Callable,
        name: Optional[str] = None,
        priority: EventPriority = EventPriority.NORMAL,
        timeout: Optional[float] = None,
        retry_count: int = 0,
        retry_delay: float = 1.0,
        required: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'LifespanManager':
        """
        Add a shutdown event handler.
        
        Args:
            handler: The event handler function (can be sync or async)
            name: Optional name for the event (for logging)
            priority: Event priority (higher priority events run first)
            timeout: Timeout in seconds for the event
            retry_count: Number of retries on failure
            retry_delay: Delay between retries in seconds
            required: Whether the event is required for shutdown
            metadata: Additional metadata for the event
            
        Returns:
            Self for method chaining
        """
        event = LifecycleEvent(
            name=name or handler.__name__,
            handler=handler,
            priority=priority,
            timeout=timeout,
            retry_count=retry_count,
            retry_delay=retry_delay,
            required=required,
            metadata=metadata or {}
        )
        self.shutdown_events.append(event)
        return self
    
    def add_database_connection(
        self,
        connect_func: Callable,
        disconnect_func: Callable,
        name: str = "database",
        timeout: float = 30.0,
        **kwargs
    ) -> 'LifespanManager':
        """Add database connection lifecycle events."""
        self.add_startup_event(
            handler=connect_func,
            name=f"{name}_connect",
            priority=EventPriority.CRITICAL,
            timeout=timeout,
            **kwargs
        )
        self.add_shutdown_event(
            handler=disconnect_func,
            name=f"{name}_disconnect",
            priority=EventPriority.CRITICAL,
            timeout=timeout,
            **kwargs
        )
        return self
    
    def add_cache_connection(
        self,
        connect_func: Callable,
        disconnect_func: Callable,
        name: str = "cache",
        timeout: float = 10.0,
        **kwargs
    ) -> 'LifespanManager':
        """Add cache connection lifecycle events."""
        self.add_startup_event(
            handler=connect_func,
            name=f"{name}_connect",
            priority=EventPriority.HIGH,
            timeout=timeout,
            **kwargs
        )
        self.add_shutdown_event(
            handler=disconnect_func,
            name=f"{name}_disconnect",
            priority=EventPriority.HIGH,
            timeout=timeout,
            **kwargs
        )
        return self
    
    def add_background_task(
        self,
        start_func: Callable,
        stop_func: Callable,
        name: str = "background_task",
        timeout: float = 10.0,
        **kwargs
    ) -> 'LifespanManager':
        """Add background task lifecycle events."""
        self.add_startup_event(
            handler=start_func,
            name=f"{name}_start",
            priority=EventPriority.LOW,
            timeout=timeout,
            **kwargs
        )
        self.add_shutdown_event(
            handler=stop_func,
            name=f"{name}_stop",
            priority=EventPriority.LOW,
            timeout=timeout,
            **kwargs
        )
        return self
    
    async def _execute_event(
        self,
        event: LifecycleEvent,
        phase: str
    ) -> bool:
        """
        Execute a single lifecycle event with error handling and retries.
        
        Args:
            event: The lifecycle event to execute
            phase: Either 'startup' or 'shutdown'
            
        Returns:
            True if successful, False otherwise
        """
        start_time = time.time()
        
        for attempt in range(event.retry_count + 1):
            try:
                logger.info(f"Executing {phase} event",
                           event_name=event.name,
                           attempt=attempt + 1,
                           priority=event.priority.value)
                
                # Execute with timeout if specified
                if event.timeout:
                    if asyncio.iscoroutinefunction(event.handler):
                        result = await asyncio.wait_for(
                            event.handler(), 
                            timeout=event.timeout
                        )
                    else:
                        # For sync functions, run in executor with timeout
                        loop = asyncio.get_event_loop()
                        result = await asyncio.wait_for(
                            loop.run_in_executor(None, event.handler),
                            timeout=event.timeout
                        )
                else:
                    if asyncio.iscoroutinefunction(event.handler):
                        result = await event.handler()
                    else:
                        result = event.handler()
                
                duration = time.time() - start_time
                logger.info(f"{phase.capitalize()} event completed",
                           event_name=event.name,
                           duration=duration,
                           result=result)
                return True
                
            except asyncio.TimeoutError:
                duration = time.time() - start_time
                logger.error(f"{phase.capitalize()} event timeout",
                            event_name=event.name,
                            timeout=event.timeout,
                            duration=duration)
                
                if attempt < event.retry_count:
                    await asyncio.sleep(event.retry_delay)
                    continue
                else:
                    if event.required:
                        raise
                    return False
                    
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"{phase.capitalize()} event failed",
                            event_name=event.name,
                            error=str(e),
                            duration=duration,
                            attempt=attempt + 1)
                
                if attempt < event.retry_count:
                    await asyncio.sleep(event.retry_delay)
                    continue
                else:
                    if event.required:
                        raise
                    return False
        
        return False
    
    async def _execute_events(
        self,
        events: List[LifecycleEvent],
        phase: str
    ) -> None:
        """
        Execute a list of lifecycle events in priority order.
        
        Args:
            events: List of lifecycle events to execute
            phase: Either 'startup' or 'shutdown'
        """
        # Sort events by priority (lower number = higher priority)
        sorted_events = sorted(events, key=lambda e: e.priority.value)
        
        logger.info(f"Starting {phase} phase",
                   event_count=len(sorted_events),
                   priorities=[e.priority.name for e in sorted_events])
        
        failed_events = []
        
        for event in sorted_events:
            try:
                success = await self._execute_event(event, phase)
                if not success:
                    failed_events.append(event.name)
            except Exception as e:
                logger.error(f"Critical {phase} event failed",
                            event_name=event.name,
                            error=str(e))
                failed_events.append(event.name)
                if event.required:
                    raise
        
        if failed_events:
            logger.warning(f"Some {phase} events failed",
                          failed_events=failed_events)
        else:
            logger.info(f"{phase.capitalize()} phase completed successfully")
    
    def create_lifespan(self) -> Callable:
        """
        Create a lifespan context manager for FastAPI.
        
        Returns:
            A lifespan context manager function
        """
        @asynccontextmanager
        async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
            # Startup phase
            try:
                await self._execute_events(self.startup_events, "startup")
                self._startup_completed = True
                yield
            finally:
                # Shutdown phase (runs even if startup failed)
                try:
                    await self._execute_events(self.shutdown_events, "shutdown")
                    self._shutdown_completed = True
                except Exception as e:
                    logger.error("Error during shutdown phase", error=str(e))
        
        return lifespan
    
    def attach_to_app(self, app: FastAPI) -> None:
        """
        Attach the lifespan manager to a FastAPI app.
        
        Args:
            app: The FastAPI application to attach to
        """
        app.router.lifespan_context = self.create_lifespan()
    
    @property
    def is_startup_completed(self) -> bool:
        """Check if startup phase has completed."""
        return self._startup_completed
    
    @property
    def is_shutdown_completed(self) -> bool:
        """Check if shutdown phase has completed."""
        return self._shutdown_completed

# Convenience functions for common patterns

def create_database_lifespan(
    connect_func: Callable,
    disconnect_func: Callable,
    name: str = "database",
    timeout: float = 30.0
) -> LifespanManager:
    """Create a lifespan manager for database connections."""
    manager = LifespanManager()
    manager.add_database_connection(connect_func, disconnect_func, name, timeout)
    return manager

def create_cache_lifespan(
    connect_func: Callable,
    disconnect_func: Callable,
    name: str = "cache",
    timeout: float = 10.0
) -> LifespanManager:
    """Create a lifespan manager for cache connections."""
    manager = LifespanManager()
    manager.add_cache_connection(connect_func, disconnect_func, name, timeout)
    return manager

def create_background_task_lifespan(
    start_func: Callable,
    stop_func: Callable,
    name: str = "background_task",
    timeout: float = 10.0
) -> LifespanManager:
    """Create a lifespan manager for background tasks."""
    manager = LifespanManager()
    manager.add_background_task(start_func, stop_func, name, timeout)
    return manager

# Migration helper functions

def migrate_on_event_to_lifespan(
    startup_handlers: Optional[List[Callable]] = None,
    shutdown_handlers: Optional[List[Callable]] = None
) -> LifespanManager:
    """
    Migrate from @app.on_event decorators to lifespan manager.
    
    Args:
        startup_handlers: List of startup event handlers
        shutdown_handlers: List of shutdown event handlers
        
    Returns:
        Configured lifespan manager
    """
    manager = LifespanManager()
    
    if startup_handlers:
        for handler in startup_handlers:
            manager.add_startup_event(handler)
    
    if shutdown_handlers:
        for handler in shutdown_handlers:
            manager.add_shutdown_event(handler)
    
    return manager 