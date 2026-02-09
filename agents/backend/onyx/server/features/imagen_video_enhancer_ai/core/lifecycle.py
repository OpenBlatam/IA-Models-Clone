"""
Lifecycle Management
===================

System for managing component lifecycle.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable, Awaitable
from enum import Enum
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class LifecycleState(Enum):
    """Component lifecycle state."""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    INITIALIZED = "initialized"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class LifecycleHook:
    """Lifecycle hook definition."""
    name: str
    callback: Callable[[], Awaitable[None]]
    priority: int = 0  # Lower priority runs first
    
    def __lt__(self, other):
        """Compare by priority."""
        return self.priority < other.priority


class LifecycleManager:
    """Manages component lifecycle."""
    
    def __init__(self, name: str = "Component"):
        """
        Initialize lifecycle manager.
        
        Args:
            name: Component name
        """
        self.name = name
        self.state = LifecycleState.UNINITIALIZED
        self.hooks: Dict[str, List[LifecycleHook]] = {
            "before_init": [],
            "after_init": [],
            "before_start": [],
            "after_start": [],
            "before_stop": [],
            "after_stop": [],
            "before_shutdown": [],
            "after_shutdown": []
        }
        self._error: Optional[Exception] = None
    
    def register_hook(
        self,
        event: str,
        callback: Callable[[], Awaitable[None]],
        priority: int = 0
    ):
        """
        Register a lifecycle hook.
        
        Args:
            event: Event name (before_init, after_init, etc.)
            callback: Async callback function
            priority: Hook priority (lower runs first)
        """
        if event not in self.hooks:
            raise ValueError(f"Unknown lifecycle event: {event}")
        
        hook = LifecycleHook(
            name=f"{self.name}_{event}",
            callback=callback,
            priority=priority
        )
        self.hooks[event].append(hook)
        self.hooks[event].sort()
    
    async def _run_hooks(self, event: str):
        """Run hooks for an event."""
        hooks = self.hooks.get(event, [])
        for hook in hooks:
            try:
                await hook.callback()
            except Exception as e:
                logger.error(f"Error in lifecycle hook {hook.name}: {e}")
                raise
    
    async def initialize(self):
        """Initialize component."""
        if self.state != LifecycleState.UNINITIALIZED:
            raise RuntimeError(f"Cannot initialize from state: {self.state}")
        
        try:
            self.state = LifecycleState.INITIALIZING
            await self._run_hooks("before_init")
            await self._do_initialize()
            await self._run_hooks("after_init")
            self.state = LifecycleState.INITIALIZED
            logger.info(f"{self.name} initialized")
        except Exception as e:
            self.state = LifecycleState.ERROR
            self._error = e
            logger.error(f"Error initializing {self.name}: {e}")
            raise
    
    async def _do_initialize(self):
        """Subclass-specific initialization."""
        pass
    
    async def start(self):
        """Start component."""
        if self.state == LifecycleState.UNINITIALIZED:
            await self.initialize()
        
        if self.state != LifecycleState.INITIALIZED:
            raise RuntimeError(f"Cannot start from state: {self.state}")
        
        try:
            self.state = LifecycleState.STARTING
            await self._run_hooks("before_start")
            await self._do_start()
            await self._run_hooks("after_start")
            self.state = LifecycleState.RUNNING
            logger.info(f"{self.name} started")
        except Exception as e:
            self.state = LifecycleState.ERROR
            self._error = e
            logger.error(f"Error starting {self.name}: {e}")
            raise
    
    async def _do_start(self):
        """Subclass-specific start logic."""
        pass
    
    async def stop(self):
        """Stop component."""
        if self.state != LifecycleState.RUNNING:
            logger.warning(f"Cannot stop from state: {self.state}")
            return
        
        try:
            self.state = LifecycleState.STOPPING
            await self._run_hooks("before_stop")
            await self._do_stop()
            await self._run_hooks("after_stop")
            self.state = LifecycleState.STOPPED
            logger.info(f"{self.name} stopped")
        except Exception as e:
            self.state = LifecycleState.ERROR
            self._error = e
            logger.error(f"Error stopping {self.name}: {e}")
            raise
    
    async def _do_stop(self):
        """Subclass-specific stop logic."""
        pass
    
    async def shutdown(self):
        """Shutdown component."""
        if self.state == LifecycleState.RUNNING:
            await self.stop()
        
        try:
            await self._run_hooks("before_shutdown")
            await self._do_shutdown()
            await self._run_hooks("after_shutdown")
            self.state = LifecycleState.UNINITIALIZED
            logger.info(f"{self.name} shutdown")
        except Exception as e:
            self.state = LifecycleState.ERROR
            self._error = e
            logger.error(f"Error shutting down {self.name}: {e}")
            raise
    
    async def _do_shutdown(self):
        """Subclass-specific shutdown logic."""
        pass
    
    def get_state(self) -> LifecycleState:
        """Get current state."""
        return self.state
    
    def get_error(self) -> Optional[Exception]:
        """Get error if in ERROR state."""
        return self._error


class LifecycleComponent(ABC):
    """Base class for components with lifecycle."""
    
    def __init__(self, name: str):
        """
        Initialize lifecycle component.
        
        Args:
            name: Component name
        """
        self.lifecycle = LifecycleManager(name)
    
    async def initialize(self):
        """Initialize component."""
        await self.lifecycle.initialize()
    
    async def start(self):
        """Start component."""
        await self.lifecycle.start()
    
    async def stop(self):
        """Stop component."""
        await self.lifecycle.stop()
    
    async def shutdown(self):
        """Shutdown component."""
        await self.lifecycle.shutdown()
    
    @abstractmethod
    async def _do_initialize(self):
        """Subclass-specific initialization."""
        pass
    
    @abstractmethod
    async def _do_start(self):
        """Subclass-specific start logic."""
        pass
    
    @abstractmethod
    async def _do_stop(self):
        """Subclass-specific stop logic."""
        pass
    
    @abstractmethod
    async def _do_shutdown(self):
        """Subclass-specific shutdown logic."""
        pass




