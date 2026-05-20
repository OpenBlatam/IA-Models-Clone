"""
Service Lifecycle Management for Color Grading AI
===================================================

Unified lifecycle management for all services.
"""

import logging
from typing import Dict, Any, Optional, List, Callable
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


class LifecyclePhase(Enum):
    """Lifecycle phases."""
    CREATED = "created"
    INITIALIZING = "initializing"
    INITIALIZED = "initialized"
    STARTING = "starting"
    STARTED = "started"
    STOPPING = "stopping"
    STOPPED = "stopped"
    DESTROYED = "destroyed"
    ERROR = "error"


@dataclass
class LifecycleEvent:
    """Lifecycle event."""
    service_name: str
    phase: LifecyclePhase
    timestamp: datetime = field(default_factory=datetime.now)
    error: Optional[Exception] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ServiceLifecycleManager:
    """
    Service lifecycle manager.
    
    Features:
    - Phase management
    - Event hooks
    - Dependency ordering
    - Error recovery
    - State tracking
    """
    
    def __init__(self):
        """Initialize lifecycle manager."""
        self._services: Dict[str, Any] = {}
        self._phases: Dict[str, LifecyclePhase] = {}
        self._hooks: Dict[LifecyclePhase, List[Callable]] = {
            phase: [] for phase in LifecyclePhase
        }
        self._events: List[LifecycleEvent] = []
        self._max_events = 1000
    
    def register_service(self, name: str, service: Any):
        """
        Register service for lifecycle management.
        
        Args:
            name: Service name
            service: Service instance
        """
        self._services[name] = service
        self._phases[name] = LifecyclePhase.CREATED
        self._record_event(name, LifecyclePhase.CREATED)
        logger.debug(f"Registered service for lifecycle: {name}")
    
    def add_hook(self, phase: LifecyclePhase, hook: Callable):
        """
        Add lifecycle hook.
        
        Args:
            phase: Lifecycle phase
            hook: Hook function
        """
        self._hooks[phase].append(hook)
        logger.debug(f"Added hook for phase: {phase.value}")
    
    async def initialize_service(self, name: str) -> bool:
        """
        Initialize service.
        
        Args:
            name: Service name
            
        Returns:
            True if successful
        """
        if name not in self._services:
            logger.error(f"Service not found: {name}")
            return False
        
        service = self._services[name]
        current_phase = self._phases.get(name, LifecyclePhase.CREATED)
        
        if current_phase in [LifecyclePhase.INITIALIZED, LifecyclePhase.STARTED]:
            logger.debug(f"Service {name} already initialized")
            return True
        
        try:
            # Transition to initializing
            self._phases[name] = LifecyclePhase.INITIALIZING
            self._record_event(name, LifecyclePhase.INITIALIZING)
            
            # Run hooks
            await self._run_hooks(LifecyclePhase.INITIALIZING, name, service)
            
            # Initialize service
            if hasattr(service, 'initialize'):
                if callable(service.initialize):
                    if inspect.iscoroutinefunction(service.initialize):
                        await service.initialize()
                    else:
                        service.initialize()
            
            # Transition to initialized
            self._phases[name] = LifecyclePhase.INITIALIZED
            self._record_event(name, LifecyclePhase.INITIALIZED)
            
            logger.info(f"Service {name} initialized")
            return True
        
        except Exception as e:
            self._phases[name] = LifecyclePhase.ERROR
            self._record_event(name, LifecyclePhase.ERROR, error=e)
            logger.error(f"Failed to initialize service {name}: {e}")
            return False
    
    async def start_service(self, name: str) -> bool:
        """
        Start service.
        
        Args:
            name: Service name
            
        Returns:
            True if successful
        """
        if name not in self._services:
            return False
        
        service = self._services[name]
        current_phase = self._phases.get(name, LifecyclePhase.CREATED)
        
        if current_phase == LifecyclePhase.STARTED:
            return True
        
        # Ensure initialized first
        if current_phase not in [LifecyclePhase.INITIALIZED, LifecyclePhase.STARTED]:
            if not await self.initialize_service(name):
                return False
        
        try:
            # Transition to starting
            self._phases[name] = LifecyclePhase.STARTING
            self._record_event(name, LifecyclePhase.STARTING)
            
            # Run hooks
            await self._run_hooks(LifecyclePhase.STARTING, name, service)
            
            # Start service
            if hasattr(service, 'start'):
                if callable(service.start):
                    if inspect.iscoroutinefunction(service.start):
                        await service.start()
                    else:
                        service.start()
            
            # Transition to started
            self._phases[name] = LifecyclePhase.STARTED
            self._record_event(name, LifecyclePhase.STARTED)
            
            logger.info(f"Service {name} started")
            return True
        
        except Exception as e:
            self._phases[name] = LifecyclePhase.ERROR
            self._record_event(name, LifecyclePhase.ERROR, error=e)
            logger.error(f"Failed to start service {name}: {e}")
            return False
    
    async def stop_service(self, name: str) -> bool:
        """
        Stop service.
        
        Args:
            name: Service name
            
        Returns:
            True if successful
        """
        if name not in self._services:
            return False
        
        service = self._services[name]
        current_phase = self._phases.get(name)
        
        if current_phase in [LifecyclePhase.STOPPED, LifecyclePhase.DESTROYED]:
            return True
        
        try:
            # Transition to stopping
            self._phases[name] = LifecyclePhase.STOPPING
            self._record_event(name, LifecyclePhase.STOPPING)
            
            # Run hooks
            await self._run_hooks(LifecyclePhase.STOPPING, name, service)
            
            # Stop service
            if hasattr(service, 'stop'):
                if callable(service.stop):
                    if inspect.iscoroutinefunction(service.stop):
                        await service.stop()
                    else:
                        service.stop()
            
            # Transition to stopped
            self._phases[name] = LifecyclePhase.STOPPED
            self._record_event(name, LifecyclePhase.STOPPED)
            
            logger.info(f"Service {name} stopped")
            return True
        
        except Exception as e:
            self._phases[name] = LifecyclePhase.ERROR
            self._record_event(name, LifecyclePhase.ERROR, error=e)
            logger.error(f"Failed to stop service {name}: {e}")
            return False
    
    async def initialize_all(self, order: Optional[List[str]] = None) -> Dict[str, bool]:
        """
        Initialize all services in order.
        
        Args:
            order: Optional service order
            
        Returns:
            Dictionary of initialization results
        """
        service_names = order or list(self._services.keys())
        results = {}
        
        for name in service_names:
            results[name] = await self.initialize_service(name)
        
        return results
    
    async def start_all(self, order: Optional[List[str]] = None) -> Dict[str, bool]:
        """Start all services in order."""
        service_names = order or list(self._services.keys())
        results = {}
        
        for name in service_names:
            results[name] = await self.start_service(name)
        
        return results
    
    async def stop_all(self, order: Optional[List[str]] = None, reverse: bool = True) -> Dict[str, bool]:
        """
        Stop all services.
        
        Args:
            order: Optional service order
            reverse: Whether to reverse order (default: True for shutdown)
            
        Returns:
            Dictionary of stop results
        """
        service_names = order or list(self._services.keys())
        if reverse:
            service_names = list(reversed(service_names))
        
        results = {}
        
        for name in service_names:
            results[name] = await self.stop_service(name)
        
        return results
    
    def get_phase(self, name: str) -> Optional[LifecyclePhase]:
        """Get service lifecycle phase."""
        return self._phases.get(name)
    
    def get_events(self, service_name: Optional[str] = None, limit: int = 100) -> List[LifecycleEvent]:
        """Get lifecycle events."""
        events = self._events
        
        if service_name:
            events = [e for e in events if e.service_name == service_name]
        
        return events[-limit:]
    
    def _record_event(self, name: str, phase: LifecyclePhase, error: Optional[Exception] = None):
        """Record lifecycle event."""
        event = LifecycleEvent(
            service_name=name,
            phase=phase,
            error=error
        )
        
        self._events.append(event)
        if len(self._events) > self._max_events:
            self._events = self._events[-self._max_events:]
    
    async def _run_hooks(self, phase: LifecyclePhase, name: str, service: Any):
        """Run lifecycle hooks."""
        hooks = self._hooks.get(phase, [])
        
        for hook in hooks:
            try:
                if inspect.iscoroutinefunction(hook):
                    await hook(name, service, phase)
                else:
                    hook(name, service, phase)
            except Exception as e:
                logger.error(f"Error in lifecycle hook for {name} at {phase.value}: {e}")


# Import for type checking
import inspect




