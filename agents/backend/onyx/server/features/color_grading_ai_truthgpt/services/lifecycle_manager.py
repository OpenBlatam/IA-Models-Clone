"""
Lifecycle Manager for Color Grading AI
=======================================

Service lifecycle management with initialization, health checks, and shutdown.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class LifecycleState(Enum):
    """Lifecycle states."""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class LifecycleHook:
    """Lifecycle hook definition."""
    name: str
    hook_func: Callable
    phase: str
    timeout: float = 10.0


class LifecycleManager:
    """
    Service lifecycle manager.
    
    Features:
    - State management
    - Lifecycle hooks
    - Initialization tracking
    - Health monitoring integration
    - Graceful shutdown integration
    """
    
    def __init__(self):
        """Initialize lifecycle manager."""
        self.state = LifecycleState.UNINITIALIZED
        self._hooks: Dict[str, List[LifecycleHook]] = {
            "before_init": [],
            "after_init": [],
            "before_start": [],
            "after_start": [],
            "before_stop": [],
            "after_stop": [],
        }
        self._services: Dict[str, Any] = {}
        self._initialization_order: List[str] = []
    
    def register_hook(
        self,
        name: str,
        hook_func: Callable,
        phase: str,
        timeout: float = 10.0
    ):
        """
        Register lifecycle hook.
        
        Args:
            name: Hook name
            hook_func: Hook function (async)
            phase: Hook phase
            timeout: Hook timeout
        """
        if phase not in self._hooks:
            raise ValueError(f"Invalid phase: {phase}")
        
        hook = LifecycleHook(
            name=name,
            hook_func=hook_func,
            phase=phase,
            timeout=timeout
        )
        
        self._hooks[phase].append(hook)
        logger.info(f"Registered lifecycle hook: {name} (phase: {phase})")
    
    def register_service(
        self,
        name: str,
        service: Any,
        dependencies: Optional[List[str]] = None
    ):
        """
        Register service for lifecycle management.
        
        Args:
            name: Service name
            service: Service instance
            dependencies: Service dependencies
        """
        self._services[name] = {
            "service": service,
            "dependencies": dependencies or [],
            "initialized": False,
        }
        logger.info(f"Registered service: {name}")
    
    async def initialize(self):
        """Initialize all services."""
        if self.state != LifecycleState.UNINITIALIZED:
            logger.warning(f"Cannot initialize from state: {self.state}")
            return
        
        self.state = LifecycleState.INITIALIZING
        logger.info("Starting initialization")
        
        try:
            # Run before_init hooks
            await self._run_hooks("before_init")
            
            # Initialize services in dependency order
            await self._initialize_services()
            
            # Run after_init hooks
            await self._run_hooks("after_init")
            
            self.state = LifecycleState.READY
            logger.info("Initialization completed")
        
        except Exception as e:
            self.state = LifecycleState.ERROR
            logger.error(f"Initialization failed: {e}")
            raise
    
    async def start(self):
        """Start all services."""
        if self.state != LifecycleState.READY:
            logger.warning(f"Cannot start from state: {self.state}")
            return
        
        self.state = LifecycleState.RUNNING
        logger.info("Starting services")
        
        try:
            # Run before_start hooks
            await self._run_hooks("before_start")
            
            # Start services
            for name, service_info in self._services.items():
                service = service_info["service"]
                if hasattr(service, "start") and callable(service.start):
                    if asyncio.iscoroutinefunction(service.start):
                        await service.start()
                    else:
                        service.start()
                    logger.info(f"Started service: {name}")
            
            # Run after_start hooks
            await self._run_hooks("after_start")
            
            logger.info("All services started")
        
        except Exception as e:
            self.state = LifecycleState.ERROR
            logger.error(f"Start failed: {e}")
            raise
    
    async def stop(self):
        """Stop all services."""
        if self.state not in [LifecycleState.RUNNING, LifecycleState.READY]:
            logger.warning(f"Cannot stop from state: {self.state}")
            return
        
        self.state = LifecycleState.STOPPING
        logger.info("Stopping services")
        
        try:
            # Run before_stop hooks
            await self._run_hooks("before_stop")
            
            # Stop services in reverse order
            for name in reversed(self._initialization_order):
                service_info = self._services.get(name)
                if service_info:
                    service = service_info["service"]
                    if hasattr(service, "stop") and callable(service.stop):
                        if asyncio.iscoroutinefunction(service.stop):
                            await service.stop()
                        else:
                            service.stop()
                        logger.info(f"Stopped service: {name}")
            
            # Run after_stop hooks
            await self._run_hooks("after_stop")
            
            self.state = LifecycleState.STOPPED
            logger.info("All services stopped")
        
        except Exception as e:
            self.state = LifecycleState.ERROR
            logger.error(f"Stop failed: {e}")
            raise
    
    async def _initialize_services(self):
        """Initialize services in dependency order."""
        # Topological sort for dependency order
        initialized = set()
        self._initialization_order = []
        
        while len(initialized) < len(self._services):
            progress = False
            
            for name, service_info in self._services.items():
                if name in initialized:
                    continue
                
                # Check if dependencies are initialized
                deps = service_info["dependencies"]
                if all(dep in initialized for dep in deps):
                    # Initialize service
                    service = service_info["service"]
                    if hasattr(service, "initialize") and callable(service.initialize):
                        if asyncio.iscoroutinefunction(service.initialize):
                            await service.initialize()
                        else:
                            service.initialize()
                    
                    service_info["initialized"] = True
                    initialized.add(name)
                    self._initialization_order.append(name)
                    progress = True
                    logger.info(f"Initialized service: {name}")
            
            if not progress:
                # Circular dependency or missing dependency
                uninitialized = [n for n in self._services.keys() if n not in initialized]
                raise ValueError(f"Cannot resolve dependencies for: {uninitialized}")
    
    async def _run_hooks(self, phase: str):
        """Run hooks for phase."""
        hooks = self._hooks.get(phase, [])
        if not hooks:
            return
        
        logger.debug(f"Running {len(hooks)} hooks for phase: {phase}")
        
        for hook in hooks:
            try:
                if asyncio.iscoroutinefunction(hook.hook_func):
                    await asyncio.wait_for(
                        hook.hook_func(),
                        timeout=hook.timeout
                    )
                else:
                    hook.hook_func()
            except Exception as e:
                logger.error(f"Hook {hook.name} failed in phase {phase}: {e}")
    
    def get_state(self) -> LifecycleState:
        """Get current lifecycle state."""
        return self.state
    
    def get_status(self) -> Dict[str, Any]:
        """Get lifecycle status."""
        return {
            "state": self.state.value,
            "services": {
                name: {
                    "initialized": info["initialized"],
                    "dependencies": info["dependencies"],
                }
                for name, info in self._services.items()
            },
            "initialization_order": self._initialization_order,
        }




