"""
Micro-frontend shell for Export IA.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json

from .registry import ComponentRegistry
from .communication import InterComponentCommunication
from .routing import MicroFrontendRouter

logger = logging.getLogger(__name__)


@dataclass
class MicroFrontendConfig:
    """Configuration for micro-frontend."""
    name: str
    version: str
    base_url: str
    components: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    permissions: List[str] = field(default_factory=list)
    theme: str = "default"
    locale: str = "en"
    debug: bool = False


class MicroFrontendShell:
    """Main shell for micro-frontend architecture."""
    
    def __init__(self, config: MicroFrontendConfig):
        self.config = config
        self.registry = ComponentRegistry()
        self.communication = InterComponentCommunication()
        self.router = MicroFrontendRouter()
        self.components: Dict[str, Any] = {}
        self.logger = logging.getLogger(f"shell.{config.name}")
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = {}
        
        # Initialize shell
        self._setup_shell()
    
    def _setup_shell(self) -> None:
        """Setup the micro-frontend shell."""
        # Register shell events
        self.communication.register_event("shell.ready", self._on_shell_ready)
        self.communication.register_event("component.loaded", self._on_component_loaded)
        self.communication.register_event("component.error", self._on_component_error)
        self.communication.register_event("navigation.requested", self._on_navigation_requested)
        
        self.logger.info(f"Micro-frontend shell initialized: {self.config.name}")
    
    async def initialize(self) -> None:
        """Initialize the micro-frontend shell."""
        try:
            # Initialize communication
            await self.communication.initialize()
            
            # Initialize router
            await self.router.initialize()
            
            # Load components
            await self._load_components()
            
            # Emit shell ready event
            await self.communication.emit_event("shell.ready", {
                "shell_name": self.config.name,
                "version": self.config.version,
                "components": list(self.components.keys())
            })
            
            self.logger.info("Micro-frontend shell initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize shell: {e}")
            raise
    
    async def _load_components(self) -> None:
        """Load all registered components."""
        for component_name in self.config.components:
            try:
                await self._load_component(component_name)
            except Exception as e:
                self.logger.error(f"Failed to load component {component_name}: {e}")
                await self.communication.emit_event("component.error", {
                    "component": component_name,
                    "error": str(e)
                })
    
    async def _load_component(self, component_name: str) -> None:
        """Load a specific component."""
        # Get component configuration
        component_config = await self.registry.get_component_config(component_name)
        if not component_config:
            raise ValueError(f"Component not found: {component_name}")
        
        # Check dependencies
        await self._check_component_dependencies(component_config)
        
        # Load component
        component = await self.registry.load_component(component_name, component_config)
        
        # Initialize component
        await component.initialize()
        
        # Register component
        self.components[component_name] = component
        
        # Emit component loaded event
        await self.communication.emit_event("component.loaded", {
            "component": component_name,
            "config": component_config
        })
        
        self.logger.info(f"Component loaded: {component_name}")
    
    async def _check_component_dependencies(self, component_config: Dict[str, Any]) -> None:
        """Check if component dependencies are satisfied."""
        dependencies = component_config.get("dependencies", [])
        
        for dependency in dependencies:
            if dependency not in self.components:
                raise ValueError(f"Component dependency not satisfied: {dependency}")
    
    async def _on_shell_ready(self, event_data: Dict[str, Any]) -> None:
        """Handle shell ready event."""
        self.logger.info("Shell ready event received")
    
    async def _on_component_loaded(self, event_data: Dict[str, Any]) -> None:
        """Handle component loaded event."""
        component_name = event_data.get("component")
        self.logger.info(f"Component loaded event: {component_name}")
    
    async def _on_component_error(self, event_data: Dict[str, Any]) -> None:
        """Handle component error event."""
        component_name = event_data.get("component")
        error = event_data.get("error")
        self.logger.error(f"Component error: {component_name} - {error}")
    
    async def _on_navigation_requested(self, event_data: Dict[str, Any]) -> None:
        """Handle navigation requested event."""
        route = event_data.get("route")
        params = event_data.get("params", {})
        
        try:
            await self.router.navigate(route, params)
        except Exception as e:
            self.logger.error(f"Navigation failed: {e}")
    
    async def get_component(self, component_name: str) -> Optional[Any]:
        """Get a loaded component."""
        return self.components.get(component_name)
    
    async def call_component_method(
        self,
        component_name: str,
        method_name: str,
        *args,
        **kwargs
    ) -> Any:
        """Call a method on a component."""
        component = await self.get_component(component_name)
        if not component:
            raise ValueError(f"Component not found: {component_name}")
        
        if not hasattr(component, method_name):
            raise ValueError(f"Method not found: {component_name}.{method_name}")
        
        method = getattr(component, method_name)
        return await method(*args, **kwargs)
    
    async def emit_event(self, event_name: str, data: Dict[str, Any]) -> None:
        """Emit an event to all components."""
        await self.communication.emit_event(event_name, data)
    
    async def register_event_handler(
        self,
        event_name: str,
        handler: Callable
    ) -> None:
        """Register an event handler."""
        if event_name not in self.event_handlers:
            self.event_handlers[event_name] = []
        
        self.event_handlers[event_name].append(handler)
        await self.communication.register_event(event_name, handler)
    
    async def unregister_event_handler(
        self,
        event_name: str,
        handler: Callable
    ) -> None:
        """Unregister an event handler."""
        if event_name in self.event_handlers:
            if handler in self.event_handlers[event_name]:
                self.event_handlers[event_name].remove(handler)
        
        await self.communication.unregister_event(event_name, handler)
    
    async def navigate(self, route: str, params: Optional[Dict[str, Any]] = None) -> None:
        """Navigate to a route."""
        await self.router.navigate(route, params or {})
    
    async def get_current_route(self) -> Dict[str, Any]:
        """Get current route information."""
        return await self.router.get_current_route()
    
    async def get_component_state(self, component_name: str) -> Optional[Dict[str, Any]]:
        """Get component state."""
        component = await self.get_component(component_name)
        if component and hasattr(component, 'get_state'):
            return await component.get_state()
        return None
    
    async def set_component_state(
        self,
        component_name: str,
        state: Dict[str, Any]
    ) -> bool:
        """Set component state."""
        component = await self.get_component(component_name)
        if component and hasattr(component, 'set_state'):
            await component.set_state(state)
            return True
        return False
    
    async def get_shell_info(self) -> Dict[str, Any]:
        """Get shell information."""
        return {
            "name": self.config.name,
            "version": self.config.version,
            "base_url": self.config.base_url,
            "components": list(self.components.keys()),
            "dependencies": self.config.dependencies,
            "permissions": self.config.permissions,
            "theme": self.config.theme,
            "locale": self.config.locale,
            "debug": self.config.debug,
            "initialized_at": datetime.now().isoformat()
        }
    
    async def cleanup(self) -> None:
        """Cleanup shell resources."""
        # Cleanup components
        for component_name, component in self.components.items():
            try:
                if hasattr(component, 'cleanup'):
                    await component.cleanup()
            except Exception as e:
                self.logger.error(f"Error cleaning up component {component_name}: {e}")
        
        # Cleanup communication
        await self.communication.cleanup()
        
        # Cleanup router
        await self.router.cleanup()
        
        self.logger.info("Micro-frontend shell cleanup completed")


# Global shell registry
_shells: Dict[str, MicroFrontendShell] = {}


def create_shell(config: MicroFrontendConfig) -> MicroFrontendShell:
    """Create a new micro-frontend shell."""
    shell = MicroFrontendShell(config)
    _shells[config.name] = shell
    return shell


def get_shell(shell_name: str) -> Optional[MicroFrontendShell]:
    """Get a micro-frontend shell by name."""
    return _shells.get(shell_name)


def list_shells() -> List[str]:
    """List all registered shells."""
    return list(_shells.keys())




