"""
System Integrator
================

System for integrating and coordinating all subsystems.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class SystemComponent:
    """System component definition."""
    name: str
    component: Any
    dependencies: List[str] = field(default_factory=list)
    priority: int = 0  # Lower = higher priority
    initialized: bool = False


class SystemIntegrator:
    """Integrator for all system components."""
    
    def __init__(self):
        """Initialize system integrator."""
        self.components: Dict[str, SystemComponent] = {}
        self.initialization_order: List[str] = []
    
    def register(
        self,
        name: str,
        component: Any,
        dependencies: Optional[List[str]] = None,
        priority: int = 0
    ):
        """
        Register a system component.
        
        Args:
            name: Component name
            component: Component instance
            dependencies: Optional list of dependency names
            priority: Initialization priority
        """
        comp = SystemComponent(
            name=name,
            component=component,
            dependencies=dependencies or [],
            priority=priority
        )
        self.components[name] = comp
        logger.debug(f"Registered component: {name}")
    
    def _calculate_initialization_order(self) -> List[str]:
        """Calculate initialization order based on dependencies and priority."""
        # Topological sort with priority
        order = []
        visited = set()
        temp_visited = set()
        
        def visit(name: str):
            if name in temp_visited:
                raise ValueError(f"Circular dependency detected involving {name}")
            if name in visited:
                return
            
            temp_visited.add(name)
            
            if name in self.components:
                comp = self.components[name]
                # Visit dependencies first
                for dep in comp.dependencies:
                    visit(dep)
            
            temp_visited.remove(name)
            visited.add(name)
            order.append(name)
        
        # Sort by priority, then visit
        sorted_components = sorted(
            self.components.items(),
            key=lambda x: (x[1].priority, x[0])
        )
        
        for name, _ in sorted_components:
            if name not in visited:
                visit(name)
        
        return order
    
    async def initialize_all(self):
        """Initialize all components in dependency order."""
        self.initialization_order = self._calculate_initialization_order()
        
        for name in self.initialization_order:
            comp = self.components[name]
            if comp.initialized:
                continue
            
            try:
                # Initialize component
                if hasattr(comp.component, 'initialize'):
                    if asyncio.iscoroutinefunction(comp.component.initialize):
                        await comp.component.initialize()
                    else:
                        comp.component.initialize()
                
                comp.initialized = True
                logger.info(f"Initialized component: {name}")
            except Exception as e:
                logger.error(f"Error initializing component {name}: {e}")
                raise
    
    async def shutdown_all(self):
        """Shutdown all components in reverse order."""
        for name in reversed(self.initialization_order):
            comp = self.components[name]
            if not comp.initialized:
                continue
            
            try:
                # Shutdown component
                if hasattr(comp.component, 'shutdown'):
                    if asyncio.iscoroutinefunction(comp.component.shutdown):
                        await comp.component.shutdown()
                    else:
                        comp.component.shutdown()
                
                comp.initialized = False
                logger.info(f"Shutdown component: {name}")
            except Exception as e:
                logger.error(f"Error shutting down component {name}: {e}")
    
    def get_component(self, name: str) -> Optional[Any]:
        """
        Get component by name.
        
        Args:
            name: Component name
            
        Returns:
            Component instance or None
        """
        comp = self.components.get(name)
        return comp.component if comp else None
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all components."""
        return {
            "components": {
                name: {
                    "initialized": comp.initialized,
                    "dependencies": comp.dependencies,
                    "priority": comp.priority
                }
                for name, comp in self.components.items()
            },
            "initialization_order": self.initialization_order
        }




