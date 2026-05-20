"""
Component Registry
Registry for training components (losses, optimizers, schedulers)
"""

from typing import Dict, Callable, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ComponentRegistry:
    """
    Registry for training components
    """
    
    def __init__(self):
        """Initialize registry"""
        self._components: Dict[str, Dict[str, Callable]] = {
            'loss': {},
            'optimizer': {},
            'scheduler': {},
            'augmentation': {},
            'metric': {},
        }
    
    def register(
        self,
        component_type: str,
        name: str,
        factory: Callable,
    ):
        """
        Register component factory
        
        Args:
            component_type: Type of component
            name: Component name
            factory: Factory function
        """
        if component_type not in self._components:
            self._components[component_type] = {}
        
        self._components[component_type][name] = factory
        logger.info(f"Registered {component_type}: {name}")
    
    def get(self, component_type: str, name: str) -> Optional[Callable]:
        """
        Get component factory
        
        Args:
            component_type: Type of component
            name: Component name
            
        Returns:
            Factory function or None
        """
        return self._components.get(component_type, {}).get(name)
    
    def create(
        self,
        component_type: str,
        name: str,
        **kwargs
    ) -> Any:
        """
        Create component using factory
        
        Args:
            component_type: Type of component
            name: Component name
            **kwargs: Component arguments
            
        Returns:
            Component instance or None
        """
        factory = self.get(component_type, name)
        if factory:
            return factory(**kwargs)
        
        logger.warning(f"Component {component_type}.{name} not found in registry")
        return None
    
    def list_components(self, component_type: Optional[str] = None) -> Dict[str, list]:
        """
        List registered components
        
        Args:
            component_type: Filter by type (optional)
            
        Returns:
            Dictionary of component types and names
        """
        if component_type:
            return {component_type: list(self._components.get(component_type, {}).keys())}
        
        return {
            comp_type: list(components.keys())
            for comp_type, components in self._components.items()
        }


# Global component registry
component_registry = ComponentRegistry()



