"""
Component Registry Module
=========================

Registro centralizado de componentes del agente.
Proporciona una forma estructurada de registrar y acceder a componentes.
"""

import logging
from typing import Dict, Any, Optional, Type, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ComponentRegistry:
    """
    Registro centralizado de componentes del agente.
    
    Proporciona una forma estructurada de registrar y acceder
    a todos los componentes del agente, facilitando la gestión
    de dependencias y el acceso centralizado.
    """
    
    def __init__(self):
        """Inicializar registro de componentes."""
        self._components: Dict[str, Any] = {}
        self._component_types: Dict[str, type] = {}
    
    def register(
        self,
        name: str,
        component: Any,
        component_type: Optional[type] = None
    ) -> None:
        """
        Registrar un componente.
        
        Args:
            name: Nombre del componente
            component: Instancia del componente
            component_type: Tipo del componente (opcional)
        """
        self._components[name] = component
        if component_type:
            self._component_types[name] = component_type
        else:
            self._component_types[name] = type(component)
        
        logger.debug(f"Registered component: {name} ({type(component).__name__})")
    
    def get(self, name: str) -> Any:
        """
        Obtener un componente por nombre.
        
        Args:
            name: Nombre del componente
            
        Returns:
            Componente registrado
            
        Raises:
            KeyError: Si el componente no está registrado
        """
        if name not in self._components:
            raise KeyError(f"Component '{name}' not found in registry")
        
        return self._components[name]
    
    def get_typed(self, name: str, component_type: Type[T]) -> T:
        """
        Obtener un componente con verificación de tipo.
        
        Args:
            name: Nombre del componente
            component_type: Tipo esperado del componente
            
        Returns:
            Componente del tipo especificado
            
        Raises:
            KeyError: Si el componente no está registrado
            TypeError: Si el componente no es del tipo esperado
        """
        component = self.get(name)
        
        if not isinstance(component, component_type):
            raise TypeError(
                f"Component '{name}' is of type {type(component).__name__}, "
                f"expected {component_type.__name__}"
            )
        
        return component
    
    def has(self, name: str) -> bool:
        """
        Verificar si un componente está registrado.
        
        Args:
            name: Nombre del componente
            
        Returns:
            True si está registrado
        """
        return name in self._components
    
    def unregister(self, name: str) -> None:
        """
        Desregistrar un componente.
        
        Args:
            name: Nombre del componente
        """
        if name in self._components:
            del self._components[name]
            if name in self._component_types:
                del self._component_types[name]
            logger.debug(f"Unregistered component: {name}")
    
    def get_all(self) -> Dict[str, Any]:
        """
        Obtener todos los componentes registrados.
        
        Returns:
            Dict con todos los componentes
        """
        return self._components.copy()
    
    def get_component_names(self) -> list[str]:
        """
        Obtener nombres de todos los componentes registrados.
        
        Returns:
            Lista de nombres de componentes
        """
        return list(self._components.keys())
    
    def get_component_info(self, name: str) -> Dict[str, Any]:
        """
        Obtener información de un componente.
        
        Args:
            name: Nombre del componente
            
        Returns:
            Dict con información del componente
        """
        if name not in self._components:
            raise KeyError(f"Component '{name}' not found in registry")
        
        component = self._components[name]
        component_type = self._component_types.get(name, type(component))
        
        return {
            "name": name,
            "type": component_type.__name__,
            "module": component_type.__module__,
            "has_methods": hasattr(component, "__dict__"),
            "methods": [
                method for method in dir(component)
                if not method.startswith("_") and callable(getattr(component, method))
            ] if hasattr(component, "__dict__") else []
        }
    
    def get_all_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Obtener información de todos los componentes.
        
        Returns:
            Dict con información de todos los componentes
        """
        return {
            name: self.get_component_info(name)
            for name in self._components.keys()
        }
    
    def clear(self) -> None:
        """Limpiar todos los componentes registrados."""
        self._components.clear()
        self._component_types.clear()
        logger.debug("Component registry cleared")
    
    def count(self) -> int:
        """
        Obtener número de componentes registrados.
        
        Returns:
            Número de componentes
        """
        return len(self._components)
    
    def register_batch(self, components: Dict[str, Any]) -> None:
        """
        Registrar múltiples componentes en batch.
        
        Args:
            components: Dict con nombre -> componente
        """
        for name, component in components.items():
            self.register(name, component)
    
    def get_by_type(self, component_type: Type[T]) -> list[T]:
        """
        Obtener todos los componentes de un tipo específico.
        
        Args:
            component_type: Tipo de componente
            
        Returns:
            Lista de componentes del tipo especificado
        """
        return [
            component for component in self._components.values()
            if isinstance(component, component_type)
        ]


def create_component_registry() -> ComponentRegistry:
    """
    Factory function para crear ComponentRegistry.
    
    Returns:
        Instancia de ComponentRegistry
    """
    return ComponentRegistry()


