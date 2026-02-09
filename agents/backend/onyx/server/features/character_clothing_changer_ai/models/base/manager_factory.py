"""
Manager Factory
===============
Factory para crear managers de manera consistente
"""

from typing import Dict, Any, Optional, Type
from .base_manager import BaseManager
from .manager_registry import manager_registry


class ManagerFactory:
    """
    Factory para crear managers
    """
    
    def __init__(self):
        self._creators: Dict[str, Type[BaseManager]] = {}
        self._configs: Dict[str, Dict[str, Any]] = {}
    
    def register_creator(
        self,
        manager_type: str,
        manager_class: Type[BaseManager],
        default_config: Optional[Dict[str, Any]] = None
    ):
        """
        Registrar un creador de manager
        
        Args:
            manager_type: Tipo de manager
            manager_class: Clase del manager
            default_config: Configuración por defecto
        """
        self._creators[manager_type] = manager_class
        if default_config:
            self._configs[manager_type] = default_config
    
    def create(
        self,
        manager_type: str,
        name: str,
        config: Optional[Dict[str, Any]] = None,
        dependencies: Optional[List[str]] = None
    ) -> Optional[BaseManager]:
        """
        Crear un manager
        
        Args:
            manager_type: Tipo de manager
            name: Nombre del manager
            config: Configuración personalizada
            dependencies: Dependencias del manager
        """
        if manager_type not in self._creators:
            return None
        
        manager_class = self._creators[manager_type]
        
        # Combinar config por defecto con config personalizada
        final_config = self._configs.get(manager_type, {}).copy()
        if config:
            final_config.update(config)
        
        # Crear instancia
        manager = manager_class(name)
        
        # Aplicar configuración si el manager la soporta
        if hasattr(manager, 'configure') and final_config:
            manager.configure(final_config)
        
        # Registrar en registry
        manager_registry.register(name, manager, dependencies)
        
        return manager
    
    def create_and_initialize(
        self,
        manager_type: str,
        name: str,
        config: Optional[Dict[str, Any]] = None,
        dependencies: Optional[List[str]] = None
    ) -> Optional[BaseManager]:
        """
        Crear e inicializar un manager
        
        Args:
            manager_type: Tipo de manager
            name: Nombre del manager
            config: Configuración personalizada
            dependencies: Dependencias del manager
        """
        manager = self.create(manager_type, name, config, dependencies)
        if manager:
            manager.initialize()
        return manager


# Instancia global
manager_factory = ManagerFactory()

