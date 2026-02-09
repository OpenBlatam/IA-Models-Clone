"""
Manager Registry
================
Registro centralizado de todos los managers del sistema
"""

from typing import Dict, List, Optional, Type, Any
from .base_manager import BaseManager


class ManagerRegistry:
    """
    Registro centralizado de managers
    """
    
    def __init__(self):
        self._managers: Dict[str, BaseManager] = {}
        self._manager_types: Dict[str, Type[BaseManager]] = {}
        self._dependencies: Dict[str, List[str]] = {}
    
    def register(
        self,
        name: str,
        manager: BaseManager,
        dependencies: Optional[List[str]] = None
    ):
        """
        Registrar un manager
        
        Args:
            name: Nombre del manager
            manager: Instancia del manager
            dependencies: Lista de nombres de managers de los que depende
        """
        self._managers[name] = manager
        self._dependencies[name] = dependencies or []
    
    def register_type(
        self,
        name: str,
        manager_class: Type[BaseManager],
        dependencies: Optional[List[str]] = None
    ):
        """
        Registrar un tipo de manager (para lazy initialization)
        
        Args:
            name: Nombre del manager
            manager_class: Clase del manager
            dependencies: Lista de nombres de managers de los que depende
        """
        self._manager_types[name] = manager_class
        self._dependencies[name] = dependencies or []
    
    def get(self, name: str) -> Optional[BaseManager]:
        """Obtener manager por nombre"""
        # Si ya está instanciado, retornarlo
        if name in self._managers:
            return self._managers[name]
        
        # Si es un tipo registrado, instanciarlo
        if name in self._manager_types:
            manager_class = self._manager_types[name]
            manager = manager_class(name)
            self._managers[name] = manager
            return manager
        
        return None
    
    def get_all(self) -> Dict[str, BaseManager]:
        """Obtener todos los managers registrados"""
        # Instanciar tipos pendientes
        for name, manager_class in self._manager_types.items():
            if name not in self._managers:
                manager = manager_class(name)
                self._managers[name] = manager
        
        return self._managers.copy()
    
    def get_dependencies(self, name: str) -> List[str]:
        """Obtener dependencias de un manager"""
        return self._dependencies.get(name, [])
    
    def initialize_all(self) -> Dict[str, bool]:
        """
        Inicializar todos los managers en orden de dependencias
        
        Returns:
            Dict con resultados de inicialización
        """
        results = {}
        
        # Ordenar por dependencias (topological sort)
        ordered = self._topological_sort()
        
        for name in ordered:
            manager = self.get(name)
            if manager:
                results[name] = manager.initialize()
            else:
                results[name] = False
        
        return results
    
    def shutdown_all(self):
        """Cerrar todos los managers"""
        for manager in self._managers.values():
            try:
                manager.shutdown()
            except Exception as e:
                print(f"Error shutting down {manager.name}: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas de todos los managers"""
        stats = {}
        for name, manager in self._managers.items():
            stats[name] = manager.get_stats()
        return stats
    
    def _topological_sort(self) -> List[str]:
        """
        Ordenar managers por dependencias (topological sort)
        """
        visited = set()
        result = []
        
        def visit(name: str):
            if name in visited:
                return
            
            # Visitar dependencias primero
            for dep in self._dependencies.get(name, []):
                if dep in self._managers or dep in self._manager_types:
                    visit(dep)
            
            visited.add(name)
            result.append(name)
        
        # Visitar todos los managers
        all_names = list(self._managers.keys()) + list(self._manager_types.keys())
        for name in all_names:
            if name not in visited:
                visit(name)
        
        return result
    
    def unregister(self, name: str) -> bool:
        """Desregistrar un manager"""
        if name in self._managers:
            manager = self._managers[name]
            manager.shutdown()
            del self._managers[name]
            return True
        
        if name in self._manager_types:
            del self._manager_types[name]
            return True
        
        return False


# Instancia global
manager_registry = ManagerRegistry()

