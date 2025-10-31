"""
Component Registry - Registro de Componentes
==========================================

Sistema de registro y gestión de componentes intercambiables con inyección de dependencias.
"""

import asyncio
import inspect
import logging
from typing import Dict, List, Any, Optional, Type, Callable, Set, Union
from datetime import datetime
from enum import Enum

from ..interfaces.base_interfaces import IComponent, IRegistry, IFactory

logger = logging.getLogger(__name__)


class ComponentScope(Enum):
    """Alcance de componentes."""
    SINGLETON = "singleton"
    TRANSIENT = "transient"
    SCOPED = "scoped"


class ComponentInfo:
    """Información de un componente."""
    
    def __init__(self, name: str, component_type: Type, implementation: Type = None,
                 scope: ComponentScope = ComponentScope.TRANSIENT, 
                 dependencies: List[str] = None, metadata: Dict[str, Any] = None):
        self.name = name
        self.component_type = component_type
        self.implementation = implementation or component_type
        self.scope = scope
        self.dependencies = dependencies or []
        self.metadata = metadata or {}
        self.registered_at = datetime.utcnow()
        self.instance = None
        self.instance_count = 0


class ComponentRegistry(IRegistry):
    """Registro de componentes."""
    
    def __init__(self):
        self._components: Dict[str, ComponentInfo] = {}
        self._instances: Dict[str, Any] = {}
        self._singletons: Dict[str, Any] = {}
        self._scoped_instances: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        self._current_scope = None
    
    async def register(self, name: str, component: Any) -> bool:
        """Registrar componente."""
        try:
            async with self._lock:
                if isinstance(component, type):
                    # Registrar tipo de componente
                    component_info = ComponentInfo(
                        name=name,
                        component_type=component,
                        implementation=component
                    )
                    self._components[name] = component_info
                    return True
                else:
                    # Registrar instancia
                    self._instances[name] = component
                    return True
        except Exception as e:
            logger.error(f"Error registering component {name}: {e}")
            return False
    
    async def unregister(self, name: str) -> bool:
        """Desregistrar componente."""
        try:
            async with self._lock:
                if name in self._components:
                    # Limpiar instancias relacionadas
                    if name in self._singletons:
                        del self._singletons[name]
                    
                    if name in self._scoped_instances:
                        del self._scoped_instances[name]
                    
                    del self._components[name]
                    return True
                
                if name in self._instances:
                    del self._instances[name]
                    return True
                
                return False
        except Exception as e:
            logger.error(f"Error unregistering component {name}: {e}")
            return False
    
    async def get(self, name: str) -> Optional[Any]:
        """Obtener componente."""
        try:
            # Verificar instancias directas primero
            if name in self._instances:
                return self._instances[name]
            
            # Verificar componentes registrados
            if name in self._components:
                component_info = self._components[name]
                return await self._create_instance(component_info)
            
            return None
        except Exception as e:
            logger.error(f"Error getting component {name}: {e}")
            return None
    
    async def list_all(self) -> List[str]:
        """Listar todos los componentes."""
        all_components = set(self._components.keys())
        all_components.update(self._instances.keys())
        return list(all_components)
    
    async def _create_instance(self, component_info: ComponentInfo) -> Any:
        """Crear instancia de componente."""
        try:
            # Verificar alcance
            if component_info.scope == ComponentScope.SINGLETON:
                if component_info.name in self._singletons:
                    return self._singletons[component_info.name]
            
            elif component_info.scope == ComponentScope.SCOPED:
                if self._current_scope:
                    if self._current_scope not in self._scoped_instances:
                        self._scoped_instances[self._current_scope] = {}
                    
                    if component_info.name in self._scoped_instances[self._current_scope]:
                        return self._scoped_instances[self._current_scope][component_info.name]
            
            # Crear nueva instancia
            instance = await self._instantiate_component(component_info)
            
            # Almacenar según alcance
            if component_info.scope == ComponentScope.SINGLETON:
                self._singletons[component_info.name] = instance
            elif component_info.scope == ComponentScope.SCOPED and self._current_scope:
                if self._current_scope not in self._scoped_instances:
                    self._scoped_instances[self._current_scope] = {}
                self._scoped_instances[self._current_scope][component_info.name] = instance
            
            component_info.instance_count += 1
            return instance
            
        except Exception as e:
            logger.error(f"Error creating instance for component {component_info.name}: {e}")
            raise
    
    async def _instantiate_component(self, component_info: ComponentInfo) -> Any:
        """Instanciar componente con inyección de dependencias."""
        try:
            # Resolver dependencias
            dependencies = await self._resolve_dependencies(component_info.dependencies)
            
            # Crear instancia
            if dependencies:
                instance = component_info.implementation(*dependencies)
            else:
                instance = component_info.implementation()
            
            # Inicializar si es un componente
            if isinstance(instance, IComponent):
                await instance.initialize()
            
            return instance
            
        except Exception as e:
            logger.error(f"Error instantiating component {component_info.name}: {e}")
            raise
    
    async def _resolve_dependencies(self, dependencies: List[str]) -> List[Any]:
        """Resolver dependencias."""
        resolved_deps = []
        
        for dep_name in dependencies:
            dep_instance = await self.get(dep_name)
            if dep_instance is None:
                raise ValueError(f"Dependency {dep_name} not found")
            resolved_deps.append(dep_instance)
        
        return resolved_deps


class ComponentManager(IComponent):
    """Gestor principal de componentes."""
    
    def __init__(self):
        self.registry = ComponentRegistry()
        self._initialized = False
        self._component_stats: Dict[str, Dict[str, Any]] = {}
    
    async def initialize(self) -> None:
        """Inicializar gestor de componentes."""
        try:
            self._initialized = True
            logger.info("Component manager initialized")
            
        except Exception as e:
            logger.error(f"Error initializing component manager: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Cerrar gestor de componentes."""
        try:
            # Cerrar todos los componentes singleton
            for name, instance in self.registry._singletons.items():
                if isinstance(instance, IComponent):
                    try:
                        await instance.shutdown()
                    except Exception as e:
                        logger.error(f"Error shutting down singleton component {name}: {e}")
            
            # Cerrar instancias directas
            for name, instance in self.registry._instances.items():
                if isinstance(instance, IComponent):
                    try:
                        await instance.shutdown()
                    except Exception as e:
                        logger.error(f"Error shutting down component {name}: {e}")
            
            self._initialized = False
            logger.info("Component manager shutdown complete")
            
        except Exception as e:
            logger.error(f"Error shutting down component manager: {e}")
    
    async def health_check(self) -> bool:
        """Verificar salud del gestor."""
        return self._initialized
    
    @property
    def name(self) -> str:
        return "ComponentManager"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    async def register_component(self, name: str, component_type: Type, 
                               implementation: Type = None, scope: ComponentScope = ComponentScope.TRANSIENT,
                               dependencies: List[str] = None, metadata: Dict[str, Any] = None) -> bool:
        """Registrar componente."""
        try:
            component_info = ComponentInfo(
                name=name,
                component_type=component_type,
                implementation=implementation,
                scope=scope,
                dependencies=dependencies or [],
                metadata=metadata or {}
            )
            
            self.registry._components[name] = component_info
            logger.info(f"Registered component {name} with scope {scope.value}")
            return True
            
        except Exception as e:
            logger.error(f"Error registering component {name}: {e}")
            return False
    
    async def register_instance(self, name: str, instance: Any) -> bool:
        """Registrar instancia de componente."""
        try:
            self.registry._instances[name] = instance
            logger.info(f"Registered component instance {name}")
            return True
            
        except Exception as e:
            logger.error(f"Error registering component instance {name}: {e}")
            return False
    
    async def get_component(self, name: str) -> Optional[Any]:
        """Obtener componente."""
        return await self.registry.get(name)
    
    async def unregister_component(self, name: str) -> bool:
        """Desregistrar componente."""
        return await self.registry.unregister(name)
    
    async def list_components(self) -> List[str]:
        """Listar componentes."""
        return await self.registry.list_all()
    
    async def get_component_info(self, name: str) -> Optional[ComponentInfo]:
        """Obtener información de componente."""
        return self.registry._components.get(name)
    
    async def set_scope(self, scope_id: str) -> None:
        """Establecer scope actual."""
        self.registry._current_scope = scope_id
    
    async def clear_scope(self, scope_id: str) -> None:
        """Limpiar scope."""
        if scope_id in self.registry._scoped_instances:
            # Cerrar componentes scoped
            for name, instance in self.registry._scoped_instances[scope_id].items():
                if isinstance(instance, IComponent):
                    try:
                        await instance.shutdown()
                    except Exception as e:
                        logger.error(f"Error shutting down scoped component {name}: {e}")
            
            del self.registry._scoped_instances[scope_id]
    
    async def get_component_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de componentes."""
        try:
            stats = {
                "total_components": len(self.registry._components),
                "total_instances": len(self.registry._instances),
                "singleton_count": len(self.registry._singletons),
                "scoped_scope_count": len(self.registry._scoped_instances),
                "component_details": {}
            }
            
            for name, component_info in self.registry._components.items():
                stats["component_details"][name] = {
                    "type": component_info.component_type.__name__,
                    "scope": component_info.scope.value,
                    "dependencies": component_info.dependencies,
                    "instance_count": component_info.instance_count,
                    "registered_at": component_info.registered_at.isoformat()
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting component stats: {e}")
            return {}
    
    async def validate_dependencies(self, component_name: str) -> Dict[str, Any]:
        """Validar dependencias de un componente."""
        try:
            component_info = self.registry._components.get(component_name)
            if not component_info:
                return {"valid": False, "error": "Component not found"}
            
            validation_results = []
            all_dependencies_resolved = True
            
            for dep_name in component_info.dependencies:
                dep_component = await self.registry.get(dep_name)
                if dep_component is None:
                    validation_results.append({
                        "dependency": dep_name,
                        "resolved": False,
                        "error": "Dependency not found"
                    })
                    all_dependencies_resolved = False
                else:
                    validation_results.append({
                        "dependency": dep_name,
                        "resolved": True
                    })
            
            return {
                "valid": all_dependencies_resolved,
                "component_name": component_name,
                "dependencies": component_info.dependencies,
                "validation_results": validation_results
            }
            
        except Exception as e:
            logger.error(f"Error validating dependencies for {component_name}: {e}")
            return {"valid": False, "error": str(e)}
    
    async def get_dependency_graph(self) -> Dict[str, Any]:
        """Obtener grafo de dependencias."""
        try:
            graph = {
                "nodes": [],
                "edges": []
            }
            
            # Agregar nodos
            for name, component_info in self.registry._components.items():
                graph["nodes"].append({
                    "id": name,
                    "type": component_info.component_type.__name__,
                    "scope": component_info.scope.value
                })
            
            # Agregar aristas
            for name, component_info in self.registry._components.items():
                for dep_name in component_info.dependencies:
                    graph["edges"].append({
                        "from": name,
                        "to": dep_name
                    })
            
            return graph
            
        except Exception as e:
            logger.error(f"Error getting dependency graph: {e}")
            return {"nodes": [], "edges": []}
    
    async def detect_circular_dependencies(self) -> List[List[str]]:
        """Detectar dependencias circulares."""
        try:
            def dfs(node, path, visited, rec_stack):
                visited.add(node)
                rec_stack.add(node)
                path.append(node)
                
                component_info = self.registry._components.get(node)
                if component_info:
                    for dep in component_info.dependencies:
                        if dep not in visited:
                            result = dfs(dep, path.copy(), visited, rec_stack)
                            if result:
                                return result
                        elif dep in rec_stack:
                            # Dependencia circular encontrada
                            cycle_start = path.index(dep)
                            return path[cycle_start:] + [dep]
                
                rec_stack.remove(node)
                return None
            
            visited = set()
            cycles = []
            
            for component_name in self.registry._components.keys():
                if component_name not in visited:
                    cycle = dfs(component_name, [], visited, set())
                    if cycle:
                        cycles.append(cycle)
            
            return cycles
            
        except Exception as e:
            logger.error(f"Error detecting circular dependencies: {e}")
            return []
    
    async def auto_register_components(self, module) -> int:
        """Registrar automáticamente componentes de un módulo."""
        try:
            registered_count = 0
            
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, IComponent) and 
                    obj != IComponent):
                    
                    # Determinar scope basado en metadatos
                    scope = ComponentScope.TRANSIENT
                    if hasattr(obj, '_component_scope'):
                        scope = obj._component_scope
                    
                    # Obtener dependencias
                    dependencies = []
                    if hasattr(obj, '_component_dependencies'):
                        dependencies = obj._component_dependencies
                    
                    # Registrar componente
                    if await self.register_component(name, obj, scope=scope, dependencies=dependencies):
                        registered_count += 1
            
            logger.info(f"Auto-registered {registered_count} components from module")
            return registered_count
            
        except Exception as e:
            logger.error(f"Error auto-registering components: {e}")
            return 0
    
    async def create_component_factory(self, component_name: str) -> Optional[IFactory]:
        """Crear factory para un componente."""
        try:
            component_info = self.registry._components.get(component_name)
            if not component_info:
                return None
            
            class ComponentFactory(IFactory):
                def __init__(self, component_info: ComponentInfo, registry: ComponentRegistry):
                    self.component_info = component_info
                    self.registry = registry
                
                async def create(self, type_name: str, **kwargs) -> Any:
                    return await self.registry._create_instance(self.component_info)
                
                async def register_type(self, type_name: str, type_class: type) -> None:
                    pass  # No implementado para factory específico
            
            return ComponentFactory(component_info, self.registry)
            
        except Exception as e:
            logger.error(f"Error creating component factory for {component_name}: {e}")
            return None




