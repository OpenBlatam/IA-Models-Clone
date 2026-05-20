"""
Registry System
Component and service registry for modular architecture
"""

import logging
import threading
import time
from typing import Dict, Any, List, Optional, Type, Callable, Protocol, TypeVar
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum

logger = logging.getLogger(__name__)

T = TypeVar('T')

class ComponentType(Enum):
    """Component types"""
    OPTIMIZER = "optimizer"
    MODEL = "model"
    TRAINER = "trainer"
    INFERENCER = "inferencer"
    MONITOR = "monitor"
    BENCHMARKER = "benchmarker"
    PLUGIN = "plugin"
    SERVICE = "service"

class ComponentStatus(Enum):
    """Component status"""
    UNREGISTERED = "unregistered"
    REGISTERED = "registered"
    INITIALIZED = "initialized"
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"

@dataclass
class ComponentInfo:
    """Component information"""
    name: str
    component_type: ComponentType
    class_type: Type
    instance: Optional[Any] = None
    status: ComponentStatus = ComponentStatus.UNREGISTERED
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

@dataclass
class ServiceInfo:
    """Service information"""
    name: str
    service_type: str
    instance: Any
    dependencies: List[str] = field(default_factory=list)
    health_check: Optional[Callable] = None
    status: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

class ComponentRegistry:
    """Registry for components"""
    
    def __init__(self):
        self._components: Dict[str, ComponentInfo] = {}
        self._by_type: Dict[ComponentType, List[str]] = {component_type: [] for component_type in ComponentType}
        self._lock = threading.Lock()
    
    def register(self, 
                name: str, 
                component_type: ComponentType, 
                class_type: Type,
                instance: Optional[Any] = None,
                dependencies: Optional[List[str]] = None,
                metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Register a component"""
        with self._lock:
            if name in self._components:
                logger.warning(f"Component {name} already registered")
                return False
            
            info = ComponentInfo(
                name=name,
                component_type=component_type,
                class_type=class_type,
                instance=instance,
                dependencies=dependencies or [],
                metadata=metadata or {}
            )
            
            self._components[name] = info
            self._by_type[component_type].append(name)
            
            logger.info(f"Registered component: {name} ({component_type.value})")
            return True
    
    def unregister(self, name: str) -> bool:
        """Unregister a component"""
        with self._lock:
            if name not in self._components:
                return False
            
            info = self._components[name]
            self._by_type[info.component_type].remove(name)
            del self._components[name]
            
            logger.info(f"Unregistered component: {name}")
            return True
    
    def get(self, name: str) -> Optional[ComponentInfo]:
        """Get component information"""
        with self._lock:
            return self._components.get(name)
    
    def get_instance(self, name: str) -> Optional[Any]:
        """Get component instance"""
        info = self.get(name)
        return info.instance if info else None
    
    def get_by_type(self, component_type: ComponentType) -> List[ComponentInfo]:
        """Get components by type"""
        with self._lock:
            return [self._components[name] for name in self._by_type[component_type]]
    
    def list_components(self) -> List[str]:
        """List all component names"""
        with self._lock:
            return list(self._components.keys())
    
    def list_by_type(self, component_type: ComponentType) -> List[str]:
        """List components by type"""
        with self._lock:
            return self._by_type[component_type].copy()
    
    def update_status(self, name: str, status: ComponentStatus) -> bool:
        """Update component status"""
        with self._lock:
            if name not in self._components:
                return False
            
            self._components[name].status = status
            self._components[name].updated_at = time.time()
            return True
    
    def update_metadata(self, name: str, metadata: Dict[str, Any]) -> bool:
        """Update component metadata"""
        with self._lock:
            if name not in self._components:
                return False
            
            self._components[name].metadata.update(metadata)
            self._components[name].updated_at = time.time()
            return True
    
    def get_dependencies(self, name: str) -> List[str]:
        """Get component dependencies"""
        info = self.get(name)
        return info.dependencies if info else []
    
    def check_dependencies(self, name: str) -> bool:
        """Check if all dependencies are available"""
        dependencies = self.get_dependencies(name)
        return all(dep in self._components for dep in dependencies)
    
    def get_status(self, name: str) -> Optional[ComponentStatus]:
        """Get component status"""
        info = self.get(name)
        return info.status if info else None
    
    def get_healthy_components(self) -> List[ComponentInfo]:
        """Get healthy components"""
        with self._lock:
            return [info for info in self._components.values() 
                   if info.status in [ComponentStatus.INITIALIZED, ComponentStatus.RUNNING]]
    
    def get_components_by_status(self, status: ComponentStatus) -> List[ComponentInfo]:
        """Get components by status"""
        with self._lock:
            return [info for info in self._components.values() if info.status == status]

class ServiceRegistry:
    """Registry for services"""
    
    def __init__(self):
        self._services: Dict[str, ServiceInfo] = {}
        self._by_type: Dict[str, List[str]] = {}
        self._lock = threading.Lock()
    
    def register(self, 
                name: str, 
                service_type: str, 
                instance: Any,
                dependencies: Optional[List[str]] = None,
                health_check: Optional[Callable] = None,
                metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Register a service"""
        with self._lock:
            if name in self._services:
                logger.warning(f"Service {name} already registered")
                return False
            
            info = ServiceInfo(
                name=name,
                service_type=service_type,
                instance=instance,
                dependencies=dependencies or [],
                health_check=health_check,
                metadata=metadata or {}
            )
            
            self._services[name] = info
            
            if service_type not in self._by_type:
                self._by_type[service_type] = []
            self._by_type[service_type].append(name)
            
            logger.info(f"Registered service: {name} ({service_type})")
            return True
    
    def unregister(self, name: str) -> bool:
        """Unregister a service"""
        with self._lock:
            if name not in self._services:
                return False
            
            info = self._services[name]
            self._by_type[info.service_type].remove(name)
            del self._services[name]
            
            logger.info(f"Unregistered service: {name}")
            return True
    
    def get(self, name: str) -> Optional[ServiceInfo]:
        """Get service information"""
        with self._lock:
            return self._services.get(name)
    
    def get_instance(self, name: str) -> Optional[Any]:
        """Get service instance"""
        info = self.get(name)
        return info.instance if info else None
    
    def get_by_type(self, service_type: str) -> List[ServiceInfo]:
        """Get services by type"""
        with self._lock:
            return [self._services[name] for name in self._by_type.get(service_type, [])]
    
    def list_services(self) -> List[str]:
        """List all service names"""
        with self._lock:
            return list(self._services.keys())
    
    def list_by_type(self, service_type: str) -> List[str]:
        """List services by type"""
        with self._lock:
            return self._by_type.get(service_type, []).copy()
    
    def check_health(self, name: str) -> bool:
        """Check service health"""
        info = self.get(name)
        if not info or not info.health_check:
            return False
        
        try:
            return info.health_check()
        except Exception as e:
            logger.error(f"Health check failed for {name}: {e}")
            return False
    
    def get_healthy_services(self) -> List[ServiceInfo]:
        """Get healthy services"""
        with self._lock:
            healthy = []
            for info in self._services.values():
                if self.check_health(info.name):
                    healthy.append(info)
            return healthy
    
    def update_status(self, name: str, status: str) -> bool:
        """Update service status"""
        with self._lock:
            if name not in self._services:
                return False
            
            self._services[name].status = status
            return True

class RegistryManager:
    """Manager for all registries"""
    
    def __init__(self):
        self.component_registry = ComponentRegistry()
        self.service_registry = ServiceRegistry()
        self._watchers: Dict[str, List[Callable]] = {}
    
    def register_component(self, 
                          name: str, 
                          component_type: ComponentType, 
                          class_type: Type,
                          **kwargs) -> bool:
        """Register a component"""
        return self.component_registry.register(name, component_type, class_type, **kwargs)
    
    def register_service(self, 
                        name: str, 
                        service_type: str, 
                        instance: Any,
                        **kwargs) -> bool:
        """Register a service"""
        return self.service_registry.register(name, service_type, instance, **kwargs)
    
    def get_component(self, name: str) -> Optional[Any]:
        """Get component instance"""
        return self.component_registry.get_instance(name)
    
    def get_service(self, name: str) -> Optional[Any]:
        """Get service instance"""
        return self.service_registry.get_instance(name)
    
    def get_components_by_type(self, component_type: ComponentType) -> List[Any]:
        """Get components by type"""
        return [info.instance for info in self.component_registry.get_by_type(component_type) 
                if info.instance is not None]
    
    def get_services_by_type(self, service_type: str) -> List[Any]:
        """Get services by type"""
        return [info.instance for info in self.service_registry.get_by_type(service_type)]
    
    def add_watcher(self, event: str, callback: Callable) -> None:
        """Add registry watcher"""
        if event not in self._watchers:
            self._watchers[event] = []
        self._watchers[event].append(callback)
    
    def _notify_watchers(self, event: str, data: Any) -> None:
        """Notify watchers"""
        for callback in self._watchers.get(event, []):
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Watcher callback error: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get registry status"""
        return {
            "components": {
                "total": len(self.component_registry.list_components()),
                "by_type": {
                    component_type.value: len(self.component_registry.list_by_type(component_type))
                    for component_type in ComponentType
                },
                "healthy": len(self.component_registry.get_healthy_components())
            },
            "services": {
                "total": len(self.service_registry.list_services()),
                "by_type": {
                    service_type: len(self.service_registry.list_by_type(service_type))
                    for service_type in self.service_registry._by_type.keys()
                },
                "healthy": len(self.service_registry.get_healthy_services())
            }
        }

class RegistryBuilder:
    """Builder for registry setup"""
    
    def __init__(self, manager: RegistryManager):
        self.manager = manager
        self.components: List[Dict[str, Any]] = []
        self.services: List[Dict[str, Any]] = []
    
    def add_component(self, 
                     name: str, 
                     component_type: ComponentType, 
                     class_type: Type,
                     **kwargs) -> 'RegistryBuilder':
        """Add component to registry"""
        self.components.append({
            "name": name,
            "component_type": component_type,
            "class_type": class_type,
            **kwargs
        })
        return self
    
    def add_service(self, 
                   name: str, 
                   service_type: str, 
                   instance: Any,
                   **kwargs) -> 'RegistryBuilder':
        """Add service to registry"""
        self.services.append({
            "name": name,
            "service_type": service_type,
            "instance": instance,
            **kwargs
        })
        return self
    
    def build(self) -> bool:
        """Build registry"""
        success = True
        
        # Register components
        for component in self.components:
            if not self.manager.register_component(**component):
                success = False
        
        # Register services
        for service in self.services:
            if not self.manager.register_service(**service):
                success = False
        
        return success

class RegistryValidator:
    """Validator for registry consistency"""
    
    def __init__(self, manager: RegistryManager):
        self.manager = manager
    
    def validate_dependencies(self) -> Dict[str, List[str]]:
        """Validate component dependencies"""
        issues = {}
        
        for name in self.manager.component_registry.list_components():
            component_info = self.manager.component_registry.get(name)
            if component_info:
                missing_deps = []
                for dep in component_info.dependencies:
                    if dep not in self.manager.component_registry.list_components():
                        missing_deps.append(dep)
                
                if missing_deps:
                    issues[name] = missing_deps
        
        return issues
    
    def validate_services(self) -> Dict[str, List[str]]:
        """Validate service health"""
        issues = {}
        
        for name in self.manager.service_registry.list_services():
            if not self.manager.service_registry.check_health(name):
                issues[name] = ["Health check failed"]
        
        return issues
    
    def validate_all(self) -> Dict[str, Any]:
        """Validate entire registry"""
        return {
            "dependency_issues": self.validate_dependencies(),
            "service_issues": self.validate_services(),
            "status": self.manager.get_status()
        }

