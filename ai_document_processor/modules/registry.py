"""
Component Registry - Service and Component Management
==================================================

Ultra-modular registry system for managing components and services dynamically.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Type, TypeVar, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid
import inspect

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ComponentStatus(str, Enum):
    """Component status enumeration."""
    REGISTERED = "registered"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    DEPRECATED = "deprecated"


class ServiceType(str, Enum):
    """Service type enumeration."""
    DOCUMENT_PROCESSOR = "document_processor"
    AI_SERVICE = "ai_service"
    TRANSFORM_SERVICE = "transform_service"
    VALIDATION_SERVICE = "validation_service"
    CACHE_SERVICE = "cache_service"
    FILE_SERVICE = "file_service"
    NOTIFICATION_SERVICE = "notification_service"
    METRICS_SERVICE = "metrics_service"
    PLUGIN_SERVICE = "plugin_service"
    CUSTOM = "custom"


@dataclass
class ComponentMetadata:
    """Component metadata."""
    name: str
    version: str
    description: str
    author: str
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    health_check_url: Optional[str] = None
    documentation_url: Optional[str] = None
    configuration_schema: Optional[Dict[str, Any]] = None


@dataclass
class ComponentInfo:
    """Component information."""
    id: str
    name: str
    component_type: str
    service_type: ServiceType
    instance: Any
    metadata: ComponentMetadata
    status: ComponentStatus = ComponentStatus.REGISTERED
    health_score: float = 1.0
    last_health_check: Optional[datetime] = None
    error_count: int = 0
    last_error: Optional[str] = None
    usage_count: int = 0
    last_used: Optional[datetime] = None


class ComponentRegistry:
    """Ultra-modular component registry."""
    
    def __init__(self):
        self._components: Dict[str, ComponentInfo] = {}
        self._components_by_type: Dict[str, List[str]] = {}
        self._components_by_service_type: Dict[ServiceType, List[str]] = {}
        self._component_instances: Dict[str, Any] = {}
        self._health_check_interval = 30  # seconds
        self._health_check_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
    
    async def register_component(
        self,
        name: str,
        component_type: str,
        service_type: ServiceType,
        instance: Any,
        metadata: ComponentMetadata,
        auto_start: bool = True
    ) -> str:
        """
        Register a component.
        
        Args:
            name: Component name
            component_type: Component type/class name
            service_type: Service type
            instance: Component instance
            metadata: Component metadata
            auto_start: Whether to auto-start the component
            
        Returns:
            Component ID
        """
        async with self._lock:
            component_id = str(uuid.uuid4())
            
            # Create component info
            component_info = ComponentInfo(
                id=component_id,
                name=name,
                component_type=component_type,
                service_type=service_type,
                instance=instance,
                metadata=metadata,
                status=ComponentStatus.REGISTERED
            )
            
            # Store component
            self._components[component_id] = component_info
            self._component_instances[component_id] = instance
            
            # Index by type
            if component_type not in self._components_by_type:
                self._components_by_type[component_type] = []
            self._components_by_type[component_type].append(component_id)
            
            # Index by service type
            if service_type not in self._components_by_service_type:
                self._components_by_service_type[service_type] = []
            self._components_by_service_type[service_type].append(component_id)
            
            # Auto-start if requested
            if auto_start:
                await self.start_component(component_id)
            
            logger.info(f"Registered component: {name} ({component_id})")
            return component_id
    
    async def unregister_component(self, component_id: str) -> bool:
        """
        Unregister a component.
        
        Args:
            component_id: Component ID
            
        Returns:
            True if unregistered, False if not found
        """
        async with self._lock:
            if component_id not in self._components:
                return False
            
            component_info = self._components[component_id]
            
            # Stop component if active
            if component_info.status == ComponentStatus.ACTIVE:
                await self.stop_component(component_id)
            
            # Remove from indexes
            if component_info.component_type in self._components_by_type:
                self._components_by_type[component_info.component_type].remove(component_id)
            
            if component_info.service_type in self._components_by_service_type:
                self._components_by_service_type[component_info.service_type].remove(component_id)
            
            # Remove from storage
            del self._components[component_id]
            del self._component_instances[component_id]
            
            logger.info(f"Unregistered component: {component_info.name} ({component_id})")
            return True
    
    async def get_component(self, component_id: str) -> Optional[ComponentInfo]:
        """Get component by ID."""
        return self._components.get(component_id)
    
    async def get_component_instance(self, component_id: str) -> Optional[Any]:
        """Get component instance by ID."""
        return self._component_instances.get(component_id)
    
    async def get_components_by_type(self, component_type: str) -> List[ComponentInfo]:
        """Get components by type."""
        component_ids = self._components_by_type.get(component_type, [])
        return [self._components[cid] for cid in component_ids if cid in self._components]
    
    async def get_components_by_service_type(self, service_type: ServiceType) -> List[ComponentInfo]:
        """Get components by service type."""
        component_ids = self._components_by_service_type.get(service_type, [])
        return [self._components[cid] for cid in component_ids if cid in self._components]
    
    async def get_active_components(self) -> List[ComponentInfo]:
        """Get all active components."""
        return [c for c in self._components.values() if c.status == ComponentStatus.ACTIVE]
    
    async def start_component(self, component_id: str) -> bool:
        """
        Start a component.
        
        Args:
            component_id: Component ID
            
        Returns:
            True if started, False if not found or already active
        """
        async with self._lock:
            if component_id not in self._components:
                return False
            
            component_info = self._components[component_id]
            
            if component_info.status == ComponentStatus.ACTIVE:
                return True
            
            try:
                # Try to start the component
                if hasattr(component_info.instance, 'start'):
                    await component_info.instance.start()
                elif hasattr(component_info.instance, 'initialize'):
                    await component_info.instance.initialize()
                
                component_info.status = ComponentStatus.ACTIVE
                component_info.last_health_check = datetime.utcnow()
                component_info.health_score = 1.0
                
                logger.info(f"Started component: {component_info.name} ({component_id})")
                return True
                
            except Exception as e:
                component_info.status = ComponentStatus.ERROR
                component_info.last_error = str(e)
                component_info.error_count += 1
                
                logger.error(f"Failed to start component {component_info.name}: {e}")
                return False
    
    async def stop_component(self, component_id: str) -> bool:
        """
        Stop a component.
        
        Args:
            component_id: Component ID
            
        Returns:
            True if stopped, False if not found
        """
        async with self._lock:
            if component_id not in self._components:
                return False
            
            component_info = self._components[component_id]
            
            try:
                # Try to stop the component
                if hasattr(component_info.instance, 'stop'):
                    await component_info.instance.stop()
                elif hasattr(component_info.instance, 'cleanup'):
                    await component_info.instance.cleanup()
                
                component_info.status = ComponentStatus.INACTIVE
                
                logger.info(f"Stopped component: {component_info.name} ({component_id})")
                return True
                
            except Exception as e:
                component_info.status = ComponentStatus.ERROR
                component_info.last_error = str(e)
                component_info.error_count += 1
                
                logger.error(f"Failed to stop component {component_info.name}: {e}")
                return False
    
    async def health_check_component(self, component_id: str) -> float:
        """
        Perform health check on a component.
        
        Args:
            component_id: Component ID
            
        Returns:
            Health score (0.0 to 1.0)
        """
        if component_id not in self._components:
            return 0.0
        
        component_info = self._components[component_id]
        
        try:
            # Try to perform health check
            if hasattr(component_info.instance, 'health_check'):
                health_score = await component_info.instance.health_check()
            elif hasattr(component_info.instance, 'is_healthy'):
                health_score = 1.0 if await component_info.instance.is_healthy() else 0.0
            else:
                # Default health check - just check if instance exists
                health_score = 1.0 if component_info.instance is not None else 0.0
            
            # Update component info
            component_info.health_score = max(0.0, min(1.0, health_score))
            component_info.last_health_check = datetime.utcnow()
            
            if health_score < 0.5:
                component_info.status = ComponentStatus.ERROR
                component_info.error_count += 1
            elif component_info.status == ComponentStatus.ERROR and health_score >= 0.8:
                component_info.status = ComponentStatus.ACTIVE
            
            return health_score
            
        except Exception as e:
            component_info.health_score = 0.0
            component_info.status = ComponentStatus.ERROR
            component_info.last_error = str(e)
            component_info.error_count += 1
            
            logger.error(f"Health check failed for component {component_info.name}: {e}")
            return 0.0
    
    async def start_health_monitoring(self):
        """Start health monitoring for all components."""
        if self._health_check_task and not self._health_check_task.done():
            return
        
        self._health_check_task = asyncio.create_task(self._health_monitoring_loop())
        logger.info("Started component health monitoring")
    
    async def stop_health_monitoring(self):
        """Stop health monitoring."""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            self._health_check_task = None
            logger.info("Stopped component health monitoring")
    
    async def _health_monitoring_loop(self):
        """Health monitoring loop."""
        while True:
            try:
                await asyncio.sleep(self._health_check_interval)
                
                # Check all active components
                for component_id in list(self._components.keys()):
                    component_info = self._components[component_id]
                    if component_info.status == ComponentStatus.ACTIVE:
                        await self.health_check_component(component_id)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
    
    async def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        total_components = len(self._components)
        active_components = len([c for c in self._components.values() if c.status == ComponentStatus.ACTIVE])
        error_components = len([c for c in self._components.values() if c.status == ComponentStatus.ERROR])
        
        service_type_counts = {}
        for service_type in ServiceType:
            service_type_counts[service_type.value] = len(
                self._components_by_service_type.get(service_type, [])
            )
        
        return {
            'total_components': total_components,
            'active_components': active_components,
            'error_components': error_components,
            'service_type_counts': service_type_counts,
            'health_check_interval': self._health_check_interval,
            'health_monitoring_active': self._health_check_task is not None and not self._health_check_task.done()
        }


class ServiceRegistry:
    """Service registry for microservices."""
    
    def __init__(self):
        self._services: Dict[str, Dict[str, Any]] = {}
        self._service_instances: Dict[str, Any] = {}
        self._service_health: Dict[str, float] = {}
        self._lock = asyncio.Lock()
    
    async def register_service(
        self,
        service_name: str,
        service_instance: Any,
        endpoints: List[str],
        metadata: Dict[str, Any]
    ) -> str:
        """Register a microservice."""
        async with self._lock:
            service_id = str(uuid.uuid4())
            
            self._services[service_id] = {
                'name': service_name,
                'endpoints': endpoints,
                'metadata': metadata,
                'registered_at': datetime.utcnow(),
                'status': 'active'
            }
            
            self._service_instances[service_id] = service_instance
            self._service_health[service_id] = 1.0
            
            logger.info(f"Registered service: {service_name} ({service_id})")
            return service_id
    
    async def unregister_service(self, service_id: str) -> bool:
        """Unregister a microservice."""
        async with self._lock:
            if service_id not in self._services:
                return False
            
            service_name = self._services[service_id]['name']
            del self._services[service_id]
            del self._service_instances[service_id]
            del self._service_health[service_id]
            
            logger.info(f"Unregistered service: {service_name} ({service_id})")
            return True
    
    async def get_service(self, service_id: str) -> Optional[Dict[str, Any]]:
        """Get service by ID."""
        return self._services.get(service_id)
    
    async def get_service_instance(self, service_id: str) -> Optional[Any]:
        """Get service instance by ID."""
        return self._service_instances.get(service_id)
    
    async def find_service_by_name(self, service_name: str) -> Optional[str]:
        """Find service ID by name."""
        for service_id, service_info in self._services.items():
            if service_info['name'] == service_name:
                return service_id
        return None
    
    async def get_all_services(self) -> Dict[str, Dict[str, Any]]:
        """Get all registered services."""
        return self._services.copy()
    
    async def update_service_health(self, service_id: str, health_score: float):
        """Update service health score."""
        if service_id in self._service_health:
            self._service_health[service_id] = max(0.0, min(1.0, health_score))
    
    async def get_healthy_services(self) -> List[str]:
        """Get list of healthy service IDs."""
        return [
            service_id for service_id, health in self._service_health.items()
            if health >= 0.5
        ]


# Global registry instances
_component_registry: Optional[ComponentRegistry] = None
_service_registry: Optional[ServiceRegistry] = None


def get_component_registry() -> ComponentRegistry:
    """Get global component registry instance."""
    global _component_registry
    if _component_registry is None:
        _component_registry = ComponentRegistry()
    return _component_registry


def get_service_registry() -> ServiceRegistry:
    """Get global service registry instance."""
    global _service_registry
    if _service_registry is None:
        _service_registry = ServiceRegistry()
    return _service_registry

















