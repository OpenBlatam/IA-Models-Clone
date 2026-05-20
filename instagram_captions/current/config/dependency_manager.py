"""
Dependency Manager for Instagram Captions API v10.0

Dependency injection and service lifecycle management.
"""

import time
from typing import Dict, Any, Optional, Callable, Type, Union
from dataclasses import dataclass
from enum import Enum

class ServiceState(Enum):
    """Service lifecycle states."""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"

@dataclass
class ServiceInfo:
    """Service information and metadata."""
    name: str
    service_type: Type
    instance: Optional[Any] = None
    state: ServiceState = ServiceState.UNINITIALIZED
    dependencies: list = None
    init_time: Optional[float] = None
    start_time: Optional[float] = None
    stop_time: Optional[float] = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []

class DependencyManager:
    """Dependency injection and service lifecycle manager."""
    
    def __init__(self):
        self.services: Dict[str, ServiceInfo] = {}
        self.singletons: Dict[str, Any] = {}
        self.factories: Dict[str, Callable] = {}
        self.startup_order: list = []
        self.shutdown_order: list = []
    
    def register_service(self, name: str, service_type: Type, 
                        dependencies: Optional[list] = None,
                        singleton: bool = True):
        """Register a service with the dependency manager."""
        if name in self.services:
            raise ValueError(f"Service '{name}' is already registered")
        
        service_info = ServiceInfo(
            name=name,
            service_type=service_type,
            dependencies=dependencies or []
        )
        
        self.services[name] = service_info
        
        if singleton:
            # For singletons, we'll create the instance when first requested
            pass
        else:
            # For non-singletons, we'll use a factory pattern
            self.factories[name] = service_type
        
        # Add to startup order (services with no dependencies first)
        if not dependencies:
            self.startup_order.insert(0, name)
        else:
            self.startup_order.append(name)
        
        # Add to shutdown order (reverse of startup)
        self.shutdown_order.insert(0, name)
        
        return service_info
    
    def get_service(self, name: str) -> Any:
        """Get a service instance, creating it if necessary."""
        if name not in self.services:
            raise ValueError(f"Service '{name}' is not registered")
        
        service_info = self.services[name]
        
        # Check if we already have an instance
        if service_info.instance is not None:
            return service_info.instance
        
        # Create new instance
        try:
            service_info.state = ServiceState.INITIALIZING
            service_info.init_time = time.time()
            
            # Resolve dependencies first
            resolved_dependencies = []
            for dep_name in service_info.dependencies:
                if dep_name not in self.services:
                    raise ValueError(f"Dependency '{dep_name}' not found for service '{name}'")
                resolved_dependencies.append(self.get_service(dep_name))
            
            # Create instance
            if name in self.factories:
                # Use factory for non-singleton services
                instance = self.factories[name](*resolved_dependencies)
            else:
                # Create singleton instance
                if resolved_dependencies:
                    instance = service_info.service_type(*resolved_dependencies)
                else:
                    instance = service_info.service_type()
            
            service_info.instance = instance
            service_info.state = ServiceState.RUNNING
            service_info.start_time = time.time()
            
            return instance
            
        except Exception as e:
            service_info.state = ServiceState.ERROR
            service_info.error_message = str(e)
            raise RuntimeError(f"Failed to create service '{name}': {e}")
    
    def start_all_services(self) -> Dict[str, bool]:
        """Start all registered services in dependency order."""
        results = {}
        
        for service_name in self.startup_order:
            try:
                # Get service (this will create and start it)
                self.get_service(service_name)
                results[service_name] = True
            except Exception as e:
                results[service_name] = False
                print(f"Failed to start service '{service_name}': {e}")
        
        return results
    
    def stop_all_services(self) -> Dict[str, bool]:
        """Stop all registered services in reverse dependency order."""
        results = {}
        
        for service_name in self.shutdown_order:
            try:
                service_info = self.services[service_name]
                if service_info.instance is not None:
                    # Try to call stop method if it exists
                    if hasattr(service_info.instance, 'stop'):
                        service_info.instance.stop()
                    elif hasattr(service_info.instance, 'close'):
                        service_info.instance.close()
                    elif hasattr(service_info.instance, 'shutdown'):
                        service_info.instance.shutdown()
                    
                    service_info.state = ServiceState.STOPPED
                    service_info.stop_time = time.time()
                    results[service_name] = True
                else:
                    results[service_name] = True  # Not started
            except Exception as e:
                results[service_name] = False
                print(f"Failed to stop service '{service_name}': {e}")
        
        return results
    
    def get_service_status(self, name: str) -> Optional[Dict[str, Any]]:
        """Get detailed status of a service."""
        if name not in self.services:
            return None
        
        service_info = self.services[name]
        
        status = {
            'name': service_info.name,
            'state': service_info.state.value,
            'type': service_info.service_type.__name__,
            'dependencies': service_info.dependencies,
            'has_instance': service_info.instance is not None,
            'error_message': service_info.error_message
        }
        
        # Add timing information
        if service_info.init_time:
            status['init_time'] = service_info.init_time
        if service_info.start_time:
            status['start_time'] = service_info.start_time
        if service_info.stop_time:
            status['stop_time'] = service_info.stop_time
        
        # Add uptime if running
        if service_info.state == ServiceState.RUNNING and service_info.start_time:
            status['uptime_seconds'] = time.time() - service_info.start_time
        
        return status
    
    def get_all_services_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all services."""
        return {
            name: self.get_service_status(name)
            for name in self.services.keys()
        }
    
    def check_service_health(self, name: str) -> Dict[str, Any]:
        """Check health of a specific service."""
        if name not in self.services:
            return {'healthy': False, 'error': 'Service not found'}
        
        service_info = self.services[name]
        
        if service_info.state == ServiceState.ERROR:
            return {
                'healthy': False,
                'error': service_info.error_message,
                'state': service_info.state.value
            }
        
        if service_info.instance is None:
            return {
                'healthy': False,
                'error': 'Service not initialized',
                'state': service_info.state.value
            }
        
        # Try to call health check method if it exists
        if hasattr(service_info.instance, 'health_check'):
            try:
                health_result = service_info.instance.health_check()
                return {
                    'healthy': True,
                    'health_data': health_result,
                    'state': service_info.state.value
                }
            except Exception as e:
                return {
                    'healthy': False,
                    'error': f'Health check failed: {e}',
                    'state': service_info.state.value
                }
        
        # If no health check method, just return basic status
        return {
            'healthy': service_info.state == ServiceState.RUNNING,
            'state': service_info.state.value
        }
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get overall health summary of all services."""
        health_results = {}
        healthy_count = 0
        total_count = len(self.services)
        
        for service_name in self.services.keys():
            health_result = self.check_service_health(service_name)
            health_results[service_name] = health_result
            
            if health_result['healthy']:
                healthy_count += 1
        
        overall_health = healthy_count == total_count
        
        return {
            'overall_healthy': overall_health,
            'healthy_services': healthy_count,
            'total_services': total_count,
            'health_percentage': (healthy_count / total_count) * 100 if total_count > 0 else 0,
            'services': health_results
        }
    
    def reset_service(self, name: str):
        """Reset a service to uninitialized state."""
        if name not in self.services:
            raise ValueError(f"Service '{name}' is not registered")
        
        service_info = self.services[name]
        service_info.instance = None
        service_info.state = ServiceState.UNINITIALIZED
        service_info.init_time = None
        service_info.start_time = None
        service_info.stop_time = None
        service_info.error_message = None
    
    def clear_all_services(self):
        """Clear all registered services."""
        self.stop_all_services()
        self.services.clear()
        self.singletons.clear()
        self.factories.clear()
        self.startup_order.clear()
        self.shutdown_order.clear()






