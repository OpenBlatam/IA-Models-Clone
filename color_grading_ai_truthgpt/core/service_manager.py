"""
Service Manager for Color Grading AI
=====================================

Unified service management combining factory, initializer, and integration.
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path

from .service_factory_refactored import RefactoredServiceFactory
from .service_initializer import ServiceInitializer, ServiceDependency, InitializationPhase
from .service_integration import ServiceIntegration
from .error_handler import ErrorHandler
from .context_manager import ContextManager
from .validation_framework import ValidationFramework, create_color_params_schema
from .performance_tracker import PerformanceTracker
from .dependency_injection import DependencyInjector, ServiceRegistry, ServiceScope
from .service_lifecycle import ServiceLifecycleManager
from ..config.color_grading_config import ColorGradingConfig

logger = logging.getLogger(__name__)


class ServiceManager:
    """
    Unified service manager.
    
    Combines:
    - Service Factory (creation)
    - Service Initializer (initialization)
    - Service Integration (wiring)
    """
    
    def __init__(
        self,
        config: ColorGradingConfig,
        output_dirs: Dict[str, Path]
    ):
        """
        Initialize service manager.
        
        Args:
            config: Configuration
            output_dirs: Output directories
        """
        self.config = config
        self.output_dirs = output_dirs
        
        # Components
        self.factory = RefactoredServiceFactory(config, output_dirs)
        self.initializer = ServiceInitializer()
        self.integration = ServiceIntegration()
        self.error_handler = ErrorHandler()
        self.context_manager = ContextManager()
        self.validation_framework = ValidationFramework()
        self.performance_tracker = PerformanceTracker()
        
        # Dependency Injection & Lifecycle
        self.injector = DependencyInjector()
        self.registry = ServiceRegistry(self.injector)
        self.lifecycle = ServiceLifecycleManager()
        
        # Setup predefined schemas
        create_color_params_schema(self.validation_framework)
        
        # Services
        self._services: Dict[str, Any] = {}
        self._initialized = False
    
    def initialize_all(self) -> Dict[str, Any]:
        """
        Initialize all services with full lifecycle.
        
        Returns:
            Dictionary of all services
        """
        if self._initialized:
            return self._services
        
        # 1. Create services via factory
        logger.info("Creating services via factory...")
        self._services = self.factory.initialize_all()
        
        # 2. Register services for integration
        self.integration.register_services(self._services)
        
        # 3. Wire integrations
        logger.info("Wiring service integrations...")
        self._wire_integrations()
        
        # 4. Register services in DI container
        logger.info("Registering services in DI container...")
        self._register_services_in_di()
        
        # 5. Register services in lifecycle manager
        logger.info("Registering services in lifecycle manager...")
        self._register_services_in_lifecycle()
        
        # 6. Setup cross-cutting concerns
        logger.info("Setting up cross-cutting concerns...")
        self._setup_cross_cutting()
        
        # 7. Inject cross-cutting services
        logger.info("Injecting cross-cutting services...")
        self._inject_cross_cutting()
        
        self._initialized = True
        logger.info(f"Service manager initialized with {len(self._services)} services")
        
        return self._services
    
    def _wire_integrations(self):
        """Wire service integrations."""
        # Wire event bus
        event_bus = self._services.get("event_bus")
        if event_bus:
            self.integration.wire_event_bus(event_bus, self._services)
        
        # Wire metrics
        metrics_collector = self._services.get("metrics_collector")
        if metrics_collector:
            self.integration.setup_metrics_collection(metrics_collector, self._services)
        
        # Wire health checks
        health_monitor = self._services.get("health_monitor")
        if health_monitor:
            self.integration.setup_health_checks(health_monitor, self._services)
    
    def _setup_cross_cutting(self):
        """Setup cross-cutting concerns."""
        # Setup audit logging for compliance
        audit_logger = self._services.get("audit_logger")
        compliance_manager = self._services.get("compliance_manager")
        
        if audit_logger and compliance_manager:
            # Connect compliance events to audit
            pass
        
        # Setup telemetry
        telemetry = self._services.get("telemetry_service")
        if telemetry:
            # Setup telemetry for key services
            pass
    
    def _register_services_in_di(self):
        """Register services in dependency injection container."""
        for name, service in self._services.items():
            service_type = type(service)
            self.injector.register_instance(name, service)
            logger.debug(f"Registered {name} in DI container")
    
    def _register_services_in_lifecycle(self):
        """Register services in lifecycle manager."""
        for name, service in self._services.items():
            self.lifecycle.register_service(name, service)
            logger.debug(f"Registered {name} in lifecycle manager")
    
    def _inject_cross_cutting(self):
        """Inject cross-cutting services into other services."""
        # Inject error handler
        for service_name, service in self._services.items():
            if hasattr(service, 'set_error_handler'):
                try:
                    service.set_error_handler(self.error_handler)
                    logger.debug(f"Injected error handler into {service_name}")
                except Exception:
                    pass
        
        # Inject context manager
        for service_name, service in self._services.items():
            if hasattr(service, 'set_context_manager'):
                try:
                    service.set_context_manager(self.context_manager)
                    logger.debug(f"Injected context manager into {service_name}")
                except Exception:
                    pass
    
    def get_service(self, name: str) -> Optional[Any]:
        """Get service by name."""
        if not self._initialized:
            self.initialize_all()
        return self._services.get(name)
    
    def get_all_services(self) -> Dict[str, Any]:
        """Get all services."""
        if not self._initialized:
            self.initialize_all()
        return self._services.copy()
    
    def get_services_by_category(self, category: str) -> Dict[str, Any]:
        """Get services by category."""
        return self.factory.get_services_by_category(category)
    
    def validate(self) -> Dict[str, Any]:
        """Validate service setup."""
        factory_validation = {
            "factory": "ok",
            "services_count": len(self._services),
        }
        
        integration_validation = self.integration.validate_integrations()
        
        return {
            **factory_validation,
            "integration": integration_validation,
            "overall_valid": integration_validation["valid"],
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get service manager status."""
        return {
            "initialized": self._initialized,
            "services_count": len(self._services),
            "categories": self.factory.get_services_by_category.__doc__,
            "integration_graph": self.integration.get_integration_graph(),
        }

