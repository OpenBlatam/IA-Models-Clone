"""
Service Integration for Color Grading AI
========================================

Integration layer for connecting services and managing interactions.
"""

import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ServiceConnection:
    """Service connection definition."""
    from_service: str
    to_service: str
    connection_type: str  # dependency, event, data_flow
    handler: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ServiceIntegration:
    """
    Service integration manager.
    
    Features:
    - Service connections
    - Event wiring
    - Data flow management
    - Integration validation
    """
    
    def __init__(self):
        """Initialize service integration."""
        self._connections: List[ServiceConnection] = []
        self._services: Dict[str, Any] = {}
    
    def register_services(self, services: Dict[str, Any]):
        """Register services for integration."""
        self._services.update(services)
        logger.info(f"Registered {len(services)} services for integration")
    
    def connect_services(
        self,
        from_service: str,
        to_service: str,
        connection_type: str = "dependency",
        handler: Optional[Callable] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Connect two services.
        
        Args:
            from_service: Source service name
            to_service: Target service name
            connection_type: Connection type
            handler: Optional connection handler
            metadata: Optional metadata
        """
        connection = ServiceConnection(
            from_service=from_service,
            to_service=to_service,
            connection_type=connection_type,
            handler=handler,
            metadata=metadata or {}
        )
        
        self._connections.append(connection)
        logger.info(f"Connected {from_service} -> {to_service} ({connection_type})")
    
    def wire_event_bus(self, event_bus: Any, services: Dict[str, Any]):
        """
        Wire services to event bus.
        
        Args:
            event_bus: Event bus instance
            services: Services dictionary
        """
        # Auto-wire services that have event handlers
        for service_name, service in services.items():
            if hasattr(service, 'subscribe_to_events'):
                try:
                    service.subscribe_to_events(event_bus)
                    logger.info(f"Wired {service_name} to event bus")
                except Exception as e:
                    logger.warning(f"Could not wire {service_name} to event bus: {e}")
    
    def setup_metrics_collection(self, metrics_collector: Any, services: Dict[str, Any]):
        """
        Setup metrics collection for services.
        
        Args:
            metrics_collector: Metrics collector instance
            services: Services dictionary
        """
        for service_name, service in services.items():
            if hasattr(service, 'set_metrics_collector'):
                try:
                    service.set_metrics_collector(metrics_collector)
                    logger.info(f"Setup metrics for {service_name}")
                except Exception as e:
                    logger.warning(f"Could not setup metrics for {service_name}: {e}")
    
    def setup_health_checks(self, health_monitor: Any, services: Dict[str, Any]):
        """
        Setup health checks for services.
        
        Args:
            health_monitor: Health monitor instance
            services: Services dictionary
        """
        for service_name, service in services.items():
            if hasattr(service, 'health_check'):
                try:
                    health_monitor.register_check(
                        name=service_name,
                        check_func=service.health_check,
                        critical=True
                    )
                    logger.info(f"Registered health check for {service_name}")
                except Exception as e:
                    logger.warning(f"Could not register health check for {service_name}: {e}")
    
    def validate_integrations(self) -> Dict[str, Any]:
        """
        Validate all service integrations.
        
        Returns:
            Validation results
        """
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "connections": len(self._connections),
        }
        
        # Validate connections
        for conn in self._connections:
            if conn.from_service not in self._services:
                results["errors"].append(f"Source service not found: {conn.from_service}")
                results["valid"] = False
            
            if conn.to_service not in self._services:
                results["errors"].append(f"Target service not found: {conn.to_service}")
                results["valid"] = False
        
        return results
    
    def get_integration_graph(self) -> Dict[str, Any]:
        """Get service integration graph."""
        graph = {
            "services": list(self._services.keys()),
            "connections": [
                {
                    "from": conn.from_service,
                    "to": conn.to_service,
                    "type": conn.connection_type,
                }
                for conn in self._connections
            ],
        }
        
        return graph




