"""
Ultimate BUL System - Enterprise Integration Hub
Comprehensive microservices architecture with enterprise-grade integrations
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import aiohttp
import redis
import yaml
from pathlib import Path
import consul
import etcd3
from prometheus_client import Counter, Histogram, Gauge
import grpc
from concurrent import futures
import threading
import time

logger = logging.getLogger(__name__)

class ServiceType(str, Enum):
    """Microservice types"""
    API_GATEWAY = "api_gateway"
    DOCUMENT_SERVICE = "document_service"
    AI_SERVICE = "ai_service"
    WORKFLOW_SERVICE = "workflow_service"
    ANALYTICS_SERVICE = "analytics_service"
    INTEGRATION_SERVICE = "integration_service"
    NOTIFICATION_SERVICE = "notification_service"
    AUTH_SERVICE = "auth_service"
    CONFIG_SERVICE = "config_service"
    MONITORING_SERVICE = "monitoring_service"

class IntegrationType(str, Enum):
    """Integration types"""
    REST_API = "rest_api"
    GRAPHQL = "graphql"
    GRPC = "grpc"
    MESSAGE_QUEUE = "message_queue"
    DATABASE = "database"
    FILE_SYSTEM = "file_system"
    CLOUD_STORAGE = "cloud_storage"
    WEBHOOK = "webhook"
    SOAP = "soap"
    FTP = "ftp"

class ServiceStatus(str, Enum):
    """Service status"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    STARTING = "starting"
    STOPPING = "stopping"
    MAINTENANCE = "maintenance"
    UNKNOWN = "unknown"

@dataclass
class Microservice:
    """Microservice definition"""
    id: str
    name: str
    service_type: ServiceType
    version: str
    status: ServiceStatus
    endpoint: str
    port: int
    health_check_url: str
    dependencies: List[str] = field(default_factory=list)
    environment: str = "production"
    replicas: int = 1
    resources: Dict[str, str] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class ServiceIntegration:
    """Service integration definition"""
    id: str
    source_service: str
    target_service: str
    integration_type: IntegrationType
    endpoint: str
    authentication: Dict[str, Any] = field(default_factory=dict)
    rate_limit: int = 1000
    timeout: int = 30
    retry_attempts: int = 3
    circuit_breaker: bool = True
    enabled: bool = True

@dataclass
class ExternalIntegration:
    """External integration definition"""
    id: str
    name: str
    integration_type: IntegrationType
    endpoint: str
    authentication: Dict[str, Any] = field(default_factory=dict)
    rate_limit: int = 100
    timeout: int = 60
    retry_attempts: int = 3
    circuit_breaker: bool = True
    enabled: bool = True
    health_check_url: Optional[str] = None

class EnterpriseIntegrationHub:
    """Enterprise Integration Hub for microservices architecture"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.microservices = {}
        self.service_integrations = {}
        self.external_integrations = {}
        self.service_discovery = None
        self.config_store = None
        self.message_broker = None
        
        # Service registry
        self.service_registry = {}
        self.health_checks = {}
        
        # Prometheus metrics
        self.prometheus_metrics = self._initialize_prometheus_metrics()
        
        # Integration monitoring
        self.monitoring_active = False
        
        # Initialize services
        self._initialize_services()
        self._initialize_integrations()
        
        # Start monitoring
        self.start_monitoring()
    
    def _initialize_services(self):
        """Initialize microservices"""
        # API Gateway
        self.microservices["api_gateway"] = Microservice(
            id="api_gateway",
            name="BUL API Gateway",
            service_type=ServiceType.API_GATEWAY,
            version="3.0.0",
            status=ServiceStatus.HEALTHY,
            endpoint="http://api-gateway:8000",
            port=8000,
            health_check_url="/health",
            dependencies=[],
            replicas=3,
            resources={"cpu": "500m", "memory": "1Gi"}
        )
        
        # Document Service
        self.microservices["document_service"] = Microservice(
            id="document_service",
            name="BUL Document Service",
            service_type=ServiceType.DOCUMENT_SERVICE,
            version="3.0.0",
            status=ServiceStatus.HEALTHY,
            endpoint="http://document-service:8001",
            port=8001,
            health_check_url="/health",
            dependencies=["ai_service"],
            replicas=2,
            resources={"cpu": "1", "memory": "2Gi"}
        )
        
        # AI Service
        self.microservices["ai_service"] = Microservice(
            id="ai_service",
            name="BUL AI Service",
            service_type=ServiceType.AI_SERVICE,
            version="3.0.0",
            status=ServiceStatus.HEALTHY,
            endpoint="http://ai-service:8002",
            port=8002,
            health_check_url="/health",
            dependencies=[],
            replicas=2,
            resources={"cpu": "2", "memory": "4Gi"}
        )
        
        # Workflow Service
        self.microservices["workflow_service"] = Microservice(
            id="workflow_service",
            name="BUL Workflow Service",
            service_type=ServiceType.WORKFLOW_SERVICE,
            version="3.0.0",
            status=ServiceStatus.HEALTHY,
            endpoint="http://workflow-service:8003",
            port=8003,
            health_check_url="/health",
            dependencies=["ai_service", "document_service"],
            replicas=2,
            resources={"cpu": "1", "memory": "2Gi"}
        )
        
        # Analytics Service
        self.microservices["analytics_service"] = Microservice(
            id="analytics_service",
            name="BUL Analytics Service",
            service_type=ServiceType.ANALYTICS_SERVICE,
            version="3.0.0",
            status=ServiceStatus.HEALTHY,
            endpoint="http://analytics-service:8004",
            port=8004,
            health_check_url="/health",
            dependencies=["document_service"],
            replicas=2,
            resources={"cpu": "1", "memory": "2Gi"}
        )
        
        # Integration Service
        self.microservices["integration_service"] = Microservice(
            id="integration_service",
            name="BUL Integration Service",
            service_type=ServiceType.INTEGRATION_SERVICE,
            version="3.0.0",
            status=ServiceStatus.HEALTHY,
            endpoint="http://integration-service:8005",
            port=8005,
            health_check_url="/health",
            dependencies=[],
            replicas=2,
            resources={"cpu": "500m", "memory": "1Gi"}
        )
        
        # Notification Service
        self.microservices["notification_service"] = Microservice(
            id="notification_service",
            name="BUL Notification Service",
            service_type=ServiceType.NOTIFICATION_SERVICE,
            version="3.0.0",
            status=ServiceStatus.HEALTHY,
            endpoint="http://notification-service:8006",
            port=8006,
            health_check_url="/health",
            dependencies=[],
            replicas=2,
            resources={"cpu": "500m", "memory": "1Gi"}
        )
        
        # Auth Service
        self.microservices["auth_service"] = Microservice(
            id="auth_service",
            name="BUL Auth Service",
            service_type=ServiceType.AUTH_SERVICE,
            version="3.0.0",
            status=ServiceStatus.HEALTHY,
            endpoint="http://auth-service:8007",
            port=8007,
            health_check_url="/health",
            dependencies=[],
            replicas=2,
            resources={"cpu": "500m", "memory": "1Gi"}
        )
        
        # Config Service
        self.microservices["config_service"] = Microservice(
            id="config_service",
            name="BUL Config Service",
            service_type=ServiceType.CONFIG_SERVICE,
            version="3.0.0",
            status=ServiceStatus.HEALTHY,
            endpoint="http://config-service:8008",
            port=8008,
            health_check_url="/health",
            dependencies=[],
            replicas=1,
            resources={"cpu": "250m", "memory": "512Mi"}
        )
        
        # Monitoring Service
        self.microservices["monitoring_service"] = Microservice(
            id="monitoring_service",
            name="BUL Monitoring Service",
            service_type=ServiceType.MONITORING_SERVICE,
            version="3.0.0",
            status=ServiceStatus.HEALTHY,
            endpoint="http://monitoring-service:8009",
            port=8009,
            health_check_url="/health",
            dependencies=[],
            replicas=1,
            resources={"cpu": "500m", "memory": "1Gi"}
        )
    
    def _initialize_integrations(self):
        """Initialize service integrations"""
        # Internal service integrations
        self.service_integrations["api_to_document"] = ServiceIntegration(
            id="api_to_document",
            source_service="api_gateway",
            target_service="document_service",
            integration_type=IntegrationType.REST_API,
            endpoint="/api/v1/documents",
            rate_limit=1000,
            timeout=30
        )
        
        self.service_integrations["document_to_ai"] = ServiceIntegration(
            id="document_to_ai",
            source_service="document_service",
            target_service="ai_service",
            integration_type=IntegrationType.REST_API,
            endpoint="/api/v1/ai/generate",
            rate_limit=500,
            timeout=60
        )
        
        self.service_integrations["workflow_to_ai"] = ServiceIntegration(
            id="workflow_to_ai",
            source_service="workflow_service",
            target_service="ai_service",
            integration_type=IntegrationType.REST_API,
            endpoint="/api/v1/ai/process",
            rate_limit=200,
            timeout=120
        )
        
        self.service_integrations["analytics_to_document"] = ServiceIntegration(
            id="analytics_to_document",
            source_service="analytics_service",
            target_service="document_service",
            integration_type=IntegrationType.REST_API,
            endpoint="/api/v1/analytics/track",
            rate_limit=2000,
            timeout=10
        )
        
        # External integrations
        self.external_integrations["google_docs"] = ExternalIntegration(
            id="google_docs",
            name="Google Docs Integration",
            integration_type=IntegrationType.REST_API,
            endpoint="https://docs.googleapis.com/v1",
            authentication={"type": "oauth2", "scopes": ["https://www.googleapis.com/auth/documents"]},
            rate_limit=100,
            timeout=30
        )
        
        self.external_integrations["office365"] = ExternalIntegration(
            id="office365",
            name="Office 365 Integration",
            integration_type=IntegrationType.REST_API,
            endpoint="https://graph.microsoft.com/v1.0",
            authentication={"type": "oauth2", "scopes": ["Files.ReadWrite"]},
            rate_limit=100,
            timeout=30
        )
        
        self.external_integrations["salesforce"] = ExternalIntegration(
            id="salesforce",
            name="Salesforce CRM Integration",
            integration_type=IntegrationType.REST_API,
            endpoint="https://api.salesforce.com/v1",
            authentication={"type": "oauth2", "scopes": ["api"]},
            rate_limit=50,
            timeout=60
        )
        
        self.external_integrations["slack"] = ExternalIntegration(
            id="slack",
            name="Slack Integration",
            integration_type=IntegrationType.REST_API,
            endpoint="https://slack.com/api",
            authentication={"type": "bearer_token"},
            rate_limit=100,
            timeout=30
        )
    
    def _initialize_prometheus_metrics(self) -> Dict[str, Any]:
        """Initialize Prometheus metrics"""
        return {
            "service_requests": Counter(
                "bul_service_requests_total",
                "Total service requests",
                ["service", "method", "status_code"]
            ),
            "service_response_time": Histogram(
                "bul_service_response_time_seconds",
                "Service response time in seconds",
                ["service", "endpoint"]
            ),
            "service_health": Gauge(
                "bul_service_health",
                "Service health status (1=healthy, 0=unhealthy)",
                ["service"]
            ),
            "integration_requests": Counter(
                "bul_integration_requests_total",
                "Total integration requests",
                ["integration", "status"]
            ),
            "integration_response_time": Histogram(
                "bul_integration_response_time_seconds",
                "Integration response time in seconds",
                ["integration"]
            ),
            "active_services": Gauge(
                "bul_active_services",
                "Number of active services"
            ),
            "active_integrations": Gauge(
                "bul_active_integrations",
                "Number of active integrations"
            )
        }
    
    async def start_monitoring(self):
        """Start integration monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        logger.info("Starting enterprise integration monitoring")
        
        # Start monitoring tasks
        asyncio.create_task(self._monitor_services())
        asyncio.create_task(self._monitor_integrations())
        asyncio.create_task(self._update_metrics())
    
    async def stop_monitoring(self):
        """Stop integration monitoring"""
        self.monitoring_active = False
        logger.info("Stopping enterprise integration monitoring")
    
    async def _monitor_services(self):
        """Monitor microservices health"""
        while self.monitoring_active:
            try:
                for service_id, service in self.microservices.items():
                    try:
                        # Check service health
                        async with aiohttp.ClientSession() as session:
                            async with session.get(
                                f"{service.endpoint}{service.health_check_url}",
                                timeout=aiohttp.ClientTimeout(total=10)
                            ) as response:
                                if response.status == 200:
                                    service.status = ServiceStatus.HEALTHY
                                else:
                                    service.status = ServiceStatus.UNHEALTHY
                    except Exception as e:
                        service.status = ServiceStatus.UNHEALTHY
                        logger.warning(f"Service {service_id} health check failed: {e}")
                    
                    # Update Prometheus metrics
                    health_value = 1 if service.status == ServiceStatus.HEALTHY else 0
                    self.prometheus_metrics["service_health"].labels(service=service_id).set(health_value)
                
                # Update active services count
                active_services = len([s for s in self.microservices.values() if s.status == ServiceStatus.HEALTHY])
                self.prometheus_metrics["active_services"].set(active_services)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error monitoring services: {e}")
                await asyncio.sleep(60)
    
    async def _monitor_integrations(self):
        """Monitor external integrations"""
        while self.monitoring_active:
            try:
                for integration_id, integration in self.external_integrations.items():
                    if not integration.enabled:
                        continue
                    
                    try:
                        # Check integration health
                        if integration.health_check_url:
                            async with aiohttp.ClientSession() as session:
                                async with session.get(
                                    integration.health_check_url,
                                    timeout=aiohttp.ClientTimeout(total=10)
                                ) as response:
                                    if response.status == 200:
                                        # Integration is healthy
                                        pass
                                    else:
                                        logger.warning(f"Integration {integration_id} health check failed")
                    except Exception as e:
                        logger.warning(f"Integration {integration_id} health check failed: {e}")
                
                # Update active integrations count
                active_integrations = len([i for i in self.external_integrations.values() if i.enabled])
                self.prometheus_metrics["active_integrations"].set(active_integrations)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error monitoring integrations: {e}")
                await asyncio.sleep(60)
    
    async def _update_metrics(self):
        """Update Prometheus metrics"""
        while self.monitoring_active:
            try:
                # Update service metrics
                for service_id, service in self.microservices.items():
                    health_value = 1 if service.status == ServiceStatus.HEALTHY else 0
                    self.prometheus_metrics["service_health"].labels(service=service_id).set(health_value)
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                logger.error(f"Error updating metrics: {e}")
                await asyncio.sleep(60)
    
    async def call_service(self, service_id: str, endpoint: str, method: str = "GET", 
                          data: Optional[Dict[str, Any]] = None, 
                          headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Call a microservice"""
        if service_id not in self.microservices:
            raise ValueError(f"Service {service_id} not found")
        
        service = self.microservices[service_id]
        
        if service.status != ServiceStatus.HEALTHY:
            raise Exception(f"Service {service_id} is not healthy")
        
        start_time = time.time()
        
        try:
            url = f"{service.endpoint}{endpoint}"
            
            async with aiohttp.ClientSession() as session:
                if method.upper() == "GET":
                    async with session.get(url, headers=headers) as response:
                        result = await response.json()
                elif method.upper() == "POST":
                    async with session.post(url, json=data, headers=headers) as response:
                        result = await response.json()
                elif method.upper() == "PUT":
                    async with session.put(url, json=data, headers=headers) as response:
                        result = await response.json()
                elif method.upper() == "DELETE":
                    async with session.delete(url, headers=headers) as response:
                        result = await response.json()
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                
                response_time = time.time() - start_time
                
                # Update Prometheus metrics
                self.prometheus_metrics["service_requests"].labels(
                    service=service_id,
                    method=method.upper(),
                    status_code=response.status
                ).inc()
                
                self.prometheus_metrics["service_response_time"].labels(
                    service=service_id,
                    endpoint=endpoint
                ).observe(response_time)
                
                return {
                    "status_code": response.status,
                    "data": result,
                    "response_time": response_time
                }
                
        except Exception as e:
            logger.error(f"Error calling service {service_id}: {e}")
            raise
    
    async def call_integration(self, integration_id: str, endpoint: str, method: str = "GET",
                              data: Optional[Dict[str, Any]] = None,
                              headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Call an external integration"""
        if integration_id not in self.external_integrations:
            raise ValueError(f"Integration {integration_id} not found")
        
        integration = self.external_integrations[integration_id]
        
        if not integration.enabled:
            raise Exception(f"Integration {integration_id} is disabled")
        
        start_time = time.time()
        
        try:
            url = f"{integration.endpoint}{endpoint}"
            
            # Add authentication headers
            auth_headers = headers or {}
            if integration.authentication:
                auth_headers.update(self._get_auth_headers(integration.authentication))
            
            async with aiohttp.ClientSession() as session:
                if method.upper() == "GET":
                    async with session.get(url, headers=auth_headers) as response:
                        result = await response.json()
                elif method.upper() == "POST":
                    async with session.post(url, json=data, headers=auth_headers) as response:
                        result = await response.json()
                elif method.upper() == "PUT":
                    async with session.put(url, json=data, headers=auth_headers) as response:
                        result = await response.json()
                elif method.upper() == "DELETE":
                    async with session.delete(url, headers=auth_headers) as response:
                        result = await response.json()
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                
                response_time = time.time() - start_time
                
                # Update Prometheus metrics
                self.prometheus_metrics["integration_requests"].labels(
                    integration=integration_id,
                    status="success"
                ).inc()
                
                self.prometheus_metrics["integration_response_time"].labels(
                    integration=integration_id
                ).observe(response_time)
                
                return {
                    "status_code": response.status,
                    "data": result,
                    "response_time": response_time
                }
                
        except Exception as e:
            # Update Prometheus metrics for failed requests
            self.prometheus_metrics["integration_requests"].labels(
                integration=integration_id,
                status="error"
            ).inc()
            
            logger.error(f"Error calling integration {integration_id}: {e}")
            raise
    
    def _get_auth_headers(self, authentication: Dict[str, Any]) -> Dict[str, str]:
        """Get authentication headers for integration"""
        auth_type = authentication.get("type")
        
        if auth_type == "bearer_token":
            token = authentication.get("token")
            return {"Authorization": f"Bearer {token}"}
        elif auth_type == "api_key":
            api_key = authentication.get("api_key")
            key_name = authentication.get("key_name", "X-API-Key")
            return {key_name: api_key}
        elif auth_type == "basic":
            username = authentication.get("username")
            password = authentication.get("password")
            import base64
            credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
            return {"Authorization": f"Basic {credentials}"}
        else:
            return {}
    
    def register_service(self, service: Microservice):
        """Register a new microservice"""
        self.microservices[service.id] = service
        logger.info(f"Registered service: {service.id}")
    
    def unregister_service(self, service_id: str):
        """Unregister a microservice"""
        if service_id in self.microservices:
            del self.microservices[service_id]
            logger.info(f"Unregistered service: {service_id}")
    
    def register_integration(self, integration: ExternalIntegration):
        """Register a new external integration"""
        self.external_integrations[integration.id] = integration
        logger.info(f"Registered integration: {integration.id}")
    
    def unregister_integration(self, integration_id: str):
        """Unregister an external integration"""
        if integration_id in self.external_integrations:
            del self.external_integrations[integration_id]
            logger.info(f"Unregistered integration: {integration_id}")
    
    def get_service(self, service_id: str) -> Optional[Microservice]:
        """Get service by ID"""
        return self.microservices.get(service_id)
    
    def list_services(self, service_type: Optional[ServiceType] = None) -> List[Microservice]:
        """List services"""
        services = list(self.microservices.values())
        
        if service_type:
            services = [s for s in services if s.service_type == service_type]
        
        return services
    
    def get_integration(self, integration_id: str) -> Optional[ExternalIntegration]:
        """Get integration by ID"""
        return self.external_integrations.get(integration_id)
    
    def list_integrations(self) -> List[ExternalIntegration]:
        """List integrations"""
        return list(self.external_integrations.values())
    
    def get_service_health(self) -> Dict[str, Any]:
        """Get service health summary"""
        total_services = len(self.microservices)
        healthy_services = len([s for s in self.microservices.values() if s.status == ServiceStatus.HEALTHY])
        unhealthy_services = total_services - healthy_services
        
        # Health by service type
        health_by_type = {}
        for service in self.microservices.values():
            service_type = service.service_type.value
            if service_type not in health_by_type:
                health_by_type[service_type] = {"healthy": 0, "unhealthy": 0}
            
            if service.status == ServiceStatus.HEALTHY:
                health_by_type[service_type]["healthy"] += 1
            else:
                health_by_type[service_type]["unhealthy"] += 1
        
        return {
            "total_services": total_services,
            "healthy_services": healthy_services,
            "unhealthy_services": unhealthy_services,
            "health_percentage": (healthy_services / total_services * 100) if total_services > 0 else 0,
            "health_by_type": health_by_type,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def get_integration_health(self) -> Dict[str, Any]:
        """Get integration health summary"""
        total_integrations = len(self.external_integrations)
        enabled_integrations = len([i for i in self.external_integrations.values() if i.enabled])
        disabled_integrations = total_integrations - enabled_integrations
        
        # Health by integration type
        health_by_type = {}
        for integration in self.external_integrations.values():
            integration_type = integration.integration_type.value
            if integration_type not in health_by_type:
                health_by_type[integration_type] = {"enabled": 0, "disabled": 0}
            
            if integration.enabled:
                health_by_type[integration_type]["enabled"] += 1
            else:
                health_by_type[integration_type]["disabled"] += 1
        
        return {
            "total_integrations": total_integrations,
            "enabled_integrations": enabled_integrations,
            "disabled_integrations": disabled_integrations,
            "enabled_percentage": (enabled_integrations / total_integrations * 100) if total_integrations > 0 else 0,
            "health_by_type": health_by_type,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def export_integration_data(self) -> Dict[str, Any]:
        """Export integration data for analysis"""
        return {
            "microservices": [
                {
                    "id": service.id,
                    "name": service.name,
                    "service_type": service.service_type.value,
                    "version": service.version,
                    "status": service.status.value,
                    "endpoint": service.endpoint,
                    "port": service.port,
                    "replicas": service.replicas,
                    "resources": service.resources,
                    "created_at": service.created_at.isoformat(),
                    "updated_at": service.updated_at.isoformat()
                }
                for service in self.microservices.values()
            ],
            "service_integrations": [
                {
                    "id": integration.id,
                    "source_service": integration.source_service,
                    "target_service": integration.target_service,
                    "integration_type": integration.integration_type.value,
                    "endpoint": integration.endpoint,
                    "rate_limit": integration.rate_limit,
                    "timeout": integration.timeout,
                    "enabled": integration.enabled
                }
                for integration in self.service_integrations.values()
            ],
            "external_integrations": [
                {
                    "id": integration.id,
                    "name": integration.name,
                    "integration_type": integration.integration_type.value,
                    "endpoint": integration.endpoint,
                    "rate_limit": integration.rate_limit,
                    "timeout": integration.timeout,
                    "enabled": integration.enabled
                }
                for integration in self.external_integrations.values()
            ],
            "service_health": self.get_service_health(),
            "integration_health": self.get_integration_health(),
            "export_timestamp": datetime.utcnow().isoformat()
        }

# Global integration hub instance
integration_hub = None

def get_integration_hub() -> EnterpriseIntegrationHub:
    """Get the global integration hub instance"""
    global integration_hub
    if integration_hub is None:
        config = {
            "consul_host": "localhost",
            "consul_port": 8500,
            "etcd_host": "localhost",
            "etcd_port": 2379
        }
        integration_hub = EnterpriseIntegrationHub(config)
    return integration_hub

if __name__ == "__main__":
    # Example usage
    async def main():
        config = {
            "consul_host": "localhost",
            "consul_port": 8500,
            "etcd_host": "localhost",
            "etcd_port": 2379
        }
        
        hub = EnterpriseIntegrationHub(config)
        
        # Call a service
        try:
            result = await hub.call_service(
                service_id="document_service",
                endpoint="/api/v1/documents",
                method="GET"
            )
            print(f"Service call result: {result}")
        except Exception as e:
            print(f"Service call failed: {e}")
        
        # Call an integration
        try:
            result = await hub.call_integration(
                integration_id="slack",
                endpoint="/chat.postMessage",
                method="POST",
                data={"channel": "#general", "text": "Hello from BUL!"}
            )
            print(f"Integration call result: {result}")
        except Exception as e:
            print(f"Integration call failed: {e}")
        
        # Get health summary
        service_health = hub.get_service_health()
        print("Service Health:")
        print(json.dumps(service_health, indent=2))
        
        integration_health = hub.get_integration_health()
        print("\nIntegration Health:")
        print(json.dumps(integration_health, indent=2))
        
        await hub.stop_monitoring()
    
    asyncio.run(main())













