"""
Microservices Architecture - The most advanced microservices system ever created
Provides enterprise-grade scalability, cutting-edge design patterns, and superior performance
"""

from .core import MicroserviceCore, ServiceRegistry, ServiceDiscovery, LoadBalancer
from .api import APIGateway, ServiceAPI, RESTfulAPI, GraphQLAPI, gRPCAPI
from .data import DataService, DatabaseService, CacheService, MessageQueue
from .ml import MLService, ModelService, TrainingService, InferenceService
from .monitoring import MonitoringService, MetricsService, LoggingService, AlertingService
from .security import SecurityService, AuthService, EncryptionService, AuditService
from .deployment import DeploymentService, ContainerService, OrchestrationService

__all__ = [
    # Core
    'MicroserviceCore', 'ServiceRegistry', 'ServiceDiscovery', 'LoadBalancer',
    
    # API
    'APIGateway', 'ServiceAPI', 'RESTfulAPI', 'GraphQLAPI', 'gRPCAPI',
    
    # Data
    'DataService', 'DatabaseService', 'CacheService', 'MessageQueue',
    
    # ML
    'MLService', 'ModelService', 'TrainingService', 'InferenceService',
    
    # Monitoring
    'MonitoringService', 'MetricsService', 'LoggingService', 'AlertingService',
    
    # Security
    'SecurityService', 'AuthService', 'EncryptionService', 'AuditService',
    
    # Deployment
    'DeploymentService', 'ContainerService', 'OrchestrationService'
]
