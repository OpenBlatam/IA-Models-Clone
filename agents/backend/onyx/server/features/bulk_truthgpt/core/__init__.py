"""
Core Module
===========

Ultra-modular core system with advanced architecture patterns.
"""

from .base import BaseComponent, BaseService, BaseEngine, BaseManager
from .registry import ComponentRegistry, ServiceRegistry, EngineRegistry
from .lifecycle import LifecycleManager, ComponentLifecycle
from .dependency import DependencyInjector, ServiceContainer
from .events import EventBus, EventHandler, EventDispatcher
from .middleware import MiddlewareManager, RequestMiddleware, ResponseMiddleware
from .validation import ValidationManager, SchemaValidator, FieldValidator
from .serialization import SerializationManager, JSONSerializer, XMLSerializer
from .caching import CacheManager, MemoryCache, RedisCache, FileCache
from .logging import LogManager, StructuredLogger, ContextLogger
from .metrics import MetricsManager, Counter, Histogram, Gauge, Summary
from .security import SecurityManager, AuthenticationManager, AuthorizationManager
from .database import DatabaseManager, ConnectionPool, TransactionManager
from .queue import QueueManager, TaskQueue, MessageQueue, EventQueue
from .scheduler import SchedulerManager, TaskScheduler, CronScheduler
from .monitoring import MonitoringManager, HealthChecker, PerformanceMonitor
from .configuration import ConfigurationManager, EnvironmentConfig, SecretManager
from .plugin import PluginManager, PluginLoader, PluginRegistry
from .api import APIManager, RouteManager, EndpointManager
from .testing import TestManager, TestRunner, MockManager
from .deployment import DeploymentManager, ContainerManager, ServiceManager

__all__ = [
    # Base classes
    'BaseComponent', 'BaseService', 'BaseEngine', 'BaseManager',
    
    # Registries
    'ComponentRegistry', 'ServiceRegistry', 'EngineRegistry',
    
    # Lifecycle
    'LifecycleManager', 'ComponentLifecycle',
    
    # Dependency injection
    'DependencyInjector', 'ServiceContainer',
    
    # Events
    'EventBus', 'EventHandler', 'EventDispatcher',
    
    # Middleware
    'MiddlewareManager', 'RequestMiddleware', 'ResponseMiddleware',
    
    # Validation
    'ValidationManager', 'SchemaValidator', 'FieldValidator',
    
    # Serialization
    'SerializationManager', 'JSONSerializer', 'XMLSerializer',
    
    # Caching
    'CacheManager', 'MemoryCache', 'RedisCache', 'FileCache',
    
    # Logging
    'LogManager', 'StructuredLogger', 'ContextLogger',
    
    # Metrics
    'MetricsManager', 'Counter', 'Histogram', 'Gauge', 'Summary',
    
    # Security
    'SecurityManager', 'AuthenticationManager', 'AuthorizationManager',
    
    # Database
    'DatabaseManager', 'ConnectionPool', 'TransactionManager',
    
    # Queue
    'QueueManager', 'TaskQueue', 'MessageQueue', 'EventQueue',
    
    # Scheduler
    'SchedulerManager', 'TaskScheduler', 'CronScheduler',
    
    # Monitoring
    'MonitoringManager', 'HealthChecker', 'PerformanceMonitor',
    
    # Configuration
    'ConfigurationManager', 'EnvironmentConfig', 'SecretManager',
    
    # Plugin
    'PluginManager', 'PluginLoader', 'PluginRegistry',
    
    # API
    'APIManager', 'RouteManager', 'EndpointManager',
    
    # Testing
    'TestManager', 'TestRunner', 'MockManager',
    
    # Deployment
    'DeploymentManager', 'ContainerManager', 'ServiceManager'
]