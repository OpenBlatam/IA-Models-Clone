"""
Consolidated Imports
====================

Centralized imports for core module.
"""

# Infrastructure
from ..infrastructure import (
    OpenRouterClient,
    TruthGPTClient,
    BaseHTTPClient
)

# Configuration
from ..config.enhancer_config import EnhancerConfig

# Core components - organized by category
# Task management
from .task_manager import TaskManager, FileTaskRepository, TaskEvent, Task, TaskStatus
from .task_creator import TaskCreator, TaskType, TaskParameters
from .parallel_executor import ParallelExecutor

# Service handling
from .service_handler import ServiceHandler, ServiceType
from .system_prompts_builder import SystemPromptsBuilder

# Processing
from .batch_processor import BatchProcessor, BatchItem, BatchResult
from .video_processor import VideoProcessor

# Management
from .cache_manager import CacheManager
from .webhook_manager import WebhookManager, WebhookEvent
from .retry_manager import RetryManager, RetryConfig
from .metrics_collector import MetricsCollector
from .event_bus import EventBus, Event, EventType
from .monitoring_dashboard import MonitoringDashboard
from .auth import AuthManager
from .notification_system import NotificationManager

# Utilities
from .helpers import create_output_directories
from ..utils.performance import PerformanceMonitor

# Benchmark and optimization
from .benchmark import BenchmarkRunner, BenchmarkResult, PerformanceProfiler
from .optimizer import PerformanceOptimizer, ResourceMonitor, OptimizationResult

# Documentation
from .doc_generator import DocumentationGenerator, DocSection

# Dynamic configuration
from .dynamic_config import DynamicConfigManager, ConfigChange

# Advanced health
from .advanced_health import AdvancedHealthChecker, HealthCheck, HealthCheckResult, HealthStatus

# Manager registry
from .manager_registry import ManagerRegistry, BaseManager

# System integrator
from .system_integrator import SystemIntegrator, SystemComponent

# Error recovery
from .error_recovery import RecoveryManager, ErrorRecovery, RecoveryConfig, RecoveryResult, RecoveryStrategy

# Async utilities
from .async_utils import (
    gather_with_exceptions,
    timeout_after,
    retry_async,
    async_to_sync,
    ensure_async,
    AsyncLock,
    AsyncSemaphore
)

# Testing helpers
from .testing_helpers import TestRunner, TestResult, AsyncTestCase, temp_directory, mock_service

# CI/CD helpers
from .ci_cd_helpers import CIHelper, DeploymentHelper, BuildInfo

# Analytics
from .analytics import AnalyticsCollector, AnalyticsReporter, AnalyticsEvent, EventType

# Reporting
from .reporting import ReportGenerator, Report, ReportType

# Data validator
from .data_validator import DataValidator, ValidationSchema, ValidationRule, SchemaBuilder

# Registry base
from .registry_base import BaseRegistry, FactoryRegistry

# Executor base
from .executor_base import BaseExecutor, AsyncExecutor, ExecutionResult, ExecutionStatus

# Storage base
from .storage_base import BaseStorage, FileStorage

# Workflow
from .workflow import Workflow, WorkflowManager, WorkflowStep, WorkflowResult, WorkflowStatus

# Pipeline
from .pipeline import Pipeline, PipelineManager, PipelineStep, PipelineResult, PipelineStage

# Orchestrator
from .orchestrator import Orchestrator, OrchestrationTask, OrchestrationResult, OrchestrationStatus

# State management
from .state_manager import StateManager, StateChange, StateEvent

# Advanced cache
from .advanced_cache import AdvancedCache, CacheEntry, CacheStrategy

# Service base
from .service_base import BaseService, AsyncService, ServiceRegistry, ServiceConfig, ServiceResult, ServiceStatus

# Handler base
from .handler_base import BaseHandler, AsyncHandler, HandlerChain, HandlerConfig, HandlerResult

# Processor base
from .processor_base import BaseProcessor, AsyncProcessor, ProcessingConfig, ProcessingResult, ProcessingStatus

# Coordinator
from .coordinator import Coordinator, CoordinationTask, CoordinationResult, CoordinationStatus

# Integration
from .integration import IntegrationAdapter, IntegrationManager, IntegrationConfig, IntegrationResult, IntegrationStatus

# Data pipeline
from .data_pipeline import DataPipeline, DataPipelineManager, TransformStep, TransformResult

# Serializer
from .serializer import Serializer, SerializationResult, SerializationFormat

# Structured logging
from .structured_logging import StructuredLogger, LogEntry, LogLevel

# Config builder
from .config_builder import ConfigBuilder, ConfigSection

# Final utilities
from .final_utils import UtilityHelper, AsyncHelper, FileHelper

# Batch operations
from .batch_operations import BatchOperationManager, BatchOperation, BatchItem, BatchResult, BatchStatus

# Scheduler
from .scheduler import Scheduler, ScheduledTask, ScheduleType

# Advanced queue
from .advanced_queue import AdvancedQueue, QueueItem, QueuePriority, QueueStatus

# Result aggregator
from .result_aggregator import ResultAggregator, AggregationResult

# Performance tuner
from .performance_tuner import PerformanceTuner, TuningRecommendation, TuningAction

# Resource manager
from .resource_manager import ResourceManager, ResourceLimit, ResourceUsage, ResourceType

# Advanced service base
from .service_base_advanced import AdvancedServiceBase, ServiceRegistry, ServiceState, ServiceMetrics

# Execution context
from .execution_context import ExecutionContext, ContextManager

# Advanced error handler
from .error_handler_advanced import ErrorHandler, ErrorHandlerDecorator, ErrorInfo, ErrorSeverity

# Agent component
from .agent_component import AgentComponent, ComponentManager, ComponentConfig, ComponentHealth, ComponentStatus

# Event handler
from .event_handler import EventHandler, EventDispatcher, Event, EventPriority

# Factory base
from .factory_base import BaseFactory, BuilderFactory

# Middleware base
from .middleware_base import BaseMiddleware, MiddlewarePipeline, Request, Response

# Constants
from .constants import (
    DEFAULT_CACHE_TTL_HOURS,
    DEFAULT_MAX_RETRIES,
    OUTPUT_DIRECTORIES,
    DEFAULT_MAX_PARALLEL_TASKS,
    DEFAULT_OUTPUT_DIR
)

__all__ = [
    # Infrastructure
    "OpenRouterClient",
    "TruthGPTClient",
    "BaseHTTPClient",
    # Configuration
    "EnhancerConfig",
    # Task management
    "TaskManager",
    "FileTaskRepository",
    "TaskEvent",
    "Task",
    "TaskStatus",
    "TaskCreator",
    "TaskType",
    "TaskParameters",
    "ParallelExecutor",
    # Service handling
    "ServiceHandler",
    "ServiceType",
    "SystemPromptsBuilder",
    # Processing
    "BatchProcessor",
    "BatchItem",
    "BatchResult",
    "VideoProcessor",
    # Management
    "CacheManager",
    "WebhookManager",
    "WebhookEvent",
    "RetryManager",
    "RetryConfig",
    "MetricsCollector",
    "EventBus",
    "Event",
    "EventType",
    "MonitoringDashboard",
    "AuthManager",
    "NotificationManager",
    # Utilities
    "create_output_directories",
    "PerformanceMonitor",
    # Benchmark and optimization
    "BenchmarkRunner",
    "BenchmarkResult",
    "PerformanceProfiler",
    "PerformanceOptimizer",
    "ResourceMonitor",
    "OptimizationResult",
    # Documentation
    "DocumentationGenerator",
    "DocSection",
    # Dynamic configuration
    "DynamicConfigManager",
    "ConfigChange",
    # Advanced health
    "AdvancedHealthChecker",
    "HealthCheck",
    "HealthCheckResult",
    "HealthStatus",
    # Manager registry
    "ManagerRegistry",
    "BaseManager",
    # System integrator
    "SystemIntegrator",
    "SystemComponent",
    # Error recovery
    "RecoveryManager",
    "ErrorRecovery",
    "RecoveryConfig",
    "RecoveryResult",
    "RecoveryStrategy",
    # Async utilities
    "gather_with_exceptions",
    "timeout_after",
    "retry_async",
    "async_to_sync",
    "ensure_async",
    "AsyncLock",
    "AsyncSemaphore",
    # Testing helpers
    "TestRunner",
    "TestResult",
    "AsyncTestCase",
    "temp_directory",
    "mock_service",
    # CI/CD helpers
    "CIHelper",
    "DeploymentHelper",
    "BuildInfo",
    # Analytics
    "AnalyticsCollector",
    "AnalyticsReporter",
    "AnalyticsEvent",
    "EventType",
    # Reporting
    "ReportGenerator",
    "Report",
    "ReportType",
    # Data validator
    "DataValidator",
    "ValidationSchema",
    "ValidationRule",
    "SchemaBuilder",
    # Registry base
    "BaseRegistry",
    "FactoryRegistry",
    # Executor base
    "BaseExecutor",
    "AsyncExecutor",
    "ExecutionResult",
    "ExecutionStatus",
    # Storage base
    "BaseStorage",
    "FileStorage",
    # Workflow
    "Workflow",
    "WorkflowManager",
    "WorkflowStep",
    "WorkflowResult",
    "WorkflowStatus",
    # Pipeline
    "Pipeline",
    "PipelineManager",
    "PipelineStep",
    "PipelineResult",
    "PipelineStage",
    # Orchestrator
    "Orchestrator",
    "OrchestrationTask",
    "OrchestrationResult",
    "OrchestrationStatus",
    # State management
    "StateManager",
    "StateChange",
    "StateEvent",
    # Advanced cache
    "AdvancedCache",
    "CacheEntry",
    "CacheStrategy",
    # Service base
    "BaseService",
    "AsyncService",
    "ServiceRegistry",
    "ServiceConfig",
    "ServiceResult",
    "ServiceStatus",
    # Handler base
    "BaseHandler",
    "AsyncHandler",
    "HandlerChain",
    "HandlerConfig",
    "HandlerResult",
    # Processor base
    "BaseProcessor",
    "AsyncProcessor",
    "ProcessingConfig",
    "ProcessingResult",
    "ProcessingStatus",
    # Coordinator
    "Coordinator",
    "CoordinationTask",
    "CoordinationResult",
    "CoordinationStatus",
    # Integration
    "IntegrationAdapter",
    "IntegrationManager",
    "IntegrationConfig",
    "IntegrationResult",
    "IntegrationStatus",
    # Data pipeline
    "DataPipeline",
    "DataPipelineManager",
    "TransformStep",
    "TransformResult",
    # Serializer
    "Serializer",
    "SerializationResult",
    "SerializationFormat",
    # Structured logging
    "StructuredLogger",
    "LogEntry",
    "LogLevel",
    # Config builder
    "ConfigBuilder",
    "ConfigSection",
    # Final utilities
    "UtilityHelper",
    "AsyncHelper",
    "FileHelper",
    # Agent component
    "AgentComponent",
    "ComponentManager",
    "ComponentConfig",
    "ComponentHealth",
    "ComponentStatus",
    # Event handler
    "EventHandler",
    "EventDispatcher",
    "Event",
    "EventPriority",
    # Factory base
    "BaseFactory",
    "BuilderFactory",
    # Middleware base
    "BaseMiddleware",
    "MiddlewarePipeline",
    "Request",
    "Response",
    # Batch operations
    "BatchOperationManager",
    "BatchOperation",
    "BatchItem",
    "BatchResult",
    "BatchStatus",
    # Scheduler
    "Scheduler",
    "ScheduledTask",
    "ScheduleType",
    # Advanced queue
    "AdvancedQueue",
    "QueueItem",
    "QueuePriority",
    "QueueStatus",
    # Result aggregator
    "ResultAggregator",
    "AggregationResult",
    # Performance tuner
    "PerformanceTuner",
    "TuningRecommendation",
    "TuningAction",
    # Resource manager
    "ResourceManager",
    "ResourceLimit",
    "ResourceUsage",
    "ResourceType",
    # Advanced service base
    "AdvancedServiceBase",
    "ServiceRegistry",
    "ServiceState",
    "ServiceMetrics",
    # Execution context
    "ExecutionContext",
    "ContextManager",
    # Advanced error handler
    "ErrorHandler",
    "ErrorHandlerDecorator",
    "ErrorInfo",
    "ErrorSeverity",
    # Constants
    "DEFAULT_CACHE_TTL_HOURS",
    "DEFAULT_MAX_RETRIES",
    "OUTPUT_DIRECTORIES",
    "DEFAULT_MAX_PARALLEL_TASKS",
    "DEFAULT_OUTPUT_DIR",
]

