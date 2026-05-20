"""
Type Checking Imports - Imports para TYPE_CHECKING
==================================================

Organiza todos los imports que solo se usan para type hints,
separándolos del código de runtime para mejor organización.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Core
    from .server import MCPServer, MCPRequest, MCPResponse
    from .connectors import (
        FileSystemConnector,
        DatabaseConnector,
        APIConnector,
        ConnectorRegistry,
        BaseConnector,
    )
    # Manifests
    from .manifests import ResourceManifest, ManifestRegistry, ManifestLoader
    # Security
    from .security import MCPSecurityManager, Scope, AccessPolicy
    # Contracts
    from .contracts import ContextFrame, PromptFrame, FrameSerializer
    # Exceptions
    from .exceptions import (
        MCPError,
        MCPAuthenticationError,
        MCPAuthorizationError,
        MCPResourceNotFoundError,
        MCPConnectorError,
        MCPOperationError,
        MCPValidationError,
        MCPRateLimitError,
        MCPContextLimitError,
    )
    # Utilities
    from .rate_limiter import RateLimiter
    from .cache import MCPCache
    from .middleware import MCPLoggingMiddleware, MCPCORSMiddleware
    from .helpers import create_mcp_server, setup_rate_limits, setup_cache
    from .retry import RetryConfig, retry_with_backoff, retryable
    from .circuit_breaker import CircuitBreaker, CircuitState, circuit_breaker
    # Features
    from .batch import BatchProcessor, BatchRequest, BatchResponse, batch_query
    from .webhooks import WebhookManager, Webhook, WebhookEvent, WebhookPayload
    from .transformers import (
        RequestTransformer,
        ResponseTransformer,
        add_timestamp_transformer,
        mask_sensitive_data_transformer,
        compress_context_transformer,
    )
    from .admin import MCPAdmin
    from .streaming import StreamResponse, StreamChunk, stream_query
    from .config import MCPConfig
    from .profiling import PerformanceProfiler, profile_function
    from .queue import AsyncTaskQueue, Task, TaskStatus
    from .graphql import MCPGraphQL
    from .plugins import Plugin, PluginManager, ConnectorPlugin, MiddlewarePlugin
    from .compression import ResponseCompressor, CompressionType, get_compression_type
    from .health import HealthChecker, HealthCheck, HealthReport, HealthStatus
    # Advanced
    from .versioning import VersionedRouter, APIVersion, versioned_route
    from .discovery import ServiceRegistry, ServiceInfo, MCPServiceStatus
    from .pooling import ConnectionPool, DatabaseConnectionPool, HTTPConnectionPool
    from .metrics_dashboard import MetricsDashboard
    from .request_queue import RequestQueue, QueuedRequest, RequestPriority
    from .multitenancy import TenantManager, Tenant, TenantContext, tenant_middleware
    from .events import EventStore, EventPublisher, Event, EventType
    from .locking import LockManager, DistributedLock
    from .docs import APIDocumentation
    from .interceptors import RequestInterceptor, ResponseInterceptor
    from .cqrs import CQRSBus, Command, Query, CommandHandler, QueryHandler
    from .saga import SagaOrchestrator, Saga, SagaStep, SagaStepStatus
    from .message_queue import MessageQueue, Message, MessagePriority
    from .advanced_cache import AdvancedCache, CacheStrategy, CacheEntry
    from .validation import AdvancedValidator, ValidationRule, validate_request
    # Infrastructure
    from .load_balancer import LoadBalancer, BackendServer, LoadBalanceStrategy
    from .gateway import APIGateway, GatewayRoute
    from .websocket import WebSocketManager, WebSocketMessage
    from .analytics import AnalyticsCollector, AnalyticsEvent
    from .testing import MCPTestClient, create_mock_connector, create_mock_manifest, assert_mcp_response
    from .backup import BackupManager, BackupMetadata
    from .migration import MigrationManager, Migration
    from .user_rate_limit import UserRateLimiter
    from .throttling import Throttler, AdaptiveThrottler, ThrottleConfig
    from .monitoring import MonitoringSystem, Alert, AlertRule, AlertLevel
    from .feature_flags import FeatureFlagManager, FeatureFlag, FeatureFlagStatus, feature_flag
    from .quotas import QuotaManager, ResourceQuota, QuotaLimit
    from .ip_rate_limit import IPRateLimiter
    from .advanced_logging import StructuredLogger, RequestLogger, LogLevel
    from .optimization import PerformanceOptimizer
    from .endpoint_rate_limit import EndpointRateLimiter
    from .signing import RequestSigner, RequestVerifier
    from .cost_tracking import CostTracker, CostEntry
    from .deduplication import RequestDeduplicator
    from .batch_optimizer import BatchOptimizer
    from .api_analytics import APIUsageAnalytics, APIUsageMetric
    from .advanced_transforms import TransformPipeline, AdvancedRequestTransformer, AdvancedResponseTransformer
    from .adaptive_rate_limit import AdaptiveRateLimiter
    from .dependency_health import DependencyHealthChecker, DependencyHealthCheck, DependencyType
    from .metrics_circuit_breaker import MetricsCircuitBreaker, CircuitBreakerMetrics

