"""
Services Module
===============
Main services package with organized exports
"""

# Core Services
from .clothing_service import ClothingChangeService, ClothingChangeRequest, ClothingChangeResult
from .batch_service import BatchProcessingService, BatchOperation, BatchItem, BatchStatus
from .comfyui_service import ComfyUIService, ComfyUIConfig

# Infrastructure Services
from .cache_service import CacheService, get_cache_service
from .webhook_service import WebhookService, WebhookEvent, WebhookConfig, get_webhook_service
from .metrics_service import MetricsService, get_metrics_service
from .health_service import HealthService

# Advanced Services
from .job_queue_service import JobQueueService, Job, JobPriority, JobStatus, get_job_queue_service
from .scheduler_service import SchedulerService, ScheduledTask, ScheduleType, get_scheduler_service
from .event_bus_service import EventBusService, Event, get_event_bus_service
from .circuit_breaker_service import (
    CircuitBreakerService, CircuitBreaker, CircuitBreakerConfig, CircuitState,
    get_circuit_breaker_service
)
from .feature_flags_service import (
    FeatureFlagsService, FeatureFlag, FeatureFlagType, get_feature_flags_service
)
from .rate_limiter_service import (
    RateLimiterService, RateLimitConfig, RateLimitStrategy, RateLimitResult,
    rate_limiter_service
)
from .notification_service import (
    NotificationService, Notification, NotificationChannel, NotificationPriority,
    notification_service
)
from .analytics_service import (
    AnalyticsService, AnalyticsEvent, AnalyticsInsight, analytics_service
)

# Base Classes
from .base import (
    BaseService, ServiceStatistics,
    ServiceRegistry, get_service_registry, register_service, get_service,
    singleton, SingletonMeta, get_or_create_service
)

__all__ = [
    # Core Services
    'ClothingChangeService',
    'ClothingChangeRequest',
    'ClothingChangeResult',
    'BatchProcessingService',
    'BatchOperation',
    'BatchItem',
    'BatchStatus',
    'ComfyUIService',
    'ComfyUIConfig',
    # Infrastructure Services
    'CacheService',
    'get_cache_service',
    'WebhookService',
    'WebhookEvent',
    'WebhookConfig',
    'get_webhook_service',
    'MetricsService',
    'get_metrics_service',
    'HealthService',
    # Advanced Services
    'JobQueueService',
    'Job',
    'JobPriority',
    'JobStatus',
    'get_job_queue_service',
    'SchedulerService',
    'ScheduledTask',
    'ScheduleType',
    'get_scheduler_service',
    'EventBusService',
    'Event',
    'get_event_bus_service',
    'CircuitBreakerService',
    'CircuitBreaker',
    'CircuitBreakerConfig',
    'CircuitState',
    'get_circuit_breaker_service',
    'FeatureFlagsService',
    'FeatureFlag',
    'FeatureFlagType',
    'get_feature_flags_service',
    'RateLimiterService',
    'RateLimitConfig',
    'RateLimitStrategy',
    'RateLimitResult',
    'rate_limiter_service',
    'NotificationService',
    'Notification',
    'NotificationChannel',
    'NotificationPriority',
    'notification_service',
    'AnalyticsService',
    'AnalyticsEvent',
    'AnalyticsInsight',
    'analytics_service',
    # Base Classes
    'BaseService',
    'ServiceStatistics',
    'ServiceRegistry',
    'get_service_registry',
    'register_service',
    'get_service',
    'singleton',
    'SingletonMeta',
    'get_or_create_service'
]
