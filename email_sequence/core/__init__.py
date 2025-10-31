"""
Core module for Email Sequence System

This module contains the core functionality including configuration,
database integration, caching, monitoring, and dependency injection.
"""

from .config import get_settings, Settings
from .database import (
    Base,
    EmailSequenceModel,
    SequenceStepModel,
    SequenceTriggerModel,
    EmailTemplateModel,
    SubscriberModel,
    SequenceSubscriberModel,
    EmailCampaignModel,
    AnalyticsEventModel,
    DatabaseManager,
    db_manager,
    get_database_session,
    init_database,
    close_database,
    check_database_health
)
from .cache import (
    CacheManager,
    cache_manager,
    SequenceCache,
    SubscriberCache,
    TemplateCache,
    cached,
    cache_key,
    init_cache,
    close_cache,
    check_cache_health
)
from .dependencies import (
    get_engine,
    get_database,
    get_redis,
    get_current_user,
    require_permission,
    rate_limit,
    get_cached_data,
    set_cached_data,
    check_database_health as deps_check_database_health,
    check_redis_health,
    check_services_health,
    lifespan
)
from .exceptions import (
    EmailSequenceError,
    SequenceNotFoundError,
    InvalidSequenceError,
    SequenceAlreadyActiveError,
    SequenceNotActiveError,
    SubscriberNotFoundError,
    InvalidSubscriberError,
    DuplicateSubscriberError,
    TemplateNotFoundError,
    InvalidTemplateError,
    CampaignNotFoundError,
    InvalidCampaignError,
    EmailDeliveryError,
    LangChainServiceError,
    AnalyticsServiceError,
    DatabaseError,
    CacheError,
    RateLimitError,
    ValidationError,
    AuthenticationError,
    AuthorizationError,
    ConfigurationError,
    ExternalServiceError,
    WebhookError,
    A_BTestError
)
from .middleware import (
    RequestIDMiddleware,
    RateLimitMiddleware,
    AuthenticationMiddleware,
    CacheMiddleware,
    PerformanceMiddleware,
    SecurityHeadersMiddleware,
    LoggingMiddleware,
    ErrorTrackingMiddleware,
    CORSMiddleware,
    setup_custom_middleware
)
from .monitoring import (
    PerformanceMonitor,
    StructuredLogger,
    HealthChecker,
    performance_monitor,
    structured_logger,
    health_checker,
    monitor_operation,
    monitor_redis_operation,
    init_monitoring,
    close_monitoring,
    get_performance_summary,
    get_health_status
)

__all__ = [
    # Configuration
    "get_settings",
    "Settings",
    
    # Database
    "Base",
    "EmailSequenceModel",
    "SequenceStepModel", 
    "SequenceTriggerModel",
    "EmailTemplateModel",
    "SubscriberModel",
    "SequenceSubscriberModel",
    "EmailCampaignModel",
    "AnalyticsEventModel",
    "DatabaseManager",
    "db_manager",
    "get_database_session",
    "init_database",
    "close_database",
    "check_database_health",
    
    # Caching
    "CacheManager",
    "cache_manager",
    "SequenceCache",
    "SubscriberCache",
    "TemplateCache",
    "cached",
    "cache_key",
    "init_cache",
    "close_cache",
    "check_cache_health",
    
    # Dependencies
    "get_engine",
    "get_database",
    "get_redis",
    "get_current_user",
    "require_permission",
    "rate_limit",
    "get_cached_data",
    "set_cached_data",
    "deps_check_database_health",
    "check_redis_health",
    "check_services_health",
    "lifespan",
    
    # Exceptions
    "EmailSequenceError",
    "SequenceNotFoundError",
    "InvalidSequenceError",
    "SequenceAlreadyActiveError",
    "SequenceNotActiveError",
    "SubscriberNotFoundError",
    "InvalidSubscriberError",
    "DuplicateSubscriberError",
    "TemplateNotFoundError",
    "InvalidTemplateError",
    "CampaignNotFoundError",
    "InvalidCampaignError",
    "EmailDeliveryError",
    "LangChainServiceError",
    "AnalyticsServiceError",
    "DatabaseError",
    "CacheError",
    "RateLimitError",
    "ValidationError",
    "AuthenticationError",
    "AuthorizationError",
    "ConfigurationError",
    "ExternalServiceError",
    "WebhookError",
    "A_BTestError",
    
    # Middleware
    "RequestIDMiddleware",
    "RateLimitMiddleware",
    "AuthenticationMiddleware",
    "CacheMiddleware",
    "PerformanceMiddleware",
    "SecurityHeadersMiddleware",
    "LoggingMiddleware",
    "ErrorTrackingMiddleware",
    "CORSMiddleware",
    "setup_custom_middleware",
    
    # Monitoring
    "PerformanceMonitor",
    "StructuredLogger",
    "HealthChecker",
    "performance_monitor",
    "structured_logger",
    "health_checker",
    "monitor_operation",
    "monitor_redis_operation",
    "init_monitoring",
    "close_monitoring",
    "get_performance_summary",
    "get_health_status"
]