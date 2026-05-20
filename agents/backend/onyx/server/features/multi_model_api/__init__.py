"""
Multi-Model API Feature
========================

Optimized multi-model API with enhanced caching, circuit breakers, and health monitoring.
"""

__version__ = "2.7.0"
__author__ = "Blatam Academy"
__license__ = "Proprietary"
__description__ = "Optimized multi-model API with enhanced caching, circuit breakers, and health monitoring"

# Legacy API (backward compatibility)
from .api import router, websocket_router

# New modular API
from .api import (
    execution_router,
    models_router,
    health_router,
    cache_router,
    rate_limit_router,
    metrics_router,
    metrics_advanced_router,
    performance_router,
    openrouter_router,
    batch_router,
    streaming_router,
    register_exception_handlers,
    get_execution_service,
    get_model_repository,
    get_cache_service,
    get_consensus_service,
    get_strategy_factory
)

# Core components
from .core import (
    ModelRegistry,
    get_registry,
    EnhancedCache,
    get_cache,
    HealthMonitor,
    get_health_monitor
)

# Services (new)
from .core.services import (
    ExecutionService,
    CacheService,
    ConsensusService,
    PerformanceService,
    get_performance_service
)

# Strategies (new)
from .core.strategies import (
    ExecutionStrategy,
    ParallelStrategy,
    SequentialStrategy,
    ConsensusStrategy,
    StrategyFactory
)

# Repositories (new)
from .core.repositories import (
    ModelRepository,
    RegistryModelRepository
)

# Middleware
from .core.middleware import (
    MetricsMiddleware,
    LoggingMiddleware,
    init_sentry
)
from .core.middleware_context import ContextMiddleware

# Performance utilities
from .core.performance import (
    fast_json_dumps,
    fast_json_loads,
    parallel_map,
    batch_process
)
from .core.response_optimizer import ResponseOptimizer

# Exceptions (new)
from .api.exceptions import (
    MultiModelAPIException,
    ModelExecutionException,
    RateLimitExceededException,
    CacheException,
    ValidationException,
    ModelNotFoundException,
    StrategyNotFoundException,
    TimeoutException
)

__all__ = [
    # Legacy routers (backward compatibility)
    "router",
    "websocket_router",
    # New modular routers
    "execution_router",
    "models_router",
    "health_router",
    "cache_router",
    "rate_limit_router",
    "metrics_router",
    "metrics_advanced_router",
    "performance_router",
    "openrouter_router",
    "batch_router",
    "streaming_router",
    # Exception handling
    "register_exception_handlers",
    "MultiModelAPIException",
    "ModelExecutionException",
    "RateLimitExceededException",
    "CacheException",
    "ValidationException",
    "ModelNotFoundException",
    "StrategyNotFoundException",
    "TimeoutException",
    # Dependencies
    "get_execution_service",
    "get_model_repository",
    "get_cache_service",
    "get_consensus_service",
    "get_strategy_factory",
    # Core components
    "ModelRegistry",
    "get_registry",
    "EnhancedCache",
    "get_cache",
    "HealthMonitor",
    "get_health_monitor",
    # Services
    "ExecutionService",
    "CacheService",
    "ConsensusService",
    "PerformanceService",
    "get_performance_service",
    # Strategies
    "ExecutionStrategy",
    "ParallelStrategy",
    "SequentialStrategy",
    "ConsensusStrategy",
    "StrategyFactory",
    # Repositories
    "ModelRepository",
    "RegistryModelRepository",
    # Middleware
    "MetricsMiddleware",
    "LoggingMiddleware",
    "init_sentry",
    "ContextMiddleware",
    # Performance
    "fast_json_dumps",
    "fast_json_loads",
    "parallel_map",
    "batch_process",
    "ResponseOptimizer"
]

