"""API layer for multi-model feature"""

# Legacy router (still available for backward compatibility)
from .router import router
from .websocket import router as websocket_router

# New modular routers
from .routers import (
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
    streaming_router
)

# Exception handling
from .exceptions import (
    MultiModelAPIException,
    ModelExecutionException,
    RateLimitExceededException,
    CacheException,
    ValidationException,
    ModelNotFoundException,
    StrategyNotFoundException,
    TimeoutException
)
from .exception_handlers import register_exception_handlers

# Dependencies
from .dependencies import (
    get_execution_service,
    get_model_repository,
    get_cache_service,
    get_consensus_service,
    get_strategy_factory,
    check_rate_limit
)

# Schemas
from .schemas import (
    ModelType,
    ModelConfig,
    MultiModelRequest,
    MultiModelResponse,
    ModelResponse,
    ModelStatus,
    ModelsListResponse,
    BatchMultiModelRequest,
    BatchMultiModelResponse
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
    "MultiModelAPIException",
    "ModelExecutionException",
    "RateLimitExceededException",
    "CacheException",
    "ValidationException",
    "ModelNotFoundException",
    "StrategyNotFoundException",
    "TimeoutException",
    "register_exception_handlers",
    # Dependencies
    "get_execution_service",
    "get_model_repository",
    "get_cache_service",
    "get_consensus_service",
    "get_strategy_factory",
    "check_rate_limit",
    # Schemas
    "ModelType",
    "ModelConfig",
    "MultiModelRequest",
    "MultiModelResponse",
    "ModelResponse",
    "ModelStatus",
    "ModelsListResponse",
    "BatchMultiModelRequest",
    "BatchMultiModelResponse"
]

