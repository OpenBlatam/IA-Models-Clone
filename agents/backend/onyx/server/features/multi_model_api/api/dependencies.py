"""
Dependency injection for Multi-Model API
Provides FastAPI dependencies for all services and repositories
"""

from functools import lru_cache
from fastapi import Depends, Request, Header
from typing import Optional

from ..core.models import ModelRegistry, get_registry
from ..core.cache import EnhancedCache, get_cache
from ..core.repositories import RegistryModelRepository, ModelRepository
from ..core.services import (
    ExecutionService,
    CacheService,
    ConsensusService,
    MetricsService,
    PerformanceService,
    get_metrics_service,
    get_performance_service
)
from ..core.strategies import StrategyFactory
from ..core.rate_limiter import get_rate_limiter, RateLimitInfo
from ..api.helpers import get_client_identifier


def get_model_registry() -> ModelRegistry:
    """Get model registry instance"""
    return get_registry()


def get_cache_instance() -> EnhancedCache:
    """Get cache instance"""
    return get_cache()


@lru_cache(maxsize=1)
def get_model_repository(
    registry: ModelRegistry = Depends(get_model_registry)
) -> ModelRepository:
    """
    Get model repository instance
    
    Args:
        registry: ModelRegistry instance
        
    Returns:
        ModelRepository instance
    """
    return RegistryModelRepository(registry)


@lru_cache(maxsize=1)
def get_strategy_factory() -> StrategyFactory:
    """
    Get strategy factory instance
    
    Returns:
        StrategyFactory instance
    """
    return StrategyFactory()


@lru_cache(maxsize=1)
def get_cache_service(
    cache: EnhancedCache = Depends(get_cache_instance)
) -> CacheService:
    """
    Get cache service instance
    
    Args:
        cache: EnhancedCache instance
        
    Returns:
        CacheService instance
    """
    return CacheService(cache)


@lru_cache(maxsize=1)
def get_consensus_service() -> ConsensusService:
    """
    Get consensus service instance
    
    Returns:
        ConsensusService instance
    """
    return ConsensusService()


@lru_cache(maxsize=1)
def get_metrics_service_dependency() -> MetricsService:
    """
    Get metrics service instance for dependency injection
    
    Returns:
        MetricsService instance
    """
    return get_metrics_service()


@lru_cache(maxsize=1)
def get_performance_service_dependency() -> PerformanceService:
    """
    Get performance service instance for dependency injection
    
    Returns:
        PerformanceService instance
    """
    return get_performance_service()


@lru_cache(maxsize=1)
def get_execution_service(
    repository: ModelRepository = Depends(get_model_repository),
    cache_service: CacheService = Depends(get_cache_service),
    consensus_service: ConsensusService = Depends(get_consensus_service),
    strategy_factory: StrategyFactory = Depends(get_strategy_factory),
    metrics_service: MetricsService = Depends(get_metrics_service_dependency),
    performance_service: PerformanceService = Depends(get_performance_service_dependency)
) -> ExecutionService:
    """
    Get execution service instance
    
    Args:
        repository: ModelRepository instance
        cache_service: CacheService instance
        consensus_service: ConsensusService instance
        strategy_factory: StrategyFactory instance
        metrics_service: MetricsService instance
        performance_service: PerformanceService instance
        
    Returns:
        ExecutionService instance
    """
    return ExecutionService(
        model_repository=repository,
        cache_service=cache_service,
        consensus_service=consensus_service,
        strategy_factory=strategy_factory,
        metrics_service=metrics_service,
        performance_service=performance_service
    )


async def check_rate_limit(
    request: Request,
    endpoint: str = "execute"
) -> RateLimitInfo:
    """
    Check rate limit for a request
    
    Args:
        request: FastAPI Request object
        endpoint: Endpoint name for rate limiting
        
    Returns:
        RateLimitInfo with rate limit status
    """
    client_id = get_client_identifier(request)
    rate_limiter = get_rate_limiter()
    return await rate_limiter.is_allowed(client_id, endpoint)

