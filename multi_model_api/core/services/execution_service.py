"""
Execution service for multi-model API
Main service for executing multi-model requests
"""

import asyncio
import logging
import time
import uuid
from typing import List, Optional, Dict, Any, Union
from fastapi import BackgroundTasks
from datetime import datetime

from ...api.schemas import (
    MultiModelRequest,
    MultiModelResponse,
    ModelConfig,
    ModelResponse
)
from ...api.exceptions import (
    ValidationException,
    TimeoutException,
    ModelExecutionException
)
from ...api.helpers import (
    build_model_kwargs,
    validate_enabled_models,
    validate_responses,
    build_response_data,
    calculate_latency_ms,
    get_enabled_models,
    calculate_response_stats
)
from ..repositories import ModelRepository
from ..strategies import StrategyFactory, ExecutionStrategy
from .cache_service import CacheService
from .consensus_service import ConsensusService
from .validation_service import ValidationService
from .metrics_service import MetricsService, get_metrics_service
from .performance_service import PerformanceService, get_performance_service

logger = logging.getLogger(__name__)


class ExecutionService:
    """Service for executing multi-model requests"""
    
    def __init__(
        self,
        model_repository: ModelRepository,
        cache_service: CacheService,
        consensus_service: ConsensusService,
        strategy_factory: StrategyFactory,
        metrics_service: Optional[MetricsService] = None,
        performance_service: Optional[PerformanceService] = None
    ):
        """
        Initialize execution service
        
        Args:
            model_repository: Repository for model operations
            cache_service: Service for cache operations
            consensus_service: Service for consensus operations
            strategy_factory: Factory for creating execution strategies
            metrics_service: Optional metrics service (uses singleton if not provided)
            performance_service: Optional performance service (uses singleton if not provided)
        """
        self.model_repository = model_repository
        self.cache_service = cache_service
        self.consensus_service = consensus_service
        self.strategy_factory = strategy_factory
        self.metrics_service = metrics_service or get_metrics_service()
        self.performance_service = performance_service or get_performance_service()
    
    
    async def _execute_model(
        self,
        model: ModelConfig,
        prompt: str,
        timeout: Optional[float] = None,
        **kwargs
    ) -> ModelResponse:
        """Execute a single model - extracted method for reusability"""
        model_kwargs = build_model_kwargs(model, timeout)
        model_kwargs.update(kwargs)
        return await self.model_repository.execute_model(
            model.model_type,
            prompt,
            **model_kwargs
        )
    
    async def _execute_models(
        self,
        models: List[ModelConfig],
        prompt: str,
        strategy: str,
        timeout: Optional[float] = None,
        consensus_method: Optional[str] = None,
        weights: Optional[Dict[str, float]] = None
    ) -> List[ModelResponse]:
        """
        Execute models using specified strategy
        
        Args:
            models: List of model configurations
            prompt: Input prompt
            strategy: Execution strategy
            timeout: Optional timeout
            consensus_method: Optional consensus method
            weights: Optional weights for consensus
            
        Returns:
            List of model responses
        """
        # Get execution strategy
        execution_strategy = self.strategy_factory.create(strategy)
        
        # Create model execution function wrapper
        async def execute_model_func(model: ModelConfig, prompt: str, **kwargs) -> ModelResponse:
            """Wrapper for model execution"""
            return await self._execute_model(model, prompt, timeout, **kwargs)
        
        # Prepare execution kwargs
        execution_kwargs = {
            "models": models,
            "prompt": prompt,
            "execute_model_func": execute_model_func,
            "timeout": timeout
        }
        
        # Add consensus-specific kwargs if needed
        if strategy == "consensus":
            execution_kwargs["consensus_method"] = consensus_method
            execution_kwargs["weights"] = weights
        
        # Execute using strategy
        return await execution_strategy.execute(**execution_kwargs)
    
    async def execute(
        self,
        request: MultiModelRequest,
        request_id: Optional[str] = None,
        background_task: Optional[BackgroundTasks] = None
    ) -> MultiModelResponse:
        """
        Execute multi-model request
        
        Args:
            request: Multi-model request
            request_id: Optional request ID (generated if not provided)
            background_task: Optional background task for async operations
            
        Returns:
            MultiModelResponse with results
            
        Raises:
            ValidationException: If request is invalid
            TimeoutException: If execution times out
            ModelExecutionException: If execution fails
        """
        request_id = request_id or str(uuid.uuid4())
        start_time = time.time()
        
        try:
            # 1. Validate request
            ValidationService.validate_request(request)
            
            enabled_models = get_enabled_models(request.models)
            
            # 2. Check cache
            cached_response = await self._check_cache(request, request_id, enabled_models, start_time)
            if cached_response:
                return cached_response
            
            # 3. Execute models
            timeout = request.timeout or ValidationService.DEFAULT_TIMEOUT
            
            try:
                responses = await self._execute_models(
                    models=enabled_models,
                    prompt=request.prompt,
                    strategy=request.strategy,
                    timeout=timeout,
                    consensus_method=request.consensus_method,
                    weights=None  # Will be calculated in consensus service
                )
            except asyncio.TimeoutError:
                logger.error(f"Request {request_id} timed out after {timeout}s")
                raise TimeoutException(
                    timeout=timeout,
                    details={"request_id": request_id, "strategy": request.strategy}
                )
            except Exception as e:
                logger.error(
                    f"Error executing models for request {request_id}: {e}",
                    exc_info=True,
                    extra={
                        "request_id": request_id,
                        "strategy": request.strategy,
                        "models_count": len(enabled_models)
                    }
                )
                raise ModelExecutionException(
                    message=f"Failed to execute models: {str(e)}",
                    details={"request_id": request_id}
                )
            
            # 4. Validate responses
            validate_responses(responses, request)
            
            # 5. Aggregate responses
            aggregated_response = self._aggregate_responses(
                responses, request, enabled_models
            )
            
            # 6. Build response
            response_data = build_response_data(
                request_id=request_id,
                request=request,
                responses=responses,
                aggregated_response=aggregated_response,
                start_time=start_time,
                enabled_models=enabled_models
            )
            
            response = MultiModelResponse(**response_data)
            
            # 7. Record metrics
            self._record_metrics(
                request_id, request, enabled_models, responses, start_time
            )
            
            # 8. Cache result (async, non-blocking)
            if request.cache_enabled:
                await self.cache_service.cache_response(
                    request,
                    response,
                    background_task
                )
            
            return response
            
        except (ValidationException, TimeoutException, ModelExecutionException):
            # Re-raise known exceptions without modification
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error in execution service for request {request_id}: {e}",
                exc_info=True,
                extra={"request_id": request_id, "error_type": type(e).__name__}
            )
            raise ModelExecutionException(
                message=f"Unexpected error: {str(e)}",
                details={"request_id": request_id}
            )
    
    async def _check_cache(
        self,
        request: MultiModelRequest,
        request_id: str,
        enabled_models: List[ModelConfig],
        start_time: float
    ) -> Optional[MultiModelResponse]:
        """Check cache and return cached response if available"""
        if not request.cache_enabled:
            return None
        
        cached_response = await self.cache_service.get_cached_response(request)
        if not cached_response:
            return None
        
        logger.info(
            f"Cache hit for request {request_id}",
            extra={"request_id": request_id, "strategy": request.strategy}
        )
        
        # Record metrics for cache hit
        cache_latency_ms = calculate_latency_ms(start_time)
        
        self.metrics_service.record_request(
            request_id=request_id,
            strategy=request.strategy,
            models_count=len(enabled_models),
            success_count=len(enabled_models),
            failure_count=0,
            total_latency_ms=cache_latency_ms,
            cache_hit=True
        )
        
        self.performance_service.record_request(
            latency_ms=cache_latency_ms,
            is_error=False,
            cache_hit=True
        )
        
        return cached_response
    
    def _aggregate_responses(
        self,
        responses: List[ModelResponse],
        request: MultiModelRequest,
        enabled_models: List[ModelConfig]
    ) -> Optional[str]:
        """Aggregate model responses using consensus service"""
        weights_map = {
            m.model_type.value: m.multiplier
            for m in enabled_models
        }
        
        return self.consensus_service.aggregate_responses(
            responses=responses,
            strategy=request.strategy,
            consensus_method=request.consensus_method or "majority",
            weights=weights_map,
            enabled_models=enabled_models
        )
    
    def _record_metrics(
        self,
        request_id: str,
        request: MultiModelRequest,
        enabled_models: List[ModelConfig],
        responses: List[ModelResponse],
        start_time: float
    ) -> None:
        """Record metrics for request execution"""
        success_count, failure_count = calculate_response_stats(responses)
        total_latency_ms = calculate_latency_ms(start_time)
        is_error = failure_count > 0 and not request.allow_partial_success
        
        self.metrics_service.record_request(
            request_id=request_id,
            strategy=request.strategy,
            models_count=len(enabled_models),
            success_count=success_count,
            failure_count=failure_count,
            total_latency_ms=total_latency_ms,
            cache_hit=False
        )
        
        self.performance_service.record_request(
            latency_ms=total_latency_ms,
            is_error=is_error,
            cache_hit=False
        )
        
        # Record individual model metrics
        for model_response in responses:
            self.metrics_service.record_model_execution(
                model_type=model_response.model_type.value,
                success=model_response.success,
                latency_ms=model_response.latency_ms or 0.0
            )

