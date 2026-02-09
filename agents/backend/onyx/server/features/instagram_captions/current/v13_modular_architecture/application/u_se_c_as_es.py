from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from typing import List, Dict, Any, Optional
from ..domain.entities import CaptionRequest, CaptionResponse, BatchRequest, BatchResponse
from ..domain.services import QualityAssessmentService, HashtagOptimizationService
from ..domain.repositories import ICacheRepository, IMetricsRepository, IAuditRepository
from ..interfaces.ai_providers import IAIProvider, IAIProviderRegistry
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Instagram Captions API v13.0 - Application Use Cases

Application layer use cases orchestrating business logic.
Clean Architecture application services.
"""



class GenerateCaptionUseCase:
    """Use case for generating single caption."""
    
    def __init__(
        self,
        ai_provider: IAIProvider,
        cache_repository: ICacheRepository,
        metrics_repository: IMetricsRepository,
        audit_repository: IAuditRepository
    ):
        
    """__init__ function."""
self.ai_provider = ai_provider
        self.cache_repository = cache_repository
        self.metrics_repository = metrics_repository
        self.audit_repository = audit_repository
    
    async def execute(self, request: CaptionRequest) -> CaptionResponse:
        """Execute caption generation use case."""
        
        # Check cache first
        cache_key = request.cache_key
        cached_response = await self.cache_repository.get(cache_key)
        if cached_response:
            cached_response.request_id = request.request_id
            return cached_response
        
        # Generate caption using AI
        caption = await self.ai_provider.generate_caption(
            content_description=request.content.description,
            style=request.style,
            custom_instructions=request.custom_instructions
        )
        
        # Generate optimized hashtags
        hashtags = HashtagOptimizationService.generate_optimized_hashtags(
            content=request.content,
            style=request.style,
            target_count=request.hashtag_count
        )
        
        # Assess quality
        quality_metrics = QualityAssessmentService.assess_caption_quality(
            caption=caption,
            content=request.content,
            style=request.style,
            hashtags=hashtags
        )
        
        # Create response
        response = CaptionResponse(
            request_id=request.request_id,
            caption=caption,
            hashtags=hashtags,
            quality_metrics=quality_metrics,
            performance_metrics=performance_metrics,
            tenant_id=request.tenant_id
        )
        
        # Cache response
        await self.cache_repository.set(cache_key, response)
        
        # Record metrics and audit
        await self.metrics_repository.record_request(request, response)
        await self.audit_repository.log_request(request, response)
        
        return response


class GenerateBatchCaptionsUseCase:
    """Use case for generating batch captions."""
    
    def __init__(
        self,
        provider_registry: IAIProviderRegistry,
        cache_repository: ICacheRepository,
        metrics_repository: IMetricsRepository
    ):
        
    """__init__ function."""
self.provider_registry = provider_registry
        self.cache_repository = cache_repository
        self.metrics_repository = metrics_repository
    
    async def execute(self, batch_request: BatchRequest) -> BatchResponse:
        """Execute batch caption generation."""
        
        # Process requests in parallel
        tasks = [
            self._process_single_request(req) 
            for req in batch_request.requests
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Separate successful and failed responses
        successful_responses = []
        failed_requests = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed_requests.append({
                    "index": i,
                    "error": str(result),
                    "request_id": batch_request.requests[i].request_id.value
                })
            else:
                successful_responses.append(result)
        
        # Create batch response
        batch_response = BatchResponse(
            batch_id=batch_request.batch_id,
            responses=successful_responses,
            failed_requests=failed_requests,
            total_requests=len(batch_request.requests),
            successful_requests=len(successful_responses),
            failed_requests_count=len(failed_requests),
            processing_time=time.time() - start_time,
            average_quality=sum(r.quality_metrics.score for r in successful_responses) / max(len(successful_responses), 1)
        )
        
        # Record batch metrics
        await self.metrics_repository.record_batch(batch_request, batch_response)
        
        return batch_response 