from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from domain.entities import (
from domain.interfaces import (
from typing import Any, List, Dict, Optional
"""
Application Use Cases
=====================

Business logic and orchestration for the copywriting system.
"""


    CopywritingRequest,
    CopywritingResponse,
    PerformanceMetrics,
    RequestStatus,
    CopywritingStyle,
    CopywritingTone
)
    CopywritingRepository,
    CacheService,
    AIService,
    EventPublisher,
    MonitoringService
)

logger = logging.getLogger(__name__)


class GenerateCopywritingUseCase:
    """Use case for generating copywriting content."""
    
    def __init__(
        self,
        repository: CopywritingRepository,
        ai_service: AIService,
        cache_service: CacheService,
        event_publisher: EventPublisher,
        monitoring_service: Optional[MonitoringService] = None
    ):
        
    """__init__ function."""
self.repository = repository
        self.ai_service = ai_service
        self.cache_service = cache_service
        self.event_publisher = event_publisher
        self.monitoring_service = monitoring_service
    
    async def execute(self, request: CopywritingRequest) -> CopywritingResponse:
        """Execute the copywriting generation use case."""
        start_time = datetime.now()
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(request)
            cached_response = await self.cache_service.get(cache_key)
            
            if cached_response:
                logger.info(f"Cache hit for request {request.id}")
                await self._record_metrics("cache_hit", start_time)
                return cached_response
            
            # Validate request
            await self._validate_request(request)
            
            # Update request status
            request.mark_processing()
            await self.repository.save_request(request)
            
            # Publish event
            await self.event_publisher.publish("copywriting.requested", {
                "request_id": request.id,
                "prompt": request.prompt,
                "style": request.style.value,
                "tone": request.tone.value
            })
            
            # Generate copywriting
            response = await self.ai_service.generate_copywriting(request)
            
            # Save response
            await self.repository.save_response(response)
            
            # Update request status
            request.mark_completed()
            await self.repository.save_request(request)
            
            # Cache the response
            await self.cache_service.set(cache_key, response, ttl=3600)
            
            # Publish completion event
            await self.event_publisher.publish("copywriting.completed", {
                "request_id": request.id,
                "response_id": response.id,
                "processing_time": response.processing_time
            })
            
            # Record metrics
            await self._record_metrics("success", start_time, response.processing_time)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating copywriting for request {request.id}: {e}")
            
            # Update request status
            request.mark_failed()
            await self.repository.save_request(request)
            
            # Publish failure event
            await self.event_publisher.publish("copywriting.failed", {
                "request_id": request.id,
                "error": str(e)
            })
            
            # Record metrics
            await self._record_metrics("error", start_time)
            
            raise
    
    def _generate_cache_key(self, request: CopywritingRequest) -> str:
        """Generate cache key for request."""
        key_parts = [
            "copywriting",
            request.style.value,
            request.tone.value,
            str(request.length),
            str(request.creativity),
            request.language,
            request.prompt[:100]  # First 100 chars of prompt
        ]
        return ":".join(key_parts)
    
    async async def _validate_request(self, request: CopywritingRequest) -> None:
        """Validate copywriting request."""
        if not request.can_be_processed():
            raise ValueError("Request cannot be processed in current status")
        
        if len(request.prompt) > 1000:
            raise ValueError("Prompt too long (max 1000 characters)")
        
        if request.length < 10 or request.length > 2000:
            raise ValueError("Invalid length (10-2000 words)")
    
    async def _record_metrics(self, status: str, start_time: datetime, processing_time: Optional[float] = None):
        """Record metrics for monitoring."""
        if not self.monitoring_service:
            return
        
        duration = (datetime.now() - start_time).total_seconds()
        
        await self.monitoring_service.increment_counter(f"copywriting.requests.{status}")
        await self.monitoring_service.record_timing("copywriting.processing_time", duration)
        
        if processing_time:
            await self.monitoring_service.record_timing("copywriting.ai_processing_time", processing_time)


class GetCopywritingHistoryUseCase:
    """Use case for retrieving copywriting history."""
    
    def __init__(self, repository: CopywritingRepository):
        
    """__init__ function."""
self.repository = repository
    
    async def execute(self, user_id: str, limit: int = 50, offset: int = 0) -> List[CopywritingRequest]:
        """Execute the history retrieval use case."""
        try:
            history = await self.repository.get_user_history(user_id, limit + offset)
            return history[offset:offset + limit]
        except Exception as e:
            logger.error(f"Error retrieving history for user {user_id}: {e}")
            raise


class AnalyzeCopywritingUseCase:
    """Use case for analyzing copywriting content."""
    
    def __init__(self, ai_service: AIService, monitoring_service: Optional[MonitoringService] = None):
        
    """__init__ function."""
self.ai_service = ai_service
        self.monitoring_service = monitoring_service
    
    async def execute(self, text: str, analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """Execute the analysis use case."""
        start_time = datetime.now()
        
        try:
            # Analyze text
            analysis = await self.ai_service.analyze_text(text)
            
            # Get suggestions
            suggestions = await self.ai_service.get_suggestions(text, analysis)
            
            # Record metrics
            if self.monitoring_service:
                duration = (datetime.now() - start_time).total_seconds()
                await self.monitoring_service.record_timing("analysis.processing_time", duration)
                await self.monitoring_service.increment_counter("analysis.requests")
            
            return {
                "analysis": analysis,
                "suggestions": suggestions,
                "processing_time": (datetime.now() - start_time).total_seconds()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing text: {e}")
            raise


class ImproveCopywritingUseCase:
    """Use case for improving existing copywriting."""
    
    def __init__(
        self,
        ai_service: AIService,
        event_publisher: EventPublisher,
        monitoring_service: Optional[MonitoringService] = None
    ):
        
    """__init__ function."""
self.ai_service = ai_service
        self.event_publisher = event_publisher
        self.monitoring_service = monitoring_service
    
    async def execute(self, original_text: str, improvements: List[str]) -> str:
        """Execute the improvement use case."""
        start_time = datetime.now()
        
        try:
            # Improve text
            improved_text = await self.ai_service.improve_copywriting(original_text, improvements)
            
            # Publish event
            await self.event_publisher.publish("copywriting.improved", {
                "original_length": len(original_text),
                "improved_length": len(improved_text),
                "improvements_applied": len(improvements)
            })
            
            # Record metrics
            if self.monitoring_service:
                duration = (datetime.now() - start_time).total_seconds()
                await self.monitoring_service.record_timing("improvement.processing_time", duration)
                await self.monitoring_service.increment_counter("improvement.requests")
            
            return improved_text
            
        except Exception as e:
            logger.error(f"Error improving text: {e}")
            raise


class BatchGenerateCopywritingUseCase:
    """Use case for batch copywriting generation."""
    
    def __init__(
        self,
        repository: CopywritingRepository,
        ai_service: AIService,
        cache_service: CacheService,
        event_publisher: EventPublisher,
        monitoring_service: Optional[MonitoringService] = None
    ):
        
    """__init__ function."""
self.repository = repository
        self.ai_service = ai_service
        self.cache_service = cache_service
        self.event_publisher = event_publisher
        self.monitoring_service = monitoring_service
    
    async def execute(self, requests: List[CopywritingRequest], max_concurrent: int = 5) -> List[CopywritingResponse]:
        """Execute batch generation use case."""
        start_time = datetime.now()
        
        try:
            # Process requests in batches
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async async def process_request(request: CopywritingRequest) -> CopywritingResponse:
                async with semaphore:
                    return await self._process_single_request(request)
            
            # Execute all requests concurrently
            tasks = [process_request(request) for request in requests]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions
            valid_responses = []
            for i, response in enumerate(responses):
                if isinstance(response, Exception):
                    logger.error(f"Error processing request {requests[i].id}: {response}")
                else:
                    valid_responses.append(response)
            
            # Record batch metrics
            if self.monitoring_service:
                duration = (datetime.now() - start_time).total_seconds()
                await self.monitoring_service.record_timing("batch.processing_time", duration)
                await self.monitoring_service.increment_counter("batch.requests", len(requests))
                await self.monitoring_service.increment_counter("batch.successful", len(valid_responses))
            
            return valid_responses
            
        except Exception as e:
            logger.error(f"Error in batch generation: {e}")
            raise
    
    async async def _process_single_request(self, request: CopywritingRequest) -> CopywritingResponse:
        """Process a single request within the batch."""
        # Reuse the single generation use case
        single_use_case = GenerateCopywritingUseCase(
            self.repository,
            self.ai_service,
            self.cache_service,
            self.event_publisher,
            self.monitoring_service
        )
        return await single_use_case.execute(request)


class GetPerformanceMetricsUseCase:
    """Use case for retrieving performance metrics."""
    
    def __init__(self, monitoring_service: MonitoringService):
        
    """__init__ function."""
self.monitoring_service = monitoring_service
    
    async def execute(self, time_range: str = "1h") -> Dict[str, Any]:
        """Execute the metrics retrieval use case."""
        try:
            metrics = await self.monitoring_service.get_metrics()
            health_status = await self.monitoring_service.get_health_status()
            
            return {
                "metrics": metrics,
                "health_status": health_status,
                "time_range": time_range,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error retrieving metrics: {e}")
            raise


class ValidatePromptUseCase:
    """Use case for validating copywriting prompts."""
    
    def __init__(self, ai_service: AIService):
        
    """__init__ function."""
self.ai_service = ai_service
    
    async def execute(self, prompt: str) -> Dict[str, Any]:
        """Execute the prompt validation use case."""
        try:
            validation = await self.ai_service.validate_prompt(prompt)
            
            # Add additional validation logic
            validation["length"] = len(prompt)
            validation["word_count"] = len(prompt.split())
            validation["has_special_chars"] = any(char in prompt for char in "!@#$%^&*()")
            
            return validation
            
        except Exception as e:
            logger.error(f"Error validating prompt: {e}")
            raise 