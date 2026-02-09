from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import asyncio
import logging
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from domain.entities import (
from domain.interfaces import (
from .use_cases import (
from typing import Any, List, Dict, Optional
"""
Application Services
===================

High-level business operations that orchestrate use cases and domain logic.
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
    GenerateCopywritingUseCase,
    GetCopywritingHistoryUseCase,
    AnalyzeCopywritingUseCase,
    ImproveCopywritingUseCase,
    BatchGenerateCopywritingUseCase,
    GetPerformanceMetricsUseCase,
    ValidatePromptUseCase
)

logger = logging.getLogger(__name__)


class CopywritingApplicationService:
    """Main application service for copywriting operations."""
    
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
        
        # Initialize use cases
        self.generate_use_case = GenerateCopywritingUseCase(
            repository, ai_service, cache_service, event_publisher, monitoring_service
        )
        self.history_use_case = GetCopywritingHistoryUseCase(repository)
        self.analyze_use_case = AnalyzeCopywritingUseCase(ai_service, monitoring_service)
        self.improve_use_case = ImproveCopywritingUseCase(ai_service, event_publisher, monitoring_service)
        self.batch_use_case = BatchGenerateCopywritingUseCase(
            repository, ai_service, cache_service, event_publisher, monitoring_service
        )
        self.metrics_use_case = GetPerformanceMetricsUseCase(monitoring_service) if monitoring_service else None
        self.validate_use_case = ValidatePromptUseCase(ai_service)
    
    async def generate_copywriting(
        self,
        prompt: str,
        style: CopywritingStyle = CopywritingStyle.PROFESSIONAL,
        tone: CopywritingTone = CopywritingTone.NEUTRAL,
        length: int = 100,
        creativity: float = 0.7,
        language: str = "en",
        target_audience: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        user_id: Optional[str] = None
    ) -> CopywritingResponse:
        """Generate copywriting content with comprehensive options."""
        try:
            # Create request
            request = CopywritingRequest(
                prompt=prompt,
                style=style,
                tone=tone,
                length=length,
                creativity=creativity,
                language=language,
                target_audience=target_audience,
                keywords=keywords or [],
                metadata={"user_id": user_id} if user_id else {}
            )
            
            # Execute generation
            response = await self.generate_use_case.execute(request)
            
            logger.info(f"Generated copywriting for prompt: {prompt[:50]}...")
            return response
            
        except Exception as e:
            logger.error(f"Error in generate_copywriting: {e}")
            raise
    
    async def generate_multiple_variations(
        self,
        prompt: str,
        variations: int = 3,
        base_style: CopywritingStyle = CopywritingStyle.PROFESSIONAL,
        base_tone: CopywritingTone = CopywritingTone.NEUTRAL,
        length: int = 100,
        creativity_range: Tuple[float, float] = (0.5, 0.9),
        user_id: Optional[str] = None
    ) -> List[CopywritingResponse]:
        """Generate multiple variations of copywriting."""
        try:
            requests = []
            
            for i in range(variations):
                # Vary creativity within range
                creativity = creativity_range[0] + (i / (variations - 1)) * (creativity_range[1] - creativity_range[0])
                
                # Vary tone slightly
                tones = list(CopywritingTone)
                tone_index = (tones.index(base_tone) + i) % len(tones)
                
                request = CopywritingRequest(
                    prompt=prompt,
                    style=base_style,
                    tone=tones[tone_index],
                    length=length,
                    creativity=creativity,
                    metadata={"user_id": user_id, "variation": i} if user_id else {"variation": i}
                )
                requests.append(request)
            
            # Execute batch generation
            responses = await self.batch_use_case.execute(requests)
            
            logger.info(f"Generated {len(responses)} variations for prompt: {prompt[:50]}...")
            return responses
            
        except Exception as e:
            logger.error(f"Error in generate_multiple_variations: {e}")
            raise
    
    async def get_user_history(
        self,
        user_id: str,
        limit: int = 50,
        offset: int = 0,
        status_filter: Optional[RequestStatus] = None
    ) -> List[CopywritingRequest]:
        """Get user's copywriting history with optional filtering."""
        try:
            history = await self.history_use_case.execute(user_id, limit + offset, offset)
            
            # Apply status filter if provided
            if status_filter:
                history = [req for req in history if req.status == status_filter]
            
            logger.info(f"Retrieved {len(history)} history items for user {user_id}")
            return history
            
        except Exception as e:
            logger.error(f"Error in get_user_history: {e}")
            raise
    
    async def analyze_copywriting(
        self,
        text: str,
        analysis_type: str = "comprehensive"
    ) -> Dict[str, Any]:
        """Analyze copywriting content."""
        try:
            analysis = await self.analyze_use_case.execute(text, analysis_type)
            
            logger.info(f"Analyzed copywriting text: {text[:50]}...")
            return analysis
            
        except Exception as e:
            logger.error(f"Error in analyze_copywriting: {e}")
            raise
    
    async def improve_copywriting(
        self,
        original_text: str,
        improvements: List[str]
    ) -> str:
        """Improve existing copywriting content."""
        try:
            improved_text = await self.improve_use_case.execute(original_text, improvements)
            
            logger.info(f"Improved copywriting text: {original_text[:50]}...")
            return improved_text
            
        except Exception as e:
            logger.error(f"Error in improve_copywriting: {e}")
            raise
    
    async def validate_prompt(self, prompt: str) -> Dict[str, Any]:
        """Validate copywriting prompt."""
        try:
            validation = await self.validate_use_case.execute(prompt)
            
            logger.info(f"Validated prompt: {prompt[:50]}...")
            return validation
            
        except Exception as e:
            logger.error(f"Error in validate_prompt: {e}")
            raise
    
    async def get_performance_metrics(self, time_range: str = "1h") -> Dict[str, Any]:
        """Get system performance metrics."""
        try:
            if not self.metrics_use_case:
                raise ValueError("Monitoring service not available")
            
            metrics = await self.metrics_use_case.execute(time_range)
            
            logger.info(f"Retrieved performance metrics for time range: {time_range}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error in get_performance_metrics: {e}")
            raise
    
    async def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        try:
            # Get repository statistics
            repo_stats = await self.repository.get_statistics()
            
            # Get cache statistics
            cache_stats = await self.cache_service.get_cache_stats()
            
            # Get AI model info
            model_info = await self.ai_service.get_model_info()
            
            # Get event statistics
            event_stats = await self.event_publisher.get_event_stats()
            
            # Combine all statistics
            stats = {
                "repository": repo_stats,
                "cache": cache_stats,
                "ai_model": model_info,
                "events": event_stats,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info("Retrieved comprehensive system statistics")
            return stats
            
        except Exception as e:
            logger.error(f"Error in get_system_statistics: {e}")
            raise
    
    async def cleanup_old_data(self, days_old: int = 30) -> Dict[str, int]:
        """Clean up old data from the system."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_old)
            
            # Get old requests
            old_requests = await self.repository.get_requests_by_status(RequestStatus.COMPLETED)
            old_requests = [req for req in old_requests if req.created_at < cutoff_date]
            
            # Delete old requests
            deleted_count = 0
            for request in old_requests:
                if await self.repository.delete_request(request.id):
                    deleted_count += 1
            
            # Clear old cache entries
            cache_cleared = await self.cache_service.clear_pattern("copywriting:*")
            
            result = {
                "requests_deleted": deleted_count,
                "cache_entries_cleared": cache_cleared,
                "cutoff_date": cutoff_date.isoformat()
            }
            
            logger.info(f"Cleaned up {deleted_count} old requests and {cache_cleared} cache entries")
            return result
            
        except Exception as e:
            logger.error(f"Error in cleanup_old_data: {e}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        try:
            health_status = {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "services": {}
            }
            
            # Check repository
            try:
                await self.repository.get_statistics()
                health_status["services"]["repository"] = "healthy"
            except Exception as e:
                health_status["services"]["repository"] = f"unhealthy: {str(e)}"
                health_status["status"] = "degraded"
            
            # Check cache
            try:
                await self.cache_service.get_cache_stats()
                health_status["services"]["cache"] = "healthy"
            except Exception as e:
                health_status["services"]["cache"] = f"unhealthy: {str(e)}"
                health_status["status"] = "degraded"
            
            # Check AI service
            try:
                is_available = await self.ai_service.is_available()
                health_status["services"]["ai_service"] = "healthy" if is_available else "unavailable"
                if not is_available:
                    health_status["status"] = "degraded"
            except Exception as e:
                health_status["services"]["ai_service"] = f"unhealthy: {str(e)}"
                health_status["status"] = "degraded"
            
            # Check event publisher
            try:
                await self.event_publisher.get_event_stats()
                health_status["services"]["event_publisher"] = "healthy"
            except Exception as e:
                health_status["services"]["event_publisher"] = f"unhealthy: {str(e)}"
                health_status["status"] = "degraded"
            
            logger.info(f"Health check completed: {health_status['status']}")
            return health_status
            
        except Exception as e:
            logger.error(f"Error in health_check: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            } 