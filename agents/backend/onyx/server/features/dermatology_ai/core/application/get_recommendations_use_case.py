from typing import List, Optional
import logging
import asyncio

from ..domain.entities import Recommendation
from ..domain.interfaces import (
    IAnalysisRepository,
    IRecommendationService,
    IUserRepository,
)
from .base import UseCase
from .exceptions import ValidationError, NotFoundError, ProcessingError
from .validators import UserIdValidator
from ...infrastructure.logging_utils import StructuredLogger
from ...infrastructure.performance_monitor import monitor_performance

logger = logging.getLogger(__name__)
structured_logger = StructuredLogger(__name__)


class GetRecommendationsUseCase(UseCase):
    
    def __init__(
        self,
        analysis_repository: IAnalysisRepository,
        recommendation_service: IRecommendationService,
        user_repository: Optional[IUserRepository] = None
    ):
        self.analysis_repository = analysis_repository
        self.recommendation_service = recommendation_service
        self.user_repository = user_repository
    
    @monitor_performance(operation_name="get_recommendations", log_threshold=2.0)
    async def execute(
        self,
        analysis_id: str,
        user_id: Optional[str] = None
    ) -> List[Recommendation]:
        if not analysis_id:
            raise ValidationError("analysis_id is required")
        
        if user_id:
            UserIdValidator.validate_user_id(user_id)
        
        structured_logger.set_context(analysis_id=analysis_id, user_id=user_id)
        
        with structured_logger.operation("get_recommendations", analysis_id=analysis_id):
            if not self.recommendation_service:
                raise ProcessingError("Recommendation service not available")
            
            tasks = [self.analysis_repository.get_by_id(analysis_id)]
            if user_id and self.user_repository:
                tasks.append(self.user_repository.get_by_id(user_id))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            analysis = results[0]
            
            if isinstance(analysis, Exception) or not analysis:
                raise NotFoundError(f"Analysis {analysis_id} not found")
            
            if not analysis.is_completed():
                raise ValidationError(f"Analysis {analysis_id} is not completed (status: {analysis.status.value})")
            
            user = results[1] if len(results) > 1 and not isinstance(results[1], Exception) else None
            
            try:
                recommendations = await self.recommendation_service.generate_recommendations(
                    analysis,
                    user
                )
                return recommendations
            except Exception as e:
                raise ProcessingError(f"Failed to generate recommendations: {e}") from e

