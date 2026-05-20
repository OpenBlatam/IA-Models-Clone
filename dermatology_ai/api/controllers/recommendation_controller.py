"""
Recommendation Controller - HTTP handlers for recommendation endpoints
"""

from fastapi import APIRouter, Depends
from typing import Optional
import logging

from ...core.application import GetRecommendationsUseCase
from ...core.domain.interfaces import IAnalysisRepository, IRecommendationService, IUserRepository
from ...utils.oauth2 import get_current_user
from ..middleware.error_handler import handle_controller_errors

logger = logging.getLogger(__name__)


class RecommendationController:
    """Controller for recommendation endpoints"""
    
    def __init__(self, get_recommendations_use_case: GetRecommendationsUseCase):
        self.get_recommendations_use_case = get_recommendations_use_case
        self.router = APIRouter(prefix="/recommendations", tags=["recommendations"])
        self._register_routes()
    
    def _serialize_recommendation(self, recommendation):
        """Serialize recommendation to dict"""
        return {
            "product_id": recommendation.product_id,
            "product_name": recommendation.product_name,
            "category": recommendation.category,
            "priority": recommendation.priority,
            "reason": recommendation.reason,
            "confidence": recommendation.confidence,
            "usage_frequency": recommendation.usage_frequency
        }
    
    def _register_routes(self):
        """Register routes"""
        
        @self.router.get(
            "/analysis/{analysis_id}",
            summary="Get product recommendations",
            description="""
            Get personalized product recommendations based on skin analysis results.
            
            **Requirements:**
            - Analysis must be completed
            - Recommendations are personalized based on:
              * Detected skin conditions
              * Skin metrics (hydration, wrinkles, etc.)
              * User's skin type
              * User preferences (if available)
            
            **Recommendation Priority:**
            1. High priority: Addresses critical conditions
            2. Medium priority: Improves skin metrics
            3. Low priority: General maintenance products
            """,
            response_description="List of recommended products with reasons and priorities"
        )
        async def get_recommendations(
            analysis_id: str = ...,
            current_user: dict = Depends(get_current_user)
        ):
            """
            Get product recommendations
            
            Generate personalized skincare product recommendations based on a completed
            skin analysis. Recommendations are prioritized by urgency and relevance
            to detected conditions and skin metrics.
            """
            async def _get_recommendations():
                recommendations = await self.get_recommendations_use_case.execute(
                    analysis_id=analysis_id,
                    user_id=current_user["sub"]
                )
                
                return {
                    "success": True,
                    "count": len(recommendations),
                    "recommendations": [
                        self._serialize_recommendation(r) for r in recommendations
                    ]
                }
            
            return await handle_controller_errors(_get_recommendations)


def create_recommendation_controller(
    analysis_repository: IAnalysisRepository,
    recommendation_service: IRecommendationService,
    user_repository: Optional[IUserRepository] = None
) -> RecommendationController:
    """Factory function to create recommendation controller"""
    use_case = GetRecommendationsUseCase(
        analysis_repository=analysis_repository,
        recommendation_service=recommendation_service,
        user_repository=user_repository
    )
    
    return RecommendationController(use_case)

