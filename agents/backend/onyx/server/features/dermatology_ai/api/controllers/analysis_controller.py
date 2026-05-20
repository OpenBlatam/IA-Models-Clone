"""
Analysis Controller - HTTP handlers for analysis endpoints.

Thin controller layer that delegates to use cases following Clean Architecture.
"""

import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, UploadFile, File, Depends

from ...core.application import AnalyzeImageUseCase, GetAnalysisHistoryUseCase
from ...core.domain.interfaces import (
    IAnalysisRepository,
    IImageProcessor,
    IAnalysisService,
    IEventPublisher,
)
from ...utils.oauth2 import get_current_user
from ..middleware.error_handler import handle_controller_errors

logger = logging.getLogger(__name__)


class AnalysisController:
    """Controller for analysis endpoints."""
    
    def __init__(
        self,
        analyze_image_use_case: AnalyzeImageUseCase,
        get_history_use_case: GetAnalysisHistoryUseCase
    ) -> None:
        """
        Initialize analysis controller.
        
        Args:
            analyze_image_use_case: Use case for analyzing images
            get_history_use_case: Use case for retrieving analysis history
        """
        self.analyze_image_use_case = analyze_image_use_case
        self.get_history_use_case = get_history_use_case
        self.router = APIRouter(prefix="/analysis", tags=["analysis"])
        self._register_routes()
    
    def _serialize_conditions(self, conditions: List[Any]) -> List[Dict[str, Any]]:
        """
        Serialize conditions to dictionary format.
        
        Args:
            conditions: List of condition objects
            
        Returns:
            List of dictionaries with condition data
        """
        return [
            {
                "name": c.name,
                "confidence": c.confidence,
                "severity": c.severity
            }
            for c in conditions
        ]
    
    def _serialize_analysis_summary(self, analysis: Any) -> Dict[str, Any]:
        """
        Serialize analysis summary to dictionary format.
        
        Args:
            analysis: Analysis entity object
            
        Returns:
            Dictionary with analysis summary data
        """
        return {
            "id": analysis.id,
            "status": analysis.status.value,
            "created_at": analysis.created_at.isoformat(),
            "metrics": analysis.metrics.to_dict() if analysis.metrics else None
        }
    
    def _register_routes(self) -> None:
        """Register all analysis routes."""
        
        @self.router.post(
            "/image",
            summary="Analyze skin image",
            description="""
            Analyze a skin image to detect conditions, measure metrics, and determine skin type.
            
            **Features:**
            - Image validation and processing
            - Skin condition detection
            - Metrics calculation (hydration, wrinkles, etc.)
            - Skin type determination
            
            **Supported formats:** JPEG, PNG, WebP
            **Max size:** 10MB
            **Min size:** 1KB
            """,
            response_description="Analysis results with metrics and detected conditions"
        )
        async def analyze_image(
            file: UploadFile = File(..., description="Skin image file to analyze"),
            current_user: Dict[str, Any] = Depends(get_current_user)
        ) -> Dict[str, Any]:
            """
            Analyze skin image.
            
            Upload a skin image for AI-powered analysis. The system will:
            1. Validate the image format and size
            2. Process the image to extract features
            3. Detect skin conditions
            4. Calculate skin metrics
            5. Determine skin type
            6. Generate recommendations
            
            Args:
                file: Uploaded image file
                current_user: Authenticated user information
                
            Returns:
                Dictionary with analysis results
            """
            async def _analyze() -> Dict[str, Any]:
                image_data = await file.read()
                
                analysis = await self.analyze_image_use_case.execute(
                    user_id=current_user["sub"],
                    image_data=image_data,
                    metadata={"filename": file.filename}
                )
                
                return {
                    "success": True,
                    "analysis_id": analysis.id,
                    "status": analysis.status.value,
                    "metrics": analysis.metrics.to_dict() if analysis.metrics else None,
                    "conditions": self._serialize_conditions(analysis.conditions)
                }
            
            return await handle_controller_errors(_analyze)
        
        @self.router.get(
            "/history",
            summary="Get analysis history",
            description="""
            Retrieve the analysis history for the current user.
            
            **Pagination:**
            - Use `limit` to control page size (1-100)
            - Use `offset` for pagination
            - Results are ordered by creation date (newest first)
            """,
            response_description="List of previous analyses with summaries"
        )
        async def get_history(
            limit: int = 10,
            offset: int = 0,
            current_user: Dict[str, Any] = Depends(get_current_user)
        ) -> Dict[str, Any]:
            """
            Get analysis history.
            
            Retrieve paginated list of previous skin analyses for the authenticated user.
            Each analysis includes status, metrics summary, and creation timestamp.
            
            Args:
                limit: Maximum number of results to return
                offset: Number of results to skip for pagination
                current_user: Authenticated user information
                
            Returns:
                Dictionary with paginated analysis history
            """
            async def _get_history() -> Dict[str, Any]:
                analyses = await self.get_history_use_case.execute(
                    user_id=current_user["sub"],
                    limit=limit,
                    offset=offset
                )
                
                return {
                    "success": True,
                    "count": len(analyses),
                    "analyses": [self._serialize_analysis_summary(a) for a in analyses]
                }
            
            return await handle_controller_errors(_get_history)
        
        @self.router.get("/{analysis_id}")
        async def get_analysis(
            analysis_id: str,
            current_user: Dict[str, Any] = Depends(get_current_user)
        ) -> Dict[str, str]:
            """
            Get analysis by ID.
            
            Args:
                analysis_id: Unique identifier for the analysis
                current_user: Authenticated user information
                
            Returns:
                Dictionary with analysis identifier
            """
            return {"analysis_id": analysis_id}


def create_analysis_controller(
    analysis_repository: IAnalysisRepository,
    image_processor: IImageProcessor,
    analysis_service: IAnalysisService,
    event_publisher: Optional[IEventPublisher] = None
) -> AnalysisController:
    """
    Factory function to create analysis controller.
    
    Args:
        analysis_repository: Repository for analysis data access
        image_processor: Image processing service
        analysis_service: Domain service for analysis logic
        event_publisher: Optional event publisher for domain events
        
    Returns:
        Configured AnalysisController instance
    """
    analyze_use_case = AnalyzeImageUseCase(
        analysis_repository=analysis_repository,
        image_processor=image_processor,
        analysis_service=analysis_service,
        event_publisher=event_publisher
    )
    
    get_history_use_case = GetAnalysisHistoryUseCase(
        analysis_repository=analysis_repository
    )
    
    return AnalysisController(analyze_use_case, get_history_use_case)

