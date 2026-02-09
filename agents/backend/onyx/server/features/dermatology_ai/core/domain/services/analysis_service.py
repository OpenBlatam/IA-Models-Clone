"""
Domain service for skin analysis business logic.

This service contains the core business logic for analyzing skin images
and creating analysis entities with metrics and conditions.
"""

import uuid
from datetime import datetime
from typing import Optional, Dict, Any

from ..entities import (
    Analysis,
    SkinMetrics,
    Condition,
    SkinType,
    AnalysisStatus,
)
from ..interfaces import IAnalysisService, IImageProcessor
from ..exceptions import InvalidImageError


class AnalysisService(IAnalysisService):
    """
    Domain service for skin analysis.
    
    Orchestrates image processing and analysis to create Analysis entities
    with calculated metrics and detected conditions.
    """
    
    def __init__(
        self,
        image_processor: IImageProcessor,
        ml_model_manager: Optional[Any] = None
    ) -> None:
        """
        Initialize analysis service.
        
        Args:
            image_processor: Service for processing and validating images
            ml_model_manager: Optional ML model manager for advanced analysis
        """
        self.image_processor = image_processor
        self.ml_model_manager = ml_model_manager
    
    async def analyze_image(
        self,
        user_id: str,
        image_data: bytes,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Analysis:
        """
        Analyze skin image and create analysis entity.
        
        Processes the image, extracts metrics, detects conditions,
        and creates a completed Analysis entity.
        
        Args:
            user_id: Identifier of the user requesting analysis
            image_data: Raw image bytes to analyze
            metadata: Optional metadata dictionary (filename, etc.)
            
        Returns:
            Completed Analysis entity with metrics and conditions
            
        Raises:
            InvalidImageError: If image data is invalid or cannot be processed
        """
        if not await self.image_processor.validate(image_data):
            raise InvalidImageError("Invalid image data")
        
        processed_data = await self.image_processor.process(image_data)
        
        metrics = SkinMetrics(
            overall_score=processed_data.get("overall_score", 0.0),
            texture_score=processed_data.get("texture_score", 0.0),
            hydration_score=processed_data.get("hydration_score", 0.0),
            elasticity_score=processed_data.get("elasticity_score", 0.0),
            pigmentation_score=processed_data.get("pigmentation_score", 0.0),
            pore_size_score=processed_data.get("pore_size_score", 0.0),
            wrinkles_score=processed_data.get("wrinkles_score", 0.0),
            redness_score=processed_data.get("redness_score", 0.0),
            dark_spots_score=processed_data.get("dark_spots_score", 0.0),
        )
        
        conditions = [
            Condition(
                name=cond.get("name", ""),
                confidence=cond.get("confidence", 0.0),
                severity=cond.get("severity", "low"),
                description=cond.get("description")
            )
            for cond in processed_data.get("conditions", [])
        ]
        
        skin_type = SkinType(processed_data.get("skin_type", "normal"))
        
        analysis = Analysis(
            id=str(uuid.uuid4()),
            user_id=user_id,
            metrics=metrics,
            conditions=conditions,
            skin_type=skin_type,
            status=AnalysisStatus.COMPLETED,
            metadata=metadata or {},
            completed_at=datetime.utcnow()
        )
        
        return analysis
    
    async def get_analysis(self, analysis_id: str) -> Optional[Analysis]:
        """
        Get analysis by ID.
        
        Note: This method is not implemented as analysis retrieval
        should be done through the repository layer.
        
        Args:
            analysis_id: Unique identifier of the analysis
            
        Raises:
            NotImplementedError: Always raises, use repository instead
        """
        raise NotImplementedError("Use repository to get analysis")

