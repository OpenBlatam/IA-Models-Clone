"""Use case for creating comparisons."""

import uuid
from datetime import datetime
from pathlib import Path
from PIL import Image

from api.schemas.comparison import ComparisonRequest, ComparisonResponse
from core.interfaces import IStorageRepository, IImageProcessor
from core.exceptions import VisualizationNotFoundError
from utils.logger import get_logger
from config.settings import settings

logger = get_logger(__name__)


class CreateComparisonUseCase:
    """Use case for creating a comparison."""
    
    def __init__(
        self,
        storage_repository: IStorageRepository,
        image_processor: IImageProcessor
    ):
        self.storage_repository = storage_repository
        self.image_processor = image_processor
    
    async def execute(
        self,
        request: ComparisonRequest,
        visualization_image: Image.Image
    ) -> ComparisonResponse:
        """
        Execute the use case.
        
        Args:
            request: Comparison request
            visualization_image: The visualization image to compare
            
        Returns:
            ComparisonResponse with comparison image URL
        """
        # Create comparison based on layout
        if request.layout == "side_by_side":
            comparison_image = self._create_side_by_side(
                visualization_image,
                include_original=request.include_original
            )
        elif request.layout == "overlay":
            comparison_image = self._create_overlay(
                visualization_image,
                include_original=request.include_original
            )
        else:
            comparison_image = visualization_image
        
        # Save comparison
        comparison_id = str(uuid.uuid4())
        await self.storage_repository.save_visualization(
            visualization_id=f"comparison_{comparison_id}",
            image=comparison_image,
            format=settings.output_format
        )
        
        return ComparisonResponse(
            comparison_id=comparison_id,
            image_url=f"/api/v1/compare/{comparison_id}",
            original_url=f"/api/v1/visualize/{request.visualization_id}" if request.include_original else None,
            visualization_url=f"/api/v1/visualize/{request.visualization_id}",
            layout=request.layout,
            created_at=datetime.utcnow().isoformat()
        )
    
    def _create_side_by_side(
        self,
        viz_image: Image.Image,
        include_original: bool = True
    ) -> Image.Image:
        """Create side-by-side comparison."""
        if not include_original:
            return viz_image
        
        # For now, return visualization image
        # In production, would load original and combine
        return viz_image
    
    def _create_overlay(
        self,
        viz_image: Image.Image,
        include_original: bool = True
    ) -> Image.Image:
        """Create overlay comparison."""
        # For now, return visualization image
        # In production, would create overlay effect
        return viz_image

