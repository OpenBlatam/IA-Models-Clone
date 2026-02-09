"""Service for creating plastic surgery visualizations (facade layer)."""

from pathlib import Path
from typing import Optional

from api.schemas.visualization import VisualizationRequest, VisualizationResponse
from domain.use_cases.create_visualization import CreateVisualizationUseCase
from domain.use_cases.get_visualization import GetVisualizationUseCase
from core.interfaces import (
    IImageProcessor,
    IAProcessor,
    IStorageRepository,
    ICacheRepository,
    IMetricsCollector
)
from utils.logger import get_logger

logger = get_logger(__name__)


class VisualizationService:
    """
    Service for handling visualization requests.
    
    This is a facade that delegates to use cases.
    """
    
    def __init__(
        self,
        image_processor: Optional[IImageProcessor] = None,
        ai_processor: Optional[IAProcessor] = None,
        storage_repository: Optional[IStorageRepository] = None,
        cache_repository: Optional[ICacheRepository] = None,
        metrics_collector: Optional[IMetricsCollector] = None
    ):
        # Use dependency injection or create defaults
        from core.factories import (
            create_image_processor,
            create_ai_processor
        )
        from infrastructure.repositories import (
            FileStorageRepository,
            FileCacheRepository
        )
        from infrastructure.adapters import MetricsCollectorAdapter
        from utils.cache_advanced import Cache
        from core.constants import DEFAULT_CACHE_TTL_HOURS
        
        self.image_processor = image_processor or create_image_processor()
        self.ai_processor = ai_processor or create_ai_processor()
        self.storage_repository = storage_repository or FileStorageRepository()
        self.cache_repository = cache_repository or FileCacheRepository(
            Cache(ttl_hours=DEFAULT_CACHE_TTL_HOURS)
        )
        self.metrics_collector = metrics_collector or MetricsCollectorAdapter()
        
        # Get event publisher
        from core.factories import create_event_publisher
        event_publisher = create_event_publisher()
        
        # Create use cases
        self.create_use_case = CreateVisualizationUseCase(
            image_processor=self.image_processor,
            ai_processor=self.ai_processor,
            storage_repository=self.storage_repository,
            cache_repository=self.cache_repository,
            metrics_collector=self.metrics_collector,
            event_publisher=event_publisher
        )
        
        self.get_use_case = GetVisualizationUseCase(
            storage_repository=self.storage_repository,
            event_publisher=event_publisher
        )
    
    async def create_visualization(
        self,
        request: VisualizationRequest
    ) -> VisualizationResponse:
        """
        Create a visualization based on the request.
        
        Args:
            request: Visualization request
            
        Returns:
            VisualizationResponse with generated image
        """
        return await self.create_use_case.execute(request)
    
    async def get_visualization(self, visualization_id: str) -> Optional[Path]:
        """
        Get a stored visualization by ID.
        
        Args:
            visualization_id: ID of the visualization
            
        Returns:
            Path to the visualization image or None if not found
        """
        return await self.get_use_case.execute(visualization_id)

