"""Use case for creating visualizations."""

import hashlib
import uuid
from datetime import datetime
from typing import Optional

from api.schemas.visualization import VisualizationRequest, VisualizationResponse
from core.interfaces import (
    IImageProcessor,
    IAProcessor,
    IStorageRepository,
    ICacheRepository,
    IMetricsCollector,
    IEventPublisher
)
from core.exceptions import ImageValidationError
from domain.events import VisualizationCreatedEvent
from domain.validators import VisualizationRequestValidator
from utils.logger import get_logger

logger = get_logger(__name__)


class CreateVisualizationUseCase:
    """Use case for creating a visualization."""
    
    def __init__(
        self,
        image_processor: IImageProcessor,
        ai_processor: IAProcessor,
        storage_repository: IStorageRepository,
        cache_repository: ICacheRepository,
        metrics_collector: IMetricsCollector,
        event_publisher: Optional[IEventPublisher] = None,
        validator: Optional[VisualizationRequestValidator] = None
    ):
        self.image_processor = image_processor
        self.ai_processor = ai_processor
        self.storage_repository = storage_repository
        self.cache_repository = cache_repository
        self.metrics_collector = metrics_collector
        self.event_publisher = event_publisher
        self.validator = validator or VisualizationRequestValidator()
    
    async def execute(
        self,
        request: VisualizationRequest
    ) -> VisualizationResponse:
        """
        Execute the use case.
        
        Args:
            request: Visualization request
            
        Returns:
            VisualizationResponse with generated image
        """
        from utils.context_utils import timing_context, error_tracking_context
        from config.settings import settings
        
        start_time = datetime.utcnow()
        
        # Validate request
        image = await self._load_image(request)
        self.validator.validate(
            surgery_type=request.surgery_type,
            intensity=request.intensity,
            target_areas=request.target_areas,
            image=image,
            supported_formats=settings.supported_formats
        )
        
        # Generate cache key
        cache_key = self._generate_cache_key(request)
        
        # Check cache first
        cached_result = await self.cache_repository.get(cache_key)
        if cached_result:
            logger.info(f"Returning cached visualization for key: {cache_key}")
            self.metrics_collector.increment("visualizations.cache_hits")
            return VisualizationResponse(**cached_result)
        
        self.metrics_collector.increment("visualizations.cache_misses")
        visualization_id = str(uuid.uuid4())
        
        async with timing_context("visualization.create"):
            async with error_tracking_context("visualization.create"):
                # Process image with AI
                processed_image = await self.ai_processor.process_surgery_visualization(
                    image=image,
                    surgery_type=request.surgery_type,
                    intensity=request.intensity,
                    target_areas=request.target_areas
                )
                
                # Save processed image
                await self.storage_repository.save_visualization(
                    visualization_id=visualization_id,
                    image=processed_image,
                    format=settings.output_format
                )
                
                # Calculate processing time
                processing_time = (datetime.utcnow() - start_time).total_seconds()
                
                # Record metrics
                self.metrics_collector.increment("visualizations.created")
                self.metrics_collector.increment(f"visualizations.{request.surgery_type.value}")
                self.metrics_collector.record_timing("visualization.processing_time", processing_time)
                
                response = VisualizationResponse(
                    visualization_id=visualization_id,
                    image_url=f"/api/v1/visualize/{visualization_id}",
                    preview_url=f"/api/v1/visualize/{visualization_id}?preview=true",
                    surgery_type=request.surgery_type,
                    intensity=request.intensity,
                    created_at=datetime.utcnow().isoformat(),
                    processing_time=processing_time
                )
                
                # Cache the result
                await self.cache_repository.set(cache_key, response.dict())
                
                # Publish event
                if self.event_publisher:
                    event = VisualizationCreatedEvent(
                        visualization_id=visualization_id,
                        surgery_type=request.surgery_type,
                        intensity=request.intensity,
                        processing_time=processing_time
                    )
                    await self.event_publisher.publish(event)
                
                return response
    
    async def _load_image(self, request: VisualizationRequest):
        """Load image from request data or URL."""
        if request.image_data:
            return await self.image_processor.load_from_bytes(request.image_data)
        elif request.image_url:
            return await self.image_processor.load_from_url(request.image_url)
        else:
            raise ImageValidationError("Either image_data or image_url must be provided")
    
    def _generate_cache_key(self, request: VisualizationRequest) -> str:
        """Generate cache key from request."""
        key_parts = [
            request.surgery_type.value,
            str(request.intensity),
            str(request.target_areas) if request.target_areas else "",
            request.image_url or "",
            request.additional_notes or ""
        ]
        
        if request.image_data:
            key_parts.append(hashlib.md5(request.image_data).hexdigest())
        
        key_string = "|".join(key_parts)
        return f"viz_{hashlib.md5(key_string.encode()).hexdigest()}"

