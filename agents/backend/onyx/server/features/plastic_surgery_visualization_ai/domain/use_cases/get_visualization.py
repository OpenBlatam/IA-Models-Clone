"""Use case for getting visualizations."""

from pathlib import Path
from typing import Optional

from core.interfaces import IStorageRepository, IEventPublisher
from core.exceptions import VisualizationNotFoundError, StorageError
from domain.events import VisualizationRetrievedEvent
from utils.logger import get_logger

logger = get_logger(__name__)


class GetVisualizationUseCase:
    """Use case for getting a visualization."""
    
    def __init__(
        self,
        storage_repository: IStorageRepository,
        event_publisher: Optional[IEventPublisher] = None
    ):
        self.storage_repository = storage_repository
        self.event_publisher = event_publisher
    
    async def execute(self, visualization_id: str) -> Path:
        """
        Execute the use case.
        
        Args:
            visualization_id: ID of the visualization
            
        Returns:
            Path to the visualization image
            
        Raises:
            VisualizationNotFoundError: If visualization not found
        """
        try:
            image_path = await self.storage_repository.get_visualization(visualization_id)
            if not image_path:
                raise VisualizationNotFoundError(
                    f"Visualization {visualization_id} not found"
                )
            
            # Publish event
            if self.event_publisher:
                event = VisualizationRetrievedEvent(visualization_id=visualization_id)
                await self.event_publisher.publish(event)
            
            return image_path
        except VisualizationNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Error retrieving visualization: {e}")
            raise StorageError(f"Failed to retrieve visualization: {str(e)}")

