"""
Stop Inspection Stream Use Case

Use case for stopping a real-time inspection stream.
"""

import logging

from ...domain.exceptions import CameraException

logger = logging.getLogger(__name__)


class StopInspectionStreamUseCase:
    """
    Use case for stopping a real-time inspection stream.
    
    This use case:
    1. Stops camera streaming
    2. Cleans up resources
    3. Returns confirmation
    """
    
    def __init__(self, camera_adapter=None):  # Will be injected from infrastructure
        """
        Initialize use case.
        
        Args:
            camera_adapter: Infrastructure adapter for camera
        """
        self.camera_adapter = camera_adapter
    
    def execute(self, camera_index: int) -> dict:
        """
        Execute stop inspection stream use case.
        
        Args:
            camera_index: Camera index to stop
        
        Returns:
            Dictionary with confirmation
        
        Raises:
            CameraException: If camera operation fails
        """
        try:
            if not self.camera_adapter:
                raise CameraException("Camera adapter not available")
            
            # Stop streaming
            if not self.camera_adapter.stop_streaming(camera_index):
                logger.warning(f"Camera {camera_index} was not streaming")
            
            # Remove callback
            self.camera_adapter.remove_inspection_callback(camera_index)
            
            logger.info(f"Inspection stream stopped for camera {camera_index}")
            
            return {
                "camera_index": camera_index,
                "status": "stopped",
            }
        
        except Exception as e:
            logger.error(f"Failed to stop inspection stream: {str(e)}", exc_info=True)
            if isinstance(e, CameraException):
                raise
            raise CameraException(f"Failed to stop inspection stream: {str(e)}")



