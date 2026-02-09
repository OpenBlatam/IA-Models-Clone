"""
Start Inspection Stream Use Case

Use case for starting a real-time inspection stream.
"""

import logging
from typing import Optional, Callable

from ...domain import Camera, CameraStatus
from ...domain.exceptions import CameraException, CameraNotInitializedException

logger = logging.getLogger(__name__)


class StartInspectionStreamUseCase:
    """
    Use case for starting a real-time inspection stream from camera.
    
    This use case:
    1. Initializes camera if needed
    2. Starts camera streaming
    3. Sets up inspection callback
    4. Returns stream information
    """
    
    def __init__(
        self,
        camera_adapter=None,  # Will be injected from infrastructure
        inspection_callback: Optional[Callable] = None,
    ):
        """
        Initialize use case.
        
        Args:
            camera_adapter: Infrastructure adapter for camera
            inspection_callback: Optional callback for inspection results
        """
        self.camera_adapter = camera_adapter
        self.inspection_callback = inspection_callback
    
    def execute(
        self,
        camera_index: int,
        resolution: Optional[tuple] = None,
        fps: Optional[int] = None
    ) -> dict:
        """
        Execute start inspection stream use case.
        
        Args:
            camera_index: Camera index to use
            resolution: Optional resolution (width, height)
            fps: Optional frames per second
        
        Returns:
            Dictionary with stream information
        
        Raises:
            CameraException: If camera operation fails
        """
        try:
            if not self.camera_adapter:
                raise CameraException("Camera adapter not available")
            
            # Initialize camera if needed
            camera = self.camera_adapter.get_camera(camera_index)
            if not camera or camera.status == CameraStatus.UNINITIALIZED:
                if not self.camera_adapter.initialize_camera(
                    camera_index, resolution, fps
                ):
                    raise CameraNotInitializedException(camera_index)
                camera = self.camera_adapter.get_camera(camera_index)
            
            # Start streaming
            if not self.camera_adapter.start_streaming(camera_index):
                raise CameraException("Failed to start camera streaming")
            
            # Set up callback if provided
            if self.inspection_callback:
                self.camera_adapter.set_inspection_callback(
                    camera_index, self.inspection_callback
                )
            
            logger.info(f"Inspection stream started for camera {camera_index}")
            
            return {
                "camera_index": camera_index,
                "status": "streaming",
                "resolution": camera.resolution if camera else resolution,
                "fps": camera.fps if camera else fps,
            }
        
        except Exception as e:
            logger.error(f"Failed to start inspection stream: {str(e)}", exc_info=True)
            if isinstance(e, CameraException):
                raise
            raise CameraException(f"Failed to start inspection stream: {str(e)}")



