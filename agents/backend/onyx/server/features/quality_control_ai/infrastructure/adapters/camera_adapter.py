"""
Camera Adapter

Adapter for camera hardware interface.
"""

import logging
from typing import Optional, Tuple, Callable
import cv2
import numpy as np

from ...domain import Camera, CameraStatus
from ...domain.exceptions import (
    CameraException,
    CameraNotInitializedException,
    CameraConnectionException,
    CameraStreamException,
)

logger = logging.getLogger(__name__)


class CameraAdapter:
    """
    Adapter for camera hardware.
    
    Provides a clean interface to camera operations while abstracting
    the underlying OpenCV implementation.
    """
    
    def __init__(self):
        """Initialize camera adapter."""
        self._cameras = {}  # camera_index -> Camera entity
        self._streams = {}  # camera_index -> VideoCapture
        self._callbacks = {}  # camera_index -> callback function
    
    def get_camera(self, camera_index: int) -> Optional[Camera]:
        """
        Get camera entity.
        
        Args:
            camera_index: Camera index
        
        Returns:
            Camera entity or None
        """
        return self._cameras.get(camera_index)
    
    def initialize_camera(
        self,
        camera_index: int,
        resolution: Optional[Tuple[int, int]] = None,
        fps: Optional[int] = None,
    ) -> bool:
        """
        Initialize a camera.
        
        Args:
            camera_index: Camera index
            resolution: Optional resolution (width, height)
            fps: Optional frames per second
        
        Returns:
            True if initialized successfully
        
        Raises:
            CameraConnectionException: If connection fails
        """
        try:
            # Open camera
            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                raise CameraConnectionException(
                    camera_index, "Failed to open camera"
                )
            
            # Set resolution if provided
            if resolution:
                width, height = resolution
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            # Set FPS if provided
            if fps:
                cap.set(cv2.CAP_PROP_FPS, fps)
            
            # Get actual resolution
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            # Create camera entity
            camera = Camera(index=camera_index)
            camera.initialize(
                resolution=(actual_width, actual_height),
                fps=actual_fps,
            )
            
            self._cameras[camera_index] = camera
            cap.release()  # Release for now, will reopen for streaming
            
            logger.info(
                f"Camera {camera_index} initialized: "
                f"{actual_width}x{actual_height} @ {actual_fps}fps"
            )
            return True
        
        except CameraConnectionException:
            raise
        except Exception as e:
            logger.error(f"Failed to initialize camera: {str(e)}", exc_info=True)
            raise CameraConnectionException(camera_index, str(e))
    
    def start_streaming(self, camera_index: int) -> bool:
        """
        Start camera streaming.
        
        Args:
            camera_index: Camera index
        
        Returns:
            True if streaming started successfully
        
        Raises:
            CameraStreamException: If streaming fails
        """
        try:
            camera = self._cameras.get(camera_index)
            if not camera or camera.status != CameraStatus.INITIALIZED:
                raise CameraNotInitializedException(camera_index)
            
            # Open video capture
            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                raise CameraStreamException("Failed to open camera for streaming")
            
            # Set resolution and FPS
            if camera.resolution:
                width, height = camera.resolution
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            if camera.fps:
                cap.set(cv2.CAP_PROP_FPS, camera.fps)
            
            self._streams[camera_index] = cap
            camera.start_streaming()
            
            logger.info(f"Camera {camera_index} streaming started")
            return True
        
        except (CameraNotInitializedException, CameraStreamException):
            raise
        except Exception as e:
            logger.error(f"Failed to start streaming: {str(e)}", exc_info=True)
            raise CameraStreamException(str(e))
    
    def stop_streaming(self, camera_index: int) -> bool:
        """
        Stop camera streaming.
        
        Args:
            camera_index: Camera index
        
        Returns:
            True if stopped successfully
        """
        try:
            if camera_index in self._streams:
                cap = self._streams[camera_index]
                cap.release()
                del self._streams[camera_index]
            
            camera = self._cameras.get(camera_index)
            if camera:
                camera.stop_streaming()
            
            logger.info(f"Camera {camera_index} streaming stopped")
            return True
        
        except Exception as e:
            logger.error(f"Failed to stop streaming: {str(e)}", exc_info=True)
            return False
    
    def capture_frame(self, camera_index: int) -> Optional[np.ndarray]:
        """
        Capture a frame from camera.
        
        Args:
            camera_index: Camera index
        
        Returns:
            Frame as numpy array or None if failed
        """
        try:
            if camera_index not in self._streams:
                raise CameraStreamException("Camera is not streaming")
            
            cap = self._streams[camera_index]
            ret, frame = cap.read()
            
            if ret:
                return frame
            else:
                logger.warning(f"Failed to capture frame from camera {camera_index}")
                return None
        
        except Exception as e:
            logger.error(f"Failed to capture frame: {str(e)}", exc_info=True)
            return None
    
    def set_inspection_callback(
        self,
        camera_index: int,
        callback: Callable[[np.ndarray], None]
    ):
        """
        Set callback for inspection on each frame.
        
        Args:
            camera_index: Camera index
            callback: Callback function that receives frame
        """
        self._callbacks[camera_index] = callback
        logger.info(f"Inspection callback set for camera {camera_index}")
    
    def remove_inspection_callback(self, camera_index: int):
        """
        Remove inspection callback.
        
        Args:
            camera_index: Camera index
        """
        if camera_index in self._callbacks:
            del self._callbacks[camera_index]
            logger.info(f"Inspection callback removed for camera {camera_index}")
    
    def release(self, camera_index: int):
        """
        Release camera resources.
        
        Args:
            camera_index: Camera index
        """
        self.stop_streaming(camera_index)
        if camera_index in self._cameras:
            camera = self._cameras[camera_index]
            camera.disconnect()
            del self._cameras[camera_index]
        
        if camera_index in self._callbacks:
            del self._callbacks[camera_index]
        
        logger.info(f"Camera {camera_index} released")



