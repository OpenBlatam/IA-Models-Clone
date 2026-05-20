"""
Face Processing Service
=======================

Service for face detection and enhancement using MediaPipe.
"""

import logging
from typing import Optional

import cv2
import numpy as np

# Third-party imports
try:
    import mediapipe as mp
    FACE_LIBS_AVAILABLE = True
except ImportError:
    FACE_LIBS_AVAILABLE = False
    logging.warning(
        "MediaPipe not available. Install with: pip install mediapipe"
    )

logger = logging.getLogger(__name__)


class FaceProcessingService:
    """Service for face detection and enhancement using MediaPipe.
    
    Features:
    - Face detection
    - Face mesh for detailed landmarks
    - Face enhancement
    """
    
    def __init__(self):
        """Initialize face processing service."""
        self.face_detection = None
        self.face_mesh = None
        self.logger = logging.getLogger(f"{__name__}.FaceProcessingService")
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize MediaPipe components."""
        if not FACE_LIBS_AVAILABLE:
            self.logger.warning("MediaPipe not available")
            return
        
        try:
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_face_mesh = mp.solutions.face_mesh
            
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=0.5,
            )
            
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            
            self.logger.info("Face processing service initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize face processing: {e}")
            raise
    
    def enhance_face(self, image: np.ndarray) -> np.ndarray:
        """Enhance facial features in image.
        
        Args:
            image: Input image as numpy array (RGB)
        
        Returns:
            Enhanced image
        """
        if not self.face_detection:
            return image
        
        try:
            # Convert RGB to BGR for MediaPipe
            rgb_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Detect faces
            results = self.face_detection.process(rgb_image)
            
            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    h, w, _ = image.shape
                    
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)
                    
                    # Enhance face region
                    face_region = image[y:y+height, x:x+width]
                    enhanced_face = self._enhance_face_region(face_region)
                    image[y:y+height, x:x+width] = enhanced_face
            
            return image
            
        except Exception as e:
            self.logger.warning(f"Face enhancement failed: {e}")
            return image
    
    @staticmethod
    def _enhance_face_region(face_region: np.ndarray) -> np.ndarray:
        """Enhance a specific face region.
        
        Args:
            face_region: Face region as numpy array
        
        Returns:
            Enhanced face region
        """
        try:
            # Apply subtle sharpening
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            enhanced = cv2.filter2D(face_region, -1, kernel)
            
            # Apply subtle color enhancement
            enhanced = cv2.convertScaleAbs(enhanced, alpha=1.1, beta=5)
            
            return enhanced
            
        except Exception as e:
            logging.warning(f"Face region enhancement failed: {e}")
            return face_region



