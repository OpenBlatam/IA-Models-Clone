"""
Object Detection Service

Service for detecting objects in images using ML models.
"""

import logging
from typing import List, Optional, Tuple
import numpy as np
import torch

from ...domain.exceptions import ModelException, ModelInferenceException
from ..adapters import MLModelLoader

logger = logging.getLogger(__name__)


class ObjectDetectionService:
    """
    Service for object detection using ML models.
    
    Supports multiple detection models:
    - YOLOv8
    - Faster R-CNN
    - SSD
    """
    
    def __init__(
        self,
        model_loader: Optional[MLModelLoader] = None,
        model_type: str = "yolov8",
        confidence_threshold: float = 0.5,
    ):
        """
        Initialize object detection service.
        
        Args:
            model_loader: Model loader adapter
            model_type: Type of detection model ('yolov8', 'faster_rcnn', 'ssd')
            confidence_threshold: Minimum confidence for detections
        """
        self.model_loader = model_loader or MLModelLoader()
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold
        self._model = None
    
    def detect_objects(
        self,
        image: np.ndarray,
    ) -> List[dict]:
        """
        Detect objects in an image.
        
        Args:
            image: Image as numpy array
        
        Returns:
            List of detected objects with bounding boxes and confidence
        
        Raises:
            ModelInferenceException: If inference fails
        """
        try:
            if self.model_type == "yolov8":
                return self._detect_with_yolov8(image)
            elif self.model_type == "faster_rcnn":
                return self._detect_with_faster_rcnn(image)
            elif self.model_type == "ssd":
                return self._detect_with_ssd(image)
            else:
                logger.warning(f"Unsupported model type: {self.model_type}")
                return []
        
        except Exception as e:
            logger.error(f"Object detection failed: {str(e)}", exc_info=True)
            raise ModelInferenceException("object_detector", str(e))
    
    def _detect_with_yolov8(self, image: np.ndarray) -> List[dict]:
        """Detect objects using YOLOv8."""
        try:
            # Try to use ultralytics if available
            try:
                from ultralytics import YOLO
                
                # Load model if needed
                if self._model is None:
                    self._model = YOLO('yolov8n.pt')  # nano model for speed
                
                # Run inference
                results = self._model(image, conf=self.confidence_threshold)
                
                # Extract detections
                detections = []
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = self._model.names[class_id]
                        
                        detections.append({
                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            "confidence": float(confidence),
                            "class_id": class_id,
                            "class_name": class_name,
                        })
                
                logger.debug(f"YOLOv8 detected {len(detections)} objects")
                return detections
            
            except ImportError:
                logger.warning("ultralytics not available, using placeholder")
                return self._placeholder_detection(image)
        
        except Exception as e:
            logger.error(f"YOLOv8 detection failed: {str(e)}", exc_info=True)
            return []
    
    def _detect_with_faster_rcnn(self, image: np.ndarray) -> List[dict]:
        """Detect objects using Faster R-CNN."""
        # Placeholder - would implement with torchvision
        logger.warning("Faster R-CNN not yet implemented")
        return self._placeholder_detection(image)
    
    def _detect_with_ssd(self, image: np.ndarray) -> List[dict]:
        """Detect objects using SSD."""
        # Placeholder - would implement with torchvision
        logger.warning("SSD not yet implemented")
        return self._placeholder_detection(image)
    
    def _placeholder_detection(self, image: np.ndarray) -> List[dict]:
        """Placeholder detection for when models are not available."""
        # Return empty list or simple detection based on image size
        height, width = image.shape[:2]
        return [{
            "bbox": [0, 0, width, height],
            "confidence": 0.5,
            "class_id": 0,
            "class_name": "object",
        }]
    
    def filter_by_confidence(
        self,
        detections: List[dict],
        min_confidence: Optional[float] = None
    ) -> List[dict]:
        """
        Filter detections by confidence threshold.
        
        Args:
            detections: List of detections
            min_confidence: Minimum confidence (uses instance threshold if None)
        
        Returns:
            Filtered list of detections
        """
        threshold = min_confidence or self.confidence_threshold
        return [
            det for det in detections
            if det.get("confidence", 0.0) >= threshold
        ]



