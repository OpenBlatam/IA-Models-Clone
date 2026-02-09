"""
Defect Classification Service

Service for classifying defects in images using ML models.
"""

import logging
from typing import List, Optional
import numpy as np
import torch

from ...domain import Defect, DefectType, DefectLocation
from ...domain.exceptions import ModelException, ModelInferenceException
from ..adapters import MLModelLoader

logger = logging.getLogger(__name__)


class DefectClassificationService:
    """
    Service for defect classification using ML models.
    
    Uses Vision Transformer (ViT) or CNN for defect classification.
    """
    
    def __init__(
        self,
        model_loader: Optional[MLModelLoader] = None,
        model_type: str = "vit",
        confidence_threshold: float = 0.5,
    ):
        """
        Initialize defect classification service.
        
        Args:
            model_loader: Model loader adapter
            model_type: Type of model ('vit', 'cnn')
            confidence_threshold: Minimum confidence for classifications
        """
        self.model_loader = model_loader or MLModelLoader()
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold
        self._model = None
        
        # Defect type mapping
        self.defect_types = [
            DefectType.SCRATCH,
            DefectType.CRACK,
            DefectType.DENT,
            DefectType.DISCOLORATION,
            DefectType.DEFORMATION,
            DefectType.MISSING_PART,
            DefectType.SURFACE_IMPERFECTION,
            DefectType.CONTAMINATION,
            DefectType.SIZE_VARIATION,
            DefectType.OTHER,
        ]
    
    def classify_defects(
        self,
        image: np.ndarray,
        defect_regions: List[dict],
    ) -> List[Defect]:
        """
        Classify defects in image regions.
        
        Args:
            image: Full image as numpy array
            defect_regions: List of defect regions with bounding boxes
        
        Returns:
            List of classified defects
        
        Raises:
            ModelInferenceException: If inference fails
        """
        try:
            defects = []
            
            for region in defect_regions:
                # Extract region
                bbox = region.get("bbox", [0, 0, image.shape[1], image.shape[0]])
                x, y, x2, y2 = bbox
                region_image = image[y:y2, x:x2]
                
                if region_image.size == 0:
                    continue
                
                # Classify region
                classification = self._classify_region(region_image)
                
                if classification:
                    defect_type, confidence = classification
                    
                    if confidence >= self.confidence_threshold:
                        # Create defect location
                        location = DefectLocation(
                            x=int(x),
                            y=int(y),
                            width=int(x2 - x),
                            height=int(y2 - y),
                        )
                        
                        # Use domain service to create defect
                        from ...domain.services import DefectClassificationService as DomainService
                        domain_service = DomainService()
                        
                        defect = domain_service.classify_defect(
                            defect_type=defect_type,
                            location=location,
                            confidence=confidence,
                            description=region.get("description"),
                        )
                        
                        defects.append(defect)
            
            logger.debug(f"Classified {len(defects)} defects")
            return defects
        
        except Exception as e:
            logger.error(f"Defect classification failed: {str(e)}", exc_info=True)
            raise ModelInferenceException("defect_classifier", str(e))
    
    def _classify_region(
        self,
        region_image: np.ndarray
    ) -> Optional[tuple[DefectType, float]]:
        """
        Classify a single region.
        
        Args:
            region_image: Image region to classify
        
        Returns:
            Tuple of (defect_type, confidence) or None
        """
        try:
            # Load model if needed
            if self._model is None:
                self._model = self.model_loader.load_model(
                    model_path="",  # Would be from repository
                    model_type="classifier"
                )
            
            # Preprocess region
            processed = self._preprocess_region(region_image)
            
            # Run inference
            with torch.no_grad():
                outputs = self._model(processed)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_class = torch.max(probabilities, 1)
                
                confidence_value = confidence.item()
                class_idx = predicted_class.item()
                
                if class_idx < len(self.defect_types):
                    defect_type = self.defect_types[class_idx]
                    return (defect_type, confidence_value)
            
            return None
        
        except Exception as e:
            logger.error(f"Region classification failed: {str(e)}", exc_info=True)
            return None
    
    def _preprocess_region(self, region_image: np.ndarray) -> torch.Tensor:
        """Preprocess image region for model input."""
        import cv2
        
        # Resize to 224x224
        resized = cv2.resize(region_image, (224, 224))
        
        # Normalize
        normalized = resized.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
        
        # Move to device
        device = self.model_loader.device
        tensor = tensor.to(device)
        
        return tensor



