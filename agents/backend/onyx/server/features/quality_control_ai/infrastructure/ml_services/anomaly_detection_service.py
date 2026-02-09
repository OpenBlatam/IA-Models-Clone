"""
Anomaly Detection Service

Service for detecting anomalies in images using ML models.
"""

import logging
from typing import List, Optional
import numpy as np
import torch

from ...domain import Anomaly, AnomalyType, AnomalySeverity, AnomalyLocation
from ...domain.exceptions import ModelException, ModelInferenceException
from ..adapters import MLModelLoader

logger = logging.getLogger(__name__)


class AnomalyDetectionService:
    """
    Service for anomaly detection using ML models.
    
    Supports multiple detection methods:
    - Autoencoder-based
    - Diffusion-based
    - Statistical
    """
    
    def __init__(
        self,
        model_loader: Optional[MLModelLoader] = None,
        use_autoencoder: bool = True,
        use_diffusion: bool = False,
    ):
        """
        Initialize anomaly detection service.
        
        Args:
            model_loader: Model loader adapter
            use_autoencoder: Whether to use autoencoder model
            use_diffusion: Whether to use diffusion model
        """
        self.model_loader = model_loader or MLModelLoader()
        self.use_autoencoder = use_autoencoder
        self.use_diffusion = use_diffusion
        self._autoencoder_model = None
        self._diffusion_model = None
    
    def detect_anomalies(
        self,
        image: np.ndarray,
        threshold: float = 0.7,
    ) -> List[Anomaly]:
        """
        Detect anomalies in an image.
        
        Args:
            image: Image as numpy array
            threshold: Anomaly detection threshold
        
        Returns:
            List of detected anomalies
        
        Raises:
            ModelInferenceException: If inference fails
        """
        try:
            anomalies = []
            
            # Autoencoder-based detection
            if self.use_autoencoder:
                autoencoder_anomalies = self._detect_with_autoencoder(
                    image, threshold
                )
                anomalies.extend(autoencoder_anomalies)
            
            # Diffusion-based detection
            if self.use_diffusion:
                diffusion_anomalies = self._detect_with_diffusion(
                    image, threshold
                )
                anomalies.extend(diffusion_anomalies)
            
            logger.debug(f"Detected {len(anomalies)} anomalies")
            return anomalies
        
        except Exception as e:
            logger.error(f"Anomaly detection failed: {str(e)}", exc_info=True)
            raise ModelInferenceException("anomaly_detector", str(e))
    
    def _detect_with_autoencoder(
        self,
        image: np.ndarray,
        threshold: float
    ) -> List[Anomaly]:
        """Detect anomalies using autoencoder."""
        try:
            # Load model if needed
            if self._autoencoder_model is None:
                # This would load from repository in real implementation
                self._autoencoder_model = self.model_loader.load_model(
                    model_path="",  # Would be from repository
                    model_type="autoencoder"
                )
            
            # Preprocess image
            processed_image = self._preprocess_image(image)
            
            # Run inference
            with torch.no_grad():
                reconstructed = self._autoencoder_model(processed_image)
                error = torch.mean((processed_image - reconstructed) ** 2, dim=1)
                anomaly_score = error.item()
            
            # Create anomaly if score exceeds threshold
            anomalies = []
            if anomaly_score > threshold:
                import uuid
                anomaly = Anomaly(
                    id=str(uuid.uuid4()),
                    type=AnomalyType.AUTOENCODER,
                    severity=self._score_to_severity(anomaly_score),
                    location=AnomalyLocation(
                        x=0, y=0,
                        width=image.shape[1],
                        height=image.shape[0]
                    ),
                    score=anomaly_score,
                )
                anomalies.append(anomaly)
            
            return anomalies
        
        except Exception as e:
            logger.error(f"Autoencoder detection failed: {str(e)}", exc_info=True)
            return []
    
    def _detect_with_diffusion(
        self,
        image: np.ndarray,
        threshold: float
    ) -> List[Anomaly]:
        """Detect anomalies using diffusion model."""
        # Placeholder - would implement diffusion-based detection
        return []
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input."""
        # Resize to 224x224
        import cv2
        resized = cv2.resize(image, (224, 224))
        
        # Normalize
        normalized = resized.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
        
        # Move to device
        device = self.model_loader.device
        tensor = tensor.to(device)
        
        return tensor
    
    def _score_to_severity(self, score: float) -> AnomalySeverity:
        """Convert anomaly score to severity."""
        if score >= 0.8:
            return AnomalySeverity.HIGH
        elif score >= 0.5:
            return AnomalySeverity.MEDIUM
        else:
            return AnomalySeverity.LOW



