"""
Enhanced Anomaly Detector with PyTorch Models
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import logging
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass

from ..config.detection_config import DetectionConfig
from ..utils.image_utils import ImageUtils
from .models.autoencoder import create_autoencoder
from .models.diffusion_anomaly import create_diffusion_detector

logger = logging.getLogger(__name__)


@dataclass
class Anomaly:
    """Anomalía detectada"""
    anomaly_type: str
    confidence: float
    location: Tuple[int, int, int, int]
    severity: str
    description: str
    features: Optional[Dict] = None


class EnhancedAnomalyDetector:
    """
    Enhanced anomaly detector using PyTorch models (Autoencoder and Diffusion)
    """
    
    def __init__(
        self,
        config: Optional[DetectionConfig] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize enhanced anomaly detector
        
        Args:
            config: Configuration
            device: PyTorch device
        """
        self.config = config or DetectionConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_utils = ImageUtils()
        
        # Initialize PyTorch models
        self.autoencoder = None
        self.diffusion_detector = None
        
        if self.config.settings.use_autoencoder:
            try:
                self.autoencoder = create_autoencoder(
                    input_channels=3,
                    latent_dim=128,
                    input_size=(224, 224),
                    device=self.device
                )
                self.autoencoder.eval()
                logger.info("PyTorch autoencoder initialized")
            except Exception as e:
                logger.warning(f"Could not initialize autoencoder: {e}")
        
        # Diffusion model (optional, more resource intensive)
        if hasattr(self.config.settings, 'use_diffusion') and self.config.settings.use_diffusion:
            try:
                self.diffusion_detector = create_diffusion_detector(
                    image_size=224,
                    in_channels=3,
                    device=self.device
                )
                if self.diffusion_detector.unet is not None:
                    self.diffusion_detector.eval()
                    logger.info("Diffusion detector initialized")
            except Exception as e:
                logger.warning(f"Could not initialize diffusion detector: {e}")
        
        # Statistical methods (fallback)
        self.use_statistical = self.config.settings.use_statistical
        
        logger.info(f"Enhanced anomaly detector initialized on {self.device}")
    
    def detect_anomalies(self, image: Union[np.ndarray, str]) -> List[Anomaly]:
        """
        Detect anomalies using PyTorch models
        
        Args:
            image: Input image
            
        Returns:
            List of detected anomalies
        """
        img = self.image_utils.load_image(image)
        if img is None:
            logger.error("Failed to load image")
            return []
        
        anomalies = []
        
        # PyTorch autoencoder detection
        if self.autoencoder is not None:
            autoencoder_anomalies = self._detect_with_autoencoder(img)
            anomalies.extend(autoencoder_anomalies)
        
        # Diffusion model detection
        if self.diffusion_detector is not None and self.diffusion_detector.unet is not None:
            diffusion_anomalies = self._detect_with_diffusion(img)
            anomalies.extend(diffusion_anomalies)
        
        # Statistical detection (fallback)
        if self.use_statistical:
            statistical_anomalies = self._detect_statistical_anomalies(img)
            anomalies.extend(statistical_anomalies)
        
        # Filter by threshold
        filtered_anomalies = [
            a for a in anomalies
            if a.confidence >= self.config.settings.anomaly_threshold
        ]
        
        return filtered_anomalies
    
    def _detect_with_autoencoder(self, image: np.ndarray) -> List[Anomaly]:
        """Detect anomalies using autoencoder"""
        anomalies = []
        
        try:
            # Preprocess image
            img_resized = cv2.resize(image, (224, 224))
            img_tensor = torch.from_numpy(img_resized).float()
            
            # Convert BGR to RGB if needed
            if len(img_tensor.shape) == 3:
                if img_tensor.shape[2] == 3:
                    img_tensor = img_tensor[:, :, ::-1]  # BGR to RGB
                img_tensor = img_tensor.permute(2, 0, 1)  # HWC to CHW
            else:
                img_tensor = img_tensor.unsqueeze(0)  # Add channel dim
            
            # Normalize to [0, 1]
            if img_tensor.max() > 1.0:
                img_tensor = img_tensor / 255.0
            
            # Normalize to [-1, 1] for autoencoder
            img_tensor = img_tensor * 2.0 - 1.0
            
            # Add batch dimension
            img_tensor = img_tensor.unsqueeze(0).to(self.device)
            
            # Forward pass
            with torch.no_grad():
                reconstructed, latent = self.autoencoder(img_tensor)
                reconstruction_error = F.mse_loss(
                    reconstructed, img_tensor, reduction='none'
                ).mean(dim=1).cpu().numpy()
            
            # Convert back to numpy
            recon_np = reconstructed[0].cpu().permute(1, 2, 0).numpy()
            recon_np = (recon_np + 1.0) / 2.0  # Denormalize
            recon_np = np.clip(recon_np, 0, 1)
            
            # Resize back to original
            h, w = image.shape[:2]
            recon_np = cv2.resize(recon_np, (w, h))
            
            # Calculate error map
            if len(image.shape) == 3:
                img_normalized = image.astype(np.float32) / 255.0
                if image.shape[2] == 3:
                    img_normalized = cv2.cvtColor(img_normalized, cv2.COLOR_BGR2RGB)
            else:
                img_normalized = image.astype(np.float32) / 255.0
                img_normalized = np.stack([img_normalized] * 3, axis=-1)
            
            error_map = np.mean(np.abs(img_normalized - recon_np), axis=2)
            
            # Threshold error map
            threshold = np.percentile(error_map, 95)
            error_mask = error_map > threshold
            
            # Find contours
            error_mask_uint8 = (error_mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(
                error_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 50:
                    continue
                
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate confidence from error
                roi_error = error_map[y:y+h, x:x+w].mean()
                confidence = min(1.0, roi_error / threshold)
                
                if confidence >= self.config.settings.anomaly_threshold:
                    severity = "high" if confidence > 0.8 else "medium" if confidence > 0.6 else "low"
                    
                    anomaly = Anomaly(
                        anomaly_type="autoencoder",
                        confidence=float(confidence),
                        location=(x, y, w, h),
                        severity=severity,
                        description=f"Autoencoder reconstruction error: {roi_error:.3f}",
                        features={"reconstruction_error": float(roi_error)}
                    )
                    anomalies.append(anomaly)
        
        except Exception as e:
            logger.error(f"Error in autoencoder detection: {e}", exc_info=True)
        
        return anomalies
    
    def _detect_with_diffusion(self, image: np.ndarray) -> List[Anomaly]:
        """Detect anomalies using diffusion model"""
        anomalies = []
        
        try:
            if self.diffusion_detector.unet is None:
                return anomalies
            
            # Preprocess
            img_resized = cv2.resize(image, (224, 224))
            img_tensor = torch.from_numpy(img_resized).float()
            
            if len(img_tensor.shape) == 3:
                if img_tensor.shape[2] == 3:
                    img_tensor = img_tensor[:, :, ::-1]  # BGR to RGB
                img_tensor = img_tensor.permute(2, 0, 1)
            
            if img_tensor.max() > 1.0:
                img_tensor = img_tensor / 255.0
            
            img_tensor = img_tensor.unsqueeze(0).to(self.device)
            
            # Compute anomaly score
            with torch.no_grad():
                anomaly_score = self.diffusion_detector.compute_anomaly_score(
                    img_tensor, num_inference_steps=20
                )
            
            score = anomaly_score.item()
            
            if score >= self.config.settings.anomaly_threshold:
                h, w = image.shape[:2]
                severity = "high" if score > 0.8 else "medium" if score > 0.6 else "low"
                
                anomaly = Anomaly(
                    anomaly_type="diffusion",
                    confidence=float(score),
                    location=(0, 0, w, h),  # Full image
                    severity=severity,
                    description=f"Diffusion anomaly score: {score:.3f}",
                    features={"anomaly_score": float(score)}
                )
                anomalies.append(anomaly)
        
        except Exception as e:
            logger.error(f"Error in diffusion detection: {e}", exc_info=True)
        
        return anomalies
    
    def _detect_statistical_anomalies(self, image: np.ndarray) -> List[Anomaly]:
        """Statistical anomaly detection (fallback)"""
        anomalies = []
        
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            h, w = gray.shape
            block_size = 64
            blocks = []
            block_locations = []
            
            for y in range(0, h, block_size):
                for x in range(0, w, block_size):
                    block = gray[y:y+block_size, x:x+block_size]
                    if block.size > 0:
                        blocks.append(block.flatten())
                        block_locations.append((x, y, min(block_size, w-x), min(block_size, h-y)))
            
            if not blocks:
                return anomalies
            
            blocks = np.array(blocks)
            means = np.mean(blocks, axis=1)
            stds = np.std(blocks, axis=1)
            
            mean_mean = np.mean(means)
            std_mean = np.std(means)
            mean_std = np.mean(stds)
            std_std = np.std(stds)
            
            threshold = self.config.settings.statistical_threshold
            
            for i, (mean_val, std_val) in enumerate(zip(means, stds)):
                z_score_mean = abs((mean_val - mean_mean) / (std_mean + 1e-6))
                z_score_std = abs((std_val - mean_std) / (std_std + 1e-6))
                
                if z_score_mean > threshold or z_score_std > threshold:
                    confidence = min(1.0, (z_score_mean + z_score_std) / (2 * threshold))
                    severity = "high" if confidence > 0.8 else "medium" if confidence > 0.6 else "low"
                    
                    x, y, w, h = block_locations[i]
                    anomaly = Anomaly(
                        anomaly_type="statistical",
                        confidence=float(confidence),
                        location=(x, y, w, h),
                        severity=severity,
                        description=f"Statistical anomaly (Z-score: {z_score_mean:.2f})",
                        features={"z_score_mean": float(z_score_mean), "z_score_std": float(z_score_std)}
                    )
                    anomalies.append(anomaly)
        
        except Exception as e:
            logger.error(f"Error in statistical detection: {e}", exc_info=True)
        
        return anomalies
    
    def load_model(self, model_path: str, model_type: str = "autoencoder"):
        """
        Load trained model weights
        
        Args:
            model_path: Path to model weights
            model_type: Type of model ("autoencoder" or "diffusion")
        """
        try:
            if model_type == "autoencoder" and self.autoencoder is not None:
                checkpoint = torch.load(model_path, map_location=self.device)
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    self.autoencoder.load_state_dict(checkpoint["model_state_dict"])
                else:
                    self.autoencoder.load_state_dict(checkpoint)
                self.autoencoder.eval()
                logger.info(f"Autoencoder model loaded from {model_path}")
            elif model_type == "diffusion" and self.diffusion_detector is not None:
                checkpoint = torch.load(model_path, map_location=self.device)
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    self.diffusion_detector.unet.load_state_dict(checkpoint["model_state_dict"])
                else:
                    self.diffusion_detector.unet.load_state_dict(checkpoint)
                self.diffusion_detector.eval()
                logger.info(f"Diffusion model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}", exc_info=True)

