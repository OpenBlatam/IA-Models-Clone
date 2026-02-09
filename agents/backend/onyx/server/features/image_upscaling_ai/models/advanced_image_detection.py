"""
Advanced Image Detection
========================

Advanced image type and quality detection using multiple techniques.
"""

import logging
import numpy as np
from typing import Dict, Any, Tuple, Optional
from PIL import Image
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


@dataclass
class ImageAnalysis:
    """Image analysis results."""
    image_type: str  # 'anime', 'photo', 'artwork', 'pixel_art', 'mixed'
    quality_score: float  # 0.0-1.0
    complexity: float  # 0.0-1.0
    has_text: bool
    has_faces: bool
    dominant_colors: List[Tuple[int, int, int]]
    recommended_model: str
    confidence: float  # 0.0-1.0


class AdvancedImageDetector:
    """
    Advanced image detection using multiple techniques.
    
    Features:
    - Image type detection (anime/photo/artwork/pixel_art)
    - Quality assessment
    - Complexity analysis
    - Text detection
    - Face detection
    - Color analysis
    - Model recommendation
    """
    
    def __init__(self):
        """Initialize detector."""
        self.face_cascade = None
        if CV2_AVAILABLE:
            try:
                self.face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
            except:
                logger.warning("Face detection not available")
    
    def analyze(
        self,
        image: Image.Image
    ) -> ImageAnalysis:
        """
        Comprehensive image analysis.
        
        Args:
            image: Input image
            
        Returns:
            ImageAnalysis with all detected features
        """
        img_array = np.array(image.convert("RGB"))
        
        # Detect image type
        image_type, type_confidence = self._detect_image_type(img_array)
        
        # Assess quality
        quality_score = self._assess_quality(img_array)
        
        # Analyze complexity
        complexity = self._analyze_complexity(img_array)
        
        # Detect text (simplified)
        has_text = self._detect_text(img_array)
        
        # Detect faces
        has_faces = self._detect_faces(img_array)
        
        # Analyze colors
        dominant_colors = self._analyze_colors(img_array)
        
        # Recommend model
        recommended_model = self._recommend_model(
            image_type, quality_score, complexity
        )
        
        return ImageAnalysis(
            image_type=image_type,
            quality_score=quality_score,
            complexity=complexity,
            has_text=has_text,
            has_faces=has_faces,
            dominant_colors=dominant_colors,
            recommended_model=recommended_model,
            confidence=type_confidence
        )
    
    def _detect_image_type(
        self,
        img_array: np.ndarray
    ) -> Tuple[str, float]:
        """Detect image type with confidence."""
        # Calculate features
        mean_color = np.mean(img_array, axis=(0, 1))
        std_color = np.std(img_array, axis=(0, 1))
        
        # Saturation
        gray = np.mean(mean_color)
        saturation = np.std([c - gray for c in mean_color])
        
        # Contrast
        contrast = np.mean(std_color)
        
        # Edge density (simplified)
        if CV2_AVAILABLE:
            gray_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray_img, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        else:
            edge_density = 0.1
        
        # Color variance
        color_variance = np.var(img_array, axis=(0, 1))
        avg_variance = np.mean(color_variance)
        
        # Pixel art detection (check for limited colors)
        unique_colors = len(np.unique(img_array.reshape(-1, 3), axis=0))
        is_pixel_art = unique_colors < 256 and img_array.shape[0] < 512
        
        # Classification
        if is_pixel_art:
            return "pixel_art", 0.9
        
        # High saturation + high contrast + high edge density = anime
        if saturation > 50 and contrast > 40 and edge_density > 0.15:
            confidence = min(0.95, 0.6 + (saturation / 100) * 0.2)
            return "anime", confidence
        
        # Low saturation + low contrast = photo
        if saturation < 30 and contrast < 30:
            confidence = min(0.95, 0.7 + (30 - saturation) / 30 * 0.2)
            return "photo", confidence
        
        # Medium values = artwork
        return "artwork", 0.7
    
    def _assess_quality(self, img_array: np.ndarray) -> float:
        """Assess image quality (0.0-1.0)."""
        if not CV2_AVAILABLE:
            return 0.7  # Default
        
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Sharpness (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        sharpness_score = min(1.0, sharpness / 500.0)
        
        # Noise estimation (simplified)
        noise = np.std(cv2.GaussianBlur(gray, (5, 5), 0) - gray)
        noise_score = max(0.0, 1.0 - noise / 20.0)
        
        # Overall quality
        quality = (sharpness_score * 0.7 + noise_score * 0.3)
        
        return min(1.0, max(0.0, quality))
    
    def _analyze_complexity(self, img_array: np.ndarray) -> float:
        """Analyze image complexity (0.0-1.0)."""
        if not CV2_AVAILABLE:
            return 0.5
        
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # Texture complexity
        # Use local binary pattern or similar (simplified)
        texture_variance = np.var(cv2.GaussianBlur(gray, (3, 3), 0))
        texture_score = min(1.0, texture_variance / 1000.0)
        
        # Combine
        complexity = (edge_density * 0.6 + texture_score * 0.4)
        
        return min(1.0, max(0.0, complexity))
    
    def _detect_text(self, img_array: np.ndarray) -> bool:
        """Detect if image contains text (simplified)."""
        # This would use OCR or text detection
        # For now, use edge density as proxy
        if not CV2_AVAILABLE:
            return False
        
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # High edge density might indicate text
        return edge_density > 0.2
    
    def _detect_faces(self, img_array: np.ndarray) -> bool:
        """Detect if image contains faces."""
        if not CV2_AVAILABLE or self.face_cascade is None:
            return False
        
        try:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            return len(faces) > 0
        except:
            return False
    
    def _analyze_colors(
        self,
        img_array: np.ndarray,
        n_colors: int = 5
    ) -> List[Tuple[int, int, int]]:
        """Extract dominant colors."""
        # Simple k-means would be better, but this is simplified
        pixels = img_array.reshape(-1, 3)
        
        # Sample pixels
        sample_size = min(10000, len(pixels))
        indices = np.random.choice(len(pixels), sample_size, replace=False)
        sample = pixels[indices]
        
        # Simple clustering (simplified)
        # In production, use k-means
        colors = []
        for _ in range(n_colors):
            idx = np.random.randint(0, len(sample))
            colors.append(tuple(sample[idx].astype(int)))
        
        return colors
    
    def _recommend_model(
        self,
        image_type: str,
        quality_score: float,
        complexity: float
    ) -> str:
        """Recommend best model based on analysis."""
        if image_type == "anime":
            return "RealESRGAN_x4plus_anime_6B"
        elif image_type == "pixel_art":
            return "RealESRNet_x4plus"  # No GAN, preserves pixel art
        elif complexity > 0.7:
            return "RealESRGAN_x4plus"  # High complexity needs GAN
        else:
            return "RealESRNet_x4plus"  # Lower complexity, faster model


