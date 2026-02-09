"""
Smart Recommender
=================

Intelligent recommendation system for upscaling parameters.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from PIL import Image
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    from .advanced_image_detection import AdvancedImageDetector, ImageAnalysis
    from .adaptive_learner import AdaptiveLearner
except ImportError:
    AdvancedImageDetector = None
    AdaptiveLearner = None


@dataclass
class UpscalingRecommendation:
    """Upscaling recommendation."""
    method: str
    scale_factor: float
    preprocessing_mode: str
    postprocessing_mode: str
    expected_quality: float
    expected_time: float
    confidence: float
    reasoning: List[str]


class SmartRecommender:
    """
    Smart recommendation system.
    
    Features:
    - Method recommendation
    - Parameter optimization
    - Quality prediction
    - Time estimation
    - Learning integration
    """
    
    def __init__(
        self,
        learner: Optional[AdaptiveLearner] = None,
        detector: Optional[AdvancedImageDetector] = None
    ):
        """
        Initialize smart recommender.
        
        Args:
            learner: Adaptive learner instance
            detector: Image detector instance
        """
        self.learner = learner or (AdaptiveLearner() if AdaptiveLearner else None)
        self.detector = detector or (AdvancedImageDetector() if AdvancedImageDetector else None)
    
    def recommend(
        self,
        image: Image.Image,
        target_scale: float,
        prioritize_speed: bool = False,
        min_quality: float = 0.7
    ) -> UpscalingRecommendation:
        """
        Recommend upscaling parameters.
        
        Args:
            image: Input image
            target_scale: Target scale factor
            prioritize_speed: Prioritize speed over quality
            min_quality: Minimum acceptable quality
            
        Returns:
            UpscalingRecommendation
        """
        reasoning = []
        
        # Analyze image
        image_type = "artwork"
        complexity = 0.5
        quality = 0.7
        
        if self.detector:
            try:
                analysis = self.detector.analyze(image)
                image_type = analysis.image_type
                complexity = analysis.complexity
                quality = analysis.quality_score
                reasoning.append(f"Image type: {image_type} (confidence: {analysis.confidence:.2f})")
                reasoning.append(f"Complexity: {complexity:.2f}, Quality: {quality:.2f}")
            except Exception as e:
                logger.warning(f"Error in image detection: {e}")
        
        # Recommend method
        if self.learner:
            method, confidence = self.learner.recommend_method(
                image_type,
                target_scale,
                prioritize_speed
            )
            reasoning.append(f"Learned recommendation: {method} (confidence: {confidence:.2f})")
        else:
            # Default recommendations
            if image_type == "anime":
                method = "RealESRGAN_x4plus_anime_6B"
            elif image_type == "pixel_art":
                method = "RealESRNet_x4plus"
            elif complexity > 0.7:
                method = "RealESRGAN_x4plus"
            else:
                method = "RealESRNet_x4plus"
            confidence = 0.5
            reasoning.append(f"Default recommendation: {method}")
        
        # Predict quality and time
        expected_quality = 0.8
        expected_time = 2.0
        
        if self.learner:
            pred_quality, qual_conf = self.learner.predict_quality(
                image_type, target_scale, method
            )
            pred_time, time_conf = self.learner.predict_time(
                image_type, target_scale, method
            )
            
            if qual_conf > 0.3:
                expected_quality = pred_quality
                reasoning.append(f"Predicted quality: {expected_quality:.2f} (confidence: {qual_conf:.2f})")
            
            if time_conf > 0.3:
                expected_time = pred_time
                reasoning.append(f"Predicted time: {expected_time:.2f}s (confidence: {time_conf:.2f})")
        
        # Adjust for image quality
        if quality < 0.5:
            expected_quality *= 0.9  # Lower quality input = lower output
            reasoning.append("Low input quality detected, adjusting expectations")
        
        # Recommend preprocessing
        if quality < 0.6:
            preprocessing_mode = "aggressive"
            reasoning.append("Aggressive preprocessing recommended for low-quality input")
        elif quality > 0.8:
            preprocessing_mode = "conservative"
            reasoning.append("Conservative preprocessing for high-quality input")
        else:
            preprocessing_mode = "auto"
            reasoning.append("Auto preprocessing mode")
        
        # Recommend postprocessing
        if complexity > 0.7:
            postprocessing_mode = "aggressive"
            reasoning.append("Aggressive postprocessing for high complexity")
        else:
            postprocessing_mode = "auto"
            reasoning.append("Auto postprocessing mode")
        
        # Check if quality meets minimum
        if expected_quality < min_quality:
            reasoning.append(f"Warning: Expected quality ({expected_quality:.2f}) below minimum ({min_quality:.2f})")
            # Suggest alternative
            if method != "RealESRGAN_x4plus":
                reasoning.append("Consider using RealESRGAN_x4plus for better quality")
        
        return UpscalingRecommendation(
            method=method,
            scale_factor=target_scale,
            preprocessing_mode=preprocessing_mode,
            postprocessing_mode=postprocessing_mode,
            expected_quality=expected_quality,
            expected_time=expected_time,
            confidence=confidence,
            reasoning=reasoning
        )
    
    def get_alternatives(
        self,
        image: Image.Image,
        target_scale: float,
        current_method: str
    ) -> List[UpscalingRecommendation]:
        """
        Get alternative recommendations.
        
        Args:
            image: Input image
            target_scale: Target scale factor
            current_method: Current method
            
        Returns:
            List of alternative recommendations
        """
        alternatives = []
        
        # Analyze image
        image_type = "artwork"
        if self.detector:
            try:
                analysis = self.detector.analyze(image)
                image_type = analysis.image_type
            except:
                pass
        
        # Get alternative methods
        all_methods = [
            "RealESRGAN_x4plus",
            "RealESRGAN_x4plus_anime_6B",
            "RealESRNet_x4plus",
            "RealESRGAN_x2plus",
            "opencv_edsr",
            "lanczos"
        ]
        
        for method in all_methods:
            if method == current_method:
                continue
            
            # Predict for this method
            expected_quality = 0.7
            expected_time = 1.5
            
            if self.learner:
                pred_quality, _ = self.learner.predict_quality(
                    image_type, target_scale, method
                )
                pred_time, _ = self.learner.predict_time(
                    image_type, target_scale, method
                )
                expected_quality = pred_quality
                expected_time = pred_time
            
            alternatives.append(UpscalingRecommendation(
                method=method,
                scale_factor=target_scale,
                preprocessing_mode="auto",
                postprocessing_mode="auto",
                expected_quality=expected_quality,
                expected_time=expected_time,
                confidence=0.5,
                reasoning=[f"Alternative method: {method}"]
            ))
        
        # Sort by quality
        alternatives.sort(key=lambda x: x.expected_quality, reverse=True)
        
        return alternatives[:3]  # Return top 3


