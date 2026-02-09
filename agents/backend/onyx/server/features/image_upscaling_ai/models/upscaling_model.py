"""
Image Upscaling Model
======================

AI-powered image upscaling using OpenRouter and optimization_core.
"""

import logging
import os
import sys
import json
from pathlib import Path
from typing import Optional, Dict, Any, Union, Tuple
from PIL import Image, ImageEnhance
import numpy as np
import time
from dataclasses import dataclass, field
import re

from .advanced_upscaling import AdvancedUpscaling
from .quality_metrics import QualityMetrics
from .helpers import ImageConverter

logger = logging.getLogger(__name__)

# Try to import optimization_core
OPTIMIZATION_CORE_AVAILABLE = False
try:
    # Try to import from the specified path or default location
    optimization_core_path = os.getenv(
        "OPTIMIZATION_CORE_PATH",
        str(Path(__file__).parent.parent.parent.parent / "Frontier-Model-run-polyglot" / "scripts" / "TruthGPT-main" / "optimization_core")
    )
    
    if os.path.exists(optimization_core_path):
        sys.path.insert(0, optimization_core_path)
        try:
            from optimization import Optimizer
            from core.validation import DataValidator
            OPTIMIZATION_CORE_AVAILABLE = True
            logger.info(f"optimization_core loaded from {optimization_core_path}")
        except ImportError as e:
            logger.warning(f"Could not import optimization_core: {e}")
    else:
        logger.warning(f"optimization_core path not found: {optimization_core_path}")
except Exception as e:
    logger.warning(f"Error loading optimization_core: {e}")


@dataclass
class UpscalingMetrics:
    """Metrics for upscaling operation."""
    original_size: Tuple[int, int]
    upscaled_size: Tuple[int, int]
    scale_factor: float
    processing_time: float
    quality_score: Optional[float] = None
    quality_metrics: Optional[Dict[str, Any]] = None
    success: bool = True
    errors: list = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class UpscalingModel:
    """
    Image upscaling model using AI enhancement and optimization_core.
    
    Features:
    - Multi-scale upscaling (2x, 4x, 8x)
    - AI-powered enhancement via OpenRouter
    - optimization_core integration for processing
    - Quality preservation
    - Anti-aliasing and sharpening
    """
    
    def __init__(
        self,
        openrouter_client=None,
        use_optimization_core: bool = True,
        quality_mode: str = "high",
    ):
        """
        Initialize upscaling model.
        
        Args:
            openrouter_client: OpenRouter client instance
            use_optimization_core: Whether to use optimization_core
            quality_mode: Quality mode ('fast', 'balanced', 'high', 'ultra')
        """
        self.openrouter_client = openrouter_client
        self.use_optimization_core = use_optimization_core and OPTIMIZATION_CORE_AVAILABLE
        self.quality_mode = quality_mode
        
        # Initialize optimization_core if available
        self.optimizer = None
        self.data_validator = None
        
        if self.use_optimization_core:
            try:
                if OPTIMIZATION_CORE_AVAILABLE:
                    self.optimizer = Optimizer()
                    self.data_validator = DataValidator()
                    logger.info("optimization_core initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize optimization_core: {e}")
                self.use_optimization_core = False
        
        # Quality settings based on mode
        self.quality_settings = {
            "fast": {
                "algorithm": "lanczos",
                "enhance_sharpness": False,
                "enhance_contrast": False,
                "use_ai": False,
                "anti_aliasing": False,
                "artifact_reduction": False,
                "multi_pass": False,
            },
            "balanced": {
                "algorithm": "lanczos",
                "enhance_sharpness": True,
                "enhance_contrast": True,
                "use_ai": False,
                "anti_aliasing": True,
                "artifact_reduction": "bilateral",
                "multi_pass": False,
            },
            "high": {
                "algorithm": "realesrgan",  # Use Real-ESRGAN for high quality
                "enhance_sharpness": True,
                "enhance_contrast": True,
                "use_ai": True,
                "anti_aliasing": True,
                "artifact_reduction": "bilateral",
                "multi_pass": False,
            },
            "ultra": {
                "algorithm": "realesrgan",  # Use Real-ESRGAN for ultra quality
                "enhance_sharpness": True,
                "enhance_contrast": True,
                "use_ai": True,
                "anti_aliasing": True,
                "artifact_reduction": "bilateral",
                "multi_pass": True,
            },
        }
        
        logger.info(f"UpscalingModel initialized (optimization_core: {self.use_optimization_core}, mode: {quality_mode})")
    
    def _validate_image(self, image: Image.Image) -> Dict[str, Any]:
        """
        Validate image using optimization_core if available.
        
        Args:
            image: Image to validate
            
        Returns:
            Validation results
        """
        if self.use_optimization_core and self.data_validator:
            try:
                # Convert PIL to numpy for validation
                img_array = np.array(image)
                # Use optimization_core validator
                validation_result = self.data_validator.validate(img_array)
                return {
                    "valid": True,
                    "details": validation_result,
                    "validator": "optimization_core"
                }
            except Exception as e:
                logger.warning(f"optimization_core validation failed: {e}")
        
        # Fallback validation
        width, height = image.size
        return {
            "valid": width > 0 and height > 0,
            "details": {"width": width, "height": height},
            "validator": "basic"
        }
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image before upscaling with advanced techniques.
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image
        """
        # Convert to RGB if needed
        image = ImageConverter.ensure_rgb(image)
        
        # Apply basic enhancements based on quality mode
        settings = self.quality_settings.get(self.quality_mode, self.quality_settings["high"])
        
        # Denoise if needed and OpenCV is available
        image = self._apply_denoising(image)
        
        # Apply contrast enhancement if enabled
        if settings.get("enhance_contrast"):
            image = self._enhance_contrast(image, factor=1.1)
        
        return image
    
    def _apply_denoising(self, image: Image.Image) -> Image.Image:
        """Apply denoising to image if OpenCV is available."""
        try:
            import cv2
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                # Apply slight denoising
                denoised = cv2.fastNlMeansDenoisingColored(img_array, None, 3, 3, 7, 21)
                return Image.fromarray(denoised)
        except ImportError:
            logger.debug("OpenCV not available for denoising")
        except Exception as e:
            logger.debug(f"Denoising skipped: {e}")
        return image
    
    def _enhance_contrast(self, image: Image.Image, factor: float = 1.1) -> Image.Image:
        """Enhance image contrast."""
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)
    
    def _upscale_basic(
        self,
        image: Image.Image,
        scale_factor: float
    ) -> Image.Image:
        """
        Advanced upscaling using multiple algorithms.
        
        Args:
            image: Input image
            scale_factor: Scale factor (e.g., 2.0 for 2x)
            
        Returns:
            Upscaled image
        """
        settings = self.quality_settings.get(self.quality_mode, self.quality_settings["high"])
        
        # Upscale using selected algorithm
        upscaled = self._apply_upscaling_algorithm(image, scale_factor, settings)
        
        # Apply post-processing
        upscaled = self._apply_post_processing(upscaled, settings)
        
        return upscaled
    
    def _apply_upscaling_algorithm(
        self,
        image: Image.Image,
        scale_factor: float,
        settings: Dict[str, Any]
    ) -> Image.Image:
        """Apply the selected upscaling algorithm."""
        algorithm = settings.get("algorithm", "lanczos")
        
        if algorithm == "realesrgan":
            # Try Real-ESRGAN first (best quality)
            try:
                return AdvancedUpscaling.upscale_realesrgan(image, scale_factor)
            except Exception as e:
                logger.warning(f"Real-ESRGAN failed: {e}, falling back to OpenCV")
                return AdvancedUpscaling.upscale_opencv_edsr(image, scale_factor)
        elif algorithm == "opencv_edsr":
            return AdvancedUpscaling.upscale_opencv_edsr(image, scale_factor)
        elif algorithm == "bicubic_enhanced":
            return AdvancedUpscaling.upscale_bicubic_enhanced(image, scale_factor)
        else:
            # Default to Lanczos
            return AdvancedUpscaling.upscale_lanczos(image, scale_factor, taps=3)
    
    def _apply_post_processing(
        self,
        image: Image.Image,
        settings: Dict[str, Any]
    ) -> Image.Image:
        """Apply post-processing enhancements."""
        processed = image
        
        # Apply sharpness enhancement
        if settings.get("enhance_sharpness"):
            processed = AdvancedUpscaling.enhance_edges(processed, strength=1.2)
        
        # Apply anti-aliasing
        if settings.get("anti_aliasing"):
            processed = AdvancedUpscaling.apply_anti_aliasing(processed, strength=0.3)
        
        # Apply artifact reduction
        artifact_method = settings.get("artifact_reduction")
        if artifact_method:
            processed = AdvancedUpscaling.reduce_artifacts(processed, method=artifact_method)
        
        return processed
    
    async def _enhance_with_ai(
        self,
        image: Image.Image,
        scale_factor: float
    ) -> Optional[Image.Image]:
        """
        Enhance image using AI via OpenRouter with actual technique application.
        
        Args:
            image: Input image
            scale_factor: Scale factor used
            
        Returns:
            Enhanced image or None if AI enhancement fails
        """
        if not self._is_ai_available():
            return None
        
        try:
            # Get AI recommendations
            response = await self._get_ai_recommendations(scale_factor)
            
            logger.debug(f"AI enhancement recommendations: {response[:200]}...")
            
            # Parse and apply AI recommendations
            enhanced_image = self._apply_ai_recommendations(image, response)
            
            return enhanced_image
            
        except Exception as e:
            logger.warning(f"AI enhancement failed: {e}")
            return None
    
    def _is_ai_available(self) -> bool:
        """Check if AI enhancement is available."""
        return (
            self.openrouter_client is not None and
            self.openrouter_client.is_configured()
        )
    
    async def _get_ai_recommendations(self, scale_factor: float) -> str:
        """Get AI recommendations for image enhancement."""
        system_prompt = (
            "You are an expert image processing AI. Analyze upscaled images and provide "
            "specific technical recommendations. Respond ONLY with a JSON object containing "
            "these fields: 'anti_aliasing_strength' (0.0-1.0), 'sharpness_factor' (1.0-2.0), "
            "'contrast_factor' (1.0-1.3), 'artifact_reduction_method' ('bilateral'|'median'|'gaussian'|'none'), "
            "'edge_enhancement' (true/false). Example: {\"anti_aliasing_strength\": 0.5, \"sharpness_factor\": 1.3, "
            "\"contrast_factor\": 1.1, \"artifact_reduction_method\": \"bilateral\", \"edge_enhancement\": true}"
        )
        
        user_prompt = (
            f"An image has been upscaled by {scale_factor}x. Analyze what post-processing techniques "
            f"would best improve quality. Consider: pixelation reduction, sharpness enhancement, "
            f"artifact removal, and edge preservation. Provide JSON recommendations."
        )
        
        model = getattr(self.openrouter_client, 'default_model', None) or "anthropic/claude-3.5-sonnet"
        return await self.openrouter_client.chat_completion_simple(
            model=model,
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.2,
            max_tokens=300
        )
    
    def _apply_ai_recommendations(
        self,
        image: Image.Image,
        ai_response: str
    ) -> Image.Image:
        """
        Apply AI recommendations to image.
        
        Args:
            image: Input image
            ai_response: AI response with recommendations
            
        Returns:
            Enhanced image
        """
        try:
            recommendations = self._parse_ai_recommendations(ai_response)
            enhanced = self._apply_recommendations_to_image(image, recommendations)
            logger.debug(f"Applied AI recommendations: {recommendations}")
            return enhanced
        except Exception as e:
            logger.warning(f"Error applying AI recommendations: {e}")
            return image
    
    def _parse_ai_recommendations(self, ai_response: str) -> Dict[str, Any]:
        """Parse AI recommendations from response."""
        json_match = re.search(r'\{[^}]+\}', ai_response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        
        # Fallback to default recommendations
        return {
            "anti_aliasing_strength": 0.5,
            "sharpness_factor": 1.3,
            "contrast_factor": 1.1,
            "artifact_reduction_method": "bilateral",
            "edge_enhancement": True
        }
    
    def _apply_recommendations_to_image(
        self,
        image: Image.Image,
        recommendations: Dict[str, Any]
    ) -> Image.Image:
        """Apply parsed recommendations to image."""
        enhanced = image
        
        # Apply anti-aliasing
        anti_aliasing_strength = recommendations.get("anti_aliasing_strength", 0)
        if anti_aliasing_strength > 0:
            enhanced = AdvancedUpscaling.apply_anti_aliasing(
                enhanced,
                strength=anti_aliasing_strength
            )
        
        # Apply artifact reduction
        artifact_method = recommendations.get("artifact_reduction_method", "none")
        if artifact_method and artifact_method != "none":
            enhanced = AdvancedUpscaling.reduce_artifacts(enhanced, method=artifact_method)
        
        # Apply edge enhancement
        if recommendations.get("edge_enhancement", False):
            sharpness_factor = recommendations.get("sharpness_factor", 1.2)
            enhanced = AdvancedUpscaling.enhance_edges(enhanced, strength=sharpness_factor)
        
        # Apply contrast enhancement
        contrast_factor = recommendations.get("contrast_factor", 1.0)
        if contrast_factor != 1.0:
            enhanced = self._enhance_contrast(enhanced, factor=contrast_factor)
        
        return enhanced
    
    def _apply_optimization_core(
        self,
        image: Image.Image
    ) -> Image.Image:
        """
        Apply optimization_core processing to image.
        
        Args:
            image: Input image
            
        Returns:
            Processed image
        """
        if not self.use_optimization_core or not self.optimizer:
            return image
        
        try:
            # Convert PIL to numpy
            img_array = np.array(image)
            
            # Use optimization_core optimizer
            # Note: This is a placeholder - actual implementation depends on optimization_core API
            optimized_array = self.optimizer.optimize(img_array)
            
            # Convert back to PIL
            optimized_image = Image.fromarray(optimized_array)
            
            return optimized_image
            
        except Exception as e:
            logger.warning(f"optimization_core processing failed: {e}")
            return image
    
    async def upscale(
        self,
        image: Union[Image.Image, str, Path],
        scale_factor: float = 2.0,
        use_ai: Optional[bool] = None,
        use_optimization_core: Optional[bool] = None,
    ) -> Tuple[Image.Image, UpscalingMetrics]:
        """
        Upscale image with specified scale factor.
        
        Args:
            image: Input image (PIL Image, path string, or Path)
            scale_factor: Scale factor (e.g., 2.0 for 2x upscaling)
            use_ai: Override AI enhancement setting
            use_optimization_core: Override optimization_core setting
            
        Returns:
            Tuple of (upscaled_image, metrics)
        """
        start_time = time.time()
        metrics = UpscalingMetrics(
            original_size=(0, 0),
            upscaled_size=(0, 0),
            scale_factor=scale_factor,
            processing_time=0.0,
            success=False
        )
        
        try:
            # Load and validate image
            pil_image = self._load_image(image)
            original_size = pil_image.size
            metrics.original_size = original_size
            
            validation = self._validate_image(pil_image)
            if not validation["valid"]:
                raise ValueError(f"Invalid image: {validation.get('details', {})}")
            
            # Preprocess
            pil_image = self._preprocess_image(pil_image)
            
            # Upscale image
            upscaled_image = self._perform_upscaling(
                pil_image,
                scale_factor,
                use_optimization_core
            )
            
            # Apply AI enhancement if enabled
            settings = self.quality_settings.get(self.quality_mode, self.quality_settings["high"])
            use_ai_enhance = use_ai if use_ai is not None else settings.get("use_ai", False)
            if use_ai_enhance:
                ai_enhanced = await self._enhance_with_ai(upscaled_image, scale_factor)
                if ai_enhanced:
                    upscaled_image = ai_enhanced
            
            # Calculate and update metrics
            self._update_metrics(metrics, pil_image, upscaled_image, start_time)
            
            logger.info(
                f"Upscaled image from {original_size} to {upscaled_image.size} "
                f"(scale: {scale_factor}x, time: {metrics.processing_time:.2f}s)"
            )
            
            return upscaled_image, metrics
            
        except Exception as e:
            metrics.processing_time = time.time() - start_time
            metrics.success = False
            metrics.errors.append(str(e))
            logger.error(f"Error upscaling image: {e}", exc_info=True)
            raise
    
    def _load_image(self, image: Union[Image.Image, str, Path]) -> Image.Image:
        """Load image from various input types."""
        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            return ImageConverter.ensure_rgb(image)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
    
    def _perform_upscaling(
        self,
        image: Image.Image,
        scale_factor: float,
        use_optimization_core: Optional[bool]
    ) -> Image.Image:
        """Perform upscaling with appropriate method."""
        settings = self.quality_settings.get(self.quality_mode, self.quality_settings["high"])
        use_multi_pass = settings.get("multi_pass", False) and scale_factor > 4.0
        
        if use_multi_pass:
            upscaled = AdvancedUpscaling.multi_scale_upscale(
                image,
                scale_factor,
                passes=min(3, int(np.ceil(scale_factor / 2.0)))
            )
        else:
            upscaled = self._upscale_basic(image, scale_factor)
        
        # Apply optimization_core if enabled
        use_opt = use_optimization_core if use_optimization_core is not None else self.use_optimization_core
        if use_opt:
            upscaled = self._apply_optimization_core(upscaled)
        
        return upscaled
    
    def _update_metrics(
        self,
        metrics: UpscalingMetrics,
        original_image: Image.Image,
        upscaled_image: Image.Image,
        start_time: float
    ) -> None:
        """Update metrics with processing results."""
        upscaled_size = upscaled_image.size
        metrics.upscaled_size = upscaled_size
        metrics.processing_time = time.time() - start_time
        metrics.success = True
        
        # Calculate comprehensive quality metrics
        quality_metrics = QualityMetrics.calculate_all_metrics(original_image, upscaled_image)
        metrics.quality_score = quality_metrics.get("overall_quality", 0.0)
        metrics.quality_metrics = quality_metrics
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_type": "Image Upscaling",
            "optimization_core_available": OPTIMIZATION_CORE_AVAILABLE,
            "optimization_core_enabled": self.use_optimization_core,
            "quality_mode": self.quality_mode,
            "openrouter_configured": (
                self.openrouter_client is not None and
                self.openrouter_client.is_configured()
            ) if self.openrouter_client else False,
        }

