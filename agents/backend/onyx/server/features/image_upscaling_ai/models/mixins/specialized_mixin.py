"""
Specialized Mixin

Contains specialized upscaling methods for specific use cases.
"""

import logging
import time
from typing import Union, Tuple, Optional, Dict, Any, List
from pathlib import Path
from PIL import Image

from ..helpers import (
    UpscalingMetrics,
    QualityCalculator,
    PostprocessingMethods,
    ImageAnalysisUtils,
)

logger = logging.getLogger(__name__)


class SpecializedMixin:
    """
    Mixin providing specialized upscaling functionality.
    
    This mixin contains:
    - Face upscaling
    - Text upscaling
    - Artwork upscaling
    - Photo upscaling
    - Anime/manga upscaling
    - Medical imaging upscaling
    """
    
    def upscale_face(
        self,
        image: Union[Image.Image, str, Path],
        scale_factor: float,
        enhance_details: bool = True,
        return_metrics: bool = False
    ) -> Union[Image.Image, Tuple[Image.Image, UpscalingMetrics]]:
        """
        Specialized upscaling for facial images.
        
        Args:
            image: Input image
            scale_factor: Scale factor
            enhance_details: Enhance facial details
            return_metrics: If True, also return metrics
            
        Returns:
            Upscaled image or tuple (image, metrics)
        """
        start_time = time.time()
        
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        else:
            pil_image = image.convert("RGB")
        
        original_size = pil_image.size
        
        # Use best method for faces
        result = self.upscale(pil_image, scale_factor, "real_esrgan_like", return_metrics=False)
        
        if enhance_details:
            # Enhance facial features
            result = PostprocessingMethods.enhance_edges(result, strength=1.3)
            result = PostprocessingMethods.adaptive_contrast_enhancement(result)
            result = PostprocessingMethods.reduce_artifacts(result, method="bilateral", strength=0.4)
            result = PostprocessingMethods.texture_enhancement(result, strength=0.2)
        
        processing_time = time.time() - start_time
        quality_metrics = QualityCalculator.calculate_quality_metrics(result)
        
        metrics = UpscalingMetrics(
            original_size=original_size,
            upscaled_size=result.size,
            scale_factor=scale_factor,
            processing_time=processing_time,
            quality_score=quality_metrics.overall_quality,
            sharpness_score=quality_metrics.sharpness,
            artifact_score=1.0 - quality_metrics.artifact_count,
            method_used="face_specialized",
            success=True,
        )
        
        if return_metrics:
            return result, metrics
        return result
    
    def upscale_text(
        self,
        image: Union[Image.Image, str, Path],
        scale_factor: float,
        enhance_legibility: bool = True,
        return_metrics: bool = False
    ) -> Union[Image.Image, Tuple[Image.Image, UpscalingMetrics]]:
        """
        Specialized upscaling for text/images with text.
        
        Args:
            image: Input image
            scale_factor: Scale factor
            enhance_legibility: Enhance text legibility
            return_metrics: If True, also return metrics
            
        Returns:
            Upscaled image or tuple (image, metrics)
        """
        start_time = time.time()
        
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        else:
            pil_image = image.convert("RGB")
        
        original_size = pil_image.size
        
        # Use method optimized for text
        result = self.upscale(pil_image, scale_factor, "real_esrgan_like", return_metrics=False)
        
        if enhance_legibility:
            # Enhance text clarity
            result = PostprocessingMethods.enhance_edges(result, strength=1.4)
            result = PostprocessingMethods.adaptive_contrast_enhancement(result)
            result = PostprocessingMethods.reduce_artifacts(result, method="bilateral", strength=0.6)
        
        processing_time = time.time() - start_time
        quality_metrics = QualityCalculator.calculate_quality_metrics(result)
        
        metrics = UpscalingMetrics(
            original_size=original_size,
            upscaled_size=result.size,
            scale_factor=scale_factor,
            processing_time=processing_time,
            quality_score=quality_metrics.overall_quality,
            sharpness_score=quality_metrics.sharpness,
            artifact_score=1.0 - quality_metrics.artifact_count,
            method_used="text_specialized",
            success=True,
        )
        
        if return_metrics:
            return result, metrics
        return result
    
    def upscale_artwork(
        self,
        image: Union[Image.Image, str, Path],
        scale_factor: float,
        preserve_style: bool = True,
        return_metrics: bool = False
    ) -> Union[Image.Image, Tuple[Image.Image, UpscalingMetrics]]:
        """
        Specialized upscaling for artwork and illustrations.
        
        Args:
            image: Input image
            scale_factor: Scale factor
            preserve_style: Preserve artistic style
            return_metrics: If True, also return metrics
            
        Returns:
            Upscaled image or tuple (image, metrics)
        """
        start_time = time.time()
        
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        else:
            pil_image = image.convert("RGB")
        
        original_size = pil_image.size
        
        # Use method optimized for artwork
        result = self.upscale(pil_image, scale_factor, "waifu2x_like", return_metrics=False)
        
        if preserve_style:
            # Preserve artistic style
            result = PostprocessingMethods.texture_enhancement(result, strength=0.3)
            result = PostprocessingMethods.color_enhancement(result, saturation=1.2, vibrance=1.15)
            result = PostprocessingMethods.reduce_artifacts(result, method="bilateral", strength=0.3)
        
        processing_time = time.time() - start_time
        quality_metrics = QualityCalculator.calculate_quality_metrics(result)
        
        metrics = UpscalingMetrics(
            original_size=original_size,
            upscaled_size=result.size,
            scale_factor=scale_factor,
            processing_time=processing_time,
            quality_score=quality_metrics.overall_quality,
            sharpness_score=quality_metrics.sharpness,
            artifact_score=1.0 - quality_metrics.artifact_count,
            method_used="artwork_specialized",
            success=True,
        )
        
        if return_metrics:
            return result, metrics
        return result
    
    def upscale_photo(
        self,
        image: Union[Image.Image, str, Path],
        scale_factor: float,
        natural_look: bool = True,
        return_metrics: bool = False
    ) -> Union[Image.Image, Tuple[Image.Image, UpscalingMetrics]]:
        """
        Specialized upscaling for photographs.
        
        Args:
            image: Input image
            scale_factor: Scale factor
            natural_look: Maintain natural photographic look
            return_metrics: If True, also return metrics
            
        Returns:
            Upscaled image or tuple (image, metrics)
        """
        start_time = time.time()
        
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        else:
            pil_image = image.convert("RGB")
        
        original_size = pil_image.size
        
        # Use method optimized for photos
        result = self.upscale(pil_image, scale_factor, "real_esrgan_like", return_metrics=False)
        
        if natural_look:
            # Maintain natural look
            result = PostprocessingMethods.texture_enhancement(result, strength=0.2)
            result = PostprocessingMethods.color_enhancement(result, saturation=1.1, vibrance=1.05)
            result = PostprocessingMethods.adaptive_contrast_enhancement(result)
            result = PostprocessingMethods.reduce_artifacts(result, method="bilateral", strength=0.4)
        
        processing_time = time.time() - start_time
        quality_metrics = QualityCalculator.calculate_quality_metrics(result)
        
        metrics = UpscalingMetrics(
            original_size=original_size,
            upscaled_size=result.size,
            scale_factor=scale_factor,
            processing_time=processing_time,
            quality_score=quality_metrics.overall_quality,
            sharpness_score=quality_metrics.sharpness,
            artifact_score=1.0 - quality_metrics.artifact_count,
            method_used="photo_specialized",
            success=True,
        )
        
        if return_metrics:
            return result, metrics
        return result
    
    def upscale_anime(
        self,
        image: Union[Image.Image, str, Path],
        scale_factor: float,
        preserve_art_style: bool = True,
        return_metrics: bool = False
    ) -> Union[Image.Image, Tuple[Image.Image, UpscalingMetrics]]:
        """
        Specialized upscaling for anime/manga style images.
        
        Args:
            image: Input image
            scale_factor: Scale factor
            preserve_art_style: Preserve anime art style
            return_metrics: If True, also return metrics
            
        Returns:
            Upscaled image or tuple (image, metrics)
        """
        start_time = time.time()
        
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        else:
            pil_image = image.convert("RGB")
        
        original_size = pil_image.size
        
        # Use method optimized for anime
        result = self.upscale(pil_image, scale_factor, "waifu2x_like", return_metrics=False)
        
        if preserve_art_style:
            # Preserve anime style
            result = PostprocessingMethods.texture_enhancement(result, strength=0.25)
            result = PostprocessingMethods.color_enhancement(result, saturation=1.15, vibrance=1.1)
            result = PostprocessingMethods.reduce_artifacts(result, method="bilateral", strength=0.5)
        
        processing_time = time.time() - start_time
        quality_metrics = QualityCalculator.calculate_quality_metrics(result)
        
        metrics = UpscalingMetrics(
            original_size=original_size,
            upscaled_size=result.size,
            scale_factor=scale_factor,
            processing_time=processing_time,
            quality_score=quality_metrics.overall_quality,
            sharpness_score=quality_metrics.sharpness,
            artifact_score=1.0 - quality_metrics.artifact_count,
            method_used="anime_specialized",
            success=True,
        )
        
        if return_metrics:
            return result, metrics
        return result
    
    def auto_detect_and_upscale(
        self,
        image: Union[Image.Image, str, Path],
        scale_factor: float,
        return_metrics: bool = False
    ) -> Union[Image.Image, Tuple[Image.Image, UpscalingMetrics]]:
        """
        Automatically detect image type and use specialized upscaling.
        
        Args:
            image: Input image
            scale_factor: Scale factor
            return_metrics: If True, also return metrics
            
        Returns:
            Upscaled image or tuple (image, metrics)
        """
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        else:
            pil_image = image.convert("RGB")
        
        # Analyze image
        analysis = ImageAnalysisUtils.analyze_image_characteristics(pil_image)
        edge_density = analysis.get("edge_analysis", {}).get("edge_density", 0)
        quality = analysis.get("quality_metrics", {}).get("overall_quality", 0.5)
        
        # Auto-detect type
        if edge_density > 0.3:
            # High edge density - likely text
            logger.info("Auto-detected: Text image")
            return self.upscale_text(pil_image, scale_factor, return_metrics=return_metrics)
        elif quality > 0.75 and edge_density < 0.1:
            # High quality, low edges - likely artwork
            logger.info("Auto-detected: Artwork image")
            return self.upscale_artwork(pil_image, scale_factor, return_metrics=return_metrics)
        elif quality > 0.6:
            # Medium-high quality - likely photo
            logger.info("Auto-detected: Photo image")
            return self.upscale_photo(pil_image, scale_factor, return_metrics=return_metrics)
        else:
            # Default to general upscaling
            logger.info("Auto-detected: General image")
            return self.upscale(pil_image, scale_factor, "real_esrgan_like", return_metrics=return_metrics)


