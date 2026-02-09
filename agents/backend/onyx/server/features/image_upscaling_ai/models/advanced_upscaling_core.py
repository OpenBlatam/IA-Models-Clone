"""
Advanced Upscaling Core
=======================

Core upscaling functionality - refactored from advanced_upscaling.py
"""

import logging
import time
from typing import Tuple, Optional, Dict, Any, List, Callable, Union
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

from .helpers import (
    UpscalingMetrics,
    QualityMetrics,
    ImageQualityValidator,
    QualityCalculator,
    MethodSelector,
    StatisticsManager,
    UpscalingCache,
    retry_on_failure,
)
from .advanced_upscaling_algorithms import UpscalingAlgorithmsStatic
from .advanced_upscaling_postprocessing import PostprocessingMethods
from .advanced_upscaling_ml import MLUpscalingMethods
from .advanced_upscaling_analysis import AnalysisMethods
from .advanced_upscaling_pipelines import PipelineMethods
from .advanced_upscaling_ensemble import EnsembleMethods
from .advanced_upscaling_adaptive import AdaptiveMethods
from .advanced_upscaling_benchmark import BenchmarkMethods
from .advanced_upscaling_enhancement import EnhancementMethods
from .advanced_upscaling_meta import MetaLearningMethods
from .advanced_upscaling_batch import BatchProcessingMethods

logger = logging.getLogger(__name__)


class AdvancedUpscaling:
    """
    Advanced upscaling techniques - refactored core.
    
    This is a refactored version that delegates to specialized modules.
    """
    
    def __init__(
        self,
        enable_cache: bool = True,
        cache_size: int = 64,
        cache_ttl: int = 3600,
        max_workers: int = 4,
        validate_images: bool = True,
        enhance_images: bool = False,
        auto_select_method: bool = False,
        max_retries: int = 3,
    ):
        """Initialize Advanced Upscaling."""
        self.enable_cache = enable_cache
        self.cache = UpscalingCache(max_size=cache_size, ttl=cache_ttl) if enable_cache else None
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.validate_images = validate_images
        self.enhance_images = enhance_images
        self.auto_select_method = auto_select_method
        self.max_retries = max_retries
        self.stats = StatisticsManager.create_default_stats()
        
        # Initialize specialized modules
        self.ml_methods = MLUpscalingMethods(self)
        self.analysis_methods = AnalysisMethods(self)
        self.pipeline_methods = PipelineMethods(self)
        self.ensemble_methods = EnsembleMethods(self)
        self.adaptive_methods = AdaptiveMethods(self)
        self.benchmark_methods = BenchmarkMethods(self)
        self.enhancement_methods = EnhancementMethods(self)
        self.meta_methods = MetaLearningMethods(self)
        self.batch_methods = BatchProcessingMethods(self)
    
    def _select_best_method(self, image: Image.Image, scale_factor: float) -> str:
        """Select best method based on image characteristics."""
        return MethodSelector.select_best_method(
            image, scale_factor, auto_select=self.auto_select_method, stats=self.stats
        )
    
    @staticmethod
    def calculate_quality_metrics(image: Image.Image) -> QualityMetrics:
        """Calculate quality metrics for an image."""
        return QualityCalculator.calculate_quality_metrics(image)
    
    def upscale(
        self,
        image: Union[Image.Image, str, Path],
        scale_factor: float,
        method: str = "lanczos",
        use_cache: Optional[bool] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        return_metrics: bool = False,
    ) -> Union[Image.Image, Tuple[Image.Image, UpscalingMetrics]]:
        """Main upscaling method."""
        start_time = time.time()
        self.stats["upscales_performed"] += 1
        
        # Load image
        image_path = None
        if isinstance(image, (str, Path)):
            image_path = str(image)
            pil_image = Image.open(image_path).convert("RGB")
        else:
            pil_image = image.convert("RGB")
        
        # Validate
        if self.validate_images:
            quality_metrics = ImageQualityValidator.validate_image(pil_image)
            self.stats["images_validated"] += 1
            
            if not quality_metrics.is_valid:
                self.stats["validation_failures"] += 1
                error_msg = f"Image validation failed: {', '.join(quality_metrics.errors)}"
                logger.warning(error_msg)
                if return_metrics:
                    metrics = UpscalingMetrics(
                        original_size=pil_image.size,
                        upscaled_size=(0, 0),
                        scale_factor=scale_factor,
                        processing_time=0.0,
                        method_used=method,
                        success=False,
                        errors=[error_msg],
                    )
                    return None, metrics
                raise ValueError(error_msg)
        
        # Enhance if enabled
        if self.enhance_images:
            pil_image = ImageQualityValidator.enhance_image(pil_image)
            self.stats["images_enhanced"] += 1
        
        # Auto-select method
        if self.auto_select_method and method == "lanczos":
            method = self._select_best_method(pil_image, scale_factor)
        
        metrics = UpscalingMetrics(
            original_size=pil_image.size,
            upscaled_size=(0, 0),
            scale_factor=scale_factor,
            processing_time=0.0,
            method_used=method,
            success=False,
        )
        
        # Check cache
        use_cache = use_cache if use_cache is not None else self.enable_cache
        if use_cache and self.cache and image_path:
            cached = self.cache.get(image_path, scale_factor, method)
            if cached is not None:
                self.stats["cache_hits"] += 1
                metrics.processing_time = time.time() - start_time
                metrics.success = True
                metrics.upscaled_size = cached.size
                metrics.quality_score = self.calculate_quality_metrics(cached).overall_quality
                self.stats["total_time"] += metrics.processing_time
                self.stats["successful_upscales"] += 1
                
                if return_metrics:
                    return cached, metrics
                return cached
            self.stats["cache_misses"] += 1
        
        try:
            if progress_callback:
                progress_callback(1, 3)
            
            # Apply upscaling algorithm
            result = self._apply_upscaling_algorithm(pil_image, scale_factor, method)
            
            if progress_callback:
                progress_callback(2, 3)
            
            # Calculate quality metrics
            quality_metrics = self.calculate_quality_metrics(result)
            metrics.quality_score = quality_metrics.overall_quality
            metrics.sharpness_score = quality_metrics.sharpness
            metrics.artifact_score = 1.0 - quality_metrics.artifact_count
            
            # Compare with original
            original_quality = self.calculate_quality_metrics(pil_image)
            quality_improvement = quality_metrics.overall_quality - original_quality.overall_quality
            
            if quality_improvement < -0.1:
                metrics.warnings.append(
                    f"Quality decreased by {abs(quality_improvement):.3f} after upscaling"
                )
                logger.warning(f"Quality decreased after upscaling: {quality_improvement:.3f}")
            elif quality_improvement > 0.1:
                logger.info(f"Quality improved after upscaling: {quality_improvement:.3f}")
            
            if progress_callback:
                progress_callback(3, 3)
            
            # Cache result
            if use_cache and self.cache and image_path:
                self.cache.set(image_path, scale_factor, method, result)
            
            # Update metrics
            metrics.processing_time = time.time() - start_time
            metrics.success = True
            metrics.upscaled_size = result.size
            self.stats["total_time"] += metrics.processing_time
            self.stats["successful_upscales"] += 1
            
            logger.info(
                f"Upscale successful: {pil_image.size} -> {result.size} "
                f"({scale_factor}x) in {metrics.processing_time:.2f}s | "
                f"Quality: {metrics.quality_score:.3f}"
            )
            
            if return_metrics:
                return result, metrics
            return result
            
        except Exception as e:
            metrics.processing_time = time.time() - start_time
            metrics.success = False
            metrics.errors.append(str(e))
            self.stats["failed_upscales"] += 1
            self.stats["total_time"] += metrics.processing_time
            logger.error(f"Upscaling failed: {e}", exc_info=True)
            
            if return_metrics:
                return None, metrics
            raise
    
    def _apply_upscaling_algorithm(
        self,
        image: Image.Image,
        scale_factor: float,
        method: str
    ) -> Image.Image:
        """Apply the selected upscaling algorithm."""
        method_map = {
            "lanczos": UpscalingAlgorithmsStatic.upscale_lanczos,
            "bicubic": UpscalingAlgorithmsStatic.upscale_bicubic_enhanced,
            "opencv": UpscalingAlgorithmsStatic.upscale_opencv_edsr,
            "multi_scale": UpscalingAlgorithmsStatic.multi_scale_upscale,
            "adaptive": UpscalingAlgorithmsStatic.upscale_adaptive,
            "esrgan_like": UpscalingAlgorithmsStatic.upscale_esrgan_like,
            "waifu2x_like": UpscalingAlgorithmsStatic.upscale_waifu2x_like,
            "real_esrgan_like": UpscalingAlgorithmsStatic.upscale_real_esrgan_like,
            "realesrgan": UpscalingAlgorithmsStatic.upscale_realesrgan,
        }
        
        upscale_func = method_map.get(method, UpscalingAlgorithmsStatic.upscale_lanczos)
        return upscale_func(image, scale_factor)
    
    # Delegate static methods
    @staticmethod
    def upscale_lanczos(image: Image.Image, scale_factor: float, taps: int = 3) -> Image.Image:
        """Upscale using Lanczos."""
        return UpscalingAlgorithmsStatic.upscale_lanczos(image, scale_factor, taps)
    
    @staticmethod
    def upscale_bicubic_enhanced(image: Image.Image, scale_factor: float) -> Image.Image:
        """Enhanced bicubic upscaling."""
        return UpscalingAlgorithmsStatic.upscale_bicubic_enhanced(image, scale_factor)
    
    @staticmethod
    def upscale_opencv_edsr(image: Image.Image, scale_factor: float) -> Image.Image:
        """OpenCV EDSR upscaling."""
        return UpscalingAlgorithmsStatic.upscale_opencv_edsr(image, scale_factor)
    
    @staticmethod
    def multi_scale_upscale(image: Image.Image, scale_factor: float, passes: int = 2) -> Image.Image:
        """Multi-scale upscaling."""
        return UpscalingAlgorithmsStatic.multi_scale_upscale(image, scale_factor, passes)
    
    @staticmethod
    def apply_anti_aliasing(image: Image.Image, strength: float = 0.5) -> Image.Image:
        """Apply anti-aliasing."""
        return PostprocessingMethods.apply_anti_aliasing(image, strength)
    
    @staticmethod
    def reduce_artifacts(image: Image.Image, method: str = "bilateral") -> Image.Image:
        """Reduce artifacts."""
        return PostprocessingMethods.reduce_artifacts(image, method)
    
    @staticmethod
    def enhance_edges(image: Image.Image, strength: float = 1.2) -> Image.Image:
        """Enhance edges."""
        return PostprocessingMethods.enhance_edges(image, strength)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return StatisticsManager.get_statistics(self.stats)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if self.cache:
            return self.cache.get_stats()
        return {"enabled": False}

