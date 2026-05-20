"""
Image Upscaling Service
=======================

Main service for image upscaling operations.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union, List
from PIL import Image
import asyncio

from ..models.upscaling_model import UpscalingModel
from ..models.batch_processor import BatchProcessor
from ..models.result_cache import ResultCache
from ..models.progress_tracker import ProgressTracker, ProgressStage
from ..models.image_comparison import ImageComparison
from ..models.validators import ImageValidator, ScaleFactorValidator, MemoryEstimator
from ..models.error_handlers import ValidationError, MemoryError, handle_upscaling_error
from ..models.performance_monitor import PerformanceMonitor
from ..infrastructure.openrouter_client import OpenRouterClient
from ..config.upscaling_config import UpscalingConfig

logger = logging.getLogger(__name__)


class UpscalingService:
    """
    Main service for image upscaling operations.
    
    Handles model initialization, image processing, and result management.
    """
    
    def __init__(self, config: Optional[UpscalingConfig] = None):
        """
        Initialize Upscaling Service.
        
        Args:
            config: Configuration instance (optional, will create default if not provided)
        """
        self.config = config or UpscalingConfig.from_env()
        self.config.validate()
        
        # Initialize OpenRouter client
        self.openrouter_client = None
        if self.config.use_ai_enhancement:
            self.openrouter_client = OpenRouterClient(
                api_key=self.config.openrouter.api_key,
                model=self.config.openrouter.model,
                timeout=self.config.openrouter.timeout,
                max_retries=self.config.openrouter.max_retries,
                retry_delay=self.config.openrouter.retry_delay,
            )
        
        # Initialize model
        self.model: Optional[UpscalingModel] = None
        
        # Ensure output directory exists
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize cache
        cache_dir = self.output_dir / "cache"
        self.cache = ResultCache(
            cache_dir=str(cache_dir),
            max_size_mb=1000,
            enabled=getattr(self.config, "enable_cache", True)
        )
        
        # Initialize batch processor
        self.batch_processor = BatchProcessor(
            max_workers=getattr(self.config, "max_workers", 4)
        )
        
        logger.info("UpscalingService initialized")
    
    def initialize_model(self) -> None:
        """Initialize the upscaling model."""
        if self.model is not None:
            logger.warning("Model already initialized")
            return
        
        logger.info("Initializing Upscaling Model...")
        
        try:
            self.model = UpscalingModel(
                openrouter_client=self.openrouter_client,
                use_optimization_core=self.config.use_optimization_core,
                quality_mode=self.config.quality_mode,
            )
            
            logger.info("Model initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise
    
    async def upscale_image(
        self,
        image: Union[str, Path, Image.Image],
        scale_factor: Optional[float] = None,
        output_filename: Optional[str] = None,
        use_ai: Optional[bool] = None,
        use_optimization_core: Optional[bool] = None,
        save_result: bool = True,
        progress_callback: Optional[callable] = None,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        Upscale an image.
        
        Args:
            image: Input image (path string, Path, or PIL Image)
            scale_factor: Scale factor (defaults to config default)
            output_filename: Optional custom output filename
            use_ai: Override AI enhancement setting
            use_optimization_core: Override optimization_core setting
            save_result: Whether to save the result
            
        Returns:
            Dict with result info and paths
        """
        if self.model is None:
            self.initialize_model()
        
        # Use default scale factor if not provided
        scale_factor = scale_factor or self.config.default_scale_factor
        
        # Validate scale factor
        is_valid, errors = self.scale_validator.validate(
            scale_factor=scale_factor,
            min_scale=self.config.min_scale_factor,
            max_scale=self.config.max_scale_factor,
            max_output_size=(self.config.max_image_size, self.config.max_image_size)
        )
        
        if not is_valid:
            raise ValidationError(
                f"Invalid scale factor: {', '.join(errors)}",
                details={"scale_factor": scale_factor, "errors": errors}
            )
        
        # Validate image
        is_valid, validation_info = self.image_validator.validate(
            image=image,
            scale_factor=scale_factor
        )
        
        if not is_valid:
            raise ValidationError(
                f"Image validation failed: {', '.join(validation_info.get('errors', []))}",
                details=validation_info
            )
        
        # Check for warnings
        warnings = validation_info.get("warnings", [])
        if warnings:
            for warning in warnings:
                logger.warning(f"Image validation warning: {warning}")
        
        # Estimate memory usage
        if isinstance(image, Image.Image):
            image_size = image.size
        elif isinstance(image, (str, Path)):
            img = Image.open(image)
            image_size = img.size
        else:
            image_size = validation_info["metrics"].get("resolution", (0, 0))
        
        memory_estimate = self.memory_estimator.estimate(image_size, scale_factor)
        logger.debug(f"Memory estimate: {memory_estimate['total_mb']:.1f}MB")
        
        if memory_estimate["total_mb"] > 1000:
            logger.warning(
                f"High memory usage estimated: {memory_estimate['total_mb']:.1f}MB. "
                "Consider reducing scale factor or image size."
            )
        
        logger.info(f"Upscaling image with scale factor {scale_factor}x")
        
        # Monitor performance
        with self.performance_monitor.monitor("upscale_image"):
            return await self._upscale_image_internal(
                image, scale_factor, output_filename, use_ai,
                use_optimization_core, save_result, progress_callback, use_cache
            )
    
    async def _upscale_image_internal(
        self,
        image: Union[str, Path, Image.Image],
        scale_factor: float,
        output_filename: Optional[str] = None,
        use_ai: Optional[bool] = None,
        use_optimization_core: Optional[bool] = None,
        save_result: bool = True,
        progress_callback: Optional[callable] = None,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """Internal upscaling method."""
        # Get image path for cache
        image_path = None
        if isinstance(image, (str, Path)):
            image_path = str(image)
        elif isinstance(image, Image.Image):
            # Generate a path for PIL images
            image_path = "pil_image"
        
        # Check cache
        if use_cache and image_path:
            cached_result = self.cache.get(
                image_path=image_path,
                scale_factor=scale_factor,
                quality_mode=self.config.quality_mode,
                use_ai=use_ai if use_ai is not None else self.config.use_ai_enhancement,
                use_optimization_core=use_optimization_core if use_optimization_core is not None else self.config.use_optimization_core,
            )
            
            if cached_result:
                logger.info("Using cached result")
                upscaled_image, metrics = cached_result
                # Convert metrics to UpscalingMetrics if needed
                from ..models.upscaling_model import UpscalingMetrics
                if not isinstance(metrics, UpscalingMetrics):
                    metrics = UpscalingMetrics(
                        original_size=metrics.get("original_size", (0, 0)),
                        upscaled_size=metrics.get("upscaled_size", (0, 0)),
                        scale_factor=scale_factor,
                        processing_time=metrics.get("processing_time", 0.0),
                        quality_score=metrics.get("quality_score"),
                        quality_metrics=metrics.get("quality_metrics"),
                        success=True
                    )
        else:
            # Initialize progress tracker
            tracker = ProgressTracker(callback=progress_callback)
            tracker.start()
            
            try:
                # Upscale image
                tracker.update(ProgressStage.UPSCALING, 0.3, "Upscaling image...")
                upscaled_image, metrics = await self.model.upscale(
                    image=image,
                    scale_factor=scale_factor,
                    use_ai=use_ai if use_ai is not None else self.config.use_ai_enhancement,
                    use_optimization_core=use_optimization_core if use_optimization_core is not None else self.config.use_optimization_core,
                )
                
                tracker.update(ProgressStage.QUALITY_CHECK, 0.9, "Calculating quality metrics...")
                tracker.complete("Upscaling completed")
                
                # Save to cache
                if use_cache and image_path:
                    self.cache.save(
                        image_path=image_path,
                        scale_factor=scale_factor,
                        quality_mode=self.config.quality_mode,
                        use_ai=use_ai if use_ai is not None else self.config.use_ai_enhancement,
                        use_optimization_core=use_optimization_core if use_optimization_core is not None else self.config.use_optimization_core,
                        image=upscaled_image,
                        metrics={
                            "original_size": metrics.original_size,
                            "upscaled_size": metrics.upscaled_size,
                            "scale_factor": metrics.scale_factor,
                            "processing_time": metrics.processing_time,
                            "quality_score": metrics.quality_score,
                            "quality_metrics": metrics.quality_metrics,
                        }
                    )
            except Exception as e:
                tracker.error(str(e))
                # Convert to appropriate error type
                raise handle_upscaling_error(e)
        
        result = {
            "scale_factor": scale_factor,
            "original_size": metrics.original_size,
            "upscaled_size": metrics.upscaled_size,
            "processing_time": metrics.processing_time,
            "quality_score": metrics.quality_score,
            "quality_metrics": metrics.quality_metrics,
            "success": metrics.success,
            "saved": False,
        }
        
        # Save result if requested
        if save_result:
            if output_filename is None:
                # Generate filename
                if isinstance(image, (str, Path)):
                    input_path = Path(image)
                    stem = input_path.stem
                else:
                    stem = "upscaled_image"
                
                output_filename = f"{stem}_x{scale_factor:.1f}.png"
            
            output_path = self.output_dir / output_filename
            upscaled_image.save(output_path, "PNG", quality=95)
            result["saved_path"] = str(output_path)
            result["saved"] = True
            logger.info(f"Saved upscaled image to {output_path}")
        
        return result
    
    async def batch_upscale(
        self,
        images: List[Union[str, Path, Image.Image]],
        scale_factor: Optional[float] = None,
        use_ai: Optional[bool] = None,
        use_optimization_core: Optional[bool] = None,
        progress_callback: Optional[callable] = None,
        parallel: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Upscale multiple images in batch with parallel processing.
        
        Args:
            images: List of input images
            scale_factor: Scale factor (defaults to config default)
            use_ai: Override AI enhancement setting
            use_optimization_core: Override optimization_core setting
            progress_callback: Optional progress callback
            parallel: Whether to process in parallel
            
        Returns:
            List of result dicts
        """
        if self.model is None:
            self.initialize_model()
        
        scale_factor = scale_factor or self.config.default_scale_factor
        
        if parallel:
            # Use batch processor for parallel execution
            async def process_single(img):
                return await self.upscale_image(
                    image=img,
                    scale_factor=scale_factor,
                    use_ai=use_ai,
                    use_optimization_core=use_optimization_core,
                    output_filename=None,  # Will be auto-generated
                    progress_callback=None,  # Batch processor handles progress
                    use_cache=True,
                )
            
            results = await self.batch_processor.process_batch(
                images=images,
                process_func=process_single,
                progress_callback=progress_callback
            )
        else:
            # Sequential processing
            results = []
            for i, image in enumerate(images):
                logger.info(f"Processing batch item {i+1}/{len(images)}")
                try:
                    result = await self.upscale_image(
                        image=image,
                        scale_factor=scale_factor,
                        use_ai=use_ai,
                        use_optimization_core=use_optimization_core,
                        output_filename=f"batch_{i+1}_x{scale_factor:.1f}.png",
                        use_cache=True,
                    )
                    result["batch_index"] = i
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing batch item {i+1}: {e}")
                    results.append({
                        "batch_index": i,
                        "error": str(e),
                        "success": False,
                    })
        
        return results
    
    def create_comparison(
        self,
        original: Union[str, Path, Image.Image],
        upscaled: Union[str, Path, Image.Image],
        metrics: Optional[Dict[str, Any]] = None,
        save_path: Optional[str] = None
    ) -> Image.Image:
        """
        Create side-by-side comparison image.
        
        Args:
            original: Original image
            upscaled: Upscaled image
            metrics: Optional metrics to display
            save_path: Optional path to save comparison
            
        Returns:
            Comparison image
        """
        # Load images if paths
        if isinstance(original, (str, Path)):
            orig_img = Image.open(original).convert("RGB")
        else:
            orig_img = original.convert("RGB")
        
        if isinstance(upscaled, (str, Path)):
            upscaled_img = Image.open(upscaled).convert("RGB")
        else:
            upscaled_img = upscaled.convert("RGB")
        
        # Create comparison
        comparison = ImageComparison.create_side_by_side(
            orig_img,
            upscaled_img,
            labels=("Original", "Upscaled"),
            metrics=metrics
        )
        
        # Save if requested
        if save_path:
            comparison.save(save_path, "PNG", quality=95)
            logger.info(f"Saved comparison to {save_path}")
        
        return comparison
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        if self.model is None:
            return {"status": "not_initialized"}
        
        info = self.model.get_model_info()
        info["config"] = {
            "default_scale_factor": self.config.default_scale_factor,
            "quality_mode": self.config.quality_mode,
            "use_ai_enhancement": self.config.use_ai_enhancement,
            "use_optimization_core": self.config.use_optimization_core,
        }
        
        return info
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check.
        
        Returns:
            Health status
        """
        health = {
            "status": "healthy",
            "model_initialized": self.model is not None,
            "openrouter_configured": (
                self.openrouter_client is not None and
                self.openrouter_client.is_configured()
            ) if self.openrouter_client else False,
        }
        
        # Check OpenRouter if configured
        if self.openrouter_client and self.openrouter_client.is_configured():
            try:
                openrouter_health = await self.openrouter_client.health_check()
                health["openrouter"] = openrouter_health
            except Exception as e:
                health["openrouter"] = {"status": "error", "error": str(e)}
        
        return health
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_stats()
    
    def clear_cache(self) -> None:
        """Clear result cache."""
        self.cache.clear()
        logger.info("Cache cleared")
    
    async def close(self) -> None:
        """Clean up resources."""
        if self.openrouter_client:
            await self.openrouter_client.close()
        
        if self.model:
            del self.model
            self.model = None
        
        logger.info("Service closed and resources cleaned up")

