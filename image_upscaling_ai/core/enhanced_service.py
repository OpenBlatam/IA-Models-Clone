"""
Enhanced Upscaling Service
==========================

Enhanced service with all intelligent features integrated.
"""

import logging
import time
from typing import Optional, Dict, Any, Union, List
from pathlib import Path
from PIL import Image

from ..config.upscaling_config import UpscalingConfig
from ..models.upscaling_model import UpscalingModel
from ..infrastructure.openrouter_client import OpenRouterClient
from ..models import (
    SmartRecommender,
    AdaptiveLearner,
    AdvancedImageDetector,
    RealtimeAnalyzer,
    AdaptivePreprocessor,
    AdaptivePostprocessor,
    RealESRGANModelManager,
    QualityValidator,
    AdvancedMetricsCollector,
    PerformanceOptimizer,
    IntelligentCache,
    FeedbackSystem
)

logger = logging.getLogger(__name__)


class EnhancedUpscalingService:
    """
    Enhanced upscaling service with all intelligent features.
    
    Features:
    - Smart recommendations
    - Adaptive learning
    - Real-time monitoring
    - Quality validation
    - Performance optimization
    - Feedback collection
    """
    
    def __init__(self, config: Optional[UpscalingConfig] = None):
        """
        Initialize enhanced service.
        
        Args:
            config: Configuration instance
        """
        self.config = config or UpscalingConfig.from_env()
        self.config.validate()
        
        # Initialize core components
        self.openrouter_client = None
        if self.config.use_ai_enhancement:
            self.openrouter_client = OpenRouterClient(
                api_key=self.config.openrouter.api_key,
                model=self.config.openrouter.model,
                timeout=self.config.openrouter.timeout,
                max_retries=self.config.openrouter.max_retries,
                retry_delay=self.config.openrouter.retry_delay,
            )
        
        self.model: Optional[UpscalingModel] = None
        
        # Initialize intelligent components
        self.learner = AdaptiveLearner()
        self.detector = AdvancedImageDetector()
        self.recommender = SmartRecommender(
            learner=self.learner,
            detector=self.detector
        )
        self.analyzer = RealtimeAnalyzer()
        self.preprocessor = AdaptivePreprocessor()
        self.postprocessor = AdaptivePostprocessor()
        
        # Initialize Real-ESRGAN manager if enabled
        self.realesrgan_manager = None
        if self.config.use_realesrgan:
            try:
                self.realesrgan_manager = RealESRGANModelManager(
                    auto_download=self.config.realesrgan_auto_download,
                    device=None  # Auto-detect
                )
            except Exception as e:
                logger.warning(f"Real-ESRGAN manager not available: {e}")
        
        # Initialize quality and performance
        self.validator = QualityValidator(min_score=0.7)
        self.metrics = AdvancedMetricsCollector()
        self.optimizer = PerformanceOptimizer(target_throughput=2.0)
        
        # Initialize cache
        cache_dir = Path(self.config.output_dir) / "cache"
        self.cache = IntelligentCache(
            cache_dir=str(cache_dir),
            max_size_mb=1000,
            max_entries=100
        )
        
        # Initialize feedback
        self.feedback = FeedbackSystem()
        
        # Setup callbacks
        self._setup_callbacks()
        
        logger.info("EnhancedUpscalingService initialized")
    
    def _setup_callbacks(self) -> None:
        """Setup real-time callbacks."""
        def on_progress(metrics):
            logger.debug(
                f"Progress: {metrics.stage} - {metrics.progress:.1%} "
                f"(Quality: {metrics.quality_estimate:.2f})"
            )
        
        self.analyzer.add_progress_callback(on_progress)
    
    def initialize_model(self) -> None:
        """Initialize the upscaling model."""
        if self.model is not None:
            return
        
        logger.info("Initializing upscaling model...")
        self.model = UpscalingModel(
            openrouter_client=self.openrouter_client,
            optimization_core_path=self.config.optimization_core_path,
            quality_mode=self.config.quality_mode,
            use_ai_enhancement=self.config.use_ai_enhancement,
            use_optimization_core=self.config.use_optimization_core,
        )
        logger.info("Model initialized")
    
    async def upscale_image_enhanced(
        self,
        image: Union[str, Path, Image.Image],
        scale_factor: Optional[float] = None,
        use_recommendations: bool = True,
        validate_quality: bool = True,
        collect_feedback: bool = True
    ) -> Dict[str, Any]:
        """
        Upscale image with all enhanced features.
        
        Args:
            image: Input image
            scale_factor: Scale factor (uses recommendation if None)
            use_recommendations: Use smart recommendations
            validate_quality: Validate output quality
            collect_feedback: Enable feedback collection
            
        Returns:
            Enhanced result dictionary
        """
        # Load image
        if isinstance(image, (str, Path)):
            image_path = Path(image)
            pil_image = Image.open(image_path).convert("RGB")
        else:
            pil_image = image.convert("RGB")
            image_path = None
        
        # Analyze image
        analysis = self.detector.analyze(pil_image)
        
        # Get recommendation
        target_scale = scale_factor or self.config.default_scale_factor
        recommendation = None
        
        if use_recommendations:
            recommendation = self.recommender.recommend(
                pil_image,
                target_scale,
                prioritize_speed=False,
                min_quality=0.7
            )
            target_scale = recommendation.scale_factor
            logger.info(f"Recommended: {recommendation.method} (quality: {recommendation.expected_quality:.2f})")
        
        # Start operation
        self.analyzer.start_operation()
        start_time = time.time()
        operation_id = None
        
        try:
            # Preprocess
            self.analyzer.update_stage("preprocessing")
            preprocessing_mode = recommendation.preprocessing_mode if recommendation else "auto"
            preprocessed = self.preprocessor.preprocess(pil_image, mode=preprocessing_mode)
            self.analyzer.update_progress(0.2, quality_estimate=analysis.quality_score)
            
            # Upscale
            self.analyzer.update_stage("upscaling")
            
            if self.realesrgan_manager and (
                recommendation and "realesrgan" in recommendation.method.lower() or
                self.config.use_realesrgan
            ):
                # Use Real-ESRGAN
                method_name = recommendation.method if recommendation else "RealESRGAN_x4plus"
                upscaled = await self.realesrgan_manager.upscale_async(
                    preprocessed,
                    target_scale,
                    model_name=method_name,
                    auto_select=True
                )
            else:
                # Use standard model
                if self.model is None:
                    self.initialize_model()
                
                result = await self.model.upscale(
                    preprocessed,
                    scale_factor=target_scale
                )
                upscaled = result[0] if isinstance(result, tuple) else result
            
            self.analyzer.update_progress(0.7, quality_estimate=0.85)
            
            # Postprocess
            self.analyzer.update_stage("postprocessing")
            postprocessing_mode = recommendation.postprocessing_mode if recommendation else "auto"
            final = self.postprocessor.postprocess(
                upscaled,
                original=pil_image,
                mode=postprocessing_mode
            )
            self.analyzer.update_progress(1.0, quality_estimate=0.90)
            
            # Validate
            processing_time = time.time() - start_time
            quality_report = None
            
            if validate_quality:
                quality_report = self.validator.validate(
                    final,
                    pil_image,
                    target_scale
                )
            
            # Record metrics
            method_used = recommendation.method if recommendation else "default"
            operation_id = self.metrics.record_operation(
                image_type=analysis.image_type,
                scale_factor=target_scale,
                method=method_used,
                processing_time=processing_time,
                quality_score=quality_report.overall_score if quality_report else 0.8,
                memory_usage_mb=512,  # Could be measured
                cache_hit=False,
                success=quality_report.passed if quality_report else True
            )
            
            # Record for learning
            self.learner.record_experience(
                image_type=analysis.image_type,
                scale_factor=target_scale,
                method_used=method_used,
                quality_score=quality_report.overall_score if quality_report else 0.8,
                processing_time=processing_time
            )
            
            # Record for optimization
            self.optimizer.record_operation(
                processing_time,
                memory_usage_mb=512,
                cache_hit=False
            )
            
            # Finish
            summary = self.analyzer.finish_operation()
            
            # Prepare result
            result = {
                "success": True,
                "operation_id": operation_id,
                "upscaled_image": final,
                "original_size": pil_image.size,
                "upscaled_size": final.size,
                "scale_factor": target_scale,
                "method_used": method_used,
                "processing_time": processing_time,
                "quality_score": quality_report.overall_score if quality_report else 0.8,
                "quality_passed": quality_report.passed if quality_report else True,
                "recommendation": {
                    "method": recommendation.method,
                    "expected_quality": recommendation.expected_quality,
                    "expected_time": recommendation.expected_time,
                    "confidence": recommendation.confidence
                } if recommendation else None,
                "analysis": {
                    "image_type": analysis.image_type,
                    "quality": analysis.quality_score,
                    "complexity": analysis.complexity,
                    "recommended_model": analysis.recommended_model
                },
                "metrics": {
                    "stages": summary.get("stages", {}),
                    "total_time": summary.get("total_time", processing_time)
                }
            }
            
            if quality_report:
                result["quality_report"] = {
                    "passed": quality_report.passed,
                    "overall_score": quality_report.overall_score,
                    "issues": quality_report.issues,
                    "recommendations": quality_report.recommendations,
                    "metrics": quality_report.metrics
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in enhanced upscaling: {e}", exc_info=True)
            
            processing_time = time.time() - start_time
            
            # Record failure
            if operation_id:
                self.metrics.record_operation(
                    image_type=analysis.image_type if 'analysis' in locals() else "unknown",
                    scale_factor=target_scale,
                    method="unknown",
                    processing_time=processing_time,
                    quality_score=0.0,
                    success=False,
                    error=str(e)
                )
            
            return {
                "success": False,
                "error": str(e),
                "processing_time": processing_time
            }
    
    def submit_feedback(
        self,
        operation_id: str,
        satisfaction: float,
        quality_rating: float,
        speed_rating: float,
        comments: Optional[str] = None
    ) -> None:
        """Submit user feedback."""
        self.feedback.submit_feedback(
            operation_id,
            satisfaction,
            quality_rating,
            speed_rating,
            comments
        )
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        system_metrics = self.metrics.get_system_metrics()
        optimizer_stats = self.optimizer.get_statistics()
        cache_stats = self.cache.get_stats()
        feedback_stats = self.feedback.get_satisfaction_stats()
        
        return {
            "system_metrics": {
                "total_operations": system_metrics.total_operations,
                "success_rate": 1.0 - system_metrics.error_rate,
                "avg_quality": system_metrics.avg_quality_score,
                "throughput": system_metrics.throughput,
                "cache_hit_rate": system_metrics.cache_hit_rate
            },
            "performance": {
                "avg_time": optimizer_stats["avg_operation_time"],
                "optimal_batch_size": optimizer_stats["optimal_batch_size"],
                "optimal_tile_size": optimizer_stats["optimal_tile_size"],
                "optimal_concurrency": optimizer_stats["optimal_concurrency"]
            },
            "cache": {
                "entries": cache_stats["entries"],
                "hit_rate": cache_stats["hit_rate"],
                "total_size_mb": cache_stats["total_size_mb"]
            },
            "feedback": {
                "total_feedback": feedback_stats["total_feedback"],
                "avg_satisfaction": feedback_stats["avg_satisfaction"],
                "avg_quality": feedback_stats["avg_quality"],
                "avg_speed": feedback_stats["avg_speed"]
            },
            "learning": {
                "total_records": self.learner.get_statistics()["total_records"],
                "methods_tested": self.learner.get_statistics()["methods_tested"]
            }
        }

