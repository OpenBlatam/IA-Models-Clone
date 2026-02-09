"""
Advanced Upscaling Analysis and Comparison
==========================================

Analysis, comparison, and recommendation methods for upscaling.
"""

import logging
import time
from typing import Tuple, Optional, Dict, Any, List, Union
from pathlib import Path
from PIL import Image

from .helpers import (
    ImageAnalysisUtils,
    MethodComparisonUtils,
    QualityCalculator,
)

logger = logging.getLogger(__name__)


class AnalysisMethods:
    """Analysis and comparison methods."""
    
    def __init__(self, base_upscaler):
        """Initialize with base upscaler."""
        self.base_upscaler = base_upscaler
    
    def analyze_image_characteristics(
        self,
        image: Union[Image.Image, str, Path]
    ) -> Dict[str, Any]:
        """Analyze image characteristics."""
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        else:
            pil_image = image.convert("RGB")
        
        return ImageAnalysisUtils.analyze_image(pil_image)
    
    def compare_methods(
        self,
        image: Union[Image.Image, str, Path],
        scale_factor: float,
        methods: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Compare different upscaling methods."""
        return MethodComparisonUtils.compare_methods(
            self.base_upscaler.upscale,
            image,
            scale_factor,
            methods=methods,
            use_cache=False
        )
    
    def compare_all_methods_comprehensive(
        self,
        image: Union[Image.Image, str, Path],
        scale_factor: float,
        methods: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Comprehensive comparison of all methods."""
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        else:
            pil_image = image.convert("RGB")
        
        if methods is None:
            methods = [
                "lanczos", "bicubic", "opencv", "multi_scale",
                "esrgan_like", "waifu2x_like", "real_esrgan_like"
            ]
        
        results = {}
        for method in methods:
            try:
                start_time = time.time()
                result = self.base_upscaler.upscale(pil_image, scale_factor, method, return_metrics=False)
                processing_time = time.time() - start_time
                
                quality = self.base_upscaler.calculate_quality_metrics(result)
                
                results[method] = {
                    "success": True,
                    "processing_time": processing_time,
                    "quality_score": quality.overall_quality,
                    "sharpness": quality.sharpness,
                    "contrast": quality.contrast,
                    "noise_level": quality.noise_level,
                    "artifact_count": quality.artifact_count,
                    "size": result.size,
                }
            except Exception as e:
                results[method] = {
                    "success": False,
                    "error": str(e),
                }
        
        # Generate recommendations
        successful = {k: v for k, v in results.items() if v.get("success")}
        
        if successful:
            best_quality = max(successful.items(), key=lambda x: x[1]["quality_score"])
            fastest = min(successful.items(), key=lambda x: x[1]["processing_time"])
            
            recommendations = {
                "best_quality": best_quality[0],
                "fastest": fastest[0],
                "balanced": self._find_balanced_method(successful),
            }
        else:
            recommendations = {}
        
        return {
            "methods": results,
            "recommendations": recommendations,
            "summary": {
                "total_methods": len(methods),
                "successful": len(successful),
                "failed": len(methods) - len(successful),
            }
        }
    
    def _find_balanced_method(self, results: Dict[str, Dict[str, Any]]) -> str:
        """Find method with best quality/time balance."""
        if not results:
            return "lanczos"
        
        best_score = 0.0
        best_method = "lanczos"
        
        for method, data in results.items():
            quality = data["quality_score"]
            time_val = data["processing_time"]
            
            if time_val > 0:
                score = quality / time_val
                if score > best_score:
                    best_score = score
                    best_method = method
        
        return best_method
    
    def get_processing_recommendations(
        self,
        image: Image.Image,
        scale_factor: float
    ) -> Dict[str, Any]:
        """Get processing recommendations based on image analysis."""
        analysis = self.analyze_image_characteristics(image)
        
        recommendations = {
            "recommended_method": "lanczos",
            "recommended_preprocessing": [],
            "recommended_postprocessing": [],
            "reasoning": [],
        }
        
        # Analyze image type
        if analysis.get("is_anime", False):
            recommendations["recommended_method"] = "waifu2x_like"
            recommendations["reasoning"].append("Anime image detected - Waifu2x recommended")
        elif analysis.get("is_photo", False):
            recommendations["recommended_method"] = "real_esrgan_like"
            recommendations["reasoning"].append("Photo detected - Real-ESRGAN recommended")
        elif analysis.get("is_artwork", False):
            recommendations["recommended_method"] = "esrgan_like"
            recommendations["reasoning"].append("Artwork detected - ESRGAN recommended")
        
        # Analyze quality
        quality_score = analysis.get("quality_score", 0.5)
        if quality_score < 0.5:
            recommendations["recommended_preprocessing"].append("denoising")
            recommendations["reasoning"].append("Low quality detected - denoising recommended")
        
        # Analyze scale factor
        if scale_factor > 4.0:
            recommendations["recommended_method"] = "multi_scale"
            recommendations["reasoning"].append("Large scale factor - multi-scale recommended")
        
        return recommendations
    
    def get_upscaling_recommendations_advanced(
        self,
        image: Image.Image,
        scale_factor: float
    ) -> Dict[str, Any]:
        """Get advanced upscaling recommendations."""
        analysis = self.analyze_image_characteristics(image)
        basic_recs = self.get_processing_recommendations(image, scale_factor)
        
        advanced = {
            **basic_recs,
            "optimal_pipeline": self._recommend_pipeline(analysis, scale_factor),
            "expected_quality": self._estimate_quality(analysis, scale_factor),
            "expected_time": self._estimate_time(analysis, scale_factor),
            "resource_requirements": self._estimate_resources(analysis, scale_factor),
        }
        
        return advanced
    
    def _recommend_pipeline(self, analysis: Dict[str, Any], scale_factor: float) -> str:
        """Recommend optimal pipeline."""
        if scale_factor <= 2.0:
            return "fast"
        elif scale_factor <= 4.0:
            return "balanced"
        else:
            return "quality"
    
    def _estimate_quality(self, analysis: Dict[str, Any], scale_factor: float) -> float:
        """Estimate expected quality."""
        base_quality = analysis.get("quality_score", 0.5)
        scale_penalty = min(scale_factor / 4.0, 0.3)
        return max(0.0, base_quality - scale_penalty)
    
    def _estimate_time(self, analysis: Dict[str, Any], scale_factor: float) -> float:
        """Estimate expected processing time."""
        base_time = 1.0
        scale_multiplier = scale_factor ** 1.5
        size_multiplier = (analysis.get("width", 512) * analysis.get("height", 512)) / (512 * 512)
        return base_time * scale_multiplier * size_multiplier
    
    def _estimate_resources(self, analysis: Dict[str, Any], scale_factor: float) -> Dict[str, Any]:
        """Estimate resource requirements."""
        width = analysis.get("width", 512)
        height = analysis.get("height", 512)
        
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        # Estimate memory (rough calculation)
        memory_mb = (new_width * new_height * 3 * 4) / (1024 * 1024)  # RGB float32
        
        return {
            "memory_mb": memory_mb,
            "cpu_intensive": scale_factor > 3.0,
            "gpu_recommended": scale_factor > 4.0 or memory_mb > 1000,
        }
    
    def get_optimal_upscaling_strategy(
        self,
        image: Image.Image,
        scale_factor: float
    ) -> Dict[str, Any]:
        """Get optimal upscaling strategy."""
        analysis = self.analyze_image_characteristics(image)
        recommendations = self.get_upscaling_recommendations_advanced(image, scale_factor)
        
        strategy = {
            "method": recommendations["recommended_method"],
            "pipeline": recommendations["optimal_pipeline"],
            "preprocessing": recommendations["recommended_preprocessing"],
            "postprocessing": recommendations["recommended_postprocessing"],
            "expected_quality": recommendations["expected_quality"],
            "expected_time": recommendations["expected_time"],
            "resource_requirements": recommendations["resource_requirements"],
            "reasoning": recommendations["reasoning"],
        }
        
        return strategy
    
    def export_comparison_report(
        self,
        image: Union[Image.Image, str, Path],
        scale_factor: float,
        methods: Optional[List[str]] = None,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Export comparison report."""
        comparison = self.compare_all_methods_comprehensive(image, scale_factor, methods)
        
        report = {
            "image_info": {
                "path": str(image) if isinstance(image, (str, Path)) else "PIL Image",
                "scale_factor": scale_factor,
            },
            "comparison": comparison,
            "generated_at": time.time(),
        }
        
        if output_path:
            import json
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Comparison report saved to {output_path}")
        
        return report


