"""
Method Comparison Utilities
============================

Utilities for comparing and benchmarking upscaling methods.
"""

import logging
import time
import json
from typing import Dict, Any, List, Optional, Union, Callable
from pathlib import Path
from PIL import Image

from .metrics_utils import UpscalingMetrics
from .quality_calculator_utils import QualityCalculator

logger = logging.getLogger(__name__)


class MethodComparisonUtils:
    """Utilities for comparing upscaling methods."""
    
    @staticmethod
    def compare_methods(
        upscale_func: Callable,
        image: Union[Image.Image, str, Path],
        scale_factor: float,
        methods: Optional[List[str]] = None,
        **upscale_kwargs
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare different upscaling methods.
        
        Args:
            upscale_func: Function to call for upscaling (should accept image, scale_factor, method, return_metrics)
            image: Input image
            scale_factor: Scale factor
            methods: List of methods to compare (default: all available)
            **upscale_kwargs: Additional arguments for upscale_func
            
        Returns:
            Dictionary with comparison results for each method
        """
        if methods is None:
            methods = ["lanczos", "bicubic", "opencv", "multi_scale", "adaptive"]
        
        results = {}
        
        for method in methods:
            try:
                start_time = time.time()
                result = upscale_func(
                    image,
                    scale_factor,
                    method,
                    return_metrics=True,
                    **upscale_kwargs
                )
                elapsed = time.time() - start_time
                
                if isinstance(result, tuple):
                    upscaled_image, metrics = result
                else:
                    upscaled_image = result
                    # Calculate metrics if not provided
                    quality = QualityCalculator.calculate_quality_metrics(upscaled_image)
                    metrics = UpscalingMetrics(
                        original_size=upscaled_image.size if hasattr(upscaled_image, 'size') else (0, 0),
                        upscaled_size=upscaled_image.size if hasattr(upscaled_image, 'size') else (0, 0),
                        scale_factor=scale_factor,
                        processing_time=elapsed,
                        quality_score=quality.overall_quality,
                        sharpness_score=quality.sharpness,
                        artifact_score=1.0 - quality.artifact_count,
                        method_used=method,
                        success=True,
                    )
                
                results[method] = {
                    "time": elapsed,
                    "quality_score": metrics.quality_score,
                    "sharpness_score": metrics.sharpness_score,
                    "artifact_score": metrics.artifact_score,
                    "success": metrics.success,
                    "size": upscaled_image.size if upscaled_image else None,
                }
            except Exception as e:
                logger.warning(f"Method {method} comparison failed: {e}")
                results[method] = {
                    "error": str(e),
                    "success": False,
                }
        
        return results
    
    @staticmethod
    def benchmark_all_methods(
        upscale_func: Callable,
        profile_func: Optional[Callable],
        image: Union[Image.Image, str, Path],
        scale_factor: float,
        methods: Optional[List[str]] = None,
        **upscale_kwargs
    ) -> Dict[str, Any]:
        """
        Comprehensive benchmark of all upscaling methods.
        
        Args:
            upscale_func: Function to call for upscaling
            profile_func: Optional function to profile performance
            image: Input image
            scale_factor: Scale factor
            methods: List of methods to benchmark
            **upscale_kwargs: Additional arguments for upscale_func
            
        Returns:
            Comprehensive benchmark report
        """
        if methods is None:
            methods = [
                "lanczos", "bicubic", "opencv", "multi_scale",
                "esrgan_like", "waifu2x_like", "real_esrgan_like",
                "adaptive"
            ]
        
        benchmark_results = {}
        
        for method in methods:
            try:
                # Profile the method if function provided
                profile = None
                if profile_func:
                    try:
                        profile = profile_func(image, scale_factor, method, iterations=3)
                    except Exception as e:
                        logger.warning(f"Profiling {method} failed: {e}")
                
                # Get single run metrics
                result = upscale_func(
                    image,
                    scale_factor,
                    method,
                    return_metrics=True,
                    **upscale_kwargs
                )
                
                if isinstance(result, tuple):
                    upscaled_image, metrics = result
                else:
                    upscaled_image = result
                    quality = QualityCalculator.calculate_quality_metrics(upscaled_image)
                    metrics = UpscalingMetrics(
                        original_size=upscaled_image.size if hasattr(upscaled_image, 'size') else (0, 0),
                        upscaled_size=upscaled_image.size if hasattr(upscaled_image, 'size') else (0, 0),
                        scale_factor=scale_factor,
                        processing_time=0.0,
                        quality_score=quality.overall_quality,
                        sharpness_score=quality.sharpness,
                        artifact_score=1.0 - quality.artifact_count,
                        method_used=method,
                        success=True,
                    )
                
                # Calculate additional metrics
                quality_metrics = QualityCalculator.calculate_quality_metrics(upscaled_image)
                
                benchmark_results[method] = {
                    "profile": profile,
                    "metrics": {
                        "quality_score": metrics.quality_score,
                        "sharpness_score": metrics.sharpness_score,
                        "artifact_score": metrics.artifact_score,
                        "processing_time": metrics.processing_time,
                    },
                    "quality_metrics": {
                        "overall_quality": quality_metrics.overall_quality,
                        "sharpness": quality_metrics.sharpness,
                        "contrast": quality_metrics.contrast,
                        "brightness": quality_metrics.brightness,
                        "noise_level": quality_metrics.noise_level,
                        "artifact_count": quality_metrics.artifact_count,
                    },
                    "success": metrics.success,
                    "size": upscaled_image.size if upscaled_image else None,
                }
            except Exception as e:
                logger.warning(f"Benchmarking {method} failed: {e}")
                benchmark_results[method] = {
                    "error": str(e),
                    "success": False,
                }
        
        # Generate recommendations
        best_quality = None
        fastest = None
        best_balanced = None
        best_quality_score = 0.0
        fastest_time = float('inf')
        best_balanced_score = 0.0
        
        for method, results in benchmark_results.items():
            if results.get("success"):
                quality = results["metrics"]["quality_score"]
                time_val = results["metrics"]["processing_time"]
                
                # Best quality
                if quality and quality > best_quality_score:
                    best_quality_score = quality
                    best_quality = method
                
                # Fastest
                if time_val and time_val < fastest_time:
                    fastest_time = time_val
                    fastest = method
                
                # Best balanced (quality / time)
                if time_val and time_val > 0:
                    balanced_score = quality / time_val if quality else 0.0
                    if balanced_score > best_balanced_score:
                        best_balanced_score = balanced_score
                        best_balanced = method
        
        report = {
            "image_info": {
                "path": str(image) if isinstance(image, (str, Path)) else "PIL Image",
                "scale_factor": scale_factor,
            },
            "methods_tested": methods,
            "results": benchmark_results,
            "recommendations": {
                "best_quality": best_quality,
                "fastest": fastest,
                "best_balanced": best_balanced,
            },
            "summary": {
                "total_methods": len(methods),
                "successful_methods": sum(1 for r in benchmark_results.values() if r.get("success")),
                "avg_quality": sum(
                    r["metrics"]["quality_score"] for r in benchmark_results.values()
                    if r.get("success") and r["metrics"].get("quality_score")
                ) / max(1, sum(1 for r in benchmark_results.values() if r.get("success"))),
            }
        }
        
        return report
    
    @staticmethod
    def export_comparison_report(
        comparison_data: Dict[str, Any],
        output_path: str,
        format: str = "json"
    ) -> None:
        """
        Export comparison report to file.
        
        Args:
            comparison_data: Comparison data dictionary
            output_path: Path to save report
            format: Format to export ('json', 'txt')
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            with open(output_file, 'w') as f:
                json.dump(comparison_data, f, indent=2)
        elif format == "txt":
            with open(output_file, 'w') as f:
                f.write("Upscaling Method Comparison Report\n")
                f.write("=" * 50 + "\n\n")
                
                if "image_info" in comparison_data:
                    f.write(f"Image: {comparison_data['image_info'].get('path', 'N/A')}\n")
                    f.write(f"Scale Factor: {comparison_data['image_info'].get('scale_factor', 'N/A')}\n\n")
                
                if "results" in comparison_data:
                    f.write("Results:\n")
                    f.write("-" * 50 + "\n")
                    for method, result in comparison_data["results"].items():
                        f.write(f"\n{method}:\n")
                        if result.get("success"):
                            f.write(f"  Quality Score: {result.get('metrics', {}).get('quality_score', 'N/A')}\n")
                            f.write(f"  Processing Time: {result.get('metrics', {}).get('processing_time', 'N/A')}s\n")
                        else:
                            f.write(f"  Error: {result.get('error', 'Unknown error')}\n")
                
                if "recommendations" in comparison_data:
                    f.write("\n\nRecommendations:\n")
                    f.write("-" * 50 + "\n")
                    recs = comparison_data["recommendations"]
                    f.write(f"Best Quality: {recs.get('best_quality', 'N/A')}\n")
                    f.write(f"Fastest: {recs.get('fastest', 'N/A')}\n")
                    f.write(f"Best Balanced: {recs.get('best_balanced', 'N/A')}\n")
        
        logger.info(f"Comparison report exported to {output_path}")


