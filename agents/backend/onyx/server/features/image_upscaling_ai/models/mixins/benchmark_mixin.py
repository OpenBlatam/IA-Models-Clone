"""
Benchmark Mixin

Contains benchmarking and performance testing functionality.
"""

import logging
import time
from typing import Union, Dict, Any, List, Optional
from pathlib import Path
from PIL import Image

from ..helpers import (
    QualityCalculator,
    MethodSelector,
)

logger = logging.getLogger(__name__)


class BenchmarkMixin:
    """
    Mixin providing benchmarking functionality.
    
    This mixin contains:
    - Performance benchmarking
    - Method comparison
    - Quality benchmarking
    - Speed benchmarking
    - Comprehensive reports
    """
    
    def benchmark_methods(
        self,
        image: Union[Image.Image, str, Path],
        scale_factor: float,
        methods: Optional[List[str]] = None,
        iterations: int = 1
    ) -> Dict[str, Any]:
        """
        Benchmark multiple upscaling methods.
        
        Args:
            image: Input image
            scale_factor: Scale factor
            methods: List of methods to benchmark (None = all)
            iterations: Number of iterations per method
            
        Returns:
            Dictionary with benchmark results
        """
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
            times = []
            qualities = []
            
            for _ in range(iterations):
                start_time = time.time()
                try:
                    result = self.upscale(pil_image, scale_factor, method, return_metrics=False)
                    processing_time = time.time() - start_time
                    
                    quality = QualityCalculator.calculate_quality_metrics(result)
                    
                    times.append(processing_time)
                    qualities.append(quality.overall_quality)
                except Exception as e:
                    logger.warning(f"Method {method} failed: {e}")
                    times.append(float('inf'))
                    qualities.append(0.0)
            
            if times and all(t != float('inf') for t in times):
                results[method] = {
                    "avg_time": sum(times) / len(times),
                    "min_time": min(times),
                    "max_time": max(times),
                    "avg_quality": sum(qualities) / len(qualities),
                    "min_quality": min(qualities),
                    "max_quality": max(qualities),
                    "iterations": iterations,
                    "success": True
                }
            else:
                results[method] = {
                    "success": False,
                    "error": "Method failed"
                }
        
        # Find best methods
        if results:
            best_quality = max(
                (m, r["avg_quality"]) for m, r in results.items() 
                if r.get("success", False)
            )
            fastest = min(
                (m, r["avg_time"]) for m, r in results.items() 
                if r.get("success", False)
            )
            
            results["_summary"] = {
                "best_quality": best_quality[0],
                "fastest": fastest[0],
                "total_methods": len(methods),
                "successful_methods": sum(1 for r in results.values() if r.get("success", False))
            }
        
        return results
    
    def benchmark_quality(
        self,
        image: Union[Image.Image, str, Path],
        scale_factor: float,
        method: str = "auto"
    ) -> Dict[str, Any]:
        """
        Benchmark quality of upscaling.
        
        Args:
            image: Input image
            scale_factor: Scale factor
            method: Method to use or 'auto'
            
        Returns:
            Dictionary with quality benchmark results
        """
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        else:
            pil_image = image.convert("RGB")
        
        if method == "auto":
            method = MethodSelector.select_best_method(pil_image, scale_factor)
        
        original_quality = QualityCalculator.calculate_quality_metrics(pil_image)
        
        start_time = time.time()
        result = self.upscale(pil_image, scale_factor, method, return_metrics=False)
        processing_time = time.time() - start_time
        
        upscaled_quality = QualityCalculator.calculate_quality_metrics(result)
        
        improvement = upscaled_quality.overall_quality - original_quality.overall_quality
        
        return {
            "method": method,
            "processing_time": processing_time,
            "original_quality": {
                "overall": original_quality.overall_quality,
                "sharpness": original_quality.sharpness,
                "contrast": original_quality.contrast,
            },
            "upscaled_quality": {
                "overall": upscaled_quality.overall_quality,
                "sharpness": upscaled_quality.sharpness,
                "contrast": upscaled_quality.contrast,
            },
            "improvement": {
                "overall": improvement,
                "sharpness": upscaled_quality.sharpness - original_quality.sharpness,
                "contrast": upscaled_quality.contrast - original_quality.contrast,
            },
            "scale_factor": scale_factor,
            "original_size": pil_image.size,
            "upscaled_size": result.size,
        }
    
    def benchmark_speed(
        self,
        image: Union[Image.Image, str, Path],
        scale_factor: float,
        methods: Optional[List[str]] = None,
        iterations: int = 5
    ) -> Dict[str, Any]:
        """
        Benchmark speed of upscaling methods.
        
        Args:
            image: Input image
            scale_factor: Scale factor
            methods: List of methods to benchmark
            iterations: Number of iterations
            
        Returns:
            Dictionary with speed benchmark results
        """
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        else:
            pil_image = image.convert("RGB")
        
        if methods is None:
            methods = ["lanczos", "bicubic", "opencv", "multi_scale", "real_esrgan_like"]
        
        results = {}
        
        for method in methods:
            times = []
            for _ in range(iterations):
                start_time = time.time()
                try:
                    self.upscale(pil_image, scale_factor, method, return_metrics=False)
                    times.append(time.time() - start_time)
                except Exception as e:
                    logger.warning(f"Method {method} failed: {e}")
                    times.append(float('inf'))
            
            if times and all(t != float('inf') for t in times):
                results[method] = {
                    "avg_time": sum(times) / len(times),
                    "min_time": min(times),
                    "max_time": max(times),
                    "iterations": iterations,
                    "images_per_second": 1.0 / (sum(times) / len(times)) if sum(times) > 0 else 0
                }
        
        if results:
            fastest = min(results.items(), key=lambda x: x[1]["avg_time"])
            results["_summary"] = {
                "fastest": fastest[0],
                "fastest_time": fastest[1]["avg_time"],
                "total_methods": len(methods),
            }
        
        return results
    
    def comprehensive_benchmark(
        self,
        image: Union[Image.Image, str, Path],
        scale_factor: float,
        methods: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive benchmark including quality and speed.
        
        Args:
            image: Input image
            scale_factor: Scale factor
            methods: List of methods to benchmark
            
        Returns:
            Dictionary with comprehensive benchmark results
        """
        quality_bench = self.benchmark_quality(image, scale_factor, "auto")
        speed_bench = self.benchmark_speed(image, scale_factor, methods)
        methods_bench = self.benchmark_methods(image, scale_factor, methods)
        
        return {
            "quality_benchmark": quality_bench,
            "speed_benchmark": speed_bench,
            "methods_benchmark": methods_bench,
            "recommendations": self._generate_benchmark_recommendations(
                quality_bench, speed_bench, methods_bench
            )
        }
    
    def _generate_benchmark_recommendations(
        self,
        quality_bench: Dict[str, Any],
        speed_bench: Dict[str, Any],
        methods_bench: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations from benchmark results."""
        recommendations = []
        
        if quality_bench.get("improvement", {}).get("overall", 0) < 0.1:
            recommendations.append("Consider using a different upscaling method for better quality")
        
        if speed_bench.get("_summary", {}).get("fastest_time", 0) > 5.0:
            recommendations.append("Processing is slow, consider optimizing or using faster methods")
        
        if methods_bench.get("_summary", {}):
            best_quality = methods_bench["_summary"].get("best_quality")
            if best_quality:
                recommendations.append(f"Best quality method: {best_quality}")
        
        return recommendations


