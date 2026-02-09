"""
Advanced Benchmarking and Performance Methods
==============================================

Benchmarking and performance analysis methods.
"""

import logging
import time
from typing import Tuple, Optional, Dict, Any, List, Union
from pathlib import Path
from PIL import Image

from .helpers import (
    UpscalingMetrics,
    MethodComparisonUtils,
)

logger = logging.getLogger(__name__)


class BenchmarkMethods:
    """Benchmarking and performance methods."""
    
    def __init__(self, base_upscaler):
        """Initialize with base upscaler."""
        self.base_upscaler = base_upscaler
    
    def benchmark_all_methods(
        self,
        image: Union[Image.Image, str, Path],
        scale_factor: float,
        methods: Optional[List[str]] = None,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Benchmark all methods on an image."""
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        else:
            pil_image = image.convert("RGB")
        
        if methods is None:
            methods = [
                "lanczos", "bicubic", "opencv", "multi_scale",
                "esrgan_like", "waifu2x_like", "real_esrgan_like"
            ]
        
        benchmark_results = {}
        
        for method in methods:
            try:
                start_time = time.time()
                result, metrics = self.base_upscaler.upscale(
                    pil_image, scale_factor, method, return_metrics=True
                )
                elapsed = time.time() - start_time
                
                quality_metrics = self.base_upscaler.calculate_quality_metrics(result)
                
                benchmark_results[method] = {
                    "success": True,
                    "processing_time": elapsed,
                    "quality_score": metrics.quality_score,
                    "sharpness_score": metrics.sharpness_score,
                    "artifact_score": metrics.artifact_score,
                    "overall_quality": quality_metrics.overall_quality,
                    "size": result.size,
                }
            except Exception as e:
                benchmark_results[method] = {
                    "success": False,
                    "error": str(e),
                }
        
        # Generate recommendations
        successful = {k: v for k, v in benchmark_results.items() if v.get("success")}
        
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
        
        report = {
            "image_info": {
                "path": str(image) if isinstance(image, (str, Path)) else "PIL Image",
                "scale_factor": scale_factor,
            },
            "methods_tested": methods,
            "results": benchmark_results,
            "recommendations": recommendations,
            "summary": {
                "total_methods": len(methods),
                "successful": len(successful),
                "failed": len(methods) - len(successful),
            },
            "generated_at": time.time(),
        }
        
        if output_path:
            import json
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Benchmark report saved to {output_path}")
        
        return report
    
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
    
    def get_performance_benchmark(
        self,
        test_images: List[Union[Image.Image, str, Path]],
        scale_factor: float,
        methods: List[str] = None,
        output_path: str = None
    ) -> Dict[str, Any]:
        """Get comprehensive performance benchmark."""
        if methods is None:
            methods = [
                "lanczos", "bicubic", "opencv", "multi_scale",
                "esrgan_like", "waifu2x_like", "real_esrgan_like"
            ]
        
        benchmark = {
            "test_config": {
                "num_images": len(test_images),
                "scale_factor": scale_factor,
                "methods": methods,
            },
            "results": {},
            "summary": {},
            "generated_at": time.time(),
        }
        
        # Benchmark each method
        for method in methods:
            method_results = {
                "processing_times": [],
                "quality_scores": [],
                "sharpness_scores": [],
                "success_count": 0,
                "failure_count": 0,
            }
            
            for img in test_images:
                try:
                    start_time = time.time()
                    result = self.base_upscaler.upscale(img, scale_factor, method, return_metrics=False)
                    processing_time = time.time() - start_time
                    
                    quality = self.base_upscaler.calculate_quality_metrics(result)
                    
                    method_results["processing_times"].append(processing_time)
                    method_results["quality_scores"].append(quality.overall_quality)
                    method_results["sharpness_scores"].append(quality.sharpness)
                    method_results["success_count"] += 1
                except Exception as e:
                    method_results["failure_count"] += 1
                    logger.warning(f"Method {method} failed on image: {e}")
            
            # Calculate statistics
            if method_results["processing_times"]:
                method_results["avg_time"] = sum(method_results["processing_times"]) / len(method_results["processing_times"])
                method_results["avg_quality"] = sum(method_results["quality_scores"]) / len(method_results["quality_scores"])
                method_results["avg_sharpness"] = sum(method_results["sharpness_scores"]) / len(method_results["sharpness_scores"])
                method_results["success_rate"] = method_results["success_count"] / len(test_images)
            else:
                method_results["avg_time"] = 0.0
                method_results["avg_quality"] = 0.0
                method_results["avg_sharpness"] = 0.0
                method_results["success_rate"] = 0.0
            
            benchmark["results"][method] = method_results
        
        # Summary statistics
        if benchmark["results"]:
            best_quality_method = max(
                benchmark["results"].items(),
                key=lambda x: x[1].get("avg_quality", 0.0)
            )[0]
            
            fastest_method = min(
                benchmark["results"].items(),
                key=lambda x: x[1].get("avg_time", float('inf'))
            )[0]
            
            benchmark["summary"] = {
                "best_quality_method": best_quality_method,
                "best_quality_score": benchmark["results"][best_quality_method].get("avg_quality", 0.0),
                "fastest_method": fastest_method,
                "fastest_time": benchmark["results"][fastest_method].get("avg_time", 0.0),
                "total_methods": len(methods),
                "total_images": len(test_images),
            }
        
        if output_path:
            import json
            with open(output_path, 'w') as f:
                json.dump(benchmark, f, indent=2)
            logger.info(f"Performance benchmark saved to {output_path}")
        
        return benchmark
    
    def profile_upscale(
        self,
        image: Union[Image.Image, str, Path],
        scale_factor: float,
        method: str = "lanczos",
        iterations: int = 5
    ) -> Dict[str, Any]:
        """Profile upscaling performance."""
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        else:
            pil_image = image.convert("RGB")
        
        times = []
        quality_scores = []
        
        for _ in range(iterations):
            start_time = time.time()
            result = self.base_upscaler.upscale(pil_image, scale_factor, method, return_metrics=False)
            elapsed = time.time() - start_time
            
            quality = self.base_upscaler.calculate_quality_metrics(result)
            
            times.append(elapsed)
            quality_scores.append(quality.overall_quality)
        
        import numpy as np
        
        return {
            "method": method,
            "iterations": iterations,
            "avg_time": sum(times) / len(times),
            "min_time": min(times),
            "max_time": max(times),
            "std_time": float(np.std(times)),
            "avg_quality": sum(quality_scores) / len(quality_scores) if quality_scores else None,
            "times": times,
            "quality_scores": quality_scores,
        }


