"""
Performance Mixin

Contains performance optimization and profiling functionality.
"""

import logging
import time
import cProfile
import pstats
from typing import Union, Dict, Any, Optional, Callable, List
from pathlib import Path
from PIL import Image
from io import StringIO

logger = logging.getLogger(__name__)


class PerformanceMixin:
    """
    Mixin providing performance optimization and profiling.
    
    This mixin contains:
    - Performance profiling
    - Bottleneck detection
    - Performance optimization
    - Resource usage tracking
    - Performance recommendations
    """
    
    def profile_operation(
        self,
        operation: Callable,
        *args,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Profile an operation for performance analysis.
        
        Args:
            operation: Function to profile
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Dictionary with profiling results
        """
        profiler = cProfile.Profile()
        start_time = time.time()
        
        profiler.enable()
        try:
            result = operation(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
        finally:
            profiler.disable()
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Get stats
        stats_stream = StringIO()
        stats = pstats.Stats(profiler, stream=stats_stream)
        stats.sort_stats('cumulative')
        stats.print_stats(20)  # Top 20 functions
        stats_output = stats_stream.getvalue()
        
        return {
            "success": success,
            "duration": duration,
            "result": result,
            "error": error,
            "stats": stats_output,
            "function_calls": stats.total_calls,
        }
    
    def profile_upscale(
        self,
        image: Union[Image.Image, str, Path],
        scale_factor: float,
        method: str = "lanczos"
    ) -> Dict[str, Any]:
        """
        Profile upscaling operation.
        
        Args:
            image: Input image
            scale_factor: Scale factor
            method: Upscaling method
            
        Returns:
            Dictionary with profiling results
        """
        def upscale_op():
            return self.upscale(image, scale_factor, method, return_metrics=False)
        
        return self.profile_operation(upscale_op)
    
    def get_performance_bottlenecks(
        self,
        profile_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Identify performance bottlenecks from profile results.
        
        Args:
            profile_results: Results from profile_operation
            
        Returns:
            List of bottleneck information
        """
        bottlenecks = []
        
        if "stats" in profile_results:
            stats_lines = profile_results["stats"].split('\n')
            for line in stats_lines[5:25]:  # Skip header, get top functions
                if line.strip() and not line.startswith('ncalls'):
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            cumtime = float(parts[3])
                            if cumtime > 0.1:  # Functions taking > 0.1s
                                bottlenecks.append({
                                    "function": ' '.join(parts[4:]) if len(parts) > 4 else line,
                                    "cumulative_time": cumtime,
                                    "percentage": (cumtime / profile_results["duration"]) * 100 if profile_results["duration"] > 0 else 0,
                                })
                        except (ValueError, IndexError):
                            continue
        
        return sorted(bottlenecks, key=lambda x: x["cumulative_time"], reverse=True)
    
    def optimize_performance(
        self,
        image: Union[Image.Image, str, Path],
        scale_factor: float,
        target_time: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Optimize performance for upscaling operation.
        
        Args:
            image: Input image
            scale_factor: Scale factor
            target_time: Target processing time in seconds
            
        Returns:
            Dictionary with optimization results
        """
        # Profile current operation
        profile = self.profile_upscale(image, scale_factor, "auto")
        
        # Find bottlenecks
        bottlenecks = self.get_performance_bottlenecks(profile)
        
        # Recommendations
        recommendations = []
        
        if profile["duration"] > 5.0:
            recommendations.append("Consider using faster upscaling method")
            recommendations.append("Enable caching for repeated operations")
        
        if bottlenecks:
            top_bottleneck = bottlenecks[0]
            if top_bottleneck["percentage"] > 50:
                recommendations.append(f"Optimize {top_bottleneck['function']} (takes {top_bottleneck['percentage']:.1f}% of time)")
        
        # Try faster methods if target time specified
        if target_time and profile["duration"] > target_time:
            fast_methods = ["lanczos", "bicubic", "opencv"]
            for method in fast_methods:
                test_profile = self.profile_upscale(image, scale_factor, method)
                if test_profile["duration"] <= target_time:
                    recommendations.append(f"Use method '{method}' for target time (takes {test_profile['duration']:.2f}s)")
                    break
        
        return {
            "current_duration": profile["duration"],
            "target_time": target_time,
            "bottlenecks": bottlenecks,
            "recommendations": recommendations,
            "optimized": len(recommendations) > 0,
        }
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """
        Get current resource usage.
        
        Returns:
            Dictionary with resource usage information
        """
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # CPU
        cpu_percent = process.cpu_percent(interval=0.1)
        
        # Memory
        memory_info = process.memory_info()
        
        # Threads
        num_threads = process.num_threads()
        
        # Cache info
        cache_info = {}
        if hasattr(self, 'cache') and self.cache:
            cache_info = {
                "size": len(self.cache.cache) if hasattr(self.cache, 'cache') else 0,
                "max_size": self.cache.max_size if hasattr(self.cache, 'max_size') else 0,
            }
        
        return {
            "cpu_percent": cpu_percent,
            "memory_rss_mb": memory_info.rss / (1024 * 1024),
            "memory_vms_mb": memory_info.vms / (1024 * 1024),
            "num_threads": num_threads,
            "cache": cache_info,
        }

