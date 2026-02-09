"""
Advanced Batch Processing Methods
=================================

Optimized batch processing and concurrent operations.
"""

import logging
import time
from typing import Tuple, Optional, Dict, Any, List, Union, Callable
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

from .helpers import (
    UpscalingMetrics,
    BatchProcessingUtils,
)

logger = logging.getLogger(__name__)


class BatchProcessingMethods:
    """Advanced batch processing methods."""
    
    def __init__(self, base_upscaler):
        """Initialize with base upscaler."""
        self.base_upscaler = base_upscaler
    
    def batch_upscale_optimized(
        self,
        images: List[Union[Image.Image, str, Path]],
        scale_factor: float,
        method: str = "lanczos",
        batch_size: int = 4,
        max_workers: int = 4,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        return_metrics: bool = False,
        **kwargs
    ) -> Union[List[Image.Image], Tuple[List[Image.Image], List[UpscalingMetrics]]]:
        """Optimized batch upscaling with error recovery."""
        start_time = time.time()
        results = []
        metrics_list = []
        failed = []
        
        def process_image(img, idx):
            try:
                if return_metrics:
                    result, metrics = self.base_upscaler.upscale(
                        img, scale_factor, method, return_metrics=True, **kwargs
                    )
                    return (idx, result, metrics, None)
                else:
                    result = self.base_upscaler.upscale(img, scale_factor, method, return_metrics=False, **kwargs)
                    return (idx, result, None, None)
            except Exception as e:
                logger.error(f"Failed to upscale image {idx}: {e}")
                return (idx, None, None, str(e))
        
        # Process in batches
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i, img in enumerate(images):
                future = executor.submit(process_image, img, i)
                futures.append(future)
            
            # Collect results
            completed = 0
            for future in as_completed(futures):
                idx, result, metrics, error = future.result()
                
                if error:
                    failed.append((idx, error))
                    if return_metrics:
                        metrics_list.append(None)
                    results.append(None)
                else:
                    results.append((idx, result))
                    if return_metrics:
                        metrics_list.append((idx, metrics))
                
                completed += 1
                if progress_callback:
                    progress_callback(completed, len(images))
        
        # Sort results by original index
        results.sort(key=lambda x: x[0] if x else -1)
        results = [r[1] if r else None for r in results]
        
        if return_metrics:
            metrics_list.sort(key=lambda x: x[0] if x else -1)
            metrics_list = [m[1] if m else None for m in metrics_list]
            return results, metrics_list
        
        processing_time = time.time() - start_time
        logger.info(f"Batch upscaled {len(images)} images in {processing_time:.2f}s ({len(failed)} failed)")
        
        return results
    
    def batch_upscale_with_adaptive_methods(
        self,
        images: List[Union[Image.Image, str, Path]],
        scale_factor: float,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        return_metrics: bool = False
    ) -> Union[List[Image.Image], Tuple[List[Image.Image], List[UpscalingMetrics]]]:
        """Batch upscale with adaptive method selection per image."""
        results = []
        metrics_list = []
        
        for i, img in enumerate(images):
            try:
                # Analyze image to select best method
                analysis = self.base_upscaler.analysis_methods.analyze_image_characteristics(img)
                
                # Select method based on analysis
                if analysis.get("is_anime", False):
                    method = "waifu2x_like"
                elif analysis.get("is_photo", False):
                    method = "real_esrgan_like"
                else:
                    method = "lanczos"
                
                if return_metrics:
                    result, metrics = self.base_upscaler.upscale(img, scale_factor, method, return_metrics=True)
                    results.append(result)
                    metrics_list.append(metrics)
                else:
                    result = self.base_upscaler.upscale(img, scale_factor, method, return_metrics=False)
                    results.append(result)
                
                if progress_callback:
                    progress_callback(i + 1, len(images))
                    
            except Exception as e:
                logger.error(f"Failed to upscale image {i}: {e}")
                results.append(None)
                if return_metrics:
                    metrics_list.append(None)
        
        if return_metrics:
            return results, metrics_list
        return results
    
    def batch_upscale_with_quality_check(
        self,
        images: List[Union[Image.Image, str, Path]],
        scale_factor: float,
        method: str = "lanczos",
        min_quality_threshold: float = 0.7,
        retry_with_different_method: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        return_metrics: bool = False
    ) -> Union[List[Image.Image], Tuple[List[Image.Image], List[UpscalingMetrics]]]:
        """Batch upscale with quality checking and retry."""
        results = []
        metrics_list = []
        alternative_methods = ["bicubic", "opencv", "multi_scale"]
        
        for i, img in enumerate(images):
            try:
                result, metrics = self.base_upscaler.upscale(img, scale_factor, method, return_metrics=True)
                
                # Check quality
                if metrics.quality_score < min_quality_threshold and retry_with_different_method:
                    # Try alternative methods
                    for alt_method in alternative_methods:
                        try:
                            alt_result, alt_metrics = self.base_upscaler.upscale(
                                img, scale_factor, alt_method, return_metrics=True
                            )
                            if alt_metrics.quality_score > metrics.quality_score:
                                result = alt_result
                                metrics = alt_metrics
                                break
                        except Exception:
                            continue
                
                results.append(result)
                if return_metrics:
                    metrics_list.append(metrics)
                
                if progress_callback:
                    progress_callback(i + 1, len(images))
                    
            except Exception as e:
                logger.error(f"Failed to upscale image {i}: {e}")
                results.append(None)
                if return_metrics:
                    metrics_list.append(None)
        
        if return_metrics:
            return results, metrics_list
        return results


