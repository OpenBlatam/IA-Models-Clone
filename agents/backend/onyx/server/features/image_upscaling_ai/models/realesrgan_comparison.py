"""
Real-ESRGAN Model Comparison
============================

Compare different Real-ESRGAN models on the same image.
"""

import logging
from typing import Dict, Any, List, Tuple, Optional
from PIL import Image
import numpy as np
import time

logger = logging.getLogger(__name__)

try:
    from .realesrgan_integration import REALESRGAN_AVAILABLE, RealESRGANWrapper
    from .quality_metrics import QualityMetrics
except ImportError:
    REALESRGAN_AVAILABLE = False
    RealESRGANWrapper = None
    QualityMetrics = None


class ModelComparison:
    """
    Compare different Real-ESRGAN models.
    
    Features:
    - Side-by-side comparison
    - Quality metrics comparison
    - Performance comparison
    - Best model recommendation
    """
    
    def __init__(self):
        """Initialize comparison tool."""
        self.quality_metrics = QualityMetrics() if QualityMetrics else None
    
    def compare_models(
        self,
        image: Image.Image,
        scale_factor: float,
        model_names: Optional[List[str]] = None,
        device: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compare multiple models on the same image.
        
        Args:
            image: Input image
            scale_factor: Scale factor
            model_names: List of model names to compare (all if None)
            device: Device to use
            
        Returns:
            Comparison results
        """
        if not REALESRGAN_AVAILABLE:
            raise ImportError("Real-ESRGAN not available")
        
        if model_names is None:
            model_names = [
                "RealESRGAN_x4plus",
                "RealESRGAN_x4plus_anime_6B",
                "RealESRNet_x4plus",
            ]
        
        results = {}
        
        for model_name in model_names:
            try:
                logger.info(f"Testing model: {model_name}")
                
                # Load model
                wrapper = RealESRGANWrapper(
                    model_name=model_name,
                    device=device
                )
                
                # Measure time
                start_time = time.time()
                upscaled = wrapper.upscale(image, scale_factor)
                elapsed_time = time.time() - start_time
                
                # Calculate quality metrics
                quality = None
                if self.quality_metrics:
                    try:
                        quality = self.quality_metrics.calculate_metrics(image, upscaled)
                    except Exception as e:
                        logger.warning(f"Error calculating quality metrics: {e}")
                
                results[model_name] = {
                    "upscaled_image": upscaled,
                    "elapsed_time": elapsed_time,
                    "quality_metrics": quality.dict() if quality else None,
                    "model_info": wrapper.get_model_info(),
                    "success": True,
                }
                
            except Exception as e:
                logger.error(f"Error testing model {model_name}: {e}")
                results[model_name] = {
                    "success": False,
                    "error": str(e),
                }
        
        # Find best model
        best_model = self._find_best_model(results)
        
        return {
            "results": results,
            "best_model": best_model,
            "comparison_summary": self._generate_summary(results),
        }
    
    def _find_best_model(self, results: Dict[str, Any]) -> Optional[str]:
        """Find best model based on quality and speed."""
        valid_results = {
            k: v for k, v in results.items()
            if v.get("success") and v.get("quality_metrics")
        }
        
        if not valid_results:
            return None
        
        # Score = quality_score / (time + 1) to balance quality and speed
        scores = {}
        for model_name, result in valid_results.items():
            quality = result["quality_metrics"]
            time_taken = result["elapsed_time"]
            
            # Weighted score (quality matters more)
            quality_score = quality.get("overall_score", 0.5)
            speed_score = 1.0 / (time_taken + 0.1)  # Avoid division by zero
            
            # Combined score (70% quality, 30% speed)
            combined_score = 0.7 * quality_score + 0.3 * speed_score
            
            scores[model_name] = combined_score
        
        if scores:
            best = max(scores.items(), key=lambda x: x[1])
            return best[0]
        
        return None
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comparison summary."""
        valid_results = {
            k: v for k, v in results.items()
            if v.get("success")
        }
        
        if not valid_results:
            return {"message": "No successful comparisons"}
        
        summary = {
            "models_tested": len(valid_results),
            "fastest": min(valid_results.items(), key=lambda x: x[1]["elapsed_time"])[0],
            "slowest": max(valid_results.items(), key=lambda x: x[1]["elapsed_time"])[0],
        }
        
        # Quality comparison
        quality_results = {
            k: v for k, v in valid_results.items()
            if v.get("quality_metrics")
        }
        
        if quality_results:
            best_quality = max(
                quality_results.items(),
                key=lambda x: x[1]["quality_metrics"].get("overall_score", 0)
            )
            summary["best_quality"] = best_quality[0]
            summary["quality_scores"] = {
                k: v["quality_metrics"].get("overall_score", 0)
                for k, v in quality_results.items()
            }
        
        return summary
    
    def create_comparison_grid(
        self,
        comparison_results: Dict[str, Any],
        labels: bool = True
    ) -> Image.Image:
        """
        Create side-by-side comparison grid.
        
        Args:
            comparison_results: Results from compare_models
            labels: Add model name labels
            
        Returns:
            Comparison grid image
        """
        results = comparison_results.get("results", {})
        valid_results = {
            k: v for k, v in results.items()
            if v.get("success") and v.get("upscaled_image")
        }
        
        if not valid_results:
            raise ValueError("No valid results to compare")
        
        images = []
        for model_name, result in valid_results.items():
            img = result["upscaled_image"]
            if labels:
                # Add label to image
                from PIL import ImageDraw, ImageFont
                draw = ImageDraw.Draw(img)
                try:
                    font = ImageFont.truetype("arial.ttf", 20)
                except:
                    font = ImageFont.load_default()
                draw.text((10, 10), model_name, fill=(255, 255, 255), font=font)
            images.append(img)
        
        # Create grid
        num_images = len(images)
        cols = min(3, num_images)
        rows = (num_images + cols - 1) // cols
        
        width = images[0].width
        height = images[0].height
        
        grid = Image.new("RGB", (width * cols, height * rows))
        
        for idx, img in enumerate(images):
            row = idx // cols
            col = idx % cols
            grid.paste(img, (col * width, row * height))
        
        return grid


