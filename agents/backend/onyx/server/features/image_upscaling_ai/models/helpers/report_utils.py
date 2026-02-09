"""
Report Utilities
================

Utilities for generating and exporting analysis reports.
"""

import logging
import time
import json
from typing import Dict, Any, Union, Optional
from pathlib import Path
from PIL import Image

from .image_analysis_utils import ImageAnalysisUtils
from .method_comparison_utils import MethodComparisonUtils

logger = logging.getLogger(__name__)


class ReportUtils:
    """Utilities for generating and exporting reports."""
    
    @staticmethod
    def generate_complete_analysis_report(
        analysis_func: callable,
        recommendations_func: callable,
        advanced_recommendations_func: callable,
        strategy_func: callable,
        compare_func: callable,
        comprehensive_compare_func: callable,
        image: Union[Image.Image, str, Path],
        scale_factor: float,
        include_comparisons: bool = True,
        include_recommendations: bool = True
    ) -> Dict[str, Any]:
        """
        Generate a complete analysis report.
        
        Args:
            analysis_func: Function to analyze image characteristics
            recommendations_func: Function to get processing recommendations
            advanced_recommendations_func: Function to get advanced recommendations
            strategy_func: Function to get optimal strategy
            compare_func: Function to compare methods
            comprehensive_compare_func: Function for comprehensive comparison
            image: Input image
            scale_factor: Scale factor
            include_comparisons: Include method comparisons
            include_recommendations: Include recommendations
            
        Returns:
            Complete analysis report dictionary
        """
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        else:
            pil_image = image.convert("RGB")
        
        report = {
            "image_info": {
                "path": str(image) if isinstance(image, (str, Path)) else "PIL Image",
                "original_size": pil_image.size,
                "scale_factor": scale_factor,
                "target_size": (
                    int(pil_image.size[0] * scale_factor),
                    int(pil_image.size[1] * scale_factor)
                ),
            },
            "analysis": analysis_func(pil_image),
            "generated_at": time.time(),
        }
        
        if include_recommendations:
            report["recommendations"] = recommendations_func(pil_image, scale_factor)
            report["advanced_recommendations"] = advanced_recommendations_func(pil_image, scale_factor)
            report["optimal_strategy"] = strategy_func(pil_image, scale_factor)
        
        if include_comparisons:
            report["method_comparison"] = compare_func(
                pil_image,
                scale_factor,
                ["lanczos", "bicubic", "opencv", "multi_scale", "esrgan_like", "real_esrgan_like"]
            )
            report["comprehensive_comparison"] = comprehensive_compare_func(
                pil_image,
                scale_factor,
                ["lanczos", "bicubic", "opencv", "multi_scale", "esrgan_like", "waifu2x_like", "real_esrgan_like"]
            )
        
        return report
    
    @staticmethod
    def save_report(report: Dict[str, Any], output_path: str) -> None:
        """
        Save report to file.
        
        Args:
            report: Report dictionary
            output_path: Path to save report
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Report saved to {output_path}")


