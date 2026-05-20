"""
Color Matcher for Color Grading AI
===================================

Matches color grading from reference images or videos.
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path
import asyncio
from PIL import Image

from .color_analyzer import ColorAnalyzer
from .image_processor import ImageProcessor

logger = logging.getLogger(__name__)


class ColorMatcher:
    """
    Matches color grading from reference media.
    
    Features:
    - Match color from reference image
    - Match color from reference video
    - Extract color palette
    - Apply color transfer
    - Generate color grading parameters
    """
    
    def __init__(self):
        """Initialize color matcher."""
        self.color_analyzer = ColorAnalyzer()
        self.image_processor = ImageProcessor()
    
    async def match_from_reference_image(
        self,
        source_path: str,
        reference_path: str
    ) -> Dict[str, Any]:
        """
        Match color grading from reference image.
        
        Args:
            source_path: Path to source image/video frame
            reference_path: Path to reference image
            
        Returns:
            Dictionary with color grading parameters
        """
        # Analyze both images
        source_analysis = await self.color_analyzer.analyze_image(source_path)
        reference_analysis = await self.color_analyzer.analyze_image(reference_path)
        
        # Calculate color transfer parameters
        params = self._calculate_color_transfer(
            source_analysis,
            reference_analysis
        )
        
        return {
            "color_params": params,
            "source_analysis": source_analysis,
            "reference_analysis": reference_analysis,
        }
    
    async def match_from_reference_video(
        self,
        source_path: str,
        reference_video_path: str,
        reference_frame_time: float = 0.0
    ) -> Dict[str, Any]:
        """
        Match color grading from reference video frame.
        
        Args:
            source_path: Path to source image/video frame
            reference_video_path: Path to reference video
            reference_frame_time: Time in seconds for reference frame
            
        Returns:
            Dictionary with color grading parameters
        """
        # Extract reference frame (simplified - would use video processor)
        # For now, assume we have a frame extractor
        # In full implementation, would use VideoProcessor
        
        # This is a placeholder - full implementation would extract frame
        logger.warning("Reference video matching requires frame extraction")
        
        # For now, return basic parameters
        return {
            "color_params": {
                "brightness": 0.0,
                "contrast": 1.0,
                "saturation": 1.0,
            },
            "method": "reference_video",
        }
    
    async def extract_color_palette(
        self,
        image_path: str,
        num_colors: int = 8
    ) -> Dict[str, Any]:
        """
        Extract color palette from image.
        
        Args:
            image_path: Path to image
            num_colors: Number of colors in palette
            
        Returns:
            Dictionary with color palette
        """
        colors = await self.image_processor.extract_dominant_colors(
            image_path,
            num_colors
        )
        
        return {
            "colors": colors,
            "count": len(colors),
        }
    
    def _calculate_color_transfer(
        self,
        source_analysis: Dict[str, Any],
        reference_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate color transfer parameters.
        
        Args:
            source_analysis: Source image analysis
            reference_analysis: Reference image analysis
            
        Returns:
            Color grading parameters
        """
        source_stats = source_analysis.get("statistics", {})
        reference_stats = reference_analysis.get("statistics", {})
        
        # Calculate adjustments for each channel
        params = {}
        
        for channel in ["R", "G", "B"]:
            source_mean = source_stats.get(channel, {}).get("mean", 128)
            reference_mean = reference_stats.get(channel, {}).get("mean", 128)
            
            # Calculate offset
            offset = reference_mean - source_mean
            params[f"{channel.lower()}_offset"] = offset / 255.0
        
        # Overall brightness adjustment
        source_brightness = source_stats.get("overall", {}).get("mean", 128)
        reference_brightness = reference_stats.get("overall", {}).get("mean", 128)
        params["brightness"] = (reference_brightness - source_brightness) / 255.0
        
        # Contrast adjustment
        source_std = source_stats.get("overall", {}).get("std", 50)
        reference_std = reference_stats.get("overall", {}).get("std", 50)
        
        if source_std > 0:
            params["contrast"] = reference_std / source_std
        else:
            params["contrast"] = 1.0
        
        # Saturation adjustment
        source_sat = source_analysis.get("color_distribution", {}).get("mean_saturation", 0.5)
        reference_sat = reference_analysis.get("color_distribution", {}).get("mean_saturation", 0.5)
        
        if source_sat > 0:
            params["saturation"] = reference_sat / source_sat
        else:
            params["saturation"] = 1.0
        
        # Color balance
        params["color_balance"] = {
            "r": params.get("r_offset", 0.0),
            "g": params.get("g_offset", 0.0),
            "b": params.get("b_offset", 0.0),
        }
        
        return params
    
    async def generate_grading_from_description(
        self,
        description: str,
        base_analysis: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate color grading parameters from text description.
        
        Args:
            description: Text description of desired look
            base_analysis: Optional base image analysis
            
        Returns:
            Color grading parameters
        """
        # This would use LLM to interpret description
        # For now, return basic parameters based on keywords
        
        params = {
            "brightness": 0.0,
            "contrast": 1.0,
            "saturation": 1.0,
            "color_balance": {"r": 0.0, "g": 0.0, "b": 0.0},
        }
        
        description_lower = description.lower()
        
        # Simple keyword matching (would be replaced with LLM)
        if "warm" in description_lower:
            params["color_balance"]["r"] = 0.1
            params["color_balance"]["b"] = -0.1
        elif "cool" in description_lower:
            params["color_balance"]["r"] = -0.1
            params["color_balance"]["b"] = 0.1
        
        if "bright" in description_lower or "brighten" in description_lower:
            params["brightness"] = 0.2
        elif "dark" in description_lower or "darken" in description_lower:
            params["brightness"] = -0.2
        
        if "vivid" in description_lower or "saturated" in description_lower:
            params["saturation"] = 1.3
        elif "desaturated" in description_lower or "muted" in description_lower:
            params["saturation"] = 0.7
        
        if "high contrast" in description_lower:
            params["contrast"] = 1.3
        elif "low contrast" in description_lower or "flat" in description_lower:
            params["contrast"] = 0.8
        
        return {
            "color_params": params,
            "description": description,
        }




