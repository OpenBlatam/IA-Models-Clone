"""
Color Analyzer for Color Grading AI
===================================

Advanced color analysis for videos and images.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Tuple
from pathlib import Path
import asyncio
from PIL import Image

logger = logging.getLogger(__name__)


class ColorAnalyzer:
    """
    Advanced color analysis for color grading.
    
    Features:
    - Histogram analysis
    - Color space analysis
    - Scene detection
    - Keyframe extraction
    - Color temperature analysis
    - Exposure analysis
    """
    
    def __init__(self, histogram_bins: int = 256, color_space: str = "RGB"):
        """
        Initialize color analyzer.
        
        Args:
            histogram_bins: Number of bins for histograms
            color_space: Color space for analysis (RGB, HSV, LAB)
        """
        self.histogram_bins = histogram_bins
        self.color_space = color_space
    
    async def analyze_image(
        self,
        image_path: str
    ) -> Dict[str, Any]:
        """
        Analyze color properties of an image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with color analysis results
        """
        def _load():
            return Image.open(image_path)
        
        image = await asyncio.to_thread(_load)
        
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        img_array = np.array(image)
        
        # Calculate histograms
        histograms = self._calculate_histograms(img_array)
        
        # Calculate statistics
        stats = self._calculate_statistics(img_array)
        
        # Color temperature
        color_temp = self._estimate_color_temperature(img_array)
        
        # Exposure analysis
        exposure = self._analyze_exposure(img_array)
        
        # Color distribution
        color_dist = self._analyze_color_distribution(img_array)
        
        return {
            "histograms": histograms,
            "statistics": stats,
            "color_temperature": color_temp,
            "exposure": exposure,
            "color_distribution": color_dist,
            "width": image.width,
            "height": image.height,
        }
    
    async def analyze_video_frames(
        self,
        frame_paths: List[str],
        scene_threshold: float = 0.3
    ) -> Dict[str, Any]:
        """
        Analyze multiple video frames for scene detection.
        
        Args:
            frame_paths: List of frame file paths
            scene_threshold: Threshold for scene change detection
            
        Returns:
            Dictionary with scene analysis results
        """
        analyses = []
        scenes = []
        current_scene = [0]
        
        for i, frame_path in enumerate(frame_paths):
            analysis = await self.analyze_image(frame_path)
            analyses.append(analysis)
            
            # Detect scene changes
            if i > 0:
                prev_analysis = analyses[i - 1]
                similarity = self._compare_analyses(prev_analysis, analysis)
                
                if similarity < (1 - scene_threshold):
                    # Scene change detected
                    scenes.append({
                        "start_frame": current_scene[0],
                        "end_frame": i - 1,
                        "frames": current_scene
                    })
                    current_scene = [i]
                else:
                    current_scene.append(i)
        
        # Add last scene
        if current_scene:
            scenes.append({
                "start_frame": current_scene[0],
                "end_frame": len(frame_paths) - 1,
                "frames": current_scene
            })
        
        return {
            "frames_analyzed": len(analyses),
            "scenes": scenes,
            "frame_analyses": analyses,
        }
    
    def _calculate_histograms(self, img_array: np.ndarray) -> Dict[str, List[int]]:
        """Calculate color histograms."""
        histograms = {}
        
        for i, channel in enumerate(["R", "G", "B"]):
            hist, _ = np.histogram(
                img_array[:, :, i],
                bins=self.histogram_bins,
                range=(0, 256)
            )
            histograms[channel] = hist.tolist()
        
        return histograms
    
    def _calculate_statistics(self, img_array: np.ndarray) -> Dict[str, Any]:
        """Calculate color statistics."""
        stats = {}
        
        for i, channel in enumerate(["R", "G", "B"]):
            channel_data = img_array[:, :, i]
            stats[channel] = {
                "mean": float(np.mean(channel_data)),
                "std": float(np.std(channel_data)),
                "min": int(np.min(channel_data)),
                "max": int(np.max(channel_data)),
                "median": float(np.median(channel_data)),
            }
        
        # Overall statistics
        stats["overall"] = {
            "mean": float(np.mean(img_array)),
            "std": float(np.std(img_array)),
            "brightness": float(np.mean(img_array)),
        }
        
        return stats
    
    def _estimate_color_temperature(self, img_array: np.ndarray) -> Dict[str, float]:
        """Estimate color temperature."""
        # Simplified color temperature estimation
        # Based on RGB ratios
        
        r_mean = np.mean(img_array[:, :, 0])
        g_mean = np.mean(img_array[:, :, 1])
        b_mean = np.mean(img_array[:, :, 2])
        
        # Calculate color temperature estimate (simplified)
        if r_mean > 0 and b_mean > 0:
            ratio = r_mean / b_mean
            # Rough estimation (not accurate, but gives an idea)
            if ratio > 1.2:
                temp_k = 5500 + (ratio - 1.2) * 2000  # Warm
            elif ratio < 0.8:
                temp_k = 5500 - (0.8 - ratio) * 2000  # Cool
            else:
                temp_k = 5500  # Neutral
        else:
            temp_k = 5500
        
        return {
            "temperature_k": float(temp_k),
            "tint": "warm" if temp_k > 5500 else "cool" if temp_k < 5500 else "neutral"
        }
    
    def _analyze_exposure(self, img_array: np.ndarray) -> Dict[str, Any]:
        """Analyze exposure levels."""
        brightness = np.mean(img_array)
        
        # Classify exposure
        if brightness < 50:
            exposure_level = "underexposed"
        elif brightness > 200:
            exposure_level = "overexposed"
        else:
            exposure_level = "normal"
        
        return {
            "brightness": float(brightness),
            "level": exposure_level,
            "histogram_peaks": self._find_histogram_peaks(img_array),
        }
    
    def _find_histogram_peaks(self, img_array: np.ndarray) -> List[int]:
        """Find peaks in brightness histogram."""
        brightness = np.mean(img_array, axis=2)
        hist, bins = np.histogram(brightness, bins=256, range=(0, 256))
        
        # Find local maxima (simplified)
        peaks = []
        for i in range(1, len(hist) - 1):
            if hist[i] > hist[i - 1] and hist[i] > hist[i + 1] and hist[i] > np.max(hist) * 0.1:
                peaks.append(int(bins[i]))
        
        return peaks[:5]  # Return top 5 peaks
    
    def _analyze_color_distribution(self, img_array: np.ndarray) -> Dict[str, Any]:
        """Analyze color distribution."""
        # Convert to HSV for better color analysis
        hsv = self._rgb_to_hsv(img_array)
        
        # Analyze hue distribution
        hue = hsv[:, :, 0] * 360
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        
        # Dominant hues
        hue_hist, _ = np.histogram(hue, bins=36, range=(0, 360))
        dominant_hues = np.argsort(hue_hist)[-5:][::-1].tolist()
        
        return {
            "mean_saturation": float(np.mean(saturation)),
            "mean_value": float(np.mean(value)),
            "dominant_hues": [int(h * 10) for h in dominant_hues],
            "color_variance": float(np.var(img_array)),
        }
    
    def _compare_analyses(
        self,
        analysis1: Dict[str, Any],
        analysis2: Dict[str, Any]
    ) -> float:
        """Compare two analyses to detect scene changes."""
        # Calculate similarity based on statistics
        stats1 = analysis1.get("statistics", {}).get("overall", {})
        stats2 = analysis2.get("statistics", {}).get("overall", {})
        
        mean1 = stats1.get("mean", 0)
        mean2 = stats2.get("mean", 0)
        
        # Simple similarity metric (can be improved)
        similarity = 1.0 - abs(mean1 - mean2) / 255.0
        
        return max(0.0, min(1.0, similarity))
    
    def _rgb_to_hsv(self, rgb: np.ndarray) -> np.ndarray:
        """Convert RGB to HSV."""
        rgb_norm = rgb / 255.0
        hsv = np.zeros_like(rgb_norm)
        
        max_val = rgb_norm.max(axis=2)
        min_val = rgb_norm.min(axis=2)
        delta = max_val - min_val
        
        # Hue
        hsv[:, :, 0] = np.where(
            delta == 0, 0,
            np.where(
                max_val == rgb_norm[:, :, 0],
                ((rgb_norm[:, :, 1] - rgb_norm[:, :, 2]) / delta) % 6,
                np.where(
                    max_val == rgb_norm[:, :, 1],
                    (rgb_norm[:, :, 2] - rgb_norm[:, :, 0]) / delta + 2,
                    (rgb_norm[:, :, 0] - rgb_norm[:, :, 1]) / delta + 4
                )
            )
        ) / 6.0
        
        # Saturation
        hsv[:, :, 1] = np.where(max_val == 0, 0, delta / max_val)
        
        # Value
        hsv[:, :, 2] = max_val
        
        return hsv




