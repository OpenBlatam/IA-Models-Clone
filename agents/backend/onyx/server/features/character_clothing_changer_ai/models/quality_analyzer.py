"""
Quality Analyzer for Flux2 Clothing Changer
============================================

Advanced quality analysis for generated images.
"""

import torch
from typing import Dict, Any, Optional, Tuple
from PIL import Image
import numpy as np
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class QualityScore:
    """Quality score breakdown."""
    overall: float
    realism: float
    consistency: float
    detail: float
    color_accuracy: float
    composition: float
    artifacts: float  # Lower is better (0 = no artifacts)


class QualityAnalyzer:
    """Advanced quality analyzer for generated images."""
    
    def __init__(self):
        """Initialize quality analyzer."""
        pass
    
    def analyze(
        self,
        original: Image.Image,
        generated: Image.Image,
        mask: Optional[Image.Image] = None,
    ) -> QualityScore:
        """
        Analyze quality of generated image.
        
        Args:
            original: Original image
            generated: Generated image
            mask: Optional mask for region of interest
            
        Returns:
            QualityScore object
        """
        # Convert to RGB if needed
        if original.mode != "RGB":
            original = original.convert("RGB")
        if generated.mode != "RGB":
            generated = generated.convert("RGB")
        
        orig_array = np.array(original)
        gen_array = np.array(generated)
        
        # Ensure same size
        if orig_array.shape != gen_array.shape:
            generated = generated.resize(original.size, Image.Resampling.LANCZOS)
            gen_array = np.array(generated)
        
        # Calculate individual scores
        realism = self._calculate_realism(orig_array, gen_array, mask)
        consistency = self._calculate_consistency(orig_array, gen_array, mask)
        detail = self._calculate_detail(gen_array, mask)
        color_accuracy = self._calculate_color_accuracy(orig_array, gen_array, mask)
        composition = self._calculate_composition(gen_array)
        artifacts = self._detect_artifacts(gen_array, mask)
        
        # Calculate overall score (weighted average)
        overall = (
            realism * 0.25 +
            consistency * 0.25 +
            detail * 0.15 +
            color_accuracy * 0.15 +
            composition * 0.10 +
            (1.0 - artifacts) * 0.10
        )
        
        return QualityScore(
            overall=overall,
            realism=realism,
            consistency=consistency,
            detail=detail,
            color_accuracy=color_accuracy,
            composition=composition,
            artifacts=artifacts,
        )
    
    def _calculate_realism(
        self,
        original: np.ndarray,
        generated: np.ndarray,
        mask: Optional[Image.Image],
    ) -> float:
        """Calculate realism score."""
        try:
            from skimage.metrics import structural_similarity as ssim
            
            # Convert to grayscale for SSIM
            orig_gray = np.mean(original, axis=2)
            gen_gray = np.mean(generated, axis=2)
            
            # Apply mask if provided
            if mask is not None:
                mask_array = np.array(mask.convert("L")) / 255.0
                if mask_array.shape != orig_gray.shape:
                    mask_array = np.array(
                        Image.fromarray(mask_array).resize(
                            (orig_gray.shape[1], orig_gray.shape[0])
                        )
                    )
                orig_gray = orig_gray * (1 - mask_array)
                gen_gray = gen_gray * (1 - mask_array)
            
            # Calculate SSIM
            score = ssim(orig_gray, gen_gray, data_range=255)
            return max(0.0, min(1.0, score))
            
        except ImportError:
            # Fallback to simple MSE-based score
            mse = np.mean((original - generated) ** 2)
            score = 1.0 / (1.0 + mse / 10000.0)
            return max(0.0, min(1.0, score))
    
    def _calculate_consistency(
        self,
        original: np.ndarray,
        generated: np.ndarray,
        mask: Optional[Image.Image],
    ) -> float:
        """Calculate consistency score (how well it blends)."""
        # Calculate edge consistency
        try:
            import cv2
            
            orig_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
            gen_gray = cv2.cvtColor(generated, cv2.COLOR_RGB2GRAY)
            
            # Calculate edges
            orig_edges = cv2.Canny(orig_gray, 50, 150)
            gen_edges = cv2.Canny(gen_gray, 50, 150)
            
            # Edge overlap
            edge_overlap = np.sum((orig_edges > 0) & (gen_edges > 0)) / max(
                np.sum(orig_edges > 0), 1
            )
            
            return max(0.0, min(1.0, edge_overlap))
            
        except ImportError:
            # Fallback
            diff = np.abs(original.astype(float) - generated.astype(float))
            consistency = 1.0 - np.mean(diff) / 255.0
            return max(0.0, min(1.0, consistency))
    
    def _calculate_detail(self, generated: np.ndarray, mask: Optional[Image.Image]) -> float:
        """Calculate detail/sharpness score."""
        try:
            import cv2
            
            gray = cv2.cvtColor(generated, cv2.COLOR_RGB2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Normalize (typical good images have var > 100)
            detail_score = min(1.0, laplacian_var / 200.0)
            return max(0.0, detail_score)
            
        except ImportError:
            # Fallback
            return 0.5
    
    def _calculate_color_accuracy(
        self,
        original: np.ndarray,
        generated: np.ndarray,
        mask: Optional[Image.Image],
    ) -> float:
        """Calculate color accuracy score."""
        # Compare color histograms
        try:
            import cv2
            
            # Calculate histograms for each channel
            hist_diff = 0.0
            for i in range(3):
                orig_hist = cv2.calcHist([original], [i], None, [256], [0, 256])
                gen_hist = cv2.calcHist([generated], [i], None, [256], [0, 256])
                
                # Normalize
                orig_hist = orig_hist / (np.sum(orig_hist) + 1e-7)
                gen_hist = gen_hist / (np.sum(gen_hist) + 1e-7)
                
                # Correlation
                correlation = cv2.compareHist(orig_hist, gen_hist, cv2.HISTCMP_CORREL)
                hist_diff += correlation
            
            return max(0.0, min(1.0, hist_diff / 3.0))
            
        except ImportError:
            # Fallback: mean color difference
            color_diff = np.mean(np.abs(original.astype(float) - generated.astype(float)))
            score = 1.0 - (color_diff / 255.0)
            return max(0.0, min(1.0, score))
    
    def _calculate_composition(self, generated: np.ndarray) -> float:
        """Calculate composition score."""
        # Simple composition check (rule of thirds, etc.)
        # For now, return a placeholder
        return 0.75
    
    def _detect_artifacts(
        self,
        generated: np.ndarray,
        mask: Optional[Image.Image],
    ) -> float:
        """Detect artifacts in generated image."""
        try:
            import cv2
            
            gray = cv2.cvtColor(generated, cv2.COLOR_RGB2GRAY)
            
            # Detect unusual patterns (potential artifacts)
            # High frequency noise
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            high_freq = np.sum(np.abs(laplacian) > 50) / laplacian.size
            
            # Unusual color transitions
            color_diff = np.std(generated, axis=2)
            unusual_transitions = np.sum(color_diff > 50) / color_diff.size
            
            artifact_score = (high_freq * 0.5 + unusual_transitions * 0.5)
            return max(0.0, min(1.0, artifact_score))
            
        except ImportError:
            return 0.1  # Assume low artifacts if can't detect


