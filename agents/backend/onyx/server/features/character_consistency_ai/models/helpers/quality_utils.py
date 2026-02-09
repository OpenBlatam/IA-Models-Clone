"""
Quality Validation Utilities
============================

Utilities for image and embedding quality validation and enhancement.
"""

import logging
from dataclasses import dataclass
from typing import List, Tuple, Union, Optional
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance

logger = logging.getLogger(__name__)


@dataclass
class ImageQualityMetrics:
    """Metrics for image quality assessment."""
    brightness: float
    contrast: float
    sharpness: float
    resolution: Tuple[int, int]
    is_valid: bool
    warnings: List[str]
    errors: List[str]


@dataclass
class EmbeddingMetrics:
    """Metrics for embedding quality."""
    norm: float
    mean: float
    std: float
    min_val: float
    max_val: float
    has_nan: bool
    has_inf: bool
    sparsity: float
    diversity: float
    quality_score: float


class ImageQualityValidator:
    """Validates and assesses image quality before processing."""
    
    @staticmethod
    def validate_image(image: Union[Image.Image, str, Path, np.ndarray]) -> ImageQualityMetrics:
        """
        Validate image quality and return metrics.
        
        Args:
            image: Input image
            
        Returns:
            ImageQualityMetrics with quality assessment
        """
        warnings = []
        errors = []
        
        # Convert to PIL if needed
        pil_image = ImageQualityValidator._to_pil_image(image)
        if pil_image is None:
            return ImageQualityMetrics(
                brightness=0.0, contrast=0.0, sharpness=0.0,
                resolution=(0, 0), is_valid=False,
                warnings=[], errors=[f"Unsupported image type: {type(image)}"]
            )
        
        width, height = pil_image.size
        resolution = (width, height)
        
        # Check minimum resolution
        if width < 64 or height < 64:
            errors.append(f"Image too small: {width}x{height}. Minimum: 64x64")
        
        # Convert to numpy for analysis
        img_array = np.array(pil_image)
        
        # Calculate metrics
        brightness, contrast = ImageQualityValidator._calculate_brightness_contrast(img_array)
        sharpness = ImageQualityValidator._calculate_sharpness(img_array)
        
        # Validate metrics
        ImageQualityValidator._validate_brightness(brightness, errors, warnings)
        ImageQualityValidator._validate_contrast(contrast, warnings)
        ImageQualityValidator._validate_sharpness(sharpness, warnings)
        
        is_valid = len(errors) == 0
        
        return ImageQualityMetrics(
            brightness=brightness,
            contrast=contrast,
            sharpness=sharpness,
            resolution=resolution,
            is_valid=is_valid,
            warnings=warnings,
            errors=errors
        )
    
    @staticmethod
    def _to_pil_image(image: Union[Image.Image, str, Path, np.ndarray]) -> Optional[Image.Image]:
        """Convert various image formats to PIL Image."""
        try:
            if isinstance(image, (str, Path)):
                return Image.open(image).convert("RGB")
            elif isinstance(image, np.ndarray):
                return Image.fromarray(image).convert("RGB")
            elif isinstance(image, Image.Image):
                return image.convert("RGB")
        except Exception as e:
            logger.error(f"Error converting image: {e}")
        return None
    
    @staticmethod
    def _calculate_brightness_contrast(img_array: np.ndarray) -> Tuple[float, float]:
        """Calculate brightness and contrast from image array."""
        gray = np.mean(img_array, axis=2) if len(img_array.shape) == 3 else img_array
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        return brightness, contrast
    
    @staticmethod
    def _calculate_sharpness(img_array: np.ndarray) -> float:
        """Calculate image sharpness using Laplacian variance."""
        gray = np.mean(img_array, axis=2) if len(img_array.shape) == 3 else img_array
        
        try:
            from scipy import ndimage
            laplacian = ndimage.laplace(gray)
            return float(np.var(laplacian))
        except ImportError:
            # Fallback if scipy not available
            contrast = float(np.std(gray))
            return contrast * 10  # Approximate
    
    @staticmethod
    def _validate_brightness(brightness: float, errors: List[str], warnings: List[str]) -> None:
        """Validate brightness and add warnings/errors."""
        if brightness < 20:
            errors.append(f"Image too dark: brightness {brightness:.1f}")
        elif brightness > 240:
            warnings.append(f"Image may be overexposed: brightness {brightness:.1f}")
    
    @staticmethod
    def _validate_contrast(contrast: float, warnings: List[str]) -> None:
        """Validate contrast and add warnings."""
        if contrast < 10:
            warnings.append(f"Low contrast: {contrast:.1f}")
    
    @staticmethod
    def _validate_sharpness(sharpness: float, warnings: List[str]) -> None:
        """Validate sharpness and add warnings."""
        if sharpness < 100:
            warnings.append(f"Image may be blurry: sharpness {sharpness:.1f}")
    
    @staticmethod
    def enhance_image(image: Image.Image) -> Image.Image:
        """Enhance image quality if needed."""
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.1)
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.05)
        
        return image


class EmbeddingQualityValidator:
    """Validates and assesses embedding quality."""
    
    @staticmethod
    def validate_embedding(embedding) -> EmbeddingMetrics:
        """
        Validate embedding quality and return metrics.
        
        Args:
            embedding: Embedding tensor
            
        Returns:
            EmbeddingMetrics with quality assessment
        """
        import torch
        
        # Basic statistics
        norm = float(torch.norm(embedding, p=2).item())
        mean = float(embedding.mean().item())
        std = float(embedding.std().item())
        min_val = float(embedding.min().item())
        max_val = float(embedding.max().item())
        
        # Check for invalid values
        has_nan = bool(torch.isnan(embedding).any().item())
        has_inf = bool(torch.isinf(embedding).any().item())
        
        # Calculate sparsity (percentage of near-zero values)
        near_zero = (torch.abs(embedding) < 0.01).sum().item()
        sparsity = float(near_zero / embedding.numel())
        
        # Calculate diversity (std normalized by mean)
        diversity = float(std / (abs(mean) + 1e-8))
        
        # Quality score (0-1, higher is better)
        quality_score = EmbeddingQualityValidator._calculate_quality_score(
            norm, std, has_nan, has_inf, sparsity
        )
        
        return EmbeddingMetrics(
            norm=norm,
            mean=mean,
            std=std,
            min_val=min_val,
            max_val=max_val,
            has_nan=has_nan,
            has_inf=has_inf,
            sparsity=sparsity,
            diversity=diversity,
            quality_score=quality_score
        )
    
    @staticmethod
    def _calculate_quality_score(
        norm: float,
        std: float,
        has_nan: bool,
        has_inf: bool,
        sparsity: float
    ) -> float:
        """Calculate overall quality score."""
        if has_nan or has_inf:
            return 0.0
        
        # Normalize norm (expected around 1.0 for normalized embeddings)
        norm_score = 1.0 - min(abs(norm - 1.0), 1.0)
        
        # Penalize high sparsity
        sparsity_score = 1.0 - min(sparsity, 0.5) * 2
        
        # Reward good std (not too low, not too high)
        std_score = min(std / 0.5, 1.0) if std < 0.5 else max(1.0 - (std - 0.5), 0.0)
        
        # Weighted average
        quality_score = (norm_score * 0.4 + sparsity_score * 0.3 + std_score * 0.3)
        
        return max(0.0, min(1.0, quality_score))

