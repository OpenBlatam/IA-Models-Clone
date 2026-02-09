"""
Clothing Change Executor
========================

Orchestrates the clothing change process, coordinating image processing,
mask generation, and pipeline execution.
"""

import logging
from typing import Optional, Dict, Any, Union, Tuple
from pathlib import Path
from PIL import Image
import numpy as np
import time

from ...metrics.quality_metrics import ProcessingMetrics
from ...validators.quality_validator import ImageQualityValidator
from ...processing import ImagePreprocessor, MaskGenerator

logger = logging.getLogger(__name__)


class ClothingChangeExecutor:
    """Executes clothing change operations."""
    
    def __init__(
        self,
        preprocessor: ImagePreprocessor,
        mask_generator: MaskGenerator,
        validate_images: bool = True,
        enhance_images: bool = False,
    ):
        """
        Initialize Clothing Change Executor.
        
        Args:
            preprocessor: Image preprocessor instance
            mask_generator: Mask generator instance
            validate_images: Whether to validate images
            enhance_images: Whether to enhance images
        """
        self.preprocessor = preprocessor
        self.mask_generator = mask_generator
        self.validate_images = validate_images
        self.enhance_images = enhance_images
    
    def prepare_image(
        self,
        image: Union[Image.Image, str, Path, np.ndarray],
        stats: Optional[Dict[str, Any]] = None
    ) -> Tuple[Image.Image, ProcessingMetrics]:
        """
        Prepare image for processing.
        
        Args:
            image: Input image
            stats: Optional statistics dictionary to update
            
        Returns:
            Tuple of (prepared_image, metrics)
        """
        metrics = ProcessingMetrics(
            processing_time=0.0,
            mask_quality=0.0,
            prompt_quality=0.0,
            success=False,
            errors=[]
        )
        
        # Convert to PIL if needed
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        else:
            error_msg = f"Unsupported image type: {type(image)}"
            metrics.errors.append(error_msg)
            raise ValueError(error_msg)
        
        # Validate image
        if self.validate_images:
            quality = ImageQualityValidator.validate_image(pil_image)
            if stats:
                stats["images_validated"] = stats.get("images_validated", 0) + 1
            
            if not quality.is_valid:
                if stats:
                    stats["validation_failures"] = stats.get("validation_failures", 0) + 1
                error_msg = f"Image validation failed: {', '.join(quality.errors)}"
                metrics.errors.append(error_msg)
                raise ValueError(error_msg)
        
        # Enhance if requested
        if self.enhance_images:
            pil_image = ImageQualityValidator.enhance_image(pil_image)
            if stats:
                stats["images_enhanced"] = stats.get("images_enhanced", 0) + 1
        
        # Preprocess for inpainting
        pil_image = self.preprocessor.preprocess_for_inpainting(pil_image)
        
        return pil_image, metrics
    
    def prepare_mask(
        self,
        pil_image: Image.Image,
        mask: Optional[Union[Image.Image, np.ndarray]] = None,
        stats: Optional[Dict[str, Any]] = None
    ) -> Tuple[Image.Image, float]:
        """
        Prepare mask for inpainting.
        
        Args:
            pil_image: Prepared image
            mask: Optional existing mask
            stats: Optional statistics dictionary
            
        Returns:
            Tuple of (mask_image, mask_quality)
        """
        if mask is None:
            try:
                mask_image = self.mask_generator.generate_smart_mask(pil_image)
                mask_image = self.mask_generator.refine_mask(mask_image, pil_image)
            except Exception as e:
                logger.warning(f"Smart mask generation failed: {e}, using simple mask")
                mask_image = self.mask_generator.generate_simple_mask(pil_image)
        else:
            if isinstance(mask, np.ndarray):
                mask_image = Image.fromarray(mask).convert("L")
            else:
                mask_image = mask.convert("L")
            
            if mask_image.size != pil_image.size:
                mask_image = mask_image.resize(pil_image.size, Image.Resampling.LANCZOS)
        
        # Calculate mask quality
        mask_array = np.array(mask_image)
        mask_coverage = np.sum(mask_array > 128) / mask_array.size
        
        return mask_image, mask_coverage
    
    def prepare_prompt(
        self,
        clothing_description: str,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        default_negative_prompt: str = ""
    ) -> Tuple[str, str, float]:
        """
        Prepare prompts for generation.
        
        Args:
            clothing_description: Clothing description
            prompt: Optional full prompt
            negative_prompt: Optional negative prompt
            default_negative_prompt: Default negative prompt
            
        Returns:
            Tuple of (prompt, negative_prompt, prompt_quality)
        """
        # Generate prompt if not provided
        if prompt is None:
            prompt = f"a character wearing {clothing_description}, high quality, detailed, professional photography"
        
        # Assess prompt quality
        prompt_length = len(prompt.split())
        prompt_quality = min(1.0, prompt_length / 20.0)
        
        if negative_prompt is None:
            negative_prompt = default_negative_prompt
        
        return prompt, negative_prompt, prompt_quality
    
    def execute_pipeline(
        self,
        pipeline,
        prompt: str,
        image: Image.Image,
        mask_image: Image.Image,
        negative_prompt: str,
        num_inference_steps: int,
        guidance_scale: float,
        strength: float,
        use_inpainting: bool = True
    ) -> Image.Image:
        """
        Execute the diffusion pipeline.
        
        Args:
            pipeline: Diffusion pipeline instance
            prompt: Generation prompt
            image: Input image
            mask_image: Mask image
            negative_prompt: Negative prompt
            num_inference_steps: Number of inference steps
            guidance_scale: Guidance scale
            strength: Inpainting strength
            use_inpainting: Whether to use inpainting
            
        Returns:
            Generated image
        """
        if use_inpainting:
            result = pipeline(
                prompt=prompt,
                image=image,
                mask_image=mask_image,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                strength=strength,
            ).images[0]
        else:
            result = pipeline(
                prompt=prompt,
                image=image,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            ).images[0]
        
        return result
    
    def update_stats(
        self,
        stats: Dict[str, Any],
        metrics: ProcessingMetrics,
        processing_time: float,
        success: bool
    ) -> None:
        """
        Update statistics dictionary.
        
        Args:
            stats: Statistics dictionary
            metrics: Processing metrics
            processing_time: Processing time
            success: Whether operation succeeded
        """
        metrics.processing_time = processing_time
        metrics.success = success
        
        if success:
            stats["successful_changes"] = stats.get("successful_changes", 0) + 1
            stats["images_processed"] = stats.get("images_processed", 0) + 1
        else:
            stats["failed_changes"] = stats.get("failed_changes", 0) + 1
        
        stats["total_time"] = stats.get("total_time", 0.0) + processing_time
        stats["clothing_changes"] = stats.get("clothing_changes", 0) + 1

