"""
Clothing Change Processor
=========================

Processes clothing change operations, handling model selection,
image processing, metrics calculation, and tensor generation.
"""

import logging
from typing import Optional, Dict, Any, Union
from pathlib import Path
from PIL import Image
import numpy as np

from ...core.utils.model_selector import ModelSelector
from ...core.utils.image_loader import ImageLoader
from ...core.utils.result_builder import ResultBuilder
from ...core.utils.generator_manager import GeneratorManager
from ...models.quality_metrics import QualityMetrics
from ...config.clothing_changer_config import ClothingChangerConfig

logger = logging.getLogger(__name__)


class ClothingChangeProcessor:
    """Processes clothing change operations."""
    
    def __init__(
        self,
        model_selector: ModelSelector,
        generator_manager: Optional[GeneratorManager],
        quality_metrics: QualityMetrics,
        config: ClothingChangerConfig,
        initialize_model_fn: callable,
    ):
        """
        Initialize Clothing Change Processor.
        
        Args:
            model_selector: Model selector instance
            generator_manager: Generator manager instance
            quality_metrics: Quality metrics calculator
            config: Configuration instance
            initialize_model_fn: Function to initialize model
        """
        self.model_selector = model_selector
        self.generator_manager = generator_manager
        self.quality_metrics = quality_metrics
        self.config = config
        self.initialize_model_fn = initialize_model_fn
    
    def process(
        self,
        image: Union[str, Path, Image.Image],
        clothing_description: str,
        character_name: Optional[str] = None,
        mask: Optional[Union[Image.Image, np.ndarray]] = None,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        strength: Optional[float] = None,
        save_tensor: bool = True,
        output_filename: Optional[str] = None,
        style: Optional[str] = None,
        calculate_metrics: bool = True,
    ) -> Dict[str, Any]:
        """
        Process clothing change operation.
        
        Args:
            image: Input character image
            clothing_description: Description of new clothing
            character_name: Optional character name
            mask: Optional mask for clothing area
            prompt: Optional full prompt
            negative_prompt: Negative prompt
            num_inference_steps: Number of inference steps
            guidance_scale: Guidance scale
            strength: Inpainting strength
            save_tensor: Whether to save as ComfyUI safe tensor
            output_filename: Optional custom output filename
            style: Optional style
            calculate_metrics: Whether to calculate quality metrics
            
        Returns:
            Dict with result info and paths
        """
        # Ensure model is initialized
        self.model_selector.ensure_model_initialized(self.initialize_model_fn)
        
        # Change clothing using appropriate model
        logger.info(f"Changing clothing: {clothing_description}")
        changed_image = self.model_selector.change_clothing(
            image=image,
            clothing_description=clothing_description,
            character_name=character_name,
            mask=mask,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
        )
        
        # Load original image for metrics
        original_pil = ImageLoader.load(image)
        
        # Calculate quality metrics if requested
        metrics = None
        if calculate_metrics:
            metrics = self._calculate_metrics(original_pil, changed_image, mask)
        
        # Save tensor if requested
        saved_path = None
        if save_tensor:
            saved_path = self._save_tensor(
                image,
                clothing_description,
                changed_image,
                character_name,
                output_filename
            )
        
        # Build result
        result = ResultBuilder.build_result(
            clothing_description=clothing_description,
            changed_image=changed_image,
            character_name=character_name,
            style=style,
            prompt=prompt,
            negative_prompt=negative_prompt,
            metrics=metrics,
            saved_path=saved_path,
            save_tensor=save_tensor,
        )
        
        return result
    
    def _calculate_metrics(
        self,
        original_image: Image.Image,
        changed_image: Image.Image,
        mask: Optional[Union[Image.Image, np.ndarray]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Calculate quality metrics.
        
        Args:
            original_image: Original image
            changed_image: Changed image
            mask: Optional mask
            
        Returns:
            Metrics dictionary or None if calculation fails
        """
        try:
            metrics = self.quality_metrics.calculate_metrics(
                original_image=original_image,
                changed_image=changed_image,
                mask=mask,
            )
            logger.info(f"Quality metrics: overall={metrics.get('overall_quality', 0):.3f}")
            return metrics
        except Exception as e:
            logger.warning(f"Error calculating metrics: {e}")
            return None
    
    def _save_tensor(
        self,
        original_image: Union[str, Path, Image.Image],
        clothing_description: str,
        changed_image: Image.Image,
        character_name: Optional[str] = None,
        output_filename: Optional[str] = None
    ) -> Optional[str]:
        """
        Save tensor if generator is available.
        
        Args:
            original_image: Original image
            clothing_description: Clothing description
            changed_image: Changed image
            character_name: Optional character name
            output_filename: Optional output filename
            
        Returns:
            Saved path or None if saving fails
        """
        if not self.generator_manager or not self.generator_manager.can_generate_tensors():
            return None
        
        try:
            self.generator_manager.ensure_generator_initialized()
            saved_path = self.generator_manager.generate_tensor(
                original_image=original_image,
                clothing_description=clothing_description,
                changed_image=changed_image,
                character_name=character_name,
                output_filename=output_filename,
            )
            return saved_path
        except Exception as e:
            logger.warning(f"Error saving tensor: {e}")
            return None

