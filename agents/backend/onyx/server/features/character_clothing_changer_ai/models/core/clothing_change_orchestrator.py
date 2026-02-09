"""
Clothing Change Orchestrator
============================

Orchestrates the complete clothing change operation.
"""

import logging
from typing import Optional, Union
from pathlib import Path
from PIL import Image
import numpy as np

from .preprocessing_orchestrator import PreprocessingOrchestrator
from .pipeline_orchestrator import PipelineOrchestrator
from .prompt_generator import PromptGenerator
from ..processing import MaskProcessor

logger = logging.getLogger(__name__)


class ClothingChangeOrchestrator:
    """Orchestrates the complete clothing change operation."""
    
    def __init__(
        self,
        preprocessing_orchestrator: PreprocessingOrchestrator,
        pipeline_orchestrator: PipelineOrchestrator,
        mask_processor: MaskProcessor,
        prompt_generator: PromptGenerator,
    ):
        """
        Initialize clothing change orchestrator.
        
        Args:
            preprocessing_orchestrator: Preprocessing orchestrator
            pipeline_orchestrator: Pipeline orchestrator
            mask_processor: Mask processor
            prompt_generator: Prompt generator
        """
        self.preprocessing = preprocessing_orchestrator
        self.pipeline = pipeline_orchestrator
        self.mask_processor = mask_processor
        self.prompt_generator = prompt_generator
    
    def change_clothing(
        self,
        image: Union[Image.Image, str, Path, np.ndarray],
        clothing_description: str,
        mask: Optional[Union[Image.Image, np.ndarray]] = None,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        strength: float = 0.8,
        auto_enhance: bool = False,
    ) -> Image.Image:
        """
        Change clothing in character image.
        
        Args:
            image: Input character image
            clothing_description: Description of new clothing
            mask: Optional mask for clothing area
            prompt: Optional full prompt
            negative_prompt: Negative prompt
            num_inference_steps: Number of inference steps
            guidance_scale: Guidance scale
            strength: Inpainting strength
            auto_enhance: Automatically enhance image
            
        Returns:
            Image with changed clothing
        """
        # Preprocess image
        pil_image = self.preprocessing.preprocess_for_inpainting(image)
        
        # Generate prompt if not provided
        if prompt is None:
            prompt = self.prompt_generator.generate_prompt(clothing_description)
        
        if negative_prompt is None:
            negative_prompt = self.prompt_generator.generate_negative_prompt()
        
        # Prepare mask
        mask_image = self.mask_processor.prepare_mask(
            image=pil_image,
            mask=mask,
            use_smart_detection=True,
        )
        
        # Generate using pipeline
        result = self.pipeline.generate(
            prompt=prompt,
            image=pil_image,
            mask_image=mask_image,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
        )
        
        logger.debug("Clothing change completed successfully")
        return result


