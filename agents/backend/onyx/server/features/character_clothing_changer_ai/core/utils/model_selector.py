"""
Model Selector
==============

Handles model selection logic for clothing changes.
"""

import logging
from typing import Optional, Tuple, Union
from PIL import Image
import numpy as np

from .model_selector_helper import ModelSelectorHelper

logger = logging.getLogger(__name__)


class ModelSelector:
    """Selects and uses the appropriate model for clothing changes."""
    
    def __init__(
        self,
        flux2_model,
        deepseek_model,
        use_deepseek_fallback: bool,
        config
    ):
        """
        Initialize model selector.
        
        Args:
            flux2_model: Flux2 model instance
            deepseek_model: DeepSeek model instance
            use_deepseek_fallback: Whether to use DeepSeek fallback
            config: Configuration object
        """
        self.flux2_model = flux2_model
        self.deepseek_model = deepseek_model
        self.use_deepseek_fallback = use_deepseek_fallback
        self.config = config
    
    def change_clothing(
        self,
        image: Union[str, Image.Image],
        clothing_description: str,
        character_name: Optional[str] = None,
        mask: Optional[Union[Image.Image, np.ndarray]] = None,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        strength: Optional[float] = None,
    ) -> Image.Image:
        """
        Change clothing using the appropriate model.
        
        Args:
            image: Input image
            clothing_description: Clothing description
            character_name: Optional character name
            mask: Optional mask
            prompt: Optional prompt
            negative_prompt: Negative prompt
            num_inference_steps: Inference steps
            guidance_scale: Guidance scale
            strength: Inpainting strength
            
        Returns:
            Changed image
        """
        # Select model
        model, model_type = ModelSelectorHelper.select_model(
            flux2_model=self.flux2_model,
            deepseek_model=self.deepseek_model,
            use_deepseek_fallback=self.use_deepseek_fallback
        )
        
        # Get default parameters
        params = ModelSelectorHelper.get_default_parameters(
            config=self.config,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            negative_prompt=negative_prompt
        )
        
        logger.info(f"Using {model_type} model for clothing change")
        
        # Call appropriate model method
        if model_type == "deepseek":
            return model.change_clothing(
                image=image,
                clothing_description=clothing_description,
                character_name=character_name,
                prompt=prompt,
                negative_prompt=params["negative_prompt"],
                num_inference_steps=params["num_inference_steps"],
                guidance_scale=params["guidance_scale"],
                strength=params["strength"],
            )
        else:  # flux2
            return model.change_clothing(
                image=image,
                clothing_description=clothing_description,
                mask=mask,
                prompt=prompt,
                negative_prompt=params["negative_prompt"],
                num_inference_steps=params["num_inference_steps"],
                guidance_scale=params["guidance_scale"],
                strength=params["strength"],
            )
    
    def ensure_model_initialized(self, initialize_fn) -> None:
        """
        Ensure model is initialized, with fallback handling.
        
        Args:
            initialize_fn: Function to initialize models
        """
        ModelSelectorHelper.ensure_model_available(
            flux2_model=self.flux2_model,
            deepseek_model=self.deepseek_model,
            initialize_fn=initialize_fn
        )

