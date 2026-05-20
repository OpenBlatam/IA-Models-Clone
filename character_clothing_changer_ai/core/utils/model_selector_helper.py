"""
Model Selector Helper Utility
=============================

Helper utilities for model selection logic.
"""

import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class ModelSelectorHelper:
    """Helper utilities for model selection."""
    
    @staticmethod
    def select_model(
        flux2_model: Optional[Any],
        deepseek_model: Optional[Any],
        use_deepseek_fallback: bool
    ) -> Tuple[Optional[Any], str]:
        """
        Select the appropriate model to use.
        
        Args:
            flux2_model: Flux2 model instance
            deepseek_model: DeepSeek model instance
            use_deepseek_fallback: Whether to use DeepSeek fallback
            
        Returns:
            Tuple of (selected_model, model_type)
            
        Raises:
            RuntimeError: If no model is available
        """
        if use_deepseek_fallback and deepseek_model:
            logger.debug("Selecting DeepSeek model (fallback mode)")
            return deepseek_model, "deepseek"
        elif flux2_model:
            logger.debug("Selecting Flux2 model")
            return flux2_model, "flux2"
        else:
            raise RuntimeError(
                "No model available. Failed to initialize both Flux2 and DeepSeek."
            )
    
    @staticmethod
    def get_default_parameters(
        config: Any,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        strength: Optional[float] = None,
        negative_prompt: Optional[str] = None
    ) -> dict:
        """
        Get default parameters from config with overrides.
        
        Args:
            config: Configuration object
            num_inference_steps: Optional override for inference steps
            guidance_scale: Optional override for guidance scale
            strength: Optional override for strength
            negative_prompt: Optional override for negative prompt
            
        Returns:
            Dictionary of parameters
        """
        return {
            "num_inference_steps": num_inference_steps or config.default_num_inference_steps,
            "guidance_scale": guidance_scale or config.default_guidance_scale,
            "strength": strength or config.default_strength,
            "negative_prompt": negative_prompt or config.default_negative_prompt,
        }
    
    @staticmethod
    def ensure_model_available(
        flux2_model: Optional[Any],
        deepseek_model: Optional[Any],
        initialize_fn: callable
    ) -> None:
        """
        Ensure at least one model is available.
        
        Args:
            flux2_model: Flux2 model instance
            deepseek_model: DeepSeek model instance
            initialize_fn: Function to initialize models
            
        Raises:
            RuntimeError: If no model is available after initialization
        """
        if flux2_model is None and deepseek_model is None:
            try:
                initialize_fn()
            except RuntimeError:
                # Both models failed, re-raise
                raise
            except Exception as e:
                # Flux2 failed, but fallback should have been activated
                # Check if DeepSeek is available
                if not deepseek_model:
                    logger.error(f"Model initialization failed and DeepSeek fallback not available: {e}")
                    raise RuntimeError(f"Failed to initialize any model: {e}")
                # DeepSeek is available, continue

