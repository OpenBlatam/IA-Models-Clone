"""
Transformers-based Music Generator

Implements music generation using Hugging Face Transformers library
with best practices for model loading, inference, and error handling.
"""

import logging
from typing import Optional, Dict, Any, List, Union
import torch
import torch.nn as nn
import numpy as np

from .base_generator import BaseMusicGenerator
from ..utils.model_utils import (
    initialize_weights,
    check_for_nan_inf,
    compile_model,
    enable_gradient_checkpointing
)
from ..utils.mixed_precision import MixedPrecisionManager

logger = logging.getLogger(__name__)


class TransformersMusicGenerator(BaseMusicGenerator):
    """
    Music generator using Hugging Face Transformers.
    
    Supports models like:
    - MusicGen (facebook/musicgen-*)
    - Other transformer-based audio models
    """
    
    def __init__(
        self,
        model_name: str = "facebook/musicgen-medium",
        device: Optional[torch.device] = None,
        use_mixed_precision: bool = True,
        use_compile: bool = True,
        compile_mode: str = "reduce-overhead",
        enable_gradient_checkpointing: bool = False
    ):
        """
        Initialize transformers-based generator.
        
        Args:
            model_name: Model name from Hugging Face
            device: Device to use
            use_mixed_precision: Enable mixed precision inference
            use_compile: Compile model for speed (PyTorch 2.0+)
            compile_mode: Compilation mode
            enable_gradient_checkpointing: Enable gradient checkpointing
        """
        super().__init__(
            device=device,
            use_mixed_precision=use_mixed_precision,
            model_name=model_name
        )
        
        self.use_compile = use_compile
        self.compile_mode = compile_mode
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        
        # Mixed precision manager
        self.amp_manager = MixedPrecisionManager(enabled=use_mixed_precision)
        
        # Initialize model
        self._load_model()
    
    def _load_model(self) -> None:
        """Load model and processor from Hugging Face."""
        try:
            from transformers import AutoProcessor, MusicgenForConditionalGeneration
            
            logger.info(f"Loading model: {self.model_name}")
            logger.info(f"Device: {self.device}")
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            
            # Load model
            self.model = MusicgenForConditionalGeneration.from_pretrained(
                self.model_name
            )
            
            # Apply weight initialization if needed (for fine-tuning)
            if hasattr(self.model, 'apply'):
                self.model.apply(initialize_weights)
            
            # Move to device
            self.model.to(self.device)
            
            # Set to evaluation mode
            self.model.eval()
            
            # Compile model for speed
            if self.use_compile:
                self.model = compile_model(self.model, mode=self.compile_mode)
            
            # Enable gradient checkpointing if requested
            if self.enable_gradient_checkpointing:
                enable_gradient_checkpointing(self.model)
            
            self._initialized = True
            logger.info("Model loaded successfully")
            
        except ImportError as e:
            logger.error(f"Transformers library not available: {e}")
            raise ImportError(
                "transformers library required. Install with: pip install transformers"
            )
        except Exception as e:
            logger.error(f"Error loading model: {e}", exc_info=True)
            raise
    
    def generate(
        self,
        prompt: Union[str, List[str]],
        duration: int = 30,
        temperature: float = 1.0,
        top_k: int = 250,
        top_p: float = 0.0,
        guidance_scale: float = 3.0,
        max_new_tokens: Optional[int] = None,
        **kwargs
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Generate music from prompt(s).
        
        Args:
            prompt: Text prompt(s)
            duration: Duration in seconds
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            guidance_scale: Guidance scale for generation
            max_new_tokens: Maximum number of tokens to generate
            **kwargs: Additional parameters
            
        Returns:
            Generated audio array(s)
        """
        if not self._initialized:
            self.initialize()
        
        # Handle single vs batch prompts
        is_single = isinstance(prompt, str)
        prompts = [prompt] if is_single else prompt
        
        # Validate inputs
        for p in prompts:
            if not p or not isinstance(p, str):
                raise ValueError("Prompt must be a non-empty string")
        
        if duration <= 0 or duration > 300:
            raise ValueError("Duration must be between 1 and 300 seconds")
        
        try:
            # Prepare inputs
            inputs = self.processor(
                text=prompts,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            
            # Check for NaN/Inf in inputs
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    check_for_nan_inf(value, f"input_{key}")
            
            # Calculate max_new_tokens if not provided
            if max_new_tokens is None:
                sample_rate = getattr(
                    self.model.config,
                    'audio_encoder',
                    type('obj', (object,), {'sample_rate': 32000})()
                ).sample_rate if hasattr(self.model.config, 'audio_encoder') else 32000
                
                hop_length = getattr(
                    self.model.config.audio_encoder,
                    'hop_length',
                    512
                ) if hasattr(self.model.config, 'audio_encoder') else 512
                
                max_new_tokens = int(duration * sample_rate / hop_length)
            
            # Generate audio
            with torch.no_grad():
                with self.amp_manager.autocast():
                    audio_values = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        guidance_scale=guidance_scale,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p if top_p > 0.0 else None,
                        **kwargs
                    )
            
            # Check for NaN/Inf in output
            check_for_nan_inf(audio_values, "audio_output")
            
            # Convert to numpy
            if is_single:
                audio = audio_values[0, 0].cpu().numpy()
            else:
                audio = [av[0].cpu().numpy() for av in audio_values]
            
            # Validate output
            if is_single:
                if np.isnan(audio).any() or np.isinf(audio).any():
                    raise ValueError("Generated audio contains NaN or Inf values")
            else:
                for i, a in enumerate(audio):
                    if np.isnan(a).any() or np.isinf(a).any():
                        raise ValueError(f"Generated audio {i} contains NaN or Inf values")
            
            logger.info(
                f"Generated audio: "
                f"shape={audio.shape if is_single else [a.shape for a in audio]}, "
                f"duration={duration}s"
            )
            
            return audio
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"GPU out of memory: {e}")
            self.clear_cache()
            raise RuntimeError(
                "GPU out of memory. Try reducing duration or batch size."
            )
        except Exception as e:
            logger.error(f"Error generating music: {e}", exc_info=True)
            raise

