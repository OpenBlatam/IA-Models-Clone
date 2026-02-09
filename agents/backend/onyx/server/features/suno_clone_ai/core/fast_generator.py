"""
Fast Music Generator with Optimizations
"""

import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any, List
import logging
import numpy as np
from functools import lru_cache
from pathlib import Path

from .music_generator import MusicGenerator

logger = logging.getLogger(__name__)


class FastMusicGenerator(MusicGenerator):
    """Optimized music generator with caching and speed improvements"""
    
    def __init__(self, use_cache: bool = True, use_compile: bool = True):
        """
        Initialize fast music generator
        
        Args:
            use_cache: Enable caching
            use_compile: Use torch.compile for optimization
        """
        super().__init__()
        self.use_cache = use_cache
        self.cache = {}
        
        # Optimize model with torch.compile if available
        if use_compile and hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                logger.info("Model compiled with torch.compile")
            except Exception as e:
                logger.warning(f"Could not compile model: {e}")
        
        # Enable mixed precision
        self.use_amp = torch.cuda.is_available()
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
            logger.info("Mixed precision enabled")
    
    @lru_cache(maxsize=100)
    def _cached_generate(
        self,
        text: str,
        duration: int,
        guidance_scale: float
    ) -> tuple:
        """Cached generation (returns hash for actual generation)"""
        return hash((text, duration, guidance_scale))
    
    def generate_from_text(
        self,
        text: str,
        duration: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        use_cache: Optional[bool] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Fast generation with caching
        
        Args:
            text: Song description
            duration: Duration in seconds
            guidance_scale: Guidance scale
            use_cache: Override cache setting
            **kwargs: Additional parameters
        
        Returns:
            Generated audio array
        """
        use_cache = use_cache if use_cache is not None else self.use_cache
        
        # Check cache
        if use_cache:
            cache_key = (text, duration, guidance_scale)
            if cache_key in self.cache:
                logger.info("Using cached result")
                return self.cache[cache_key]
        
        # Generate with optimizations
        try:
            from config.settings import settings
            
            duration = duration or settings.default_duration
            guidance_scale = guidance_scale or settings.cfg_coef
            
            # Prepare inputs
            inputs = self.processor(
                text=[text],
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            
            # Generate with mixed precision if available
            with torch.no_grad():
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        audio_values = self.model.generate(
                            **inputs,
                            max_new_tokens=int(duration * settings.sample_rate / self.model.config.audio_encoder.hop_length),
                            guidance_scale=guidance_scale,
                            temperature=settings.temperature,
                            top_k=settings.top_k,
                            top_p=settings.top_p,
                        )
                else:
                    audio_values = self.model.generate(
                        **inputs,
                        max_new_tokens=int(duration * settings.sample_rate / self.model.config.audio_encoder.hop_length),
                        guidance_scale=guidance_scale,
                        temperature=settings.temperature,
                        top_k=settings.top_k,
                        top_p=settings.top_p,
                    )
            
            # Convert to numpy
            audio = audio_values[0, 0].cpu().numpy()
            
            # Cache result
            if use_cache:
                cache_key = (text, duration, guidance_scale)
                self.cache[cache_key] = audio
            
            logger.info(f"Fast generation completed: shape={audio.shape}")
            return audio
            
        except Exception as e:
            logger.error(f"Error in fast generation: {e}")
            raise
    
    def generate_batch(
        self,
        texts: List[str],
        duration: Optional[int] = None,
        **kwargs
    ) -> List[np.ndarray]:
        """
        Generate multiple songs in batch
        
        Args:
            texts: List of song descriptions
            duration: Duration in seconds
            **kwargs: Additional parameters
        
        Returns:
            List of audio arrays
        """
        results = []
        
        for text in texts:
            try:
                audio = self.generate_from_text(text, duration=duration, **kwargs)
                results.append(audio)
            except Exception as e:
                logger.error(f"Error generating for text '{text[:50]}...': {e}")
                results.append(None)
        
        return results
    
    def clear_cache(self):
        """Clear generation cache"""
        self.cache.clear()
        self._cached_generate.cache_clear()
        logger.info("Cache cleared")

