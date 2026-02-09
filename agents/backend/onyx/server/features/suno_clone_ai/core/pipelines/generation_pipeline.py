"""
Generation Pipeline

Functional pipeline for music generation with proper composition.
"""

import logging
from typing import Optional, Dict, Any, List, Callable
import numpy as np

from ..generators import BaseMusicGenerator

logger = logging.getLogger(__name__)


class GenerationPipeline:
    """
    Functional pipeline for music generation.
    
    Composes multiple processing steps in a clear, functional way.
    """
    
    def __init__(
        self,
        generator: BaseMusicGenerator,
        preprocessors: Optional[List[Callable]] = None,
        postprocessors: Optional[List[Callable]] = None
    ):
        """
        Initialize generation pipeline.
        
        Args:
            generator: Music generator
            preprocessors: List of preprocessing functions
            postprocessors: List of postprocessing functions
        """
        self.generator = generator
        self.preprocessors = preprocessors or []
        self.postprocessors = postprocessors or []
    
    def __call__(
        self,
        prompt: str,
        **kwargs
    ) -> np.ndarray:
        """
        Execute pipeline.
        
        Args:
            prompt: Text prompt
            **kwargs: Generation parameters
            
        Returns:
            Generated audio
        """
        # Preprocessing
        processed_prompt = prompt
        for preprocessor in self.preprocessors:
            processed_prompt = preprocessor(processed_prompt)
        
        # Generation
        audio = self.generator.generate(processed_prompt, **kwargs)
        
        # Postprocessing
        for postprocessor in self.postprocessors:
            audio = postprocessor(audio)
        
        return audio
    
    def add_preprocessor(self, preprocessor: Callable) -> 'GenerationPipeline':
        """Add preprocessing step."""
        self.preprocessors.append(preprocessor)
        return self
    
    def add_postprocessor(self, postprocessor: Callable) -> 'GenerationPipeline':
        """Add postprocessing step."""
        self.postprocessors.append(postprocessor)
        return self


def compose_pipeline(
    generator: BaseMusicGenerator,
    *processors: Callable
) -> GenerationPipeline:
    """
    Compose pipeline from generator and processors.
    
    Args:
        generator: Music generator
        *processors: Processing functions
        
    Returns:
        Composed pipeline
    """
    pipeline = GenerationPipeline(generator)
    
    for processor in processors:
        # Determine if pre or post processor based on signature
        # For simplicity, add as postprocessor
        pipeline.add_postprocessor(processor)
    
    return pipeline



