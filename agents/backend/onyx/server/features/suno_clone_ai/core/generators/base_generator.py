"""
Base Generator Interface

Defines the base interface for all music generators following best practices.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Union
import torch
import numpy as np

logger = logging.getLogger(__name__)


class BaseMusicGenerator(ABC):
    """
    Abstract base class for music generators.
    
    All music generators should inherit from this class and implement
    the required methods following best practices.
    """
    
    def __init__(
        self,
        device: Optional[torch.device] = None,
        use_mixed_precision: bool = True,
        model_name: Optional[str] = None
    ):
        """
        Initialize base generator.
        
        Args:
            device: Device to use (auto-detected if None)
            use_mixed_precision: Whether to use mixed precision
            model_name: Model name/identifier
        """
        from ..utils.model_utils import get_device, setup_gpu_optimizations
        
        self.device = device or get_device()
        self.use_mixed_precision = use_mixed_precision and torch.cuda.is_available()
        self.model_name = model_name
        
        # Setup GPU optimizations
        if torch.cuda.is_available():
            setup_gpu_optimizations()
        
        self.model = None
        self.processor = None
        self._initialized = False
    
    @abstractmethod
    def _load_model(self) -> None:
        """
        Load the model and processor.
        
        This method should be implemented by subclasses to load
        the specific model architecture.
        """
        raise NotImplementedError
    
    @abstractmethod
    def generate(
        self,
        prompt: Union[str, List[str]],
        duration: int = 30,
        **kwargs
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Generate music from prompt(s).
        
        Args:
            prompt: Text prompt(s) for generation
            duration: Duration in seconds
            **kwargs: Additional generation parameters
            
        Returns:
            Generated audio array(s)
        """
        raise NotImplementedError
    
    def generate_batch(
        self,
        prompts: List[str],
        duration: int = 30,
        **kwargs
    ) -> List[np.ndarray]:
        """
        Generate music for multiple prompts.
        
        Args:
            prompts: List of text prompts
            duration: Duration in seconds for each
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated audio arrays
        """
        results = []
        for prompt in prompts:
            try:
                audio = self.generate(prompt, duration=duration, **kwargs)
                results.append(audio)
            except Exception as e:
                logger.error(f"Error generating for prompt '{prompt}': {e}")
                results.append(None)
        
        return results
    
    def initialize(self) -> None:
        """
        Initialize the generator (lazy loading).
        
        This method ensures the model is loaded before use.
        """
        if not self._initialized:
            self._load_model()
            self._initialized = True
    
    def clear_cache(self) -> None:
        """Clear GPU cache."""
        from ..utils.model_utils import clear_gpu_cache
        clear_gpu_cache()
    
    def to(self, device: torch.device) -> 'BaseMusicGenerator':
        """
        Move model to device.
        
        Args:
            device: Target device
            
        Returns:
            Self for chaining
        """
        self.device = device
        if self.model is not None:
            self.model.to(device)
        return self
    
    def eval(self) -> 'BaseMusicGenerator':
        """Set model to evaluation mode."""
        if self.model is not None:
            self.model.eval()
        return self
    
    def train(self) -> 'BaseMusicGenerator':
        """Set model to training mode."""
        if self.model is not None:
            self.model.train()
        return self

