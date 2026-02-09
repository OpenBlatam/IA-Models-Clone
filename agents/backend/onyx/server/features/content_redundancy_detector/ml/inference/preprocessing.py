"""
Preprocessing Module
Input preprocessing utilities
"""

import torch
import numpy as np
from PIL import Image
from typing import Union, Optional, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """
    Image preprocessing utilities
    """
    
    def __init__(
        self,
        image_size: int = 224,
        mean: Optional[list] = None,
        std: Optional[list] = None,
        normalize: bool = True,
    ):
        """
        Initialize preprocessor
        
        Args:
            image_size: Target image size
            mean: Normalization mean
            std: Normalization std
            normalize: Whether to normalize
        """
        self.image_size = image_size
        self.mean = mean or [0.485, 0.456, 0.406]
        self.std = std or [0.229, 0.224, 0.225]
        self.normalize = normalize
    
    def preprocess(
        self,
        image: Union[str, Path, Image.Image, np.ndarray, torch.Tensor],
    ) -> torch.Tensor:
        """
        Preprocess image
        
        Args:
            image: Input image (path, PIL Image, numpy array, or tensor)
            
        Returns:
            Preprocessed tensor
        """
        # Load image
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
        elif isinstance(image, torch.Tensor):
            # Assume already preprocessed
            return image
        
        # Resize
        image = image.resize((self.image_size, self.image_size))
        
        # Convert to tensor
        image_array = np.array(image).transpose(2, 0, 1) / 255.0
        tensor = torch.from_numpy(image_array).float()
        
        # Normalize
        if self.normalize:
            tensor = self._normalize(tensor)
        
        return tensor
    
    def preprocess_batch(
        self,
        images: list,
    ) -> torch.Tensor:
        """
        Preprocess batch of images
        
        Args:
            images: List of images
            
        Returns:
            Batch tensor
        """
        tensors = [self.preprocess(img) for img in images]
        return torch.stack(tensors)
    
    def _normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalize tensor"""
        mean = torch.tensor(self.mean).view(3, 1, 1)
        std = torch.tensor(self.std).view(3, 1, 1)
        return (tensor - mean) / std



