"""
Advanced Data Processing
========================
Efficient data processing pipelines
"""

from typing import Dict, Any, List, Optional, Callable
import torch
from torch.utils.data import Dataset, DataLoader
import structlog
import numpy as np
from functools import partial

logger = structlog.get_logger()


class DataPipeline:
    """
    Data processing pipeline with transformations
    """
    
    def __init__(self):
        """Initialize pipeline"""
        self.transformations: List[Callable] = []
        logger.info("DataPipeline initialized")
    
    def add_transformation(self, transform: Callable) -> 'DataPipeline':
        """
        Add transformation to pipeline
        
        Args:
            transform: Transformation function
            
        Returns:
            Self for chaining
        """
        self.transformations.append(transform)
        return self
    
    def apply(self, data: Any) -> Any:
        """
        Apply all transformations
        
        Args:
            data: Input data
            
        Returns:
            Transformed data
        """
        result = data
        for transform in self.transformations:
            result = transform(result)
        return result


class TextNormalizer:
    """Text normalization utilities"""
    
    @staticmethod
    def normalize(text: str) -> str:
        """
        Normalize text
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Lowercase
        text = text.lower()
        
        return text
    
    @staticmethod
    def remove_special_chars(text: str, keep_punctuation: bool = True) -> str:
        """
        Remove special characters
        
        Args:
            text: Input text
            keep_punctuation: Keep punctuation marks
            
        Returns:
            Cleaned text
        """
        import re
        
        if keep_punctuation:
            # Keep letters, numbers, and basic punctuation
            text = re.sub(r'[^a-zA-Z0-9\s.,!?;:]', '', text)
        else:
            # Keep only letters and numbers
            text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        
        return text


class BatchProcessor:
    """Batch processing utilities"""
    
    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Custom collate function for batching
        
        Args:
            batch: List of samples
            
        Returns:
            Batched data
        """
        # Get all keys
        keys = batch[0].keys()
        
        batched = {}
        for key in keys:
            values = [sample[key] for sample in batch]
            
            # Stack if tensors
            if isinstance(values[0], torch.Tensor):
                batched[key] = torch.stack(values)
            elif isinstance(values[0], (int, float)):
                batched[key] = torch.tensor(values)
            else:
                batched[key] = values
        
        return batched


class DataAugmentationPipeline:
    """Pipeline for data augmentation"""
    
    def __init__(self):
        """Initialize augmentation pipeline"""
        self.augmentations: List[Callable] = []
    
    def add_augmentation(self, augmentation: Callable, probability: float = 0.5) -> 'DataAugmentationPipeline':
        """
        Add augmentation with probability
        
        Args:
            augmentation: Augmentation function
            probability: Probability of applying
            
        Returns:
            Self for chaining
        """
        def prob_augment(data):
            import random
            if random.random() < probability:
                return augmentation(data)
            return data
        
        self.augmentations.append(prob_augment)
        return self
    
    def apply(self, data: Any) -> Any:
        """Apply augmentations"""
        result = data
        for aug in self.augmentations:
            result = aug(result)
        return result


# Global instances
data_pipeline = DataPipeline()
text_normalizer = TextNormalizer()
batch_processor = BatchProcessor()
augmentation_pipeline = DataAugmentationPipeline()




