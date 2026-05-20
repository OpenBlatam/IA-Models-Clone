"""
Data Service - High-level Data Management
==========================================

Service for managing data operations:
- Dataset creation
- DataLoader creation
- Data splitting
- Preprocessing pipelines
"""

import logging
from typing import Dict, Any, Optional, List
from torch.utils.data import Dataset, DataLoader

from ..core.base import BaseComponent
from ..data import (
    create_dataloader, train_val_test_split,
    TextDataset, ImageDataset
)
from ..data.preprocessing import TextPreprocessor, ImagePreprocessor
from ..architecture.strategy import DataStrategy, StandardDataStrategy

logger = logging.getLogger(__name__)


class DataService(BaseComponent):
    """
    High-level service for data management.
    
    Provides unified interface for data operations.
    """
    
    def _initialize(self) -> None:
        """Initialize service."""
        self.text_preprocessor: Optional[TextPreprocessor] = None
        self.image_preprocessor: Optional[ImagePreprocessor] = None
        self.strategy: Optional[DataStrategy] = None
    
    def setup_text_preprocessing(
        self,
        lowercase: bool = True,
        remove_stopwords: bool = False,
        **kwargs
    ) -> 'DataService':
        """
        Setup text preprocessing.
        
        Args:
            lowercase: Convert to lowercase
            remove_stopwords: Remove stopwords
            **kwargs: Additional preprocessing options
            
        Returns:
            Self for method chaining
        """
        self.text_preprocessor = TextPreprocessor(
            lowercase=lowercase,
            remove_stopwords=remove_stopwords,
            **kwargs
        )
        return self
    
    def setup_image_preprocessing(
        self,
        resize: Optional[tuple] = None,
        normalize: bool = True,
        **kwargs
    ) -> 'DataService':
        """
        Setup image preprocessing.
        
        Args:
            resize: Target size
            normalize: Apply normalization
            **kwargs: Additional preprocessing options
            
        Returns:
            Self for method chaining
        """
        self.image_preprocessor = ImagePreprocessor(
            resize=resize,
            normalize=normalize,
            **kwargs
        )
        return self
    
    def create_text_dataset(
        self,
        texts: List[str],
        labels: Optional[List[int]] = None,
        tokenizer: Optional[Any] = None,
        **kwargs
    ) -> TextDataset:
        """
        Create text dataset.
        
        Args:
            texts: List of texts
            labels: List of labels
            tokenizer: Tokenizer function
            **kwargs: Additional dataset options
            
        Returns:
            TextDataset
        """
        # Preprocess texts if preprocessor available
        if self.text_preprocessor:
            texts = [self.text_preprocessor.preprocess(text) for text in texts]
        
        return TextDataset(
            texts=texts,
            labels=labels,
            tokenizer=tokenizer,
            **kwargs
        )
    
    def create_image_dataset(
        self,
        image_paths: List[str],
        labels: Optional[List[int]] = None,
        **kwargs
    ) -> ImageDataset:
        """
        Create image dataset.
        
        Args:
            image_paths: List of image paths
            labels: List of labels
            **kwargs: Additional dataset options
            
        Returns:
            ImageDataset
        """
        transform = None
        if self.image_preprocessor:
            transform = self.image_preprocessor.preprocess
        
        return ImageDataset(
            image_paths=image_paths,
            labels=labels,
            transform=transform,
            **kwargs
        )
    
    def prepare_data(
        self,
        dataset: Dataset,
        strategy: Optional[DataStrategy] = None,
        **kwargs
    ) -> Dict[str, DataLoader]:
        """
        Prepare data using strategy.
        
        Args:
            dataset: Dataset
            strategy: Data strategy (uses default if None)
            **kwargs: Strategy arguments
            
        Returns:
            Dictionary with data loaders
        """
        if strategy is None:
            strategy = StandardDataStrategy()
        
        return strategy.prepare_data(dataset, **kwargs)
    
    def set_strategy(self, strategy: DataStrategy) -> 'DataService':
        """
        Set data strategy.
        
        Args:
            strategy: Data strategy
            
        Returns:
            Self for method chaining
        """
        self.strategy = strategy
        return self



