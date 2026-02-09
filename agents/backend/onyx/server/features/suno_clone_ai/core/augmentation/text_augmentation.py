"""
Text Augmentation

Utilities for text data augmentation.
"""

import logging
import random
from typing import List, Callable, Optional

logger = logging.getLogger(__name__)


class TextAugmenter:
    """Base class for text augmentation."""
    
    def __call__(self, text: str) -> str:
        """
        Apply augmentation.
        
        Args:
            text: Input text
            
        Returns:
            Augmented text
        """
        raise NotImplementedError


class SynonymReplacement(TextAugmenter):
    """Replace words with synonyms."""
    
    def __init__(self, p: float = 0.3):
        """
        Initialize synonym replacement.
        
        Args:
            p: Probability of replacement
        """
        self.p = p
        # Simple synonym dictionary (in production, use WordNet or similar)
        self.synonyms = {
            'good': ['great', 'excellent', 'fine'],
            'bad': ['poor', 'terrible', 'awful'],
            'music': ['song', 'tune', 'melody'],
            'generate': ['create', 'produce', 'make']
        }
    
    def __call__(self, text: str) -> str:
        """Apply synonym replacement."""
        words = text.split()
        augmented = []
        
        for word in words:
            if random.random() < self.p and word.lower() in self.synonyms:
                synonym = random.choice(self.synonyms[word.lower()])
                augmented.append(synonym)
            else:
                augmented.append(word)
        
        return ' '.join(augmented)


class RandomInsertion(TextAugmenter):
    """Randomly insert words."""
    
    def __init__(self, p: float = 0.2, n: int = 1):
        """
        Initialize random insertion.
        
        Args:
            p: Probability of insertion
            n: Number of words to insert
        """
        self.p = p
        self.n = n
        self.common_words = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at']
    
    def __call__(self, text: str) -> str:
        """Apply random insertion."""
        if random.random() > self.p:
            return text
        
        words = text.split()
        for _ in range(self.n):
            insert_pos = random.randint(0, len(words))
            word = random.choice(self.common_words)
            words.insert(insert_pos, word)
        
        return ' '.join(words)


class RandomDeletion(TextAugmenter):
    """Randomly delete words."""
    
    def __init__(self, p: float = 0.2):
        """
        Initialize random deletion.
        
        Args:
            p: Probability of deletion
        """
        self.p = p
    
    def __call__(self, text: str) -> str:
        """Apply random deletion."""
        words = text.split()
        
        if len(words) <= 1:
            return text
        
        augmented = [word for word in words if random.random() > self.p]
        
        return ' '.join(augmented) if augmented else text


def create_text_augmentation_pipeline(
    augmentations: List[TextAugmenter],
    p: float = 1.0
) -> Callable:
    """
    Create text augmentation pipeline.
    
    Args:
        augmentations: List of augmentation functions
        p: Probability of applying augmentation
        
    Returns:
        Augmentation function
    """
    def augment(text: str) -> str:
        if random.random() < p:
            for aug in augmentations:
                text = aug(text)
        return text
    
    return augment



