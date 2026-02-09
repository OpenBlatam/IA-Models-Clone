"""
Data Augmentation for Text
==========================
Text augmentation techniques for psychological analysis
"""

from typing import List, Optional, Callable
import random
import structlog
import re

logger = structlog.get_logger()


class TextAugmenter:
    """
    Text augmentation for training data
    """
    
    def __init__(self):
        """Initialize augmenter"""
        self.augmentation_methods = []
        logger.info("TextAugmenter initialized")
    
    def synonym_replacement(
        self,
        text: str,
        n: int = 1
    ) -> str:
        """
        Replace words with synonyms
        
        Args:
            text: Input text
            n: Number of replacements
            
        Returns:
            Augmented text
        """
        # Simplified - in production use WordNet or similar
        words = text.split()
        if len(words) < n:
            return text
        
        # Random word replacement (simplified)
        indices = random.sample(range(len(words)), min(n, len(words)))
        for idx in indices:
            # In production, use actual synonym dictionary
            words[idx] = words[idx]  # Placeholder
        
        return ' '.join(words)
    
    def random_deletion(
        self,
        text: str,
        p: float = 0.1
    ) -> str:
        """
        Randomly delete words
        
        Args:
            text: Input text
            p: Probability of deletion
            
        Returns:
            Augmented text
        """
        words = text.split()
        if len(words) == 1:
            return text
        
        new_words = [word for word in words if random.random() > p]
        
        if len(new_words) == 0:
            return random.choice(words)
        
        return ' '.join(new_words)
    
    def random_swap(
        self,
        text: str,
        n: int = 1
    ) -> str:
        """
        Randomly swap words
        
        Args:
            text: Input text
            n: Number of swaps
            
        Returns:
            Augmented text
        """
        words = text.split()
        if len(words) < 2:
            return text
        
        for _ in range(n):
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]
        
        return ' '.join(words)
    
    def back_translation(
        self,
        text: str,
        target_language: str = "es"
    ) -> str:
        """
        Back translation augmentation
        
        Args:
            text: Input text
            target_language: Target language for translation
            
        Returns:
            Augmented text
        """
        # In production, use translation API
        # For now, return original
        logger.debug("Back translation not implemented, returning original")
        return text
    
    def augment(
        self,
        text: str,
        methods: Optional[List[str]] = None,
        num_augmentations: int = 1
    ) -> List[str]:
        """
        Apply augmentation methods
        
        Args:
            text: Input text
            methods: List of methods to apply (None = all)
            num_augmentations: Number of augmented versions
            
        Returns:
            List of augmented texts
        """
        if methods is None:
            methods = ["synonym_replacement", "random_deletion", "random_swap"]
        
        augmented_texts = []
        
        for _ in range(num_augmentations):
            augmented = text
            
            # Apply random method
            method = random.choice(methods)
            
            if method == "synonym_replacement":
                augmented = self.synonym_replacement(augmented)
            elif method == "random_deletion":
                augmented = self.random_deletion(augmented)
            elif method == "random_swap":
                augmented = self.random_swap(augmented)
            elif method == "back_translation":
                augmented = self.back_translation(augmented)
            
            if augmented != text:  # Only add if changed
                augmented_texts.append(augmented)
        
        return augmented_texts if augmented_texts else [text]


class AugmentedDataset:
    """Dataset with built-in augmentation"""
    
    def __init__(
        self,
        base_dataset,
        augmenter: TextAugmenter,
        augmentation_prob: float = 0.5
    ):
        """
        Initialize augmented dataset
        
        Args:
            base_dataset: Base dataset
            augmenter: Text augmenter
            augmentation_prob: Probability of augmentation
        """
        self.base_dataset = base_dataset
        self.augmenter = augmenter
        self.augmentation_prob = augmentation_prob
    
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int):
        """Get item with optional augmentation"""
        item = self.base_dataset[idx]
        
        if random.random() < self.augmentation_prob:
            text = item.get("text", "")
            if text:
                augmented = self.augmenter.augment(text, num_augmentations=1)
                if augmented:
                    item["text"] = augmented[0]
        
        return item


# Global augmenter instance
text_augmenter = TextAugmenter()




