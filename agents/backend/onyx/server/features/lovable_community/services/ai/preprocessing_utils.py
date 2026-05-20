"""
Advanced Preprocessing Utilities

Provides comprehensive text preprocessing for deep learning:
- Text cleaning and normalization
- Tokenization utilities
- Data augmentation
- Feature extraction
"""

import logging
import re
from typing import List, Dict, Any, Optional, Callable
import numpy as np
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """
    Advanced text preprocessing for deep learning models
    """
    
    def __init__(
        self,
        lowercase: bool = True,
        remove_urls: bool = True,
        remove_mentions: bool = True,
        remove_hashtags: bool = False,
        remove_extra_whitespace: bool = True,
        remove_punctuation: bool = False,
        normalize_unicode: bool = True,
        max_length: Optional[int] = None
    ):
        """
        Initialize text preprocessor
        
        Args:
            lowercase: Convert to lowercase
            remove_urls: Remove URLs
            remove_mentions: Remove @mentions
            remove_hashtags: Remove #hashtags
            remove_extra_whitespace: Remove extra whitespace
            remove_punctuation: Remove punctuation
            normalize_unicode: Normalize unicode characters
            max_length: Maximum text length
        """
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.remove_mentions = remove_mentions
        self.remove_hashtags = remove_hashtags
        self.remove_extra_whitespace = remove_extra_whitespace
        self.remove_punctuation = remove_punctuation
        self.normalize_unicode = normalize_unicode
        self.max_length = max_length
    
    def preprocess(self, text: str) -> str:
        """
        Preprocess text
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        if not text:
            return ""
        
        # Normalize unicode
        if self.normalize_unicode:
            import unicodedata
            text = unicodedata.normalize('NFKD', text)
        
        # Remove URLs
        if self.remove_urls:
            text = re.sub(r'http\S+|www\.\S+', '', text)
        
        # Remove mentions
        if self.remove_mentions:
            text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags
        if self.remove_hashtags:
            text = re.sub(r'#\w+', '', text)
        
        # Lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove punctuation
        if self.remove_punctuation:
            text = re.sub(r'[^\w\s]', '', text)
        
        # Remove extra whitespace
        if self.remove_extra_whitespace:
            text = re.sub(r'\s+', ' ', text).strip()
        
        # Truncate
        if self.max_length and len(text) > self.max_length:
            text = text[:self.max_length]
        
        return text
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """
        Preprocess batch of texts
        
        Args:
            texts: List of texts
            
        Returns:
            List of preprocessed texts
        """
        return [self.preprocess(text) for text in texts]


class TokenizationUtils:
    """
    Utilities for tokenization
    """
    
    @staticmethod
    def tokenize_with_truncation(
        tokenizer: PreTrainedTokenizer,
        text: str,
        max_length: int = 512,
        padding: bool = True,
        truncation: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize text with proper truncation and padding
        
        Args:
            tokenizer: Tokenizer to use
            text: Text to tokenize
            max_length: Maximum sequence length
            padding: Whether to pad
            truncation: Whether to truncate
            
        Returns:
            Dictionary with tokenized inputs
        """
        return tokenizer(
            text,
            max_length=max_length,
            padding="max_length" if padding else False,
            truncation=truncation,
            return_tensors="pt"
        )
    
    @staticmethod
    def batch_tokenize(
        tokenizer: PreTrainedTokenizer,
        texts: List[str],
        max_length: int = 512,
        padding: bool = True,
        truncation: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize batch of texts
        
        Args:
            tokenizer: Tokenizer to use
            texts: List of texts
            max_length: Maximum sequence length
            padding: Whether to pad
            truncation: Whether to truncate
            
        Returns:
            Dictionary with batched tokenized inputs
        """
        return tokenizer(
            texts,
            max_length=max_length,
            padding="max_length" if padding else False,
            truncation=truncation,
            return_tensors="pt"
        )
    
    @staticmethod
    def get_attention_mask(input_ids: torch.Tensor, pad_token_id: int = 0) -> torch.Tensor:
        """
        Create attention mask from input ids
        
        Args:
            input_ids: Input token ids
            pad_token_id: Padding token id
            
        Returns:
            Attention mask
        """
        return (input_ids != pad_token_id).long()


class DataAugmentation:
    """
    Data augmentation for text data
    """
    
    @staticmethod
    def synonym_replacement(text: str, n: int = 1) -> str:
        """
        Replace words with synonyms
        
        Args:
            text: Input text
            n: Number of replacements
            
        Returns:
            Augmented text
        """
        # This is a placeholder - would need a synonym library
        # For production, use libraries like nltk or spacy
        return text
    
    @staticmethod
    def random_deletion(text: str, p: float = 0.1) -> str:
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
        
        new_words = [word for word in words if np.random.random() > p]
        
        if len(new_words) == 0:
            return words[np.random.randint(0, len(words))]
        
        return ' '.join(new_words)
    
    @staticmethod
    def random_swap(text: str, n: int = 1) -> str:
        """
        Randomly swap words
        
        Args:
            text: Input text
            n: Number of swaps
            
        Returns:
            Augmented text
        """
        words = text.split()
        if len(words) <= 1:
            return text
        
        for _ in range(n):
            idx1, idx2 = np.random.choice(len(words), 2, replace=False)
            words[idx1], words[idx2] = words[idx2], words[idx1]
        
        return ' '.join(words)
    
    @staticmethod
    def back_translation(
        text: str,
        source_lang: str = "en",
        target_lang: str = "de"
    ) -> str:
        """
        Back translation augmentation
        
        Args:
            text: Input text
            source_lang: Source language
            target_lang: Target language
            
        Returns:
            Augmented text
        """
        # Placeholder - would need translation model
        return text


class FeatureExtractor:
    """
    Extract features from text for analysis
    """
    
    @staticmethod
    def extract_basic_features(text: str) -> Dict[str, Any]:
        """
        Extract basic text features
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with features
        """
        return {
            "length": len(text),
            "word_count": len(text.split()),
            "char_count": len(text),
            "sentence_count": len(re.split(r'[.!?]+', text)),
            "avg_word_length": np.mean([len(word) for word in text.split()]) if text.split() else 0,
            "has_url": bool(re.search(r'http\S+|www\.\S+', text)),
            "has_mention": bool(re.search(r'@\w+', text)),
            "has_hashtag": bool(re.search(r'#\w+', text)),
            "uppercase_ratio": sum(1 for c in text if c.isupper()) / len(text) if text else 0,
            "digit_ratio": sum(1 for c in text if c.isdigit()) / len(text) if text else 0
        }
    
    @staticmethod
    def extract_ngrams(text: str, n: int = 2) -> List[str]:
        """
        Extract n-grams from text
        
        Args:
            text: Input text
            n: N-gram size
            
        Returns:
            List of n-grams
        """
        words = text.split()
        if len(words) < n:
            return []
        
        return [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]


def create_preprocessing_pipeline(
    steps: List[Callable[[str], str]]
) -> Callable[[str], str]:
    """
    Create a preprocessing pipeline from a list of functions
    
    Args:
        steps: List of preprocessing functions
        
    Returns:
        Pipeline function
    """
    def pipeline(text: str) -> str:
        result = text
        for step in steps:
            result = step(result)
        return result
    
    return pipeline















