"""
Text Preprocessing

Utilities for preprocessing text data.
"""

import logging
import re
from typing import Optional, Callable, List

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Preprocess text data."""
    
    def __init__(self):
        """Initialize text preprocessor."""
        pass
    
    def clean(
        self,
        text: str,
        lowercase: bool = True,
        remove_punctuation: bool = False,
        remove_numbers: bool = False,
        remove_extra_spaces: bool = True
    ) -> str:
        """
        Clean text.
        
        Args:
            text: Input text
            lowercase: Convert to lowercase
            remove_punctuation: Remove punctuation
            remove_numbers: Remove numbers
            remove_extra_spaces: Remove extra spaces
            
        Returns:
            Cleaned text
        """
        if lowercase:
            text = text.lower()
        
        if remove_punctuation:
            text = re.sub(r'[^\w\s]', '', text)
        
        if remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        if remove_extra_spaces:
            text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(
        self,
        text: str,
        method: str = "whitespace"
    ) -> List[str]:
        """
        Tokenize text.
        
        Args:
            text: Input text
            method: Tokenization method ('whitespace', 'word', 'char')
            
        Returns:
            List of tokens
        """
        if method == "whitespace":
            return text.split()
        elif method == "word":
            # Word tokenization with punctuation handling
            return re.findall(r'\b\w+\b', text)
        elif method == "char":
            return list(text)
        else:
            raise ValueError(f"Unknown tokenization method: {method}")
    
    def remove_stopwords(
        self,
        text: str,
        stopwords: Optional[List[str]] = None
    ) -> str:
        """
        Remove stopwords.
        
        Args:
            text: Input text
            stopwords: List of stopwords (uses default if None)
            
        Returns:
            Text without stopwords
        """
        if stopwords is None:
            # Default English stopwords
            stopwords = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
                'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was'
            }
        
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in stopwords]
        
        return ' '.join(filtered_words)


def clean_text(
    text: str,
    **kwargs
) -> str:
    """Convenience function to clean text."""
    preprocessor = TextPreprocessor()
    return preprocessor.clean(text, **kwargs)


def tokenize_text(
    text: str,
    method: str = "whitespace"
) -> List[str]:
    """Convenience function to tokenize text."""
    preprocessor = TextPreprocessor()
    return preprocessor.tokenize(text, method)


def create_text_preprocessing_pipeline(
    steps: List[str],
    **kwargs
) -> Callable:
    """
    Create text preprocessing pipeline.
    
    Args:
        steps: List of preprocessing steps
        **kwargs: Additional preprocessing arguments
        
    Returns:
        Preprocessing function
    """
    preprocessor = TextPreprocessor()
    
    def preprocess(text: str) -> str:
        for step in steps:
            if step == "clean":
                text = preprocessor.clean(text, **kwargs)
            elif step == "remove_stopwords":
                text = preprocessor.remove_stopwords(text)
            else:
                logger.warning(f"Unknown preprocessing step: {step}")
        
        return text
    
    return preprocess



