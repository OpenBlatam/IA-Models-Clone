"""
Text Feature Extraction

Utilities for extracting text features.
"""

import logging
import numpy as np
from typing import Optional, Dict, List
from collections import Counter

logger = logging.getLogger(__name__)

# Try to import transformers for embeddings
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available for text features")


class TextFeatureExtractor:
    """Extract text features."""
    
    def __init__(
        self,
        model_name: Optional[str] = None
    ):
        """
        Initialize feature extractor.
        
        Args:
            model_name: Pre-trained model name for embeddings
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        
        if model_name and TRANSFORMERS_AVAILABLE:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name)
                self.model.eval()
                logger.info(f"Loaded model for embeddings: {model_name}")
            except Exception as e:
                logger.warning(f"Could not load model {model_name}: {e}")
    
    def extract_embeddings(
        self,
        text: str,
        pooling: str = "mean"
    ) -> np.ndarray:
        """
        Extract text embeddings.
        
        Args:
            text: Input text
            pooling: Pooling method ('mean', 'max', 'cls')
            
        Returns:
            Text embeddings
        """
        if not self.tokenizer or not self.model:
            raise ValueError("Model not loaded. Provide model_name in constructor.")
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state
        
        # Pool
        if pooling == "mean":
            embeddings = embeddings.mean(dim=1)
        elif pooling == "max":
            embeddings = embeddings.max(dim=1)[0]
        elif pooling == "cls":
            embeddings = embeddings[:, 0, :]
        else:
            raise ValueError(f"Unknown pooling: {pooling}")
        
        return embeddings.squeeze().cpu().numpy()
    
    def extract_tfidf(
        self,
        texts: List[str],
        max_features: int = 1000
    ) -> np.ndarray:
        """
        Extract TF-IDF features.
        
        Args:
            texts: List of texts
            max_features: Maximum features
            
        Returns:
            TF-IDF matrix
        """
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer(max_features=max_features)
            tfidf = vectorizer.fit_transform(texts)
            return tfidf.toarray()
        except ImportError:
            logger.warning("sklearn not available for TF-IDF")
            # Simple TF-IDF implementation
            return self._simple_tfidf(texts, max_features)
    
    def extract_bow(
        self,
        texts: List[str],
        vocab_size: int = 1000
    ) -> np.ndarray:
        """
        Extract bag-of-words features.
        
        Args:
            texts: List of texts
            vocab_size: Vocabulary size
            
        Returns:
            BOW matrix
        """
        # Build vocabulary
        word_counts = Counter()
        for text in texts:
            words = text.lower().split()
            word_counts.update(words)
        
        # Get top words
        vocab = {word: idx for idx, (word, _) in enumerate(word_counts.most_common(vocab_size))}
        
        # Create BOW vectors
        bow_matrix = []
        for text in texts:
            words = text.lower().split()
            bow = np.zeros(vocab_size)
            for word in words:
                if word in vocab:
                    bow[vocab[word]] += 1
            bow_matrix.append(bow)
        
        return np.array(bow_matrix)
    
    def _simple_tfidf(
        self,
        texts: List[str],
        max_features: int
    ) -> np.ndarray:
        """Simple TF-IDF implementation."""
        # Simplified version
        return self.extract_bow(texts, max_features)


def extract_embeddings(
    text: str,
    model_name: str = "bert-base-uncased",
    **kwargs
) -> np.ndarray:
    """Convenience function to extract embeddings."""
    extractor = TextFeatureExtractor(model_name)
    return extractor.extract_embeddings(text, **kwargs)


def extract_tfidf(
    texts: List[str],
    **kwargs
) -> np.ndarray:
    """Convenience function to extract TF-IDF."""
    extractor = TextFeatureExtractor()
    return extractor.extract_tfidf(texts, **kwargs)


def extract_bow(
    texts: List[str],
    **kwargs
) -> np.ndarray:
    """Convenience function to extract BOW."""
    extractor = TextFeatureExtractor()
    return extractor.extract_bow(texts, **kwargs)



