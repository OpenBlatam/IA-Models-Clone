"""
Topic Modeling Model
Uses LDA and other techniques for topic extraction
"""

import logging
import torch
import numpy as np
from typing import Dict, Any, Optional, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import spacy

from .base import BaseModel

logger = logging.getLogger(__name__)


class TopicModelingModel(BaseModel):
    """
    Topic modeling using LDA and TF-IDF
    """
    
    def __init__(
        self,
        model_name: str = "lda_topic_model",
        device: Optional[torch.device] = None,
        use_mixed_precision: bool = False,
        n_components: int = 10,
    ):
        """
        Initialize topic modeling
        
        Args:
            model_name: Model identifier
            device: PyTorch device (not used for LDA but kept for consistency)
            use_mixed_precision: Not applicable for LDA
            n_components: Number of topics
        """
        super().__init__(model_name, device, use_mixed_precision)
        self.n_components = n_components
        self.vectorizer = None
        self.lda_model = None
        self.nlp = None
    
    async def load(self) -> None:
        """Load spaCy model and initialize vectorizer"""
        if self.is_loaded:
            return
        
        try:
            logger.info("Loading spaCy model for topic modeling")
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy English model not found. Please install: python -m spacy download en_core_web_sm")
                raise
            
            # Initialize TF-IDF vectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            # Initialize LDA model
            self.lda_model = LatentDirichletAllocation(
                n_components=self.n_components,
                random_state=42,
                max_iter=10
            )
            
            self.is_loaded = True
            logger.info("Successfully initialized topic modeling components")
        except Exception as e:
            logger.error(f"Error loading topic modeling components: {e}")
            raise
    
    def _preprocess_texts(self, texts: List[str]) -> List[str]:
        """Preprocess texts using spaCy"""
        processed_texts = []
        for text in texts:
            doc = self.nlp(text)
            tokens = [
                token.lemma_.lower()
                for token in doc
                if not token.is_stop and not token.is_punct and token.is_alpha
            ]
            processed_texts.append(' '.join(tokens))
        return processed_texts
    
    async def predict(
        self,
        inputs: List[str],
        num_topics: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Extract topics from texts
        
        Args:
            inputs: List of texts
            num_topics: Number of topics to extract (overrides n_components if provided)
            
        Returns:
            Dictionary with extracted topics
        """
        if not self.is_loaded:
            await self.load()
        
        try:
            # Use provided num_topics or default
            n_topics = num_topics or self.n_components
            
            # Preprocess texts
            processed_texts = self._preprocess_texts(inputs)
            
            # Create TF-IDF matrix
            tfidf_matrix = self.vectorizer.fit_transform(processed_texts)
            
            # Fit LDA model with specified number of topics
            lda_model = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                max_iter=10
            )
            lda_model.fit(tfidf_matrix)
            
            # Extract topics
            feature_names = self.vectorizer.get_feature_names_out()
            topics = []
            
            for topic_idx, topic in enumerate(lda_model.components_):
                top_words_idx = topic.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                topics.append({
                    "topic_id": topic_idx,
                    "words": top_words,
                    "weights": topic[top_words_idx].tolist()
                })
            
            return {
                "topics": topics,
                "num_topics": n_topics,
            }
        except Exception as e:
            logger.error(f"Error in topic extraction: {e}")
            raise



