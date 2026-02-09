"""
Sentiment Analysis Model
Uses transformer-based models for accurate sentiment detection
"""

import logging
import torch
from typing import Dict, Any, Optional, List
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from textblob import TextBlob

from .base import BaseModel

logger = logging.getLogger(__name__)


class SentimentModel(BaseModel):
    """
    Transformer-based sentiment analysis model
    """
    
    def __init__(
        self,
        model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
        device: Optional[torch.device] = None,
        use_mixed_precision: bool = False,
    ):
        """
        Initialize sentiment model
        
        Args:
            model_name: HuggingFace model identifier
            device: PyTorch device
            use_mixed_precision: Use mixed precision for inference
        """
        super().__init__(model_name, device, use_mixed_precision)
        self.transformer_model_name = model_name
    
    async def load(self) -> None:
        """Load sentiment analysis pipeline"""
        if self.is_loaded:
            return
        
        try:
            logger.info(f"Loading sentiment model: {self.transformer_model_name}")
            self.model = pipeline(
                "sentiment-analysis",
                model=self.transformer_model_name,
                device=0 if self.device.type == "cuda" else -1,
                return_all_scores=True,
            )
            self.is_loaded = True
            logger.info(f"Successfully loaded sentiment model: {self.transformer_model_name}")
        except Exception as e:
            logger.error(f"Error loading sentiment model: {e}")
            raise
    
    async def predict(self, inputs: str) -> Dict[str, Any]:
        """
        Analyze sentiment of input text
        
        Args:
            inputs: Text to analyze
            
        Returns:
            Dictionary with sentiment analysis results
        """
        if not self.is_loaded:
            await self.load()
        
        try:
            # Use transformer model
            results = self.model(inputs)
            
            # Extract sentiment scores
            sentiment_scores = {}
            for result in results[0]:
                sentiment_scores[result['label']] = result['score']
            
            # Determine dominant sentiment
            dominant_sentiment = max(sentiment_scores, key=sentiment_scores.get)
            
            # Also use TextBlob for additional metrics
            blob = TextBlob(inputs)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            return {
                "dominant_sentiment": dominant_sentiment,
                "sentiment_scores": sentiment_scores,
                "polarity": float(polarity),
                "subjectivity": float(subjectivity),
                "confidence": float(sentiment_scores[dominant_sentiment]),
            }
        except Exception as e:
            logger.error(f"Error in sentiment prediction: {e}")
            raise
    
    async def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze sentiment for multiple texts
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of sentiment analysis results
        """
        if not self.is_loaded:
            await self.load()
        
        results = []
        for text in texts:
            result = await self.predict(text)
            results.append(result)
        
        return results



