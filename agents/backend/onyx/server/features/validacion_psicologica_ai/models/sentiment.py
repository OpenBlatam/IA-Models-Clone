"""
Sentiment Analysis Model
========================
Sentiment analysis using transformers
"""

from typing import List, Optional, Dict
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import structlog

from .base import BaseModel
from ..config_loader import config_loader

logger = structlog.get_logger()


class SentimentTransformerModel(BaseModel):
    """
    Sentiment analysis model using transformers
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        num_labels: Optional[int] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize sentiment model
        
        Args:
            model_name: Model name (uses config if None)
            num_labels: Number of labels (uses config if None)
            device: Device (auto-detect if None)
        """
        super().__init__(device)
        
        # Load config
        model_config = config_loader.get_model_config("sentiment")
        model_name = model_name or model_config.get("name", "cardiffnlp/twitter-roberta-base-sentiment-latest")
        num_labels = num_labels or model_config.get("num_labels", 3)
        self.max_length = model_config.get("max_length", 512)
        
        # Load model
        self.tokenizer = None
        self.model = None
        self._load_model(model_name, num_labels)
    
    def _load_model(self, model_name: str, num_labels: int) -> None:
        """Load model with error handling"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels
            )
            self.model = self.model.to(self.device)
            self.model.eval()
            logger.info("Sentiment model loaded", model_name=model_name)
        except Exception as e:
            logger.error("Error loading sentiment model", error=str(e), model_name=model_name)
            self.tokenizer = None
            self.model = None
    
    def forward(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            texts: List of texts
            
        Returns:
            Dictionary with logits and predictions
        """
        if not texts or self.model is None:
            batch_size = len(texts) if texts else 1
            return {
                "logits": torch.zeros(batch_size, 3, device=self.device),
                "predictions": torch.zeros(batch_size, device=self.device, dtype=torch.long)
            }
        
        try:
            # Tokenize
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**encoded)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
            
            return {
                "logits": logits,
                "predictions": predictions
            }
        except Exception as e:
            logger.error("Error in sentiment forward", error=str(e))
            batch_size = len(texts)
            return {
                "logits": torch.zeros(batch_size, 3, device=self.device),
                "predictions": torch.zeros(batch_size, device=self.device, dtype=torch.long)
            }




