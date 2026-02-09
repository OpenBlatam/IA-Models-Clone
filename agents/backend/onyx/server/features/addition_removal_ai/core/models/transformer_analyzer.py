"""
Transformer-based Content Analyzer using HuggingFace Transformers
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import logging

try:
    from transformers import (
        AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
        AutoModelForTokenClassification, pipeline
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


class TransformerContentAnalyzer(nn.Module):
    """
    Transformer-based analyzer for content understanding and analysis
    """
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        device: Optional[torch.device] = None,
        use_gpu: bool = True
    ):
        """
        Initialize transformer analyzer
        
        Args:
            model_name: HuggingFace model name
            device: PyTorch device
            use_gpu: Whether to use GPU
        """
        super().__init__()
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library is required")
        
        self.device = device or torch.device(
            "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        )
        self.model_name = model_name
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        logger.info(f"TransformerContentAnalyzer initialized with {model_name} on {self.device}")
    
    def encode(self, text: str, max_length: int = 512) -> torch.Tensor:
        """
        Encode text to embeddings
        
        Args:
            text: Input text
            max_length: Maximum sequence length
            
        Returns:
            Text embeddings
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use [CLS] token embedding
            embeddings = outputs.last_hidden_state[:, 0, :]
        
        return embeddings
    
    def analyze_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        emb1 = self.encode(text1)
        emb2 = self.encode(text2)
        
        # Cosine similarity
        similarity = torch.nn.functional.cosine_similarity(emb1, emb2)
        return similarity.item()
    
    def extract_features(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Extract features from text
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of features
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        return {
            "embeddings": outputs.last_hidden_state,
            "pooler_output": outputs.pooler_output,
            "attention": outputs.attentions if hasattr(outputs, 'attentions') else None
        }


class SentimentTransformerAnalyzer(nn.Module):
    """Transformer-based sentiment analysis"""
    
    def __init__(
        self,
        model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
        device: Optional[torch.device] = None
    ):
        super().__init__()
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library is required")
        
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        
        # Load model for classification
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()
        
        logger.info(f"SentimentTransformerAnalyzer initialized on {self.device}")
    
    def analyze(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment
        
        Args:
            text: Input text
            
        Returns:
            Sentiment scores
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
        
        # Map to labels (adjust based on model)
        labels = ["negative", "neutral", "positive"]
        scores = {label: float(prob) for label, prob in zip(labels, probs[0])}
        
        return scores


class NERTransformerAnalyzer(nn.Module):
    """Transformer-based Named Entity Recognition"""
    
    def __init__(
        self,
        model_name: str = "dbmdz/bert-large-cased-finetuned-conll03-english",
        device: Optional[torch.device] = None
    ):
        super().__init__()
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library is required")
        
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Use pipeline for NER
        self.ner_pipeline = pipeline(
            "ner",
            model=model_name,
            device=0 if self.device.type == "cuda" else -1,
            aggregation_strategy="simple"
        )
        
        logger.info(f"NERTransformerAnalyzer initialized on {self.device}")
    
    def extract_entities(self, text: str) -> List[Dict]:
        """
        Extract named entities
        
        Args:
            text: Input text
            
        Returns:
            List of entities
        """
        results = self.ner_pipeline(text)
        return results


def create_transformer_analyzer(
    model_name: str = "bert-base-uncased",
    device: Optional[torch.device] = None
) -> TransformerContentAnalyzer:
    """Factory function to create transformer analyzer"""
    return TransformerContentAnalyzer(model_name=model_name, device=device)

