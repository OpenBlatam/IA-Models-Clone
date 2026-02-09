"""
Core Analyzer
=============
Main analyzer that integrates all models
Refactored for better organization
"""

from typing import Dict, Any, List, Optional
import torch
import structlog

from ..models import (
    PsychologicalEmbeddingModel,
    PersonalityClassifier,
    SentimentTransformerModel
)
from ..config_loader import config_loader

logger = structlog.get_logger()


class DeepLearningAnalyzer:
    """
    Main analyzer that integrates all deep learning models
    Refactored with better organization and error handling
    """
    
    def __init__(self):
        """Initialize analyzer"""
        self.device = self._get_device()
        
        # Initialize models
        self.embedding_model = PsychologicalEmbeddingModel(device=self.device)
        self.personality_classifier = PersonalityClassifier(device=self.device)
        self.sentiment_model = SentimentTransformerModel(device=self.device)
        
        logger.info("DeepLearningAnalyzer initialized", device=str(self.device))
    
    def _get_device(self) -> torch.device:
        """Get appropriate device"""
        device_config = config_loader.get_device_config()
        
        if device_config.get("use_cuda", True) and torch.cuda.is_available():
            device_id = device_config.get("cuda_device", 0)
            return torch.device(f"cuda:{device_id}")
        elif device_config.get("allow_mps", False) and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    async def analyze_comprehensive(
        self,
        texts: List[str],
        include_llm: bool = True
    ) -> Dict[str, Any]:
        """
        Comprehensive analysis using all models
        
        Args:
            texts: List of texts to analyze
            include_llm: Include LLM analysis (if available)
            
        Returns:
            Comprehensive analysis results
        """
        try:
            results = {
                "embeddings": None,
                "personality": None,
                "sentiment": None,
                "llm_analysis": None
            }
            
            # Get embeddings
            embeddings = self.embedding_model(texts)
            results["embeddings"] = embeddings.cpu().numpy().tolist()
            
            # Get personality traits
            personality = self.personality_classifier(texts)
            results["personality"] = {
                trait: pred.cpu().numpy().tolist()
                for trait, pred in personality.items()
            }
            
            # Get sentiment
            sentiment = self.sentiment_model(texts)
            results["sentiment"] = {
                "logits": sentiment["logits"].cpu().numpy().tolist(),
                "predictions": sentiment["predictions"].cpu().numpy().tolist()
            }
            
            # LLM analysis (if enabled and available)
            if include_llm:
                try:
                    from ..ai_integrations import ai_service_manager
                    llm_results = await ai_service_manager.analyze_texts(texts)
                    results["llm_analysis"] = llm_results
                except Exception as e:
                    logger.warning("LLM analysis not available", error=str(e))
            
            return results
        except Exception as e:
            logger.error("Error in comprehensive analysis", error=str(e))
            raise


# Global instance
deep_learning_analyzer = DeepLearningAnalyzer()




