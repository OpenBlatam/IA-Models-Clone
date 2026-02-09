"""
Enhanced AI Engine with PyTorch, Transformers, and Diffusion Models
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
import logging

from .transformer_analyzer import (
    TransformerContentAnalyzer,
    SentimentTransformerAnalyzer,
    NERTransformerAnalyzer,
    create_transformer_analyzer
)
from .content_generator import (
    TextGenerator,
    T5ContentGenerator,
    DiffusionContentGenerator,
    create_text_generator,
    create_t5_generator
)

logger = logging.getLogger(__name__)


class EnhancedAIEngine:
    """
    Enhanced AI Engine with deep learning capabilities
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        use_gpu: bool = True
    ):
        """
        Initialize enhanced AI engine
        
        Args:
            config: Configuration dictionary
            device: PyTorch device
            use_gpu: Whether to use GPU
        """
        self.config = config or {}
        self.device = device or torch.device(
            "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        )
        
        # Initialize transformers
        self.transformer_analyzer = None
        self.sentiment_analyzer = None
        self.ner_analyzer = None
        self.text_generator = None
        self.t5_generator = None
        self.diffusion_generator = None
        
        # Load models based on config
        self._initialize_models()
        
        logger.info(f"EnhancedAIEngine initialized on {self.device}")
    
    def _initialize_models(self):
        """Initialize all models"""
        try:
            # Transformer analyzer
            if self.config.get("use_transformer_analyzer", True):
                self.transformer_analyzer = create_transformer_analyzer(
                    model_name=self.config.get("transformer_model", "bert-base-uncased"),
                    device=self.device
                )
            
            # Sentiment analyzer
            if self.config.get("use_sentiment_analyzer", True):
                self.sentiment_analyzer = SentimentTransformerAnalyzer(device=self.device)
            
            # NER analyzer
            if self.config.get("use_ner_analyzer", True):
                self.ner_analyzer = NERTransformerAnalyzer(device=self.device)
            
            # Text generator
            if self.config.get("use_text_generator", True):
                self.text_generator = create_text_generator(
                    model_name=self.config.get("text_model", "gpt2"),
                    device=self.device
                )
            
            # T5 generator
            if self.config.get("use_t5_generator", False):
                self.t5_generator = create_t5_generator(
                    model_name=self.config.get("t5_model", "t5-small"),
                    device=self.device
                )
            
            # Diffusion generator (optional, resource intensive)
            if self.config.get("use_diffusion", False):
                try:
                    self.diffusion_generator = DiffusionContentGenerator(device=self.device)
                except Exception as e:
                    logger.warning(f"Diffusion generator not available: {e}")
        
        except Exception as e:
            logger.error(f"Error initializing models: {e}", exc_info=True)
    
    def analyze_content(self, content: str) -> Dict[str, Any]:
        """
        Comprehensive content analysis
        
        Args:
            content: Content to analyze
            
        Returns:
            Analysis results
        """
        results = {}
        
        # Semantic analysis
        if self.transformer_analyzer:
            try:
                features = self.transformer_analyzer.extract_features(content)
                results["embeddings"] = features["embeddings"].cpu().numpy().tolist()
                results["semantic_features"] = features["pooler_output"].cpu().numpy().tolist()
            except Exception as e:
                logger.error(f"Semantic analysis failed: {e}")
        
        # Sentiment analysis
        if self.sentiment_analyzer:
            try:
                results["sentiment"] = self.sentiment_analyzer.analyze(content)
            except Exception as e:
                logger.error(f"Sentiment analysis failed: {e}")
        
        # Named entity recognition
        if self.ner_analyzer:
            try:
                results["entities"] = self.ner_analyzer.extract_entities(content)
            except Exception as e:
                logger.error(f"NER failed: {e}")
        
        return results
    
    def generate_content(
        self,
        prompt: str,
        max_length: int = 100,
        generator_type: str = "gpt2"
    ) -> str:
        """
        Generate content from prompt
        
        Args:
            prompt: Input prompt
            max_length: Maximum length
            generator_type: Type of generator ("gpt2" or "t5")
            
        Returns:
            Generated content
        """
        if generator_type == "t5" and self.t5_generator:
            return self.t5_generator.generate("complete:", prompt, max_length=max_length)
        elif self.text_generator:
            generated = self.text_generator.generate(
                prompt, max_length=max_length, num_return_sequences=1
            )
            return generated[0] if generated else prompt
        
        return prompt
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        if self.transformer_analyzer:
            return self.transformer_analyzer.analyze_similarity(text1, text2)
        return 0.0
    
    def suggest_additions(
        self,
        content: str,
        context: Optional[str] = None
    ) -> List[str]:
        """
        Suggest content additions
        
        Args:
            content: Current content
            context: Additional context
            
        Returns:
            List of suggested additions
        """
        suggestions = []
        
        if self.text_generator:
            # Generate suggestions based on content
            prompt = f"Continue: {content}"
            generated = self.text_generator.generate(
                prompt,
                max_length=50,
                temperature=0.8,
                num_return_sequences=3
            )
            suggestions.extend(generated)
        
        return suggestions
    
    def optimize_content(self, content: str) -> str:
        """
        Optimize content using AI
        
        Args:
            content: Content to optimize
            
        Returns:
            Optimized content
        """
        if self.t5_generator:
            # Use T5 for content improvement
            optimized = self.t5_generator.generate("improve:", content)
            return optimized
        
        return content
    
    def summarize(self, content: str) -> str:
        """
        Summarize content
        
        Args:
            content: Content to summarize
            
        Returns:
            Summary
        """
        if self.t5_generator:
            return self.t5_generator.summarize(content)
        
        return content[:200]  # Fallback
    
    def expand_content(self, content: str) -> str:
        """
        Expand content
        
        Args:
            content: Content to expand
            
        Returns:
            Expanded content
        """
        if self.t5_generator:
            return self.t5_generator.expand(content)
        
        if self.text_generator:
            return self.text_generator.complete(content, max_new_tokens=100)
        
        return content

