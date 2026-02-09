"""
Refactored AI/ML Engine
Following PyTorch best practices with proper model management, GPU support, and async patterns
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import hashlib
import torch

from core.config import settings
from ml.models import (
    ModelManager,
    EmbeddingModel,
    SentimentModel,
    SummarizationModel,
    TopicModelingModel,
)

logger = logging.getLogger(__name__)


class AIMLEngine:
    """
    Enhanced AI/ML Engine with proper PyTorch patterns
    - GPU utilization with automatic device detection
    - Mixed precision support
    - Model caching and lifecycle management
    - Proper async/await patterns
    """
    
    def __init__(
        self,
        enable_gpu: Optional[bool] = None,
        use_mixed_precision: Optional[bool] = None,
        model_cache_size: int = 10,
    ):
        """
        Initialize AI/ML Engine
        
        Args:
            enable_gpu: Enable GPU if available (auto-detect if None)
            use_mixed_precision: Use mixed precision training/inference
            model_cache_size: Maximum number of models to cache
        """
        self.enable_gpu = enable_gpu if enable_gpu is not None else settings.enable_gpu
        self.use_mixed_precision = (
            use_mixed_precision
            if use_mixed_precision is not None
            else getattr(settings, "use_mixed_precision", False)
        )
        
        # Device management
        self.device = self._get_device()
        
        # Model manager
        self.model_manager = ModelManager(max_cache_size=model_cache_size)
        
        # Initialize models lazily
        self.initialized = False
        
        logger.info(
            f"Initialized AI/ML Engine - Device: {self.device}, "
            f"Mixed Precision: {self.use_mixed_precision}"
        )
    
    def _get_device(self) -> torch.device:
        """Get appropriate device based on settings and availability"""
        if self.enable_gpu and torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU")
        return device
    
    async def initialize(self) -> None:
        """Initialize AI/ML models and components"""
        if self.initialized:
            return
        
        try:
            logger.info("Initializing AI/ML Engine...")
            
            # Pre-load commonly used models
            if getattr(settings, "preload_models", False):
                await self._preload_models()
            
            self.initialized = True
            logger.info("AI/ML Engine initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing AI/ML Engine: {e}", exc_info=True)
            raise
    
    async def _preload_models(self) -> None:
        """Pre-load commonly used models"""
        logger.info("Pre-loading commonly used models...")
        try:
            # Pre-load embedding model (most commonly used)
            await self.model_manager.get_model(
                EmbeddingModel,
                model_name=settings.embedding_model,
                device=self.device,
                use_mixed_precision=self.use_mixed_precision,
            )
            logger.info("Pre-loaded embedding model")
        except Exception as e:
            logger.warning(f"Error pre-loading models: {e}")
    
    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of text using transformer models
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment analysis results
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            model = await self.model_manager.get_model(
                SentimentModel,
                model_name=settings.sentiment_model if hasattr(settings, "sentiment_model") else "cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=self.device,
                use_mixed_precision=self.use_mixed_precision,
            )
            
            result = await model.predict(text)
            result["timestamp"] = datetime.now().isoformat()
            
            return result
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}", exc_info=True)
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def detect_language(self, text: str) -> Dict[str, Any]:
        """
        Detect language of text
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with language detection results
        """
        try:
            from langdetect import detect, LangDetectException
            
            language = detect(text)
            confidence = 1.0  # langdetect doesn't provide confidence scores
            
            return {
                "language": language,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat()
            }
        except LangDetectException:
            return {
                "language": "unknown",
                "confidence": 0.0,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error in language detection: {e}", exc_info=True)
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def extract_topics(self, texts: List[str], num_topics: int = 5) -> Dict[str, Any]:
        """
        Extract topics from a collection of texts using LDA
        
        Args:
            texts: List of texts
            num_topics: Number of topics to extract
            
        Returns:
            Dictionary with extracted topics
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            model = await self.model_manager.get_model(
                TopicModelingModel,
                model_name="lda_topic_model",
                device=self.device,
                use_mixed_precision=self.use_mixed_precision,
                n_components=num_topics,
            )
            
            result = await model.predict(texts, num_topics=num_topics)
            result["timestamp"] = datetime.now().isoformat()
            
            return result
        except Exception as e:
            logger.error(f"Error in topic extraction: {e}", exc_info=True)
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def calculate_semantic_similarity(
        self,
        text1: str,
        text2: str,
    ) -> Dict[str, Any]:
        """
        Calculate semantic similarity using sentence transformers
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Dictionary with similarity results
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            model = await self.model_manager.get_model(
                EmbeddingModel,
                model_name=settings.embedding_model,
                device=self.device,
                use_mixed_precision=self.use_mixed_precision,
            )
            
            similarity = await model.compute_similarity(text1, text2)
            
            return {
                "similarity_score": float(similarity),
                "similarity_percentage": float(similarity * 100),
                "method": "sentence_transformer",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error in semantic similarity calculation: {e}", exc_info=True)
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def detect_plagiarism(
        self,
        text: str,
        reference_texts: List[str],
        threshold: float = 0.8,
    ) -> Dict[str, Any]:
        """
        Detect potential plagiarism by comparing with reference texts
        
        Args:
            text: Text to check
            reference_texts: Reference texts to compare against
            threshold: Similarity threshold
            
        Returns:
            Dictionary with plagiarism detection results
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            model = await self.model_manager.get_model(
                EmbeddingModel,
                model_name=settings.embedding_model,
                device=self.device,
                use_mixed_precision=self.use_mixed_precision,
            )
            
            similarities = []
            for i, ref_text in enumerate(reference_texts):
                similarity = await model.compute_similarity(text, ref_text)
                similarities.append({
                    "reference_index": i,
                    "similarity_score": float(similarity),
                    "is_plagiarized": float(similarity) >= threshold
                })
            
            # Find highest similarity
            max_similarity = max(similarities, key=lambda x: x["similarity_score"]) if similarities else None
            
            return {
                "is_plagiarized": max_similarity["is_plagiarized"] if max_similarity else False,
                "max_similarity": max_similarity["similarity_score"] if max_similarity else 0.0,
                "similarities": similarities,
                "threshold": threshold,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error in plagiarism detection: {e}", exc_info=True)
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def extract_entities(self, text: str) -> Dict[str, Any]:
        """
        Extract named entities from text using spaCy
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with extracted entities
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            import spacy
            
            # Load spaCy model
            try:
                nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.error("spaCy English model not found. Please install: python -m spacy download en_core_web_sm")
                raise
            
            doc = nlp(text)
            
            entities = []
            for ent in doc.ents:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "description": spacy.explain(ent.label_)
                })
            
            return {
                "entities": entities,
                "entity_count": len(entities),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error in entity extraction: {e}", exc_info=True)
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def generate_summary(
        self,
        text: str,
        max_length: int = 150,
    ) -> Dict[str, Any]:
        """
        Generate text summary using BART model
        
        Args:
            text: Text to summarize
            max_length: Maximum length of summary
            
        Returns:
            Dictionary with summary results
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            model = await self.model_manager.get_model(
                SummarizationModel,
                model_name=settings.summarization_model if hasattr(settings, "summarization_model") else "facebook/bart-large-cnn",
                device=self.device,
                use_mixed_precision=self.use_mixed_precision,
            )
            
            result = await model.predict(text, max_length=max_length)
            result["timestamp"] = datetime.now().isoformat()
            
            return result
        except Exception as e:
            logger.error(f"Error in text summarization: {e}", exc_info=True)
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def analyze_readability(self, text: str) -> Dict[str, Any]:
        """
        Advanced readability analysis using multiple metrics
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with readability metrics
        """
        try:
            from textblob import TextBlob
            
            blob = TextBlob(text)
            
            # Basic metrics
            sentences = blob.sentences
            words = blob.words
            characters = len(text)
            
            # Calculate readability metrics
            avg_sentence_length = (
                sum(len(sentence.words) for sentence in sentences) / len(sentences)
                if sentences else 0
            )
            avg_word_length = (
                sum(len(word) for word in words) / len(words)
                if words else 0
            )
            
            # Flesch Reading Ease (simplified)
            flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_word_length)
            
            # Grade level estimation
            grade_level = (0.39 * avg_sentence_length) + (11.8 * avg_word_length) - 15.59
            
            return {
                "flesch_score": flesch_score,
                "grade_level": grade_level,
                "avg_sentence_length": avg_sentence_length,
                "avg_word_length": avg_word_length,
                "sentence_count": len(sentences),
                "word_count": len(words),
                "character_count": characters,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error in readability analysis: {e}", exc_info=True)
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def comprehensive_analysis(self, text: str) -> Dict[str, Any]:
        """
        Perform comprehensive analysis combining all AI/ML features
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with comprehensive analysis results
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            # Run all analyses in parallel for better performance
            tasks = [
                self.analyze_sentiment(text),
                self.detect_language(text),
                self.extract_entities(text),
                self.generate_summary(text),
                self.analyze_readability(text)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Combine results
            analysis = {
                "text_hash": hashlib.md5(text.encode()).hexdigest(),
                "text_length": len(text),
                "sentiment": (
                    results[0] if not isinstance(results[0], Exception)
                    else {"error": str(results[0])}
                ),
                "language": (
                    results[1] if not isinstance(results[1], Exception)
                    else {"error": str(results[1])}
                ),
                "entities": (
                    results[2] if not isinstance(results[2], Exception)
                    else {"error": str(results[2])}
                ),
                "summary": (
                    results[3] if not isinstance(results[3], Exception)
                    else {"error": str(results[3])}
                ),
                "readability": (
                    results[4] if not isinstance(results[4], Exception)
                    else {"error": str(results[4])}
                ),
                "timestamp": datetime.now().isoformat()
            }
            
            return analysis
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {e}", exc_info=True)
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_model_cache_stats(self) -> Dict[str, Any]:
        """Get model cache statistics"""
        return self.model_manager.get_cache_stats()
    
    def clear_model_cache(self) -> None:
        """Clear model cache"""
        self.model_manager.clear_cache()


# Global AI/ML Engine instance
ai_ml_engine = AIMLEngine()



