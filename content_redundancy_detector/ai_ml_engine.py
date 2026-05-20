"""
Advanced AI/ML Engine for Content Analysis
"""

import asyncio
import time
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import Counter
import re
import math

logger = logging.getLogger(__name__)


class MLModelType(Enum):
    """ML Model types"""
    SIMILARITY = "similarity"
    QUALITY = "quality"
    SENTIMENT = "sentiment"
    TOPIC_MODELING = "topic_modeling"
    CLASSIFICATION = "classification"
    CLUSTERING = "clustering"


class AIModelType(Enum):
    """AI Model types"""
    GPT = "gpt"
    BERT = "bert"
    TRANSFORMER = "transformer"
    EMBEDDING = "embedding"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"


@dataclass
class MLModel:
    """ML Model configuration"""
    id: str
    name: str
    model_type: MLModelType
    version: str
    accuracy: float
    training_data_size: int
    last_trained: float
    parameters: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True


@dataclass
class AIModel:
    """AI Model configuration"""
    id: str
    name: str
    model_type: AIModelType
    provider: str
    api_key: Optional[str] = None
    endpoint: Optional[str] = None
    max_tokens: int = 1000
    temperature: float = 0.7
    is_active: bool = True


@dataclass
class PredictionResult:
    """ML/AI Prediction result"""
    model_id: str
    model_name: str
    prediction: Any
    confidence: float
    processing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdvancedMLProcessor:
    """Advanced ML processing engine"""
    
    def __init__(self):
        self._models: Dict[str, MLModel] = {}
        self._ai_models: Dict[str, AIModel] = {}
        self._embeddings_cache: Dict[str, List[float]] = {}
        self._similarity_matrix: Dict[str, Dict[str, float]] = {}
    
    def register_ml_model(self, model: MLModel) -> None:
        """Register an ML model"""
        self._models[model.id] = model
        logger.info(f"ML model registered: {model.name} ({model.model_type.value})")
    
    def register_ai_model(self, model: AIModel) -> None:
        """Register an AI model"""
        self._ai_models[model.id] = model
        logger.info(f"AI model registered: {model.name} ({model.model_type.value})")
    
    async def predict_similarity(self, text1: str, text2: str, model_id: str = "default") -> PredictionResult:
        """Predict similarity using ML model"""
        start_time = time.time()
        
        try:
            # Get embeddings
            embedding1 = await self._get_embedding(text1)
            embedding2 = await self._get_embedding(text2)
            
            # Calculate cosine similarity
            similarity = self._cosine_similarity(embedding1, embedding2)
            
            # Apply ML model if available
            if model_id in self._models:
                model = self._models[model_id]
                if model.model_type == MLModelType.SIMILARITY:
                    # Apply model-specific transformations
                    similarity = self._apply_ml_model(similarity, model)
            
            processing_time = time.time() - start_time
            
            return PredictionResult(
                model_id=model_id,
                model_name=self._models.get(model_id, MLModel("default", "Default", MLModelType.SIMILARITY, "1.0", 0.0, 0.0)).name,
                prediction=similarity,
                confidence=min(similarity * 1.2, 1.0),
                processing_time=processing_time,
                metadata={
                    "embedding_dimension": len(embedding1),
                    "model_type": "similarity"
                }
            )
            
        except Exception as e:
            logger.error(f"Error in similarity prediction: {e}")
            raise
    
    async def predict_quality(self, content: str, model_id: str = "default") -> PredictionResult:
        """Predict content quality using ML model"""
        start_time = time.time()
        
        try:
            # Extract features
            features = self._extract_quality_features(content)
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(features)
            
            # Apply ML model if available
            if model_id in self._models:
                model = self._models[model_id]
                if model.model_type == MLModelType.QUALITY:
                    quality_score = self._apply_ml_model(quality_score, model)
            
            processing_time = time.time() - start_time
            
            return PredictionResult(
                model_id=model_id,
                model_name=self._models.get(model_id, MLModel("default", "Default", MLModelType.QUALITY, "1.0", 0.0, 0.0)).name,
                prediction=quality_score,
                confidence=0.85,  # Quality predictions are generally reliable
                processing_time=processing_time,
                metadata={
                    "features_extracted": len(features),
                    "model_type": "quality"
                }
            )
            
        except Exception as e:
            logger.error(f"Error in quality prediction: {e}")
            raise
    
    async def predict_sentiment(self, content: str, model_id: str = "default") -> PredictionResult:
        """Predict sentiment using ML model"""
        start_time = time.time()
        
        try:
            # Extract sentiment features
            sentiment_features = self._extract_sentiment_features(content)
            
            # Calculate sentiment score
            sentiment_score = self._calculate_sentiment_score(sentiment_features)
            
            # Apply ML model if available
            if model_id in self._models:
                model = self._models[model_id]
                if model.model_type == MLModelType.SENTIMENT:
                    sentiment_score = self._apply_ml_model(sentiment_score, model)
            
            processing_time = time.time() - start_time
            
            return PredictionResult(
                model_id=model_id,
                model_name=self._models.get(model_id, MLModel("default", "Default", MLModelType.SENTIMENT, "1.0", 0.0, 0.0)).name,
                prediction=sentiment_score,
                confidence=0.80,
                processing_time=processing_time,
                metadata={
                    "sentiment_features": len(sentiment_features),
                    "model_type": "sentiment"
                }
            )
            
        except Exception as e:
            logger.error(f"Error in sentiment prediction: {e}")
            raise
    
    async def predict_topics(self, content: str, num_topics: int = 5, model_id: str = "default") -> PredictionResult:
        """Predict topics using ML model"""
        start_time = time.time()
        
        try:
            # Extract topic features
            topic_features = self._extract_topic_features(content)
            
            # Calculate topic distribution
            topics = self._calculate_topic_distribution(topic_features, num_topics)
            
            # Apply ML model if available
            if model_id in self._models:
                model = self._models[model_id]
                if model.model_type == MLModelType.TOPIC_MODELING:
                    topics = self._apply_ml_model(topics, model)
            
            processing_time = time.time() - start_time
            
            return PredictionResult(
                model_id=model_id,
                model_name=self._models.get(model_id, MLModel("default", "Default", MLModelType.TOPIC_MODELING, "1.0", 0.0, 0.0)).name,
                prediction=topics,
                confidence=0.75,
                processing_time=processing_time,
                metadata={
                    "num_topics": num_topics,
                    "model_type": "topic_modeling"
                }
            )
            
        except Exception as e:
            logger.error(f"Error in topic prediction: {e}")
            raise
    
    async def cluster_content(self, contents: List[str], num_clusters: int = 3, model_id: str = "default") -> PredictionResult:
        """Cluster content using ML model"""
        start_time = time.time()
        
        try:
            # Get embeddings for all contents
            embeddings = []
            for content in contents:
                embedding = await self._get_embedding(content)
                embeddings.append(embedding)
            
            # Perform clustering
            clusters = self._perform_clustering(embeddings, num_clusters)
            
            # Apply ML model if available
            if model_id in self._models:
                model = self._models[model_id]
                if model.model_type == MLModelType.CLUSTERING:
                    clusters = self._apply_ml_model(clusters, model)
            
            processing_time = time.time() - start_time
            
            return PredictionResult(
                model_id=model_id,
                model_name=self._models.get(model_id, MLModel("default", "Default", MLModelType.CLUSTERING, "1.0", 0.0, 0.0)).name,
                prediction=clusters,
                confidence=0.70,
                processing_time=processing_time,
                metadata={
                    "num_clusters": num_clusters,
                    "num_contents": len(contents),
                    "model_type": "clustering"
                }
            )
            
        except Exception as e:
            logger.error(f"Error in content clustering: {e}")
            raise
    
    async def generate_ai_response(self, prompt: str, model_id: str = "default") -> PredictionResult:
        """Generate AI response using AI model"""
        start_time = time.time()
        
        try:
            # Get AI model
            if model_id not in self._ai_models:
                raise ValueError(f"AI model {model_id} not found")
            
            ai_model = self._ai_models[model_id]
            
            # Generate response based on model type
            if ai_model.model_type == AIModelType.SUMMARIZATION:
                response = await self._generate_summary(prompt, ai_model)
            elif ai_model.model_type == AIModelType.TRANSLATION:
                response = await self._generate_translation(prompt, ai_model)
            else:
                response = await self._generate_general_response(prompt, ai_model)
            
            processing_time = time.time() - start_time
            
            return PredictionResult(
                model_id=model_id,
                model_name=ai_model.name,
                prediction=response,
                confidence=0.85,
                processing_time=processing_time,
                metadata={
                    "model_type": ai_model.model_type.value,
                    "provider": ai_model.provider
                }
            )
            
        except Exception as e:
            logger.error(f"Error in AI response generation: {e}")
            raise
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get text embedding (simplified implementation)"""
        # Check cache first
        text_hash = str(hash(text))
        if text_hash in self._embeddings_cache:
            return self._embeddings_cache[text_hash]
        
        # Generate embedding (simplified TF-IDF-like approach)
        words = re.findall(r'\b\w+\b', text.lower())
        word_counts = Counter(words)
        total_words = len(words)
        
        # Create embedding vector (simplified)
        embedding = []
        for i in range(100):  # 100-dimensional embedding
            if i < len(words):
                word = words[i % len(words)]
                tf = word_counts[word] / total_words
                embedding.append(tf)
            else:
                embedding.append(0.0)
        
        # Cache embedding
        self._embeddings_cache[text_hash] = embedding
        
        return embedding
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have the same length")
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(a * a for a in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def _extract_quality_features(self, content: str) -> Dict[str, Any]:
        """Extract quality features from content"""
        words = re.findall(r'\b\w+\b', content)
        sentences = re.split(r'[.!?]+', content)
        
        return {
            "word_count": len(words),
            "sentence_count": len([s for s in sentences if s.strip()]),
            "avg_sentence_length": len(words) / max(len([s for s in sentences if s.strip()]), 1),
            "unique_words": len(set(words)),
            "vocabulary_richness": len(set(words)) / max(len(words), 1),
            "punctuation_ratio": len(re.findall(r'[.!?,;:]', content)) / max(len(content), 1),
            "capitalization_ratio": len(re.findall(r'[A-Z]', content)) / max(len(content), 1)
        }
    
    def _calculate_quality_score(self, features: Dict[str, Any]) -> float:
        """Calculate quality score from features"""
        score = 0.0
        
        # Word count score (optimal range: 50-500 words)
        word_count = features["word_count"]
        if 50 <= word_count <= 500:
            score += 0.2
        elif word_count > 500:
            score += 0.1
        
        # Sentence length score (optimal: 10-20 words per sentence)
        avg_sentence_length = features["avg_sentence_length"]
        if 10 <= avg_sentence_length <= 20:
            score += 0.2
        elif 5 <= avg_sentence_length <= 30:
            score += 0.1
        
        # Vocabulary richness score
        vocabulary_richness = features["vocabulary_richness"]
        if vocabulary_richness > 0.7:
            score += 0.2
        elif vocabulary_richness > 0.5:
            score += 0.1
        
        # Punctuation score
        punctuation_ratio = features["punctuation_ratio"]
        if 0.01 <= punctuation_ratio <= 0.05:
            score += 0.2
        elif 0.005 <= punctuation_ratio <= 0.1:
            score += 0.1
        
        # Capitalization score
        capitalization_ratio = features["capitalization_ratio"]
        if 0.05 <= capitalization_ratio <= 0.15:
            score += 0.2
        elif 0.02 <= capitalization_ratio <= 0.25:
            score += 0.1
        
        return min(score, 1.0)
    
    def _extract_sentiment_features(self, content: str) -> Dict[str, Any]:
        """Extract sentiment features from content"""
        positive_words = ["good", "great", "excellent", "amazing", "wonderful", "fantastic", "love", "like", "happy", "joy"]
        negative_words = ["bad", "terrible", "awful", "horrible", "hate", "dislike", "sad", "angry", "frustrated", "disappointed"]
        
        words = re.findall(r'\b\w+\b', content.lower())
        
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        return {
            "positive_words": positive_count,
            "negative_words": negative_count,
            "total_words": len(words),
            "positive_ratio": positive_count / max(len(words), 1),
            "negative_ratio": negative_count / max(len(words), 1),
            "sentiment_balance": (positive_count - negative_count) / max(len(words), 1)
        }
    
    def _calculate_sentiment_score(self, features: Dict[str, Any]) -> float:
        """Calculate sentiment score from features"""
        sentiment_balance = features["sentiment_balance"]
        
        # Normalize to 0-1 range
        score = (sentiment_balance + 1) / 2
        return max(0.0, min(1.0, score))
    
    def _extract_topic_features(self, content: str) -> Dict[str, Any]:
        """Extract topic features from content"""
        # Simple topic extraction based on word frequency
        words = re.findall(r'\b\w+\b', content.lower())
        word_counts = Counter(words)
        
        # Remove common stop words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should"}
        filtered_words = {word: count for word, count in word_counts.items() if word not in stop_words and len(word) > 2}
        
        return {
            "word_frequencies": dict(filtered_words),
            "total_words": len(words),
            "unique_words": len(filtered_words),
            "top_words": dict(Counter(filtered_words).most_common(10))
        }
    
    def _calculate_topic_distribution(self, features: Dict[str, Any], num_topics: int) -> List[Dict[str, Any]]:
        """Calculate topic distribution from features"""
        top_words = features["top_words"]
        
        # Simple topic modeling (in real implementation, use LDA or similar)
        topics = []
        words_per_topic = max(1, len(top_words) // num_topics)
        
        for i in range(num_topics):
            start_idx = i * words_per_topic
            end_idx = start_idx + words_per_topic
            
            topic_words = list(top_words.keys())[start_idx:end_idx]
            topic_weight = sum(top_words[word] for word in topic_words) / sum(top_words.values())
            
            topics.append({
                "topic_id": i,
                "words": topic_words,
                "weight": topic_weight,
                "name": f"Topic {i+1}"
            })
        
        return topics
    
    def _perform_clustering(self, embeddings: List[List[float]], num_clusters: int) -> List[int]:
        """Perform clustering on embeddings (simplified k-means)"""
        if len(embeddings) < num_clusters:
            return list(range(len(embeddings)))
        
        # Simple k-means clustering
        clusters = [0] * len(embeddings)
        
        # Initialize centroids randomly
        centroids = []
        for i in range(num_clusters):
            centroid = embeddings[i % len(embeddings)]
            centroids.append(centroid)
        
        # Iterate until convergence (simplified)
        for _ in range(10):  # Max 10 iterations
            # Assign points to closest centroid
            for i, embedding in enumerate(embeddings):
                distances = [self._cosine_similarity(embedding, centroid) for centroid in centroids]
                clusters[i] = distances.index(max(distances))
            
            # Update centroids
            for k in range(num_clusters):
                cluster_points = [embeddings[i] for i, cluster in enumerate(clusters) if cluster == k]
                if cluster_points:
                    centroids[k] = [sum(point[j] for point in cluster_points) / len(cluster_points) for j in range(len(cluster_points[0]))]
        
        return clusters
    
    def _apply_ml_model(self, input_data: Any, model: MLModel) -> Any:
        """Apply ML model to input data"""
        # Simplified model application
        # In real implementation, load and apply actual ML model
        
        if model.model_type == MLModelType.SIMILARITY:
            # Apply similarity model adjustments
            return input_data * model.accuracy
        elif model.model_type == MLModelType.QUALITY:
            # Apply quality model adjustments
            return input_data * (0.8 + 0.2 * model.accuracy)
        elif model.model_type == MLModelType.SENTIMENT:
            # Apply sentiment model adjustments
            return max(0.0, min(1.0, input_data * model.accuracy))
        else:
            return input_data
    
    async def _generate_summary(self, text: str, ai_model: AIModel) -> str:
        """Generate summary using AI model"""
        # Simplified summary generation
        sentences = re.split(r'[.!?]+', text)
        summary_sentences = sentences[:3]  # Take first 3 sentences
        return '. '.join(summary_sentences) + '.'
    
    async def _generate_translation(self, text: str, ai_model: AIModel) -> str:
        """Generate translation using AI model"""
        # Simplified translation (just return original text)
        return f"[Translated] {text}"
    
    async def _generate_general_response(self, prompt: str, ai_model: AIModel) -> str:
        """Generate general response using AI model"""
        # Simplified response generation
        return f"AI Response to: {prompt[:50]}..."
    
    def get_models(self) -> Dict[str, MLModel]:
        """Get all registered ML models"""
        return self._models.copy()
    
    def get_ai_models(self) -> Dict[str, AIModel]:
        """Get all registered AI models"""
        return self._ai_models.copy()
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get model statistics"""
        return {
            "ml_models": len(self._models),
            "ai_models": len(self._ai_models),
            "embeddings_cached": len(self._embeddings_cache),
            "similarity_matrix_size": len(self._similarity_matrix)
        }


# Global AI/ML engine
ai_ml_engine = AdvancedMLProcessor()


# Initialize with default models
def initialize_default_models():
    """Initialize default ML and AI models"""
    # Default ML models
    default_ml_models = [
        MLModel("similarity_v1", "Similarity Model v1", MLModelType.SIMILARITY, "1.0", 0.95, time.time()),
        MLModel("quality_v1", "Quality Model v1", MLModelType.QUALITY, "1.0", 0.90, time.time()),
        MLModel("sentiment_v1", "Sentiment Model v1", MLModelType.SENTIMENT, "1.0", 0.85, time.time()),
        MLModel("topics_v1", "Topic Model v1", MLModelType.TOPIC_MODELING, "1.0", 0.80, time.time()),
        MLModel("clustering_v1", "Clustering Model v1", MLModelType.CLUSTERING, "1.0", 0.75, time.time())
    ]
    
    for model in default_ml_models:
        ai_ml_engine.register_ml_model(model)
    
    # Default AI models
    default_ai_models = [
        AIModel("gpt_v1", "GPT Model v1", AIModelType.GPT, "openai", max_tokens=1000, temperature=0.7),
        AIModel("bert_v1", "BERT Model v1", AIModelType.BERT, "huggingface", max_tokens=512, temperature=0.5),
        AIModel("summarizer_v1", "Summarizer v1", AIModelType.SUMMARIZATION, "custom", max_tokens=200, temperature=0.3),
        AIModel("translator_v1", "Translator v1", AIModelType.TRANSLATION, "custom", max_tokens=500, temperature=0.2)
    ]
    
    for model in default_ai_models:
        ai_ml_engine.register_ai_model(model)


# Initialize default models
initialize_default_models()


