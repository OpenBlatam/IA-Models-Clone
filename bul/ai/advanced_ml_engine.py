"""
Advanced Machine Learning Engine for BUL System
Implements sophisticated ML capabilities including document analysis, content optimization, and predictive analytics
"""

from typing import Dict, List, Optional, Any, Union, Tuple
from pydantic import BaseModel, Field
from enum import Enum
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import json
import pickle
from pathlib import Path
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import LatentDirichletAllocation, PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score
import spacy
import nltk
from textstat import flesch_reading_ease, flesch_kincaid_grade
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
from sentence_transformers import SentenceTransformer
import openai
import anthropic

logger = logging.getLogger(__name__)


class MLTaskType(str, Enum):
    """ML task types"""
    TEXT_CLASSIFICATION = "text_classification"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    TOPIC_MODELING = "topic_modeling"
    TEXT_SUMMARIZATION = "text_summarization"
    CONTENT_OPTIMIZATION = "content_optimization"
    READABILITY_ANALYSIS = "readability_analysis"
    DOCUMENT_SIMILARITY = "document_similarity"
    CONTENT_GENERATION = "content_generation"
    QUALITY_ASSESSMENT = "quality_assessment"
    PREDICTIVE_ANALYTICS = "predictive_analytics"


class ModelType(str, Enum):
    """Model types"""
    TRANSFORMER = "transformer"
    SKLEARN = "sklearn"
    CUSTOM = "custom"
    HYBRID = "hybrid"


class QualityScore(BaseModel):
    """Content quality score"""
    overall_score: float = Field(..., ge=0.0, le=1.0, description="Overall quality score")
    readability_score: float = Field(..., ge=0.0, le=1.0, description="Readability score")
    coherence_score: float = Field(..., ge=0.0, le=1.0, description="Coherence score")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    engagement_score: float = Field(..., ge=0.0, le=1.0, description="Engagement score")
    technical_score: float = Field(..., ge=0.0, le=1.0, description="Technical accuracy score")
    suggestions: List[str] = Field(default_factory=list, description="Improvement suggestions")


class DocumentAnalysis(BaseModel):
    """Document analysis results"""
    document_id: str = Field(..., description="Document ID")
    word_count: int = Field(..., description="Word count")
    sentence_count: int = Field(..., description="Sentence count")
    paragraph_count: int = Field(..., description="Paragraph count")
    reading_time: float = Field(..., description="Estimated reading time in minutes")
    readability_score: float = Field(..., description="Flesch reading ease score")
    grade_level: float = Field(..., description="Flesch-Kincaid grade level")
    sentiment: str = Field(..., description="Overall sentiment")
    sentiment_score: float = Field(..., description="Sentiment score")
    topics: List[str] = Field(default_factory=list, description="Identified topics")
    keywords: List[str] = Field(default_factory=list, description="Key terms")
    entities: List[Dict[str, Any]] = Field(default_factory=list, description="Named entities")
    quality_score: QualityScore = Field(..., description="Content quality assessment")
    language: str = Field(..., description="Detected language")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ContentOptimization(BaseModel):
    """Content optimization suggestions"""
    document_id: str = Field(..., description="Document ID")
    original_score: float = Field(..., description="Original quality score")
    optimized_score: float = Field(..., description="Optimized quality score")
    improvements: List[Dict[str, Any]] = Field(default_factory=list, description="Suggested improvements")
    optimized_content: str = Field(..., description="Optimized content")
    metrics: Dict[str, float] = Field(default_factory=dict, description="Optimization metrics")


class PredictiveInsight(BaseModel):
    """Predictive analytics insight"""
    insight_type: str = Field(..., description="Type of insight")
    prediction: Any = Field(..., description="Prediction value")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence")
    factors: List[str] = Field(default_factory=list, description="Key factors")
    timeframe: str = Field(..., description="Prediction timeframe")
    recommendations: List[str] = Field(default_factory=list, description="Actionable recommendations")


@dataclass
class MLModel:
    """ML model wrapper"""
    name: str
    model_type: ModelType
    task_type: MLTaskType
    model: Any
    tokenizer: Optional[Any] = None
    accuracy: Optional[float] = None
    last_trained: Optional[datetime] = None
    version: str = "1.0.0"
    metadata: Dict[str, Any] = None


class AdvancedMLEngine:
    """Advanced Machine Learning Engine for BUL System"""
    
    def __init__(self):
        self.models: Dict[str, MLModel] = {}
        self.nlp = None
        self.sentence_transformer = None
        self.vectorizer = None
        self.topic_model = None
        self.sentiment_analyzer = None
        self.summarizer = None
        self.classifier = None
        self.regressor = None
        self._initialize_models()
        self._load_pretrained_models()
    
    def _initialize_models(self):
        """Initialize ML models and components"""
        try:
            # Initialize spaCy
            self.nlp = spacy.load("en_core_web_sm")
            
            # Initialize sentence transformer
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Initialize vectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            # Initialize topic model
            self.topic_model = LatentDirichletAllocation(
                n_components=10,
                random_state=42
            )
            
            # Initialize sentiment analyzer
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest"
            )
            
            # Initialize summarizer
            self.summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn"
            )
            
            logger.info("ML models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing ML models: {e}")
    
    def _load_pretrained_models(self):
        """Load pretrained models"""
        try:
            # Load custom models if available
            models_path = Path("models")
            if models_path.exists():
                for model_file in models_path.glob("*.pkl"):
                    model_name = model_file.stem
                    with open(model_file, 'rb') as f:
                        model_data = pickle.load(f)
                        self.models[model_name] = MLModel(
                            name=model_name,
                            model_type=ModelType.SKLEARN,
                            task_type=MLTaskType.TEXT_CLASSIFICATION,
                            model=model_data['model'],
                            accuracy=model_data.get('accuracy'),
                            last_trained=model_data.get('last_trained'),
                            version=model_data.get('version', '1.0.0'),
                            metadata=model_data.get('metadata', {})
                        )
            
            logger.info(f"Loaded {len(self.models)} pretrained models")
            
        except Exception as e:
            logger.error(f"Error loading pretrained models: {e}")
    
    async def analyze_document(self, content: str, document_id: str = None) -> DocumentAnalysis:
        """Comprehensive document analysis"""
        try:
            if not document_id:
                document_id = f"doc_{datetime.utcnow().timestamp()}"
            
            # Basic text statistics
            doc = self.nlp(content)
            word_count = len([token for token in doc if not token.is_space])
            sentence_count = len(list(doc.sents))
            paragraph_count = len(content.split('\n\n'))
            reading_time = word_count / 200  # Average reading speed: 200 words per minute
            
            # Readability analysis
            readability_score = flesch_reading_ease(content)
            grade_level = flesch_kincaid_grade(content)
            
            # Sentiment analysis
            sentiment_result = self.sentiment_analyzer(content[:512])  # Limit for API
            sentiment = sentiment_result[0]['label']
            sentiment_score = sentiment_result[0]['score']
            
            # Topic modeling
            topics = await self._extract_topics(content)
            
            # Keyword extraction
            keywords = await self._extract_keywords(content)
            
            # Named entity recognition
            entities = await self._extract_entities(content)
            
            # Language detection
            language = await self._detect_language(content)
            
            # Quality assessment
            quality_score = await self._assess_content_quality(content)
            
            return DocumentAnalysis(
                document_id=document_id,
                word_count=word_count,
                sentence_count=sentence_count,
                paragraph_count=paragraph_count,
                reading_time=reading_time,
                readability_score=readability_score,
                grade_level=grade_level,
                sentiment=sentiment,
                sentiment_score=sentiment_score,
                topics=topics,
                keywords=keywords,
                entities=entities,
                quality_score=quality_score,
                language=language
            )
            
        except Exception as e:
            logger.error(f"Error analyzing document: {e}")
            raise
    
    async def _extract_topics(self, content: str) -> List[str]:
        """Extract topics from content"""
        try:
            # Simple topic extraction using TF-IDF
            if len(content.split()) < 10:
                return []
            
            # Vectorize content
            tfidf_matrix = self.vectorizer.fit_transform([content])
            feature_names = self.vectorizer.get_feature_names_out()
            
            # Get top terms
            scores = tfidf_matrix.toarray()[0]
            top_indices = scores.argsort()[-5:][::-1]
            topics = [feature_names[i] for i in top_indices if scores[i] > 0]
            
            return topics
            
        except Exception as e:
            logger.error(f"Error extracting topics: {e}")
            return []
    
    async def _extract_keywords(self, content: str) -> List[str]:
        """Extract keywords from content"""
        try:
            doc = self.nlp(content)
            
            # Extract noun phrases and important terms
            keywords = []
            for chunk in doc.noun_chunks:
                if len(chunk.text.split()) <= 3:  # Limit to 3-word phrases
                    keywords.append(chunk.text.lower())
            
            # Remove duplicates and return top keywords
            unique_keywords = list(set(keywords))
            return unique_keywords[:10]  # Return top 10 keywords
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []
    
    async def _extract_entities(self, content: str) -> List[Dict[str, Any]]:
        """Extract named entities"""
        try:
            doc = self.nlp(content)
            entities = []
            
            for ent in doc.ents:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "description": spacy.explain(ent.label_)
                })
            
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return []
    
    async def _detect_language(self, content: str) -> str:
        """Detect document language"""
        try:
            # Simple language detection based on common words
            english_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
            spanish_words = ['el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'se', 'no', 'te', 'lo', 'le']
            french_words = ['le', 'de', 'et', 'à', 'un', 'il', 'être', 'et', 'en', 'avoir', 'que', 'pour', 'dans']
            
            content_lower = content.lower()
            
            english_count = sum(1 for word in english_words if word in content_lower)
            spanish_count = sum(1 for word in spanish_words if word in content_lower)
            french_count = sum(1 for word in french_words if word in content_lower)
            
            if english_count > spanish_count and english_count > french_count:
                return "en"
            elif spanish_count > french_count:
                return "es"
            elif french_count > 0:
                return "fr"
            else:
                return "en"  # Default to English
                
        except Exception as e:
            logger.error(f"Error detecting language: {e}")
            return "en"
    
    async def _assess_content_quality(self, content: str) -> QualityScore:
        """Assess content quality"""
        try:
            # Readability assessment
            readability_score = flesch_reading_ease(content) / 100.0  # Normalize to 0-1
            
            # Coherence assessment (simplified)
            sentences = [sent.text for sent in self.nlp(content).sents]
            coherence_score = min(1.0, len(sentences) / 20.0)  # More sentences = better coherence
            
            # Relevance assessment (placeholder)
            relevance_score = 0.8  # Would be calculated based on context
            
            # Engagement assessment
            engagement_words = ['you', 'your', 'imagine', 'think', 'consider', 'discover', 'explore']
            engagement_count = sum(1 for word in engagement_words if word in content.lower())
            engagement_score = min(1.0, engagement_count / 10.0)
            
            # Technical accuracy (placeholder)
            technical_score = 0.7  # Would be calculated based on domain knowledge
            
            # Overall score
            overall_score = (
                readability_score * 0.2 +
                coherence_score * 0.2 +
                relevance_score * 0.2 +
                engagement_score * 0.2 +
                technical_score * 0.2
            )
            
            # Generate suggestions
            suggestions = []
            if readability_score < 0.5:
                suggestions.append("Consider simplifying sentence structure for better readability")
            if engagement_score < 0.3:
                suggestions.append("Add more engaging language to connect with readers")
            if coherence_score < 0.5:
                suggestions.append("Improve document structure and flow")
            
            return QualityScore(
                overall_score=overall_score,
                readability_score=readability_score,
                coherence_score=coherence_score,
                relevance_score=relevance_score,
                engagement_score=engagement_score,
                technical_score=technical_score,
                suggestions=suggestions
            )
            
        except Exception as e:
            logger.error(f"Error assessing content quality: {e}")
            return QualityScore(
                overall_score=0.5,
                readability_score=0.5,
                coherence_score=0.5,
                relevance_score=0.5,
                engagement_score=0.5,
                technical_score=0.5,
                suggestions=["Unable to assess content quality"]
            )
    
    async def optimize_content(self, content: str, document_id: str = None) -> ContentOptimization:
        """Optimize content for better quality"""
        try:
            if not document_id:
                document_id = f"doc_{datetime.utcnow().timestamp()}"
            
            # Analyze original content
            original_analysis = await self.analyze_document(content, document_id)
            original_score = original_analysis.quality_score.overall_score
            
            # Apply optimizations
            optimized_content = await self._apply_optimizations(content, original_analysis)
            
            # Analyze optimized content
            optimized_analysis = await self.analyze_document(optimized_content, f"{document_id}_optimized")
            optimized_score = optimized_analysis.quality_score.overall_score
            
            # Generate improvements list
            improvements = []
            if optimized_analysis.quality_score.readability_score > original_analysis.quality_score.readability_score:
                improvements.append({
                    "type": "readability",
                    "description": "Improved sentence structure and word choice",
                    "impact": optimized_analysis.quality_score.readability_score - original_analysis.quality_score.readability_score
                })
            
            if optimized_analysis.quality_score.engagement_score > original_analysis.quality_score.engagement_score:
                improvements.append({
                    "type": "engagement",
                    "description": "Added more engaging language and calls to action",
                    "impact": optimized_analysis.quality_score.engagement_score - original_analysis.quality_score.engagement_score
                })
            
            # Calculate metrics
            metrics = {
                "score_improvement": optimized_score - original_score,
                "readability_improvement": optimized_analysis.quality_score.readability_score - original_analysis.quality_score.readability_score,
                "engagement_improvement": optimized_analysis.quality_score.engagement_score - original_analysis.quality_score.engagement_score,
                "word_count_change": optimized_analysis.word_count - original_analysis.word_count
            }
            
            return ContentOptimization(
                document_id=document_id,
                original_score=original_score,
                optimized_score=optimized_score,
                improvements=improvements,
                optimized_content=optimized_content,
                metrics=metrics
            )
            
        except Exception as e:
            logger.error(f"Error optimizing content: {e}")
            raise
    
    async def _apply_optimizations(self, content: str, analysis: DocumentAnalysis) -> str:
        """Apply content optimizations"""
        try:
            optimized_content = content
            
            # Improve readability if needed
            if analysis.quality_score.readability_score < 0.6:
                optimized_content = await self._improve_readability(optimized_content)
            
            # Improve engagement if needed
            if analysis.quality_score.engagement_score < 0.5:
                optimized_content = await self._improve_engagement(optimized_content)
            
            # Improve coherence if needed
            if analysis.quality_score.coherence_score < 0.6:
                optimized_content = await self._improve_coherence(optimized_content)
            
            return optimized_content
            
        except Exception as e:
            logger.error(f"Error applying optimizations: {e}")
            return content
    
    async def _improve_readability(self, content: str) -> str:
        """Improve content readability"""
        # Simple readability improvements
        # In a real implementation, this would use more sophisticated NLP techniques
        
        # Replace complex words with simpler alternatives
        replacements = {
            'utilize': 'use',
            'facilitate': 'help',
            'implement': 'put in place',
            'comprehensive': 'complete',
            'substantial': 'large',
            'consequently': 'so',
            'furthermore': 'also',
            'nevertheless': 'but'
        }
        
        for complex_word, simple_word in replacements.items():
            content = content.replace(complex_word, simple_word)
        
        return content
    
    async def _improve_engagement(self, content: str) -> str:
        """Improve content engagement"""
        # Add engaging elements
        sentences = content.split('. ')
        
        # Add questions to engage readers
        if len(sentences) > 3:
            engaging_questions = [
                "Have you ever wondered about this?",
                "What do you think about this approach?",
                "Can you imagine the possibilities?"
            ]
            
            # Insert question at strategic points
            for i, question in enumerate(engaging_questions):
                if i * 2 < len(sentences):
                    sentences.insert(i * 2 + 1, question)
        
        return '. '.join(sentences)
    
    async def _improve_coherence(self, content: str) -> str:
        """Improve content coherence"""
        # Add transition words for better flow
        sentences = content.split('. ')
        
        transition_words = ['Additionally', 'Moreover', 'Furthermore', 'However', 'Therefore', 'Consequently']
        
        for i in range(1, len(sentences)):
            if i % 3 == 0 and i < len(transition_words):
                sentences[i] = f"{transition_words[i % len(transition_words)]}, {sentences[i].lower()}"
        
        return '. '.join(sentences)
    
    async def generate_predictive_insights(self, data: Dict[str, Any]) -> List[PredictiveInsight]:
        """Generate predictive analytics insights"""
        try:
            insights = []
            
            # Document performance prediction
            if 'document_metrics' in data:
                performance_insight = await self._predict_document_performance(data['document_metrics'])
                insights.append(performance_insight)
            
            # User engagement prediction
            if 'user_behavior' in data:
                engagement_insight = await self._predict_user_engagement(data['user_behavior'])
                insights.append(engagement_insight)
            
            # Content trend prediction
            if 'content_history' in data:
                trend_insight = await self._predict_content_trends(data['content_history'])
                insights.append(trend_insight)
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating predictive insights: {e}")
            return []
    
    async def _predict_document_performance(self, metrics: Dict[str, Any]) -> PredictiveInsight:
        """Predict document performance"""
        # Simple prediction based on quality metrics
        quality_score = metrics.get('quality_score', 0.5)
        readability_score = metrics.get('readability_score', 0.5)
        engagement_score = metrics.get('engagement_score', 0.5)
        
        # Calculate predicted performance
        predicted_performance = (quality_score * 0.4 + readability_score * 0.3 + engagement_score * 0.3)
        confidence = min(0.9, predicted_performance + 0.1)
        
        factors = []
        if quality_score > 0.7:
            factors.append("High content quality")
        if readability_score > 0.7:
            factors.append("Good readability")
        if engagement_score > 0.7:
            factors.append("High engagement potential")
        
        recommendations = []
        if predicted_performance < 0.6:
            recommendations.append("Improve content quality and readability")
            recommendations.append("Add more engaging elements")
        else:
            recommendations.append("Content is well-optimized for performance")
        
        return PredictiveInsight(
            insight_type="document_performance",
            prediction=predicted_performance,
            confidence=confidence,
            factors=factors,
            timeframe="30 days",
            recommendations=recommendations
        )
    
    async def _predict_user_engagement(self, behavior: Dict[str, Any]) -> PredictiveInsight:
        """Predict user engagement"""
        # Simple engagement prediction
        session_duration = behavior.get('avg_session_duration', 0)
        page_views = behavior.get('avg_page_views', 0)
        bounce_rate = behavior.get('bounce_rate', 0.5)
        
        # Calculate engagement score
        engagement_score = (session_duration / 300) * 0.4 + (page_views / 5) * 0.3 + (1 - bounce_rate) * 0.3
        engagement_score = min(1.0, max(0.0, engagement_score))
        
        confidence = 0.7  # Medium confidence for user behavior prediction
        
        factors = []
        if session_duration > 300:
            factors.append("Long session duration")
        if page_views > 3:
            factors.append("Multiple page views")
        if bounce_rate < 0.3:
            factors.append("Low bounce rate")
        
        recommendations = []
        if engagement_score < 0.5:
            recommendations.append("Improve content relevance and quality")
            recommendations.append("Optimize user experience and navigation")
        else:
            recommendations.append("Maintain current engagement strategies")
        
        return PredictiveInsight(
            insight_type="user_engagement",
            prediction=engagement_score,
            confidence=confidence,
            factors=factors,
            timeframe="7 days",
            recommendations=recommendations
        )
    
    async def _predict_content_trends(self, history: Dict[str, Any]) -> PredictiveInsight:
        """Predict content trends"""
        # Simple trend prediction based on historical data
        recent_views = history.get('recent_views', 0)
        historical_views = history.get('historical_views', 0)
        
        if historical_views > 0:
            growth_rate = (recent_views - historical_views) / historical_views
        else:
            growth_rate = 0
        
        # Predict future trend
        predicted_trend = "stable"
        if growth_rate > 0.2:
            predicted_trend = "growing"
        elif growth_rate < -0.2:
            predicted_trend = "declining"
        
        confidence = min(0.8, abs(growth_rate) + 0.3)
        
        factors = []
        if growth_rate > 0:
            factors.append("Positive growth trend")
        elif growth_rate < 0:
            factors.append("Declining trend")
        else:
            factors.append("Stable performance")
        
        recommendations = []
        if predicted_trend == "growing":
            recommendations.append("Continue current content strategy")
            recommendations.append("Consider expanding similar content")
        elif predicted_trend == "declining":
            recommendations.append("Review and update content")
            recommendations.append("Consider content refresh or new topics")
        else:
            recommendations.append("Monitor trends and optimize as needed")
        
        return PredictiveInsight(
            insight_type="content_trends",
            prediction=predicted_trend,
            confidence=confidence,
            factors=factors,
            timeframe="14 days",
            recommendations=recommendations
        )
    
    async def train_custom_model(self, training_data: List[Dict[str, Any]], task_type: MLTaskType) -> MLModel:
        """Train a custom ML model"""
        try:
            if task_type == MLTaskType.TEXT_CLASSIFICATION:
                return await self._train_classifier(training_data)
            elif task_type == MLTaskType.SENTIMENT_ANALYSIS:
                return await self._train_sentiment_model(training_data)
            else:
                raise ValueError(f"Unsupported task type: {task_type}")
                
        except Exception as e:
            logger.error(f"Error training custom model: {e}")
            raise
    
    async def _train_classifier(self, training_data: List[Dict[str, Any]]) -> MLModel:
        """Train a text classifier"""
        try:
            # Prepare training data
            texts = [item['text'] for item in training_data]
            labels = [item['label'] for item in training_data]
            
            # Vectorize texts
            X = self.vectorizer.fit_transform(texts)
            
            # Train classifier
            self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            self.classifier.fit(X, labels)
            
            # Evaluate model
            X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
            self.classifier.fit(X_train, y_train)
            y_pred = self.classifier.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Create model wrapper
            model = MLModel(
                name=f"classifier_{datetime.utcnow().timestamp()}",
                model_type=ModelType.SKLEARN,
                task_type=MLTaskType.TEXT_CLASSIFICATION,
                model=self.classifier,
                accuracy=accuracy,
                last_trained=datetime.utcnow(),
                version="1.0.0",
                metadata={"vectorizer": self.vectorizer}
            )
            
            # Save model
            await self._save_model(model)
            
            return model
            
        except Exception as e:
            logger.error(f"Error training classifier: {e}")
            raise
    
    async def _train_sentiment_model(self, training_data: List[Dict[str, Any]]) -> MLModel:
        """Train a sentiment analysis model"""
        try:
            # Prepare training data
            texts = [item['text'] for item in training_data]
            sentiments = [item['sentiment'] for item in training_data]
            
            # Vectorize texts
            X = self.vectorizer.fit_transform(texts)
            
            # Train regressor for sentiment scores
            self.regressor = GradientBoostingRegressor(n_estimators=100, random_state=42)
            self.regressor.fit(X, sentiments)
            
            # Evaluate model
            X_train, X_test, y_train, y_test = train_test_split(X, sentiments, test_size=0.2, random_state=42)
            self.regressor.fit(X_train, y_train)
            y_pred = self.regressor.predict(X_test)
            
            # Calculate accuracy (simplified)
            accuracy = 1.0 - np.mean(np.abs(y_test - y_pred))
            
            # Create model wrapper
            model = MLModel(
                name=f"sentiment_{datetime.utcnow().timestamp()}",
                model_type=ModelType.SKLEARN,
                task_type=MLTaskType.SENTIMENT_ANALYSIS,
                model=self.regressor,
                accuracy=accuracy,
                last_trained=datetime.utcnow(),
                version="1.0.0",
                metadata={"vectorizer": self.vectorizer}
            )
            
            # Save model
            await self._save_model(model)
            
            return model
            
        except Exception as e:
            logger.error(f"Error training sentiment model: {e}")
            raise
    
    async def _save_model(self, model: MLModel):
        """Save trained model"""
        try:
            models_path = Path("models")
            models_path.mkdir(exist_ok=True)
            
            model_data = {
                'model': model.model,
                'accuracy': model.accuracy,
                'last_trained': model.last_trained,
                'version': model.version,
                'metadata': model.metadata
            }
            
            model_file = models_path / f"{model.name}.pkl"
            with open(model_file, 'wb') as f:
                pickle.dump(model_data, f)
            
            # Add to models registry
            self.models[model.name] = model
            
            logger.info(f"Model {model.name} saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    async def get_model_performance(self, model_name: str) -> Dict[str, Any]:
        """Get model performance metrics"""
        try:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")
            
            model = self.models[model_name]
            
            return {
                "name": model.name,
                "type": model.model_type.value,
                "task": model.task_type.value,
                "accuracy": model.accuracy,
                "last_trained": model.last_trained.isoformat() if model.last_trained else None,
                "version": model.version,
                "metadata": model.metadata
            }
            
        except Exception as e:
            logger.error(f"Error getting model performance: {e}")
            raise
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List all available models"""
        return [
            {
                "name": model.name,
                "type": model.model_type.value,
                "task": model.task_type.value,
                "accuracy": model.accuracy,
                "last_trained": model.last_trained.isoformat() if model.last_trained else None,
                "version": model.version
            }
            for model in self.models.values()
        ]


# Global ML engine instance
ml_engine = AdvancedMLEngine()














