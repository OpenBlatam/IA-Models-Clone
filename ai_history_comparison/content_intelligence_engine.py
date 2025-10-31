"""
Content Intelligence Engine - Advanced AI Content Analysis and Intelligence
======================================================================

This module provides comprehensive content intelligence capabilities including:
- Content sentiment analysis and emotion detection
- Content categorization and topic modeling
- Content readability and complexity analysis
- Content engagement prediction
- Content performance analytics
- Content recommendation engine
- Content trend analysis
- Content competitive analysis
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textstat import flesch_reading_ease, flesch_kincaid_grade
import spacy
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
from collections import Counter, defaultdict
import re
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContentType(Enum):
    """Content type enumeration"""
    ARTICLE = "article"
    BLOG_POST = "blog_post"
    SOCIAL_MEDIA = "social_media"
    EMAIL = "email"
    PRODUCT_DESCRIPTION = "product_description"
    NEWS = "news"
    TECHNICAL_DOCUMENT = "technical_document"
    MARKETING_COPY = "marketing_copy"
    CREATIVE_WRITING = "creative_writing"
    ACADEMIC_PAPER = "academic_paper"

class SentimentType(Enum):
    """Sentiment type enumeration"""
    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"

class ComplexityLevel(Enum):
    """Content complexity level enumeration"""
    VERY_SIMPLE = "very_simple"
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"

@dataclass
class ContentMetrics:
    """Content metrics data structure"""
    word_count: int
    sentence_count: int
    paragraph_count: int
    character_count: int
    average_sentence_length: float
    average_word_length: float
    readability_score: float
    complexity_level: ComplexityLevel
    sentiment_score: float
    sentiment_type: SentimentType
    emotion_scores: Dict[str, float] = field(default_factory=dict)
    topic_scores: Dict[str, float] = field(default_factory=dict)
    keyword_density: Dict[str, float] = field(default_factory=dict)
    engagement_score: float = 0.0
    virality_potential: float = 0.0
    seo_score: float = 0.0
    quality_score: float = 0.0

@dataclass
class ContentInsight:
    """Content insight data structure"""
    content_id: str
    insight_type: str
    insight_value: Any
    confidence: float
    explanation: str
    recommendations: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class ContentRecommendation:
    """Content recommendation data structure"""
    content_id: str
    recommended_content: List[str]
    similarity_scores: Dict[str, float]
    recommendation_reason: str
    confidence: float
    created_at: datetime = field(default_factory=datetime.utcnow)

class ContentIntelligenceEngine:
    """
    Advanced Content Intelligence Engine
    
    Provides comprehensive content analysis, intelligence, and recommendations
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the Content Intelligence Engine"""
        self.config = config
        self.nlp = None
        self.sentiment_analyzer = None
        self.topic_model = None
        self.engagement_predictor = None
        self.recommendation_engine = None
        self.trend_analyzer = None
        self.competitive_analyzer = None
        
        # Initialize models
        self._initialize_models()
        
        # Content cache
        self.content_cache = {}
        self.insights_cache = {}
        self.recommendations_cache = {}
        
        logger.info("Content Intelligence Engine initialized successfully")
    
    def _initialize_models(self):
        """Initialize all AI models and analyzers"""
        try:
            # Initialize spaCy model
            self.nlp = spacy.load("en_core_web_sm")
            
            # Initialize sentiment analyzer
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
            
            # Initialize topic modeling
            self.topic_model = LatentDirichletAllocation(
                n_components=10,
                random_state=42
            )
            
            # Initialize engagement predictor
            self.engagement_predictor = RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
            
            # Initialize recommendation engine
            self.recommendation_engine = None  # Will be initialized with data
            
            # Initialize trend analyzer
            self.trend_analyzer = None  # Will be initialized with data
            
            # Initialize competitive analyzer
            self.competitive_analyzer = None  # Will be initialized with data
            
            logger.info("All models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise
    
    async def analyze_content(self, content: str, content_type: ContentType = ContentType.ARTICLE) -> ContentMetrics:
        """
        Comprehensive content analysis
        
        Args:
            content: The content to analyze
            content_type: Type of content being analyzed
            
        Returns:
            ContentMetrics object with comprehensive analysis
        """
        try:
            # Basic metrics
            word_count = len(content.split())
            sentence_count = len(sent_tokenize(content))
            paragraph_count = len(content.split('\n\n'))
            character_count = len(content)
            
            # Calculate averages
            average_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
            average_word_length = character_count / word_count if word_count > 0 else 0
            
            # Readability analysis
            readability_score = flesch_reading_ease(content)
            complexity_level = self._determine_complexity(readability_score)
            
            # Sentiment analysis
            sentiment_scores = self.sentiment_analyzer.polarity_scores(content)
            sentiment_score = sentiment_scores['compound']
            sentiment_type = self._determine_sentiment_type(sentiment_score)
            
            # Emotion analysis
            emotion_scores = await self._analyze_emotions(content)
            
            # Topic analysis
            topic_scores = await self._analyze_topics(content)
            
            # Keyword analysis
            keyword_density = await self._analyze_keywords(content)
            
            # Engagement prediction
            engagement_score = await self._predict_engagement(content, content_type)
            
            # Virality potential
            virality_potential = await self._predict_virality(content, content_type)
            
            # SEO score
            seo_score = await self._calculate_seo_score(content, content_type)
            
            # Quality score
            quality_score = await self._calculate_quality_score(content, content_type)
            
            return ContentMetrics(
                word_count=word_count,
                sentence_count=sentence_count,
                paragraph_count=paragraph_count,
                character_count=character_count,
                average_sentence_length=average_sentence_length,
                average_word_length=average_word_length,
                readability_score=readability_score,
                complexity_level=complexity_level,
                sentiment_score=sentiment_score,
                sentiment_type=sentiment_type,
                emotion_scores=emotion_scores,
                topic_scores=topic_scores,
                keyword_density=keyword_density,
                engagement_score=engagement_score,
                virality_potential=virality_potential,
                seo_score=seo_score,
                quality_score=quality_score
            )
            
        except Exception as e:
            logger.error(f"Error analyzing content: {e}")
            raise
    
    async def _analyze_emotions(self, content: str) -> Dict[str, float]:
        """Analyze emotional content"""
        try:
            # Use spaCy for emotion analysis
            doc = self.nlp(content)
            
            # Simple emotion detection based on keywords
            emotions = {
                'joy': 0.0,
                'sadness': 0.0,
                'anger': 0.0,
                'fear': 0.0,
                'surprise': 0.0,
                'disgust': 0.0
            }
            
            # Emotion keywords
            emotion_keywords = {
                'joy': ['happy', 'joyful', 'excited', 'pleased', 'delighted', 'thrilled'],
                'sadness': ['sad', 'depressed', 'melancholy', 'gloomy', 'sorrowful', 'mournful'],
                'anger': ['angry', 'furious', 'rage', 'irritated', 'annoyed', 'mad'],
                'fear': ['afraid', 'scared', 'terrified', 'worried', 'anxious', 'nervous'],
                'surprise': ['surprised', 'amazed', 'astonished', 'shocked', 'stunned', 'bewildered'],
                'disgust': ['disgusted', 'revolted', 'repulsed', 'sickened', 'appalled', 'horrified']
            }
            
            words = [token.text.lower() for token in doc if token.is_alpha]
            
            for emotion, keywords in emotion_keywords.items():
                count = sum(1 for word in words if word in keywords)
                emotions[emotion] = count / len(words) if words else 0.0
            
            return emotions
            
        except Exception as e:
            logger.error(f"Error analyzing emotions: {e}")
            return {}
    
    async def _analyze_topics(self, content: str) -> Dict[str, float]:
        """Analyze content topics"""
        try:
            # Simple topic analysis using TF-IDF
            vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform([content])
            feature_names = vectorizer.get_feature_names_out()
            
            # Get top topics
            topic_scores = {}
            for i, score in enumerate(tfidf_matrix.toarray()[0]):
                if score > 0.1:  # Threshold for significant topics
                    topic_scores[feature_names[i]] = float(score)
            
            return topic_scores
            
        except Exception as e:
            logger.error(f"Error analyzing topics: {e}")
            return {}
    
    async def _analyze_keywords(self, content: str) -> Dict[str, float]:
        """Analyze keyword density"""
        try:
            # Extract keywords using spaCy
            doc = self.nlp(content)
            
            # Get important keywords (nouns, adjectives, verbs)
            keywords = []
            for token in doc:
                if (token.pos_ in ['NOUN', 'ADJ', 'VERB'] and 
                    not token.is_stop and 
                    not token.is_punct and 
                    len(token.text) > 2):
                    keywords.append(token.lemma_.lower())
            
            # Calculate density
            keyword_counts = Counter(keywords)
            total_words = len(keywords)
            
            keyword_density = {}
            for keyword, count in keyword_counts.most_common(20):
                keyword_density[keyword] = count / total_words if total_words > 0 else 0.0
            
            return keyword_density
            
        except Exception as e:
            logger.error(f"Error analyzing keywords: {e}")
            return {}
    
    async def _predict_engagement(self, content: str, content_type: ContentType) -> float:
        """Predict content engagement score"""
        try:
            # Simple engagement prediction based on content features
            features = []
            
            # Word count feature
            word_count = len(content.split())
            features.append(min(word_count / 1000, 1.0))  # Normalize to 0-1
            
            # Question count feature
            question_count = content.count('?')
            features.append(min(question_count / 10, 1.0))  # Normalize to 0-1
            
            # Exclamation count feature
            exclamation_count = content.count('!')
            features.append(min(exclamation_count / 10, 1.0))  # Normalize to 0-1
            
            # Sentiment feature
            sentiment_scores = self.sentiment_analyzer.polarity_scores(content)
            features.append((sentiment_scores['compound'] + 1) / 2)  # Normalize to 0-1
            
            # Content type feature
            type_scores = {
                ContentType.SOCIAL_MEDIA: 0.9,
                ContentType.BLOG_POST: 0.7,
                ContentType.ARTICLE: 0.6,
                ContentType.EMAIL: 0.5,
                ContentType.PRODUCT_DESCRIPTION: 0.4,
                ContentType.TECHNICAL_DOCUMENT: 0.3,
                ContentType.ACADEMIC_PAPER: 0.2
            }
            features.append(type_scores.get(content_type, 0.5))
            
            # Simple weighted average
            weights = [0.2, 0.2, 0.2, 0.2, 0.2]
            engagement_score = sum(f * w for f, w in zip(features, weights))
            
            return min(max(engagement_score, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error predicting engagement: {e}")
            return 0.5
    
    async def _predict_virality(self, content: str, content_type: ContentType) -> float:
        """Predict content virality potential"""
        try:
            # Virality prediction based on content characteristics
            features = []
            
            # Emotional intensity
            sentiment_scores = self.sentiment_analyzer.polarity_scores(content)
            emotional_intensity = abs(sentiment_scores['compound'])
            features.append(emotional_intensity)
            
            # Controversy indicators
            controversy_words = ['controversial', 'shocking', 'amazing', 'incredible', 'unbelievable']
            controversy_count = sum(1 for word in controversy_words if word.lower() in content.lower())
            features.append(min(controversy_count / 5, 1.0))
            
            # Question count (encourages interaction)
            question_count = content.count('?')
            features.append(min(question_count / 10, 1.0))
            
            # Content type virality potential
            virality_scores = {
                ContentType.SOCIAL_MEDIA: 0.9,
                ContentType.BLOG_POST: 0.7,
                ContentType.ARTICLE: 0.6,
                ContentType.EMAIL: 0.3,
                ContentType.PRODUCT_DESCRIPTION: 0.4,
                ContentType.TECHNICAL_DOCUMENT: 0.2,
                ContentType.ACADEMIC_PAPER: 0.1
            }
            features.append(virality_scores.get(content_type, 0.5))
            
            # Simple weighted average
            weights = [0.3, 0.3, 0.2, 0.2]
            virality_score = sum(f * w for f, w in zip(features, weights))
            
            return min(max(virality_score, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error predicting virality: {e}")
            return 0.5
    
    async def _calculate_seo_score(self, content: str, content_type: ContentType) -> float:
        """Calculate SEO score for content"""
        try:
            features = []
            
            # Word count (optimal range: 300-2000 words)
            word_count = len(content.split())
            if 300 <= word_count <= 2000:
                features.append(1.0)
            else:
                features.append(max(0.0, 1.0 - abs(word_count - 1000) / 1000))
            
            # Keyword density (optimal: 1-3%)
            # This is a simplified version
            features.append(0.7)  # Placeholder
            
            # Readability score
            readability = flesch_reading_ease(content)
            if 60 <= readability <= 80:
                features.append(1.0)
            else:
                features.append(max(0.0, 1.0 - abs(readability - 70) / 70))
            
            # Content type SEO potential
            seo_scores = {
                ContentType.ARTICLE: 0.9,
                ContentType.BLOG_POST: 0.8,
                ContentType.PRODUCT_DESCRIPTION: 0.7,
                ContentType.TECHNICAL_DOCUMENT: 0.6,
                ContentType.EMAIL: 0.3,
                ContentType.SOCIAL_MEDIA: 0.4,
                ContentType.ACADEMIC_PAPER: 0.5
            }
            features.append(seo_scores.get(content_type, 0.5))
            
            # Simple weighted average
            weights = [0.3, 0.2, 0.3, 0.2]
            seo_score = sum(f * w for f, w in zip(features, weights))
            
            return min(max(seo_score, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating SEO score: {e}")
            return 0.5
    
    async def _calculate_quality_score(self, content: str, content_type: ContentType) -> float:
        """Calculate overall content quality score"""
        try:
            features = []
            
            # Grammar and spelling (simplified)
            features.append(0.8)  # Placeholder for grammar check
            
            # Readability
            readability = flesch_reading_ease(content)
            features.append(min(readability / 100, 1.0))
            
            # Word variety
            words = content.lower().split()
            unique_words = set(words)
            word_variety = len(unique_words) / len(words) if words else 0
            features.append(word_variety)
            
            # Sentence structure variety
            sentences = sent_tokenize(content)
            sentence_lengths = [len(s.split()) for s in sentences]
            if sentence_lengths:
                length_variety = 1.0 - (np.std(sentence_lengths) / np.mean(sentence_lengths)) if np.mean(sentence_lengths) > 0 else 0
                features.append(max(0.0, length_variety))
            else:
                features.append(0.0)
            
            # Content completeness
            completeness_score = min(len(content) / 500, 1.0)  # Assume 500 chars is minimum
            features.append(completeness_score)
            
            # Simple weighted average
            weights = [0.2, 0.2, 0.2, 0.2, 0.2]
            quality_score = sum(f * w for f, w in zip(features, weights))
            
            return min(max(quality_score, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating quality score: {e}")
            return 0.5
    
    def _determine_complexity(self, readability_score: float) -> ComplexityLevel:
        """Determine content complexity level based on readability score"""
        if readability_score >= 80:
            return ComplexityLevel.VERY_SIMPLE
        elif readability_score >= 60:
            return ComplexityLevel.SIMPLE
        elif readability_score >= 40:
            return ComplexityLevel.MODERATE
        elif readability_score >= 20:
            return ComplexityLevel.COMPLEX
        else:
            return ComplexityLevel.VERY_COMPLEX
    
    def _determine_sentiment_type(self, sentiment_score: float) -> SentimentType:
        """Determine sentiment type based on sentiment score"""
        if sentiment_score >= 0.6:
            return SentimentType.VERY_POSITIVE
        elif sentiment_score >= 0.2:
            return SentimentType.POSITIVE
        elif sentiment_score >= -0.2:
            return SentimentType.NEUTRAL
        elif sentiment_score >= -0.6:
            return SentimentType.NEGATIVE
        else:
            return SentimentType.VERY_NEGATIVE
    
    async def generate_insights(self, content_id: str, content: str, metrics: ContentMetrics) -> List[ContentInsight]:
        """Generate actionable insights from content analysis"""
        try:
            insights = []
            
            # Engagement insight
            if metrics.engagement_score < 0.4:
                insights.append(ContentInsight(
                    content_id=content_id,
                    insight_type="engagement",
                    insight_value=metrics.engagement_score,
                    confidence=0.8,
                    explanation="Content has low engagement potential",
                    recommendations=[
                        "Add more questions to encourage interaction",
                        "Use more emotional language",
                        "Include call-to-action statements",
                        "Make content more conversational"
                    ]
                ))
            
            # Readability insight
            if metrics.readability_score < 40:
                insights.append(ContentInsight(
                    content_id=content_id,
                    insight_type="readability",
                    insight_value=metrics.readability_score,
                    confidence=0.9,
                    explanation="Content is difficult to read",
                    recommendations=[
                        "Use shorter sentences",
                        "Replace complex words with simpler alternatives",
                        "Break up long paragraphs",
                        "Use bullet points and subheadings"
                    ]
                ))
            
            # SEO insight
            if metrics.seo_score < 0.5:
                insights.append(ContentInsight(
                    content_id=content_id,
                    insight_type="seo",
                    insight_value=metrics.seo_score,
                    confidence=0.7,
                    explanation="Content has low SEO potential",
                    recommendations=[
                        "Optimize keyword density",
                        "Add relevant headings",
                        "Include meta descriptions",
                        "Improve content structure"
                    ]
                ))
            
            # Quality insight
            if metrics.quality_score < 0.6:
                insights.append(ContentInsight(
                    content_id=content_id,
                    insight_type="quality",
                    insight_value=metrics.quality_score,
                    confidence=0.8,
                    explanation="Content quality needs improvement",
                    recommendations=[
                        "Check grammar and spelling",
                        "Improve sentence structure variety",
                        "Add more detailed information",
                        "Ensure content completeness"
                    ]
                ))
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return []
    
    async def get_content_recommendations(self, content_id: str, content: str, 
                                        similar_content: List[Dict[str, Any]]) -> ContentRecommendation:
        """Get content recommendations based on similarity and performance"""
        try:
            # Calculate similarity scores
            similarity_scores = {}
            recommended_content = []
            
            for similar in similar_content:
                sim_id = similar.get('id', '')
                sim_content = similar.get('content', '')
                
                # Calculate similarity using TF-IDF
                vectorizer = TfidfVectorizer()
                tfidf_matrix = vectorizer.fit_transform([content, sim_content])
                similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
                
                similarity_scores[sim_id] = float(similarity)
                
                # Add to recommendations if similarity is high enough
                if similarity > 0.3:
                    recommended_content.append(sim_id)
            
            # Sort by similarity
            recommended_content.sort(key=lambda x: similarity_scores.get(x, 0), reverse=True)
            recommended_content = recommended_content[:10]  # Top 10 recommendations
            
            return ContentRecommendation(
                content_id=content_id,
                recommended_content=recommended_content,
                similarity_scores=similarity_scores,
                recommendation_reason="Based on content similarity and performance metrics",
                confidence=0.8
            )
            
        except Exception as e:
            logger.error(f"Error getting content recommendations: {e}")
            return ContentRecommendation(
                content_id=content_id,
                recommended_content=[],
                similarity_scores={},
                recommendation_reason="Error generating recommendations",
                confidence=0.0
            )
    
    async def analyze_content_trends(self, content_data: List[Dict[str, Any]], 
                                   time_period: str = "30d") -> Dict[str, Any]:
        """Analyze content trends over time"""
        try:
            # Convert to DataFrame for analysis
            df = pd.DataFrame(content_data)
            
            if df.empty:
                return {"error": "No content data available"}
            
            # Convert date columns
            if 'created_at' in df.columns:
                df['created_at'] = pd.to_datetime(df['created_at'])
                df['date'] = df['created_at'].dt.date
            
            # Trend analysis
            trends = {
                "total_content": len(df),
                "average_engagement": df.get('engagement_score', pd.Series([0.5] * len(df))).mean(),
                "average_quality": df.get('quality_score', pd.Series([0.5] * len(df))).mean(),
                "top_topics": [],
                "sentiment_distribution": {},
                "complexity_distribution": {},
                "performance_trends": {}
            }
            
            # Topic trends
            if 'topics' in df.columns:
                all_topics = []
                for topics in df['topics'].dropna():
                    if isinstance(topics, str):
                        all_topics.extend(topics.split(','))
                    elif isinstance(topics, list):
                        all_topics.extend(topics)
                
                topic_counts = Counter(all_topics)
                trends["top_topics"] = [{"topic": topic, "count": count} 
                                      for topic, count in topic_counts.most_common(10)]
            
            # Sentiment distribution
            if 'sentiment_type' in df.columns:
                sentiment_counts = df['sentiment_type'].value_counts()
                trends["sentiment_distribution"] = sentiment_counts.to_dict()
            
            # Complexity distribution
            if 'complexity_level' in df.columns:
                complexity_counts = df['complexity_level'].value_counts()
                trends["complexity_distribution"] = complexity_counts.to_dict()
            
            # Performance trends over time
            if 'date' in df.columns and 'engagement_score' in df.columns:
                daily_performance = df.groupby('date')['engagement_score'].mean()
                trends["performance_trends"] = {
                    "dates": daily_performance.index.tolist(),
                    "scores": daily_performance.values.tolist()
                }
            
            return trends
            
        except Exception as e:
            logger.error(f"Error analyzing content trends: {e}")
            return {"error": str(e)}
    
    async def competitive_analysis(self, content: str, competitor_content: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform competitive content analysis"""
        try:
            analysis = {
                "content_metrics": {},
                "competitive_positioning": {},
                "gaps_and_opportunities": [],
                "recommendations": []
            }
            
            # Analyze our content
            our_metrics = await self.analyze_content(content)
            analysis["content_metrics"] = {
                "engagement_score": our_metrics.engagement_score,
                "quality_score": our_metrics.quality_score,
                "seo_score": our_metrics.seo_score,
                "virality_potential": our_metrics.virality_potential,
                "readability_score": our_metrics.readability_score
            }
            
            # Analyze competitor content
            competitor_metrics = []
            for comp_content in competitor_content:
                comp_metrics = await self.analyze_content(comp_content.get('content', ''))
                competitor_metrics.append({
                    "id": comp_content.get('id', ''),
                    "metrics": comp_metrics
                })
            
            # Calculate averages
            if competitor_metrics:
                avg_engagement = np.mean([m["metrics"].engagement_score for m in competitor_metrics])
                avg_quality = np.mean([m["metrics"].quality_score for m in competitor_metrics])
                avg_seo = np.mean([m["metrics"].seo_score for m in competitor_metrics])
                avg_virality = np.mean([m["metrics"].virality_potential for m in competitor_metrics])
                avg_readability = np.mean([m["metrics"].readability_score for m in competitor_metrics])
                
                analysis["competitive_positioning"] = {
                    "engagement_vs_competitors": our_metrics.engagement_score - avg_engagement,
                    "quality_vs_competitors": our_metrics.quality_score - avg_quality,
                    "seo_vs_competitors": our_metrics.seo_score - avg_seo,
                    "virality_vs_competitors": our_metrics.virality_potential - avg_virality,
                    "readability_vs_competitors": our_metrics.readability_score - avg_readability
                }
                
                # Identify gaps and opportunities
                if our_metrics.engagement_score < avg_engagement:
                    analysis["gaps_and_opportunities"].append({
                        "area": "engagement",
                        "gap": avg_engagement - our_metrics.engagement_score,
                        "recommendation": "Improve engagement through interactive elements and emotional content"
                    })
                
                if our_metrics.quality_score < avg_quality:
                    analysis["gaps_and_opportunities"].append({
                        "area": "quality",
                        "gap": avg_quality - our_metrics.quality_score,
                        "recommendation": "Enhance content quality through better structure and grammar"
                    })
                
                if our_metrics.seo_score < avg_seo:
                    analysis["gaps_and_opportunities"].append({
                        "area": "seo",
                        "gap": avg_seo - our_metrics.seo_score,
                        "recommendation": "Optimize for search engines with better keywords and structure"
                    })
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error performing competitive analysis: {e}")
            return {"error": str(e)}
    
    async def batch_analyze_content(self, content_batch: List[Dict[str, Any]]) -> List[ContentMetrics]:
        """Analyze multiple content pieces in batch"""
        try:
            results = []
            
            for content_item in content_batch:
                content = content_item.get('content', '')
                content_type = ContentType(content_item.get('type', 'article'))
                
                metrics = await self.analyze_content(content, content_type)
                results.append(metrics)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch content analysis: {e}")
            return []
    
    async def export_analysis_report(self, content_id: str, metrics: ContentMetrics, 
                                   insights: List[ContentInsight]) -> Dict[str, Any]:
        """Export comprehensive analysis report"""
        try:
            report = {
                "content_id": content_id,
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "metrics": {
                    "word_count": metrics.word_count,
                    "sentence_count": metrics.sentence_count,
                    "paragraph_count": metrics.paragraph_count,
                    "character_count": metrics.character_count,
                    "average_sentence_length": metrics.average_sentence_length,
                    "average_word_length": metrics.average_word_length,
                    "readability_score": metrics.readability_score,
                    "complexity_level": metrics.complexity_level.value,
                    "sentiment_score": metrics.sentiment_score,
                    "sentiment_type": metrics.sentiment_type.value,
                    "emotion_scores": metrics.emotion_scores,
                    "topic_scores": metrics.topic_scores,
                    "keyword_density": metrics.keyword_density,
                    "engagement_score": metrics.engagement_score,
                    "virality_potential": metrics.virality_potential,
                    "seo_score": metrics.seo_score,
                    "quality_score": metrics.quality_score
                },
                "insights": [
                    {
                        "type": insight.insight_type,
                        "value": insight.insight_value,
                        "confidence": insight.confidence,
                        "explanation": insight.explanation,
                        "recommendations": insight.recommendations
                    }
                    for insight in insights
                ],
                "summary": {
                    "overall_score": (metrics.engagement_score + metrics.quality_score + 
                                    metrics.seo_score + metrics.virality_potential) / 4,
                    "strengths": [],
                    "weaknesses": [],
                    "priority_actions": []
                }
            }
            
            # Determine strengths and weaknesses
            if metrics.engagement_score > 0.7:
                report["summary"]["strengths"].append("High engagement potential")
            elif metrics.engagement_score < 0.4:
                report["summary"]["weaknesses"].append("Low engagement potential")
            
            if metrics.quality_score > 0.7:
                report["summary"]["strengths"].append("High content quality")
            elif metrics.quality_score < 0.4:
                report["summary"]["weaknesses"].append("Content quality needs improvement")
            
            if metrics.seo_score > 0.7:
                report["summary"]["strengths"].append("Good SEO optimization")
            elif metrics.seo_score < 0.4:
                report["summary"]["weaknesses"].append("Poor SEO optimization")
            
            # Priority actions
            if insights:
                report["summary"]["priority_actions"] = insights[0].recommendations[:3]
            
            return report
            
        except Exception as e:
            logger.error(f"Error exporting analysis report: {e}")
            return {"error": str(e)}

# Example usage and testing
async def main():
    """Example usage of the Content Intelligence Engine"""
    try:
        # Initialize engine
        config = {
            "models": {
                "sentiment": "vader",
                "topic_modeling": "lda",
                "engagement_prediction": "random_forest"
            },
            "cache_size": 1000,
            "batch_size": 50
        }
        
        engine = ContentIntelligenceEngine(config)
        
        # Sample content
        sample_content = """
        Artificial Intelligence is revolutionizing the way we work and live. 
        From healthcare to finance, AI is transforming industries at an unprecedented pace. 
        But what does this mean for the future of work? Will AI replace human workers, 
        or will it augment our capabilities? The answer is complex and depends on how 
        we choose to integrate AI into our society. Companies that embrace AI responsibly 
        and ethically will thrive, while those that resist change may find themselves 
        left behind. The key is to focus on human-AI collaboration rather than replacement.
        """
        
        # Analyze content
        print("Analyzing content...")
        metrics = await engine.analyze_content(sample_content, ContentType.ARTICLE)
        
        print(f"Word count: {metrics.word_count}")
        print(f"Readability score: {metrics.readability_score}")
        print(f"Sentiment: {metrics.sentiment_type.value}")
        print(f"Engagement score: {metrics.engagement_score}")
        print(f"Quality score: {metrics.quality_score}")
        
        # Generate insights
        print("\nGenerating insights...")
        insights = await engine.generate_insights("sample_1", sample_content, metrics)
        
        for insight in insights:
            print(f"\nInsight: {insight.insight_type}")
            print(f"Explanation: {insight.explanation}")
            print(f"Recommendations: {insight.recommendations}")
        
        # Export report
        print("\nExporting analysis report...")
        report = await engine.export_analysis_report("sample_1", metrics, insights)
        print(f"Overall score: {report['summary']['overall_score']:.2f}")
        print(f"Strengths: {report['summary']['strengths']}")
        print(f"Weaknesses: {report['summary']['weaknesses']}")
        
        print("\nContent Intelligence Engine demo completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main demo: {e}")

if __name__ == "__main__":
    asyncio.run(main())
























