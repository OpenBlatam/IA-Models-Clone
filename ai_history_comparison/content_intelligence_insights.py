"""
Content Intelligence & Insights Engine - Advanced Content Analytics and Intelligence
=================================================================================

This module provides comprehensive content intelligence capabilities including:
- Advanced content analytics and insights
- Predictive content performance modeling
- Content trend analysis and forecasting
- Content optimization recommendations
- Content audience analysis
- Content ROI and business impact analysis
- Content competitive analysis
- Content sentiment and emotion analysis
- Content topic modeling and clustering
- Content engagement prediction
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
import hashlib
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import redis
import sqlalchemy as sa
from sqlalchemy import create_engine, text
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import LatentDirichletAllocation, NMF, PCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
import textstat
import requests
from bs4 import BeautifulSoup
import openai
import anthropic
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import networkx as nx
from scipy import stats
from scipy.spatial.distance import cosine
import yfinance as yf
import tweepy
from googleapiclient.discovery import build
import boto3
from google.cloud import language_v1
import openai
import anthropic

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InsightType(Enum):
    """Insight type enumeration"""
    PERFORMANCE = "performance"
    TREND = "trend"
    OPTIMIZATION = "optimization"
    AUDIENCE = "audience"
    COMPETITIVE = "competitive"
    SENTIMENT = "sentiment"
    TOPIC = "topic"
    ENGAGEMENT = "engagement"
    ROI = "roi"
    PREDICTIVE = "predictive"

class ContentMetric(Enum):
    """Content metric enumeration"""
    VIEWS = "views"
    ENGAGEMENT = "engagement"
    SHARES = "shares"
    COMMENTS = "comments"
    LIKES = "likes"
    CLICKS = "clicks"
    CONVERSIONS = "conversions"
    BOUNCE_RATE = "bounce_rate"
    TIME_ON_PAGE = "time_on_page"
    REVENUE = "revenue"

class AudienceSegment(Enum):
    """Audience segment enumeration"""
    DEMOGRAPHIC = "demographic"
    BEHAVIORAL = "behavioral"
    PSYCHOGRAPHIC = "psychographic"
    GEOGRAPHIC = "geographic"
    TECHNOLOGICAL = "technological"

@dataclass
class ContentInsight:
    """Content insight data structure"""
    insight_id: str
    content_id: str
    insight_type: InsightType
    title: str
    description: str
    confidence_score: float
    impact_score: float
    recommendations: List[str] = field(default_factory=list)
    supporting_data: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None

@dataclass
class ContentAnalytics:
    """Content analytics data structure"""
    analytics_id: str
    content_id: str
    metrics: Dict[str, float] = field(default_factory=dict)
    trends: Dict[str, List[float]] = field(default_factory=dict)
    benchmarks: Dict[str, float] = field(default_factory=dict)
    calculated_at: datetime = field(default_factory=datetime.utcnow)
    time_period: str = "30d"

@dataclass
class PredictiveModel:
    """Predictive model data structure"""
    model_id: str
    model_type: str
    target_metric: ContentMetric
    features: List[str] = field(default_factory=list)
    accuracy: float = 0.0
    model_data: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_trained: datetime = field(default_factory=datetime.utcnow)

@dataclass
class ContentTrend:
    """Content trend data structure"""
    trend_id: str
    trend_name: str
    trend_type: str
    direction: str  # increasing, decreasing, stable
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    timeframe: str
    affected_content: List[str] = field(default_factory=list)
    description: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class AudienceInsight:
    """Audience insight data structure"""
    insight_id: str
    content_id: str
    segment_type: AudienceSegment
    segment_data: Dict[str, Any] = field(default_factory=dict)
    engagement_metrics: Dict[str, float] = field(default_factory=dict)
    preferences: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class CompetitiveAnalysis:
    """Competitive analysis data structure"""
    analysis_id: str
    content_id: str
    competitors: List[str] = field(default_factory=list)
    competitive_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    market_position: str = ""
    opportunities: List[str] = field(default_factory=list)
    threats: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)

class ContentIntelligenceInsights:
    """
    Advanced Content Intelligence & Insights Engine
    
    Provides comprehensive content intelligence and analytics capabilities
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the Content Intelligence & Insights Engine"""
        self.config = config
        self.content_insights = {}
        self.content_analytics = {}
        self.predictive_models = {}
        self.content_trends = {}
        self.audience_insights = {}
        self.competitive_analyses = {}
        self.redis_client = None
        self.database_engine = None
        
        # Initialize NLP components
        self._initialize_nlp()
        self._initialize_ml_models()
        self._initialize_external_apis()
        
        # Initialize components
        self._initialize_database()
        self._initialize_redis()
        
        # Start background tasks
        self._start_background_tasks()
        
        logger.info("Content Intelligence & Insights Engine initialized successfully")
    
    def _initialize_nlp(self):
        """Initialize NLP components"""
        try:
            # Initialize NLTK
            nltk.download('vader_lexicon', quiet=True)
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
            
            # Initialize spaCy
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy model not found, using basic tokenization")
                self.nlp = None
            
            # Initialize transformers
            try:
                self.sentiment_pipeline = pipeline("sentiment-analysis")
                self.emotion_pipeline = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
            except Exception as e:
                logger.warning(f"Transformers models not available: {e}")
                self.sentiment_pipeline = None
                self.emotion_pipeline = None
            
            logger.info("NLP components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing NLP components: {e}")
    
    def _initialize_ml_models(self):
        """Initialize ML models"""
        try:
            # Initialize vectorizers
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            self.count_vectorizer = CountVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            # Initialize clustering models
            self.kmeans = KMeans(n_clusters=5, random_state=42)
            self.dbscan = DBSCAN(eps=0.5, min_samples=5)
            
            # Initialize topic modeling
            self.lda = LatentDirichletAllocation(n_components=10, random_state=42)
            self.nmf = NMF(n_components=10, random_state=42)
            
            # Initialize regression models
            self.performance_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
            self.engagement_predictor = GradientBoostingRegressor(n_estimators=100, random_state=42)
            
            # Initialize scaler
            self.scaler = StandardScaler()
            
            logger.info("ML models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing ML models: {e}")
    
    def _initialize_external_apis(self):
        """Initialize external APIs"""
        try:
            # Initialize OpenAI
            if self.config.get("openai_api_key"):
                openai.api_key = self.config["openai_api_key"]
                logger.info("OpenAI API initialized")
            
            # Initialize Anthropic
            if self.config.get("anthropic_api_key"):
                self.anthropic_client = anthropic.Anthropic(api_key=self.config["anthropic_api_key"])
                logger.info("Anthropic API initialized")
            
            # Initialize Google Cloud Language API
            if self.config.get("gcp_credentials_path"):
                self.gcp_language_client = language_v1.LanguageServiceClient.from_service_account_file(
                    self.config["gcp_credentials_path"]
                )
                logger.info("Google Cloud Language API initialized")
            
            # Initialize social media APIs
            if self.config.get("twitter_api_key"):
                self.twitter_api = tweepy.API(tweepy.OAuthHandler(
                    self.config["twitter_api_key"],
                    self.config["twitter_api_secret"]
                ))
                logger.info("Twitter API initialized")
            
            # Initialize YouTube API
            if self.config.get("youtube_api_key"):
                self.youtube_api = build('youtube', 'v3', developerKey=self.config["youtube_api_key"])
                logger.info("YouTube API initialized")
            
            logger.info("External APIs initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing external APIs: {e}")
    
    def _initialize_database(self):
        """Initialize database connection"""
        try:
            if self.config.get("database_url"):
                self.database_engine = create_engine(self.config["database_url"])
                logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
    
    def _initialize_redis(self):
        """Initialize Redis connection"""
        try:
            if self.config.get("redis_url"):
                self.redis_client = redis.Redis.from_url(self.config["redis_url"])
                logger.info("Redis connection established")
        except Exception as e:
            logger.error(f"Error initializing Redis: {e}")
    
    def _start_background_tasks(self):
        """Start background tasks"""
        try:
            # Start analytics processing task
            asyncio.create_task(self._process_analytics_periodically())
            
            # Start trend analysis task
            asyncio.create_task(self._analyze_trends_periodically())
            
            # Start model training task
            asyncio.create_task(self._train_models_periodically())
            
            # Start insight generation task
            asyncio.create_task(self._generate_insights_periodically())
            
            logger.info("Background tasks started")
            
        except Exception as e:
            logger.error(f"Error starting background tasks: {e}")
    
    async def analyze_content_performance(self, content_id: str, metrics_data: Dict[str, Any]) -> ContentAnalytics:
        """Analyze content performance"""
        try:
            analytics_id = str(uuid.uuid4())
            
            # Calculate performance metrics
            metrics = await self._calculate_performance_metrics(metrics_data)
            
            # Calculate trends
            trends = await self._calculate_performance_trends(content_id, metrics_data)
            
            # Calculate benchmarks
            benchmarks = await self._calculate_performance_benchmarks(content_id, metrics)
            
            analytics = ContentAnalytics(
                analytics_id=analytics_id,
                content_id=content_id,
                metrics=metrics,
                trends=trends,
                benchmarks=benchmarks
            )
            
            # Store analytics
            self.content_analytics[analytics_id] = analytics
            
            logger.info(f"Content performance analytics completed for {content_id}")
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error analyzing content performance: {e}")
            raise
    
    async def _calculate_performance_metrics(self, metrics_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate performance metrics"""
        try:
            metrics = {}
            
            # Basic metrics
            metrics["views"] = metrics_data.get("views", 0)
            metrics["engagement"] = metrics_data.get("engagement", 0)
            metrics["shares"] = metrics_data.get("shares", 0)
            metrics["comments"] = metrics_data.get("comments", 0)
            metrics["likes"] = metrics_data.get("likes", 0)
            metrics["clicks"] = metrics_data.get("clicks", 0)
            metrics["conversions"] = metrics_data.get("conversions", 0)
            
            # Calculated metrics
            if metrics["views"] > 0:
                metrics["engagement_rate"] = metrics["engagement"] / metrics["views"]
                metrics["share_rate"] = metrics["shares"] / metrics["views"]
                metrics["comment_rate"] = metrics["comments"] / metrics["views"]
                metrics["like_rate"] = metrics["likes"] / metrics["views"]
                metrics["click_rate"] = metrics["clicks"] / metrics["views"]
                metrics["conversion_rate"] = metrics["conversions"] / metrics["views"]
            else:
                metrics["engagement_rate"] = 0.0
                metrics["share_rate"] = 0.0
                metrics["comment_rate"] = 0.0
                metrics["like_rate"] = 0.0
                metrics["click_rate"] = 0.0
                metrics["conversion_rate"] = 0.0
            
            # Time-based metrics
            if "time_on_page" in metrics_data:
                metrics["avg_time_on_page"] = np.mean(metrics_data["time_on_page"])
            
            if "bounce_rate" in metrics_data:
                metrics["bounce_rate"] = metrics_data["bounce_rate"]
            
            # Revenue metrics
            if "revenue" in metrics_data:
                metrics["revenue"] = metrics_data["revenue"]
                if metrics["views"] > 0:
                    metrics["revenue_per_view"] = metrics["revenue"] / metrics["views"]
                if metrics["conversions"] > 0:
                    metrics["revenue_per_conversion"] = metrics["revenue"] / metrics["conversions"]
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {}
    
    async def _calculate_performance_trends(self, content_id: str, metrics_data: Dict[str, Any]) -> Dict[str, List[float]]:
        """Calculate performance trends"""
        try:
            trends = {}
            
            # Get historical data (this would come from database in production)
            historical_data = await self._get_historical_metrics(content_id)
            
            if historical_data:
                # Calculate trends for each metric
                for metric in ["views", "engagement", "shares", "comments", "likes", "clicks", "conversions"]:
                    if metric in historical_data:
                        values = historical_data[metric]
                        if len(values) > 1:
                            # Calculate trend direction and strength
                            trend_direction = np.polyfit(range(len(values)), values, 1)[0]
                            trends[metric] = {
                                "values": values,
                                "direction": "increasing" if trend_direction > 0 else "decreasing",
                                "strength": abs(trend_direction),
                                "change_percent": ((values[-1] - values[0]) / values[0] * 100) if values[0] > 0 else 0
                            }
            
            return trends
            
        except Exception as e:
            logger.error(f"Error calculating performance trends: {e}")
            return {}
    
    async def _calculate_performance_benchmarks(self, content_id: str, metrics: Dict[str, float]) -> Dict[str, float]:
        """Calculate performance benchmarks"""
        try:
            benchmarks = {}
            
            # Get industry benchmarks (this would come from external data sources)
            industry_benchmarks = await self._get_industry_benchmarks()
            
            # Get competitor benchmarks
            competitor_benchmarks = await self._get_competitor_benchmarks(content_id)
            
            # Calculate relative performance
            for metric, value in metrics.items():
                if metric in industry_benchmarks:
                    benchmarks[f"{metric}_vs_industry"] = value / industry_benchmarks[metric] if industry_benchmarks[metric] > 0 else 0
                
                if metric in competitor_benchmarks:
                    benchmarks[f"{metric}_vs_competitors"] = value / competitor_benchmarks[metric] if competitor_benchmarks[metric] > 0 else 0
            
            return benchmarks
            
        except Exception as e:
            logger.error(f"Error calculating performance benchmarks: {e}")
            return {}
    
    async def _get_historical_metrics(self, content_id: str) -> Dict[str, List[float]]:
        """Get historical metrics for content"""
        try:
            # This would query the database for historical data
            # For demo purposes, return mock data
            return {
                "views": [100, 150, 200, 180, 220, 250, 280],
                "engagement": [10, 15, 20, 18, 22, 25, 28],
                "shares": [5, 8, 12, 10, 15, 18, 20]
            }
            
        except Exception as e:
            logger.error(f"Error getting historical metrics: {e}")
            return {}
    
    async def _get_industry_benchmarks(self) -> Dict[str, float]:
        """Get industry benchmarks"""
        try:
            # This would come from external data sources or industry reports
            return {
                "engagement_rate": 0.05,
                "share_rate": 0.02,
                "comment_rate": 0.01,
                "like_rate": 0.03,
                "click_rate": 0.08,
                "conversion_rate": 0.02
            }
            
        except Exception as e:
            logger.error(f"Error getting industry benchmarks: {e}")
            return {}
    
    async def _get_competitor_benchmarks(self, content_id: str) -> Dict[str, float]:
        """Get competitor benchmarks"""
        try:
            # This would analyze competitor content performance
            return {
                "engagement_rate": 0.06,
                "share_rate": 0.025,
                "comment_rate": 0.015,
                "like_rate": 0.035,
                "click_rate": 0.09,
                "conversion_rate": 0.025
            }
            
        except Exception as e:
            logger.error(f"Error getting competitor benchmarks: {e}")
            return {}
    
    async def analyze_content_sentiment(self, content_id: str, content: str) -> Dict[str, Any]:
        """Analyze content sentiment and emotions"""
        try:
            sentiment_analysis = {}
            
            # Basic sentiment analysis
            if self.sentiment_analyzer:
                sentiment_scores = self.sentiment_analyzer.polarity_scores(content)
                sentiment_analysis["basic_sentiment"] = {
                    "positive": sentiment_scores["pos"],
                    "negative": sentiment_scores["neg"],
                    "neutral": sentiment_scores["neu"],
                    "compound": sentiment_scores["compound"]
                }
            
            # Advanced sentiment analysis using transformers
            if self.sentiment_pipeline:
                try:
                    sentiment_result = self.sentiment_pipeline(content[:512])  # Limit length
                    sentiment_analysis["advanced_sentiment"] = sentiment_result[0]
                except Exception as e:
                    logger.warning(f"Advanced sentiment analysis failed: {e}")
            
            # Emotion analysis
            if self.emotion_pipeline:
                try:
                    emotion_result = self.emotion_pipeline(content[:512])
                    sentiment_analysis["emotions"] = emotion_result[0]
                except Exception as e:
                    logger.warning(f"Emotion analysis failed: {e}")
            
            # Readability analysis
            sentiment_analysis["readability"] = {
                "flesch_reading_ease": textstat.flesch_reading_ease(content),
                "flesch_kincaid_grade": textstat.flesch_kincaid_grade(content),
                "gunning_fog": textstat.gunning_fog(content),
                "smog_index": textstat.smog_index(content),
                "automated_readability_index": textstat.automated_readability_index(content)
            }
            
            # Text complexity analysis
            sentiment_analysis["complexity"] = {
                "avg_sentence_length": textstat.avg_sentence_length(content),
                "avg_syllables_per_word": textstat.avg_syllables_per_word(content),
                "difficult_words": textstat.difficult_words(content),
                "lexicon_count": textstat.lexicon_count(content)
            }
            
            logger.info(f"Content sentiment analysis completed for {content_id}")
            
            return sentiment_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing content sentiment: {e}")
            return {}
    
    async def perform_topic_modeling(self, content_list: List[str], num_topics: int = 10) -> Dict[str, Any]:
        """Perform topic modeling on content"""
        try:
            topic_analysis = {}
            
            # Prepare text data
            processed_texts = [self._preprocess_text(text) for text in content_list]
            
            # TF-IDF vectorization
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(processed_texts)
            
            # LDA topic modeling
            lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
            lda_topics = lda_model.fit_transform(tfidf_matrix)
            
            # NMF topic modeling
            nmf_model = NMF(n_components=num_topics, random_state=42)
            nmf_topics = nmf_model.fit_transform(tfidf_matrix)
            
            # Extract topic words
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            
            lda_topic_words = []
            for topic_idx, topic in enumerate(lda_model.components_):
                top_words_idx = topic.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                lda_topic_words.append({
                    "topic_id": topic_idx,
                    "words": top_words,
                    "weights": topic[top_words_idx].tolist()
                })
            
            nmf_topic_words = []
            for topic_idx, topic in enumerate(nmf_model.components_):
                top_words_idx = topic.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                nmf_topic_words.append({
                    "topic_id": topic_idx,
                    "words": top_words,
                    "weights": topic[top_words_idx].tolist()
                })
            
            topic_analysis = {
                "lda_topics": lda_topic_words,
                "nmf_topics": nmf_topic_words,
                "lda_topic_distributions": lda_topics.tolist(),
                "nmf_topic_distributions": nmf_topics.tolist(),
                "num_topics": num_topics,
                "num_documents": len(content_list)
            }
            
            logger.info(f"Topic modeling completed: {num_topics} topics from {len(content_list)} documents")
            
            return topic_analysis
            
        except Exception as e:
            logger.error(f"Error performing topic modeling: {e}")
            return {}
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for analysis"""
        try:
            # Convert to lowercase
            text = text.lower()
            
            # Remove special characters and digits
            import re
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            
            # Tokenize and remove stop words
            tokens = word_tokenize(text)
            tokens = [token for token in tokens if token not in self.stop_words]
            
            # Lemmatize
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
            
            return ' '.join(tokens)
            
        except Exception as e:
            logger.error(f"Error preprocessing text: {e}")
            return text
    
    async def cluster_content(self, content_list: List[str], num_clusters: int = 5) -> Dict[str, Any]:
        """Cluster content based on similarity"""
        try:
            clustering_analysis = {}
            
            # Prepare text data
            processed_texts = [self._preprocess_text(text) for text in content_list]
            
            # TF-IDF vectorization
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(processed_texts)
            
            # K-means clustering
            kmeans_model = KMeans(n_clusters=num_clusters, random_state=42)
            kmeans_labels = kmeans_model.fit_predict(tfidf_matrix)
            
            # DBSCAN clustering
            dbscan_model = DBSCAN(eps=0.5, min_samples=2)
            dbscan_labels = dbscan_model.fit_predict(tfidf_matrix)
            
            # Calculate silhouette scores
            kmeans_silhouette = silhouette_score(tfidf_matrix, kmeans_labels)
            dbscan_silhouette = silhouette_score(tfidf_matrix, dbscan_labels) if len(set(dbscan_labels)) > 1 else 0
            
            # Organize clusters
            kmeans_clusters = {}
            for i, label in enumerate(kmeans_labels):
                if label not in kmeans_clusters:
                    kmeans_clusters[label] = []
                kmeans_clusters[label].append(i)
            
            dbscan_clusters = {}
            for i, label in enumerate(dbscan_labels):
                if label not in dbscan_clusters:
                    dbscan_clusters[label] = []
                dbscan_clusters[label].append(i)
            
            clustering_analysis = {
                "kmeans_clusters": kmeans_clusters,
                "dbscan_clusters": dbscan_clusters,
                "kmeans_silhouette_score": kmeans_silhouette,
                "dbscan_silhouette_score": dbscan_silhouette,
                "num_clusters": num_clusters,
                "num_documents": len(content_list)
            }
            
            logger.info(f"Content clustering completed: {num_clusters} clusters from {len(content_list)} documents")
            
            return clustering_analysis
            
        except Exception as e:
            logger.error(f"Error clustering content: {e}")
            return {}
    
    async def predict_content_performance(self, content_id: str, content_features: Dict[str, Any]) -> Dict[str, Any]:
        """Predict content performance"""
        try:
            predictions = {}
            
            # Prepare features for prediction
            feature_vector = await self._prepare_feature_vector(content_features)
            
            # Predict engagement
            if hasattr(self, 'engagement_predictor') and self.engagement_predictor:
                engagement_prediction = self.engagement_predictor.predict([feature_vector])[0]
                predictions["predicted_engagement"] = max(0, engagement_prediction)
            
            # Predict views
            if hasattr(self, 'performance_predictor') and self.performance_predictor:
                views_prediction = self.performance_predictor.predict([feature_vector])[0]
                predictions["predicted_views"] = max(0, views_prediction)
            
            # Predict conversion rate
            if "predicted_engagement" in predictions and "predicted_views" in predictions:
                if predictions["predicted_views"] > 0:
                    predictions["predicted_conversion_rate"] = predictions["predicted_engagement"] / predictions["predicted_views"]
                else:
                    predictions["predicted_conversion_rate"] = 0.0
            
            # Add confidence intervals (simplified)
            for key in ["predicted_engagement", "predicted_views"]:
                if key in predictions:
                    predictions[f"{key}_confidence_lower"] = predictions[key] * 0.8
                    predictions[f"{key}_confidence_upper"] = predictions[key] * 1.2
            
            logger.info(f"Content performance prediction completed for {content_id}")
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error predicting content performance: {e}")
            return {}
    
    async def _prepare_feature_vector(self, content_features: Dict[str, Any]) -> List[float]:
        """Prepare feature vector for prediction"""
        try:
            features = []
            
            # Text features
            if "content" in content_features:
                content = content_features["content"]
                features.extend([
                    len(content),  # Content length
                    len(content.split()),  # Word count
                    len(content.split('.')) - 1,  # Sentence count
                    content.count('!'),  # Exclamation count
                    content.count('?'),  # Question count
                ])
            
            # Metadata features
            features.extend([
                content_features.get("word_count", 0),
                content_features.get("image_count", 0),
                content_features.get("video_count", 0),
                content_features.get("link_count", 0),
                content_features.get("hashtag_count", 0),
                content_features.get("mention_count", 0),
            ])
            
            # Time features
            if "publish_time" in content_features:
                publish_time = content_features["publish_time"]
                if isinstance(publish_time, str):
                    publish_time = datetime.fromisoformat(publish_time)
                
                features.extend([
                    publish_time.hour,
                    publish_time.weekday(),
                    publish_time.month,
                ])
            
            # Category features (one-hot encoded)
            categories = ["news", "entertainment", "sports", "technology", "business", "lifestyle"]
            for category in categories:
                features.append(1 if content_features.get("category") == category else 0)
            
            return features
            
        except Exception as e:
            logger.error(f"Error preparing feature vector: {e}")
            return [0.0] * 20  # Return default feature vector
    
    async def generate_content_insights(self, content_id: str, analytics: ContentAnalytics, 
                                      sentiment_analysis: Dict[str, Any]) -> List[ContentInsight]:
        """Generate content insights"""
        try:
            insights = []
            
            # Performance insights
            performance_insights = await self._generate_performance_insights(content_id, analytics)
            insights.extend(performance_insights)
            
            # Sentiment insights
            sentiment_insights = await self._generate_sentiment_insights(content_id, sentiment_analysis)
            insights.extend(sentiment_insights)
            
            # Optimization insights
            optimization_insights = await self._generate_optimization_insights(content_id, analytics)
            insights.extend(optimization_insights)
            
            # Store insights
            for insight in insights:
                self.content_insights[insight.insight_id] = insight
            
            logger.info(f"Generated {len(insights)} insights for content {content_id}")
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating content insights: {e}")
            return []
    
    async def _generate_performance_insights(self, content_id: str, analytics: ContentAnalytics) -> List[ContentInsight]:
        """Generate performance insights"""
        try:
            insights = []
            
            # High performance insight
            if analytics.metrics.get("engagement_rate", 0) > 0.1:
                insight = ContentInsight(
                    insight_id=str(uuid.uuid4()),
                    content_id=content_id,
                    insight_type=InsightType.PERFORMANCE,
                    title="High Engagement Performance",
                    description=f"Content achieved {analytics.metrics['engagement_rate']:.1%} engagement rate, significantly above average",
                    confidence_score=0.9,
                    impact_score=0.8,
                    recommendations=[
                        "Analyze what made this content successful",
                        "Create similar content to replicate success",
                        "Share insights with content team"
                    ],
                    supporting_data={"engagement_rate": analytics.metrics["engagement_rate"]}
                )
                insights.append(insight)
            
            # Low performance insight
            if analytics.metrics.get("engagement_rate", 0) < 0.02:
                insight = ContentInsight(
                    insight_id=str(uuid.uuid4()),
                    content_id=content_id,
                    insight_type=InsightType.PERFORMANCE,
                    title="Low Engagement Performance",
                    description=f"Content achieved only {analytics.metrics['engagement_rate']:.1%} engagement rate, below industry average",
                    confidence_score=0.8,
                    impact_score=0.6,
                    recommendations=[
                        "Review content quality and relevance",
                        "Improve headline and introduction",
                        "Add more engaging visuals",
                        "Consider different publishing time"
                    ],
                    supporting_data={"engagement_rate": analytics.metrics["engagement_rate"]}
                )
                insights.append(insight)
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating performance insights: {e}")
            return []
    
    async def _generate_sentiment_insights(self, content_id: str, sentiment_analysis: Dict[str, Any]) -> List[ContentInsight]:
        """Generate sentiment insights"""
        try:
            insights = []
            
            # Sentiment insight
            if "basic_sentiment" in sentiment_analysis:
                sentiment = sentiment_analysis["basic_sentiment"]
                
                if sentiment["compound"] > 0.5:
                    insight = ContentInsight(
                        insight_id=str(uuid.uuid4()),
                        content_id=content_id,
                        insight_type=InsightType.SENTIMENT,
                        title="Positive Sentiment Content",
                        description=f"Content has strong positive sentiment (compound score: {sentiment['compound']:.2f})",
                        confidence_score=0.8,
                        impact_score=0.7,
                        recommendations=[
                            "Leverage positive sentiment for brand building",
                            "Share on social media for positive engagement",
                            "Use as case study for successful content"
                        ],
                        supporting_data={"sentiment_scores": sentiment}
                    )
                    insights.append(insight)
                
                elif sentiment["compound"] < -0.5:
                    insight = ContentInsight(
                        insight_id=str(uuid.uuid4()),
                        content_id=content_id,
                        insight_type=InsightType.SENTIMENT,
                        title="Negative Sentiment Content",
                        description=f"Content has negative sentiment (compound score: {sentiment['compound']:.2f})",
                        confidence_score=0.8,
                        impact_score=0.6,
                        recommendations=[
                            "Review content for potential issues",
                            "Consider content revision",
                            "Monitor audience feedback carefully"
                        ],
                        supporting_data={"sentiment_scores": sentiment}
                    )
                    insights.append(insight)
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating sentiment insights: {e}")
            return []
    
    async def _generate_optimization_insights(self, content_id: str, analytics: ContentAnalytics) -> List[ContentInsight]:
        """Generate optimization insights"""
        try:
            insights = []
            
            # Benchmark comparison insights
            for benchmark_key, benchmark_value in analytics.benchmarks.items():
                if benchmark_value < 0.8:  # Below 80% of benchmark
                    insight = ContentInsight(
                        insight_id=str(uuid.uuid4()),
                        content_id=content_id,
                        insight_type=InsightType.OPTIMIZATION,
                        title=f"Below Benchmark Performance",
                        description=f"Content performs at {benchmark_value:.1%} of {benchmark_key.replace('_vs_', ' vs ')} benchmark",
                        confidence_score=0.7,
                        impact_score=0.6,
                        recommendations=[
                            f"Analyze {benchmark_key.replace('_vs_', ' vs ')} best practices",
                            "Implement optimization strategies",
                            "A/B test improvements"
                        ],
                        supporting_data={"benchmark": benchmark_key, "value": benchmark_value}
                    )
                    insights.append(insight)
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating optimization insights: {e}")
            return []
    
    async def _process_analytics_periodically(self):
        """Process analytics periodically"""
        while True:
            try:
                await asyncio.sleep(3600)  # Process every hour
                
                # In production, this would process new analytics data
                logger.info("Analytics processing completed")
                
            except Exception as e:
                logger.error(f"Error in analytics processing: {e}")
                await asyncio.sleep(3600)
    
    async def _analyze_trends_periodically(self):
        """Analyze trends periodically"""
        while True:
            try:
                await asyncio.sleep(86400)  # Analyze daily
                
                # In production, this would analyze content trends
                logger.info("Trend analysis completed")
                
            except Exception as e:
                logger.error(f"Error in trend analysis: {e}")
                await asyncio.sleep(86400)
    
    async def _train_models_periodically(self):
        """Train models periodically"""
        while True:
            try:
                await asyncio.sleep(604800)  # Train weekly
                
                # In production, this would retrain ML models
                logger.info("Model training completed")
                
            except Exception as e:
                logger.error(f"Error in model training: {e}")
                await asyncio.sleep(604800)
    
    async def _generate_insights_periodically(self):
        """Generate insights periodically"""
        while True:
            try:
                await asyncio.sleep(1800)  # Generate every 30 minutes
                
                # In production, this would generate new insights
                logger.info("Insight generation completed")
                
            except Exception as e:
                logger.error(f"Error in insight generation: {e}")
                await asyncio.sleep(1800)

# Example usage and testing
async def main():
    """Example usage of the Content Intelligence & Insights Engine"""
    try:
        # Initialize engine
        config = {
            "database_url": "postgresql://user:password@localhost/intelligencedb",
            "redis_url": "redis://localhost:6379",
            "openai_api_key": "your-openai-key",
            "anthropic_api_key": "your-anthropic-key"
        }
        
        engine = ContentIntelligenceInsights(config)
        
        # Analyze content performance
        print("Analyzing content performance...")
        metrics_data = {
            "views": 1000,
            "engagement": 150,
            "shares": 50,
            "comments": 25,
            "likes": 100,
            "clicks": 200,
            "conversions": 10,
            "time_on_page": [120, 150, 180, 200, 160],
            "bounce_rate": 0.3,
            "revenue": 500.0
        }
        
        analytics = await engine.analyze_content_performance("content_001", metrics_data)
        print(f"Analytics completed: {analytics.analytics_id}")
        print(f"Engagement rate: {analytics.metrics.get('engagement_rate', 0):.1%}")
        
        # Analyze content sentiment
        print("Analyzing content sentiment...")
        test_content = "This is an amazing product that has completely transformed our workflow! The features are incredible and the user experience is outstanding. I highly recommend it to everyone."
        
        sentiment_analysis = await engine.analyze_content_sentiment("content_001", test_content)
        print(f"Sentiment analysis completed")
        if "basic_sentiment" in sentiment_analysis:
            sentiment = sentiment_analysis["basic_sentiment"]
            print(f"Sentiment: {sentiment['compound']:.2f} (positive: {sentiment['pos']:.2f}, negative: {sentiment['neg']:.2f})")
        
        # Perform topic modeling
        print("Performing topic modeling...")
        content_list = [
            "Machine learning and artificial intelligence are revolutionizing technology",
            "Climate change and environmental sustainability are critical global issues",
            "Digital transformation is changing how businesses operate",
            "Renewable energy sources are becoming more cost-effective",
            "Data science and analytics are driving business decisions"
        ]
        
        topic_analysis = await engine.perform_topic_modeling(content_list, num_topics=3)
        print(f"Topic modeling completed: {topic_analysis['num_topics']} topics")
        print(f"LDA topics: {len(topic_analysis['lda_topics'])}")
        
        # Cluster content
        print("Clustering content...")
        clustering_analysis = await engine.cluster_content(content_list, num_clusters=3)
        print(f"Clustering completed: {clustering_analysis['num_clusters']} clusters")
        print(f"K-means silhouette score: {clustering_analysis['kmeans_silhouette_score']:.2f}")
        
        # Predict content performance
        print("Predicting content performance...")
        content_features = {
            "content": test_content,
            "word_count": len(test_content.split()),
            "image_count": 2,
            "video_count": 0,
            "link_count": 1,
            "hashtag_count": 0,
            "mention_count": 0,
            "publish_time": datetime.utcnow().isoformat(),
            "category": "technology"
        }
        
        predictions = await engine.predict_content_performance("content_001", content_features)
        print(f"Performance prediction completed")
        if "predicted_engagement" in predictions:
            print(f"Predicted engagement: {predictions['predicted_engagement']:.0f}")
        if "predicted_views" in predictions:
            print(f"Predicted views: {predictions['predicted_views']:.0f}")
        
        # Generate content insights
        print("Generating content insights...")
        insights = await engine.generate_content_insights("content_001", analytics, sentiment_analysis)
        print(f"Generated {len(insights)} insights")
        
        for insight in insights:
            print(f"- {insight.title}: {insight.description}")
            print(f"  Confidence: {insight.confidence_score:.1%}, Impact: {insight.impact_score:.1%}")
            print(f"  Recommendations: {insight.recommendations}")
        
        print("\nContent Intelligence & Insights Engine demo completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main demo: {e}")

if __name__ == "__main__":
    asyncio.run(main())
























