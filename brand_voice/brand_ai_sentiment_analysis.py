"""
Advanced Brand Sentiment Analysis System
========================================

This module provides comprehensive sentiment analysis capabilities for brand monitoring,
including multi-modal sentiment analysis, emotion detection, and brand reputation tracking.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import re
from collections import defaultdict, Counter
import aiohttp
import aiofiles
from pathlib import Path

# Deep Learning and NLP
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    pipeline, BertTokenizer, BertModel, RobertaTokenizer, RobertaModel,
    XLNetTokenizer, XLNetModel, DebertaTokenizer, DebertaModel
)
from sentence_transformers import SentenceTransformer
import spacy
from textblob import TextBlob
import vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Computer Vision for Visual Sentiment
import cv2
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import resnet50, efficientnet_b0

# Audio Processing for Voice Sentiment
import librosa
import soundfile as sf
from speech_recognition import Recognizer, Microphone
import whisper

# Advanced Analytics
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Database and Caching
import redis
import sqlite3
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Configuration
import yaml
from pydantic import BaseModel, Field

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration Models
class SentimentConfig(BaseModel):
    """Configuration for sentiment analysis system"""
    
    # Model configurations
    text_models: List[str] = Field(default=[
        "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "nlptown/bert-base-multilingual-uncased-sentiment",
        "ProsusAI/finbert",
        "microsoft/DialoGPT-medium"
    ])
    
    emotion_models: List[str] = Field(default=[
        "j-hartmann/emotion-english-distilroberta-base",
        "microsoft/DialoGPT-medium",
        "facebook/blenderbot-400M-distill"
    ])
    
    visual_models: List[str] = Field(default=[
        "google/vit-base-patch16-224",
        "microsoft/resnet-50",
        "facebook/deit-base-patch16-224"
    ])
    
    audio_models: List[str] = Field(default=[
        "openai/whisper-base",
        "facebook/wav2vec2-base-960h",
        "microsoft/speecht5_tts"
    ])
    
    # Analysis parameters
    confidence_threshold: float = 0.7
    batch_size: int = 32
    max_sequence_length: int = 512
    sampling_rate: int = 16000
    
    # Real-time parameters
    update_interval: int = 300  # 5 minutes
    alert_threshold: float = -0.5
    trend_window: int = 24  # hours
    
    # Database settings
    redis_url: str = "redis://localhost:6379"
    sqlite_path: str = "brand_sentiment.db"
    
    # API settings
    social_media_apis: Dict[str, str] = Field(default={
        "twitter": "https://api.twitter.com/2",
        "facebook": "https://graph.facebook.com/v18.0",
        "instagram": "https://graph.instagram.com/v18.0",
        "linkedin": "https://api.linkedin.com/v2",
        "youtube": "https://www.googleapis.com/youtube/v3",
        "tiktok": "https://open-api.tiktok.com"
    })

class SentimentType(Enum):
    """Types of sentiment analysis"""
    TEXT = "text"
    VISUAL = "visual"
    AUDIO = "audio"
    MULTIMODAL = "multimodal"
    BRAND = "brand"
    COMPETITIVE = "competitive"

class EmotionType(Enum):
    """Emotion categories"""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    NEUTRAL = "neutral"
    EXCITEMENT = "excitement"
    TRUST = "trust"
    ANTICIPATION = "anticipation"

@dataclass
class SentimentResult:
    """Result of sentiment analysis"""
    content: str
    sentiment_type: SentimentType
    sentiment_score: float  # -1 to 1
    confidence: float
    emotions: Dict[EmotionType, float]
    keywords: List[str]
    timestamp: datetime
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BrandSentimentProfile:
    """Comprehensive brand sentiment profile"""
    brand_name: str
    overall_sentiment: float
    sentiment_trend: List[float]
    emotion_distribution: Dict[EmotionType, float]
    key_topics: List[Tuple[str, float]]
    competitor_comparison: Dict[str, float]
    crisis_indicators: List[str]
    recommendations: List[str]
    last_updated: datetime

class AdvancedSentimentAnalyzer:
    """Advanced sentiment analysis system with multi-modal capabilities"""
    
    def __init__(self, config: SentimentConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize models
        self.text_models = {}
        self.emotion_models = {}
        self.visual_models = {}
        self.audio_models = {}
        
        # Initialize databases
        self.redis_client = redis.from_url(config.redis_url)
        self.db_engine = create_engine(f"sqlite:///{config.sqlite_path}")
        self.SessionLocal = sessionmaker(bind=self.db_engine)
        
        # Initialize NLP models
        self.nlp = spacy.load("en_core_web_sm")
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize audio models
        self.whisper_model = whisper.load_model("base")
        
        # Initialize visual models
        self.visual_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        logger.info("Advanced Sentiment Analyzer initialized successfully")
    
    async def initialize_models(self):
        """Initialize all sentiment analysis models"""
        try:
            # Load text sentiment models
            for model_name in self.config.text_models:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModelForSequenceClassification.from_pretrained(model_name)
                    self.text_models[model_name] = {
                        'tokenizer': tokenizer,
                        'model': model.to(self.device)
                    }
                    logger.info(f"Loaded text model: {model_name}")
                except Exception as e:
                    logger.warning(f"Failed to load text model {model_name}: {e}")
            
            # Load emotion models
            for model_name in self.config.emotion_models:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModelForSequenceClassification.from_pretrained(model_name)
                    self.emotion_models[model_name] = {
                        'tokenizer': tokenizer,
                        'model': model.to(self.device)
                    }
                    logger.info(f"Loaded emotion model: {model_name}")
                except Exception as e:
                    logger.warning(f"Failed to load emotion model {model_name}: {e}")
            
            # Load visual models
            for model_name in self.config.visual_models:
                try:
                    model = AutoModel.from_pretrained(model_name)
                    self.visual_models[model_name] = model.to(self.device)
                    logger.info(f"Loaded visual model: {model_name}")
                except Exception as e:
                    logger.warning(f"Failed to load visual model {model_name}: {e}")
            
            logger.info("All models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise
    
    async def analyze_text_sentiment(self, text: str, source: str = "unknown") -> SentimentResult:
        """Analyze sentiment of text content"""
        try:
            # Clean and preprocess text
            cleaned_text = self._preprocess_text(text)
            
            # Get sentiment scores from multiple models
            sentiment_scores = []
            emotion_scores = defaultdict(list)
            
            # VADER sentiment
            vader_scores = self.vader_analyzer.polarity_scores(cleaned_text)
            sentiment_scores.append(vader_scores['compound'])
            
            # TextBlob sentiment
            blob = TextBlob(cleaned_text)
            sentiment_scores.append(blob.sentiment.polarity)
            
            # Transformer models
            for model_name, model_data in self.text_models.items():
                try:
                    inputs = model_data['tokenizer'](
                        cleaned_text,
                        return_tensors="pt",
                        truncation=True,
                        padding=True,
                        max_length=self.config.max_sequence_length
                    ).to(self.device)
                    
                    with torch.no_grad():
                        outputs = model_data['model'](**inputs)
                        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                        
                        # Convert to sentiment score (-1 to 1)
                        if predictions.shape[1] == 3:  # negative, neutral, positive
                            sentiment = predictions[0][2] - predictions[0][0]  # positive - negative
                        elif predictions.shape[1] == 2:  # negative, positive
                            sentiment = predictions[0][1] - predictions[0][0]
                        else:
                            sentiment = predictions[0].mean() - 0.5
                        
                        sentiment_scores.append(sentiment.item())
                        
                except Exception as e:
                    logger.warning(f"Error with model {model_name}: {e}")
            
            # Emotion analysis
            for model_name, model_data in self.emotion_models.items():
                try:
                    inputs = model_data['tokenizer'](
                        cleaned_text,
                        return_tensors="pt",
                        truncation=True,
                        padding=True,
                        max_length=self.config.max_sequence_length
                    ).to(self.device)
                    
                    with torch.no_grad():
                        outputs = model_data['model'](**inputs)
                        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                        
                        # Map predictions to emotions (simplified)
                        emotion_labels = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust']
                        for i, emotion in enumerate(emotion_labels):
                            if i < predictions.shape[1]:
                                emotion_scores[emotion].append(predictions[0][i].item())
                        
                except Exception as e:
                    logger.warning(f"Error with emotion model {model_name}: {e}")
            
            # Calculate final scores
            final_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0.0
            confidence = 1.0 - np.std(sentiment_scores) if len(sentiment_scores) > 1 else 0.5
            
            # Calculate emotion distribution
            emotion_distribution = {}
            for emotion, scores in emotion_scores.items():
                if scores:
                    emotion_distribution[emotion] = np.mean(scores)
            
            # Extract keywords
            keywords = self._extract_keywords(cleaned_text)
            
            return SentimentResult(
                content=text,
                sentiment_type=SentimentType.TEXT,
                sentiment_score=final_sentiment,
                confidence=confidence,
                emotions=emotion_distribution,
                keywords=keywords,
                timestamp=datetime.now(),
                source=source,
                metadata={
                    'model_scores': sentiment_scores,
                    'emotion_scores': dict(emotion_scores),
                    'text_length': len(cleaned_text),
                    'word_count': len(cleaned_text.split())
                }
            )
            
        except Exception as e:
            logger.error(f"Error analyzing text sentiment: {e}")
            raise
    
    async def analyze_visual_sentiment(self, image_path: str, source: str = "unknown") -> SentimentResult:
        """Analyze sentiment of visual content"""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.visual_transforms(image).unsqueeze(0).to(self.device)
            
            # Analyze with visual models
            visual_scores = []
            emotion_scores = defaultdict(list)
            
            for model_name, model in self.visual_models.items():
                try:
                    with torch.no_grad():
                        features = model(image_tensor)
                        
                        # Simple sentiment mapping based on visual features
                        # This is a simplified approach - in practice, you'd need trained models
                        visual_sentiment = self._extract_visual_sentiment(features)
                        visual_scores.append(visual_sentiment)
                        
                except Exception as e:
                    logger.warning(f"Error with visual model {model_name}: {e}")
            
            # Color analysis for sentiment
            color_sentiment = self._analyze_color_sentiment(image)
            visual_scores.append(color_sentiment)
            
            # Composition analysis
            composition_sentiment = self._analyze_composition_sentiment(image)
            visual_scores.append(composition_sentiment)
            
            # Calculate final scores
            final_sentiment = np.mean(visual_scores) if visual_scores else 0.0
            confidence = 1.0 - np.std(visual_scores) if len(visual_scores) > 1 else 0.5
            
            # Extract visual keywords
            visual_keywords = self._extract_visual_keywords(image)
            
            return SentimentResult(
                content=image_path,
                sentiment_type=SentimentType.VISUAL,
                sentiment_score=final_sentiment,
                confidence=confidence,
                emotions=emotion_distribution,
                keywords=visual_keywords,
                timestamp=datetime.now(),
                source=source,
                metadata={
                    'visual_scores': visual_scores,
                    'color_sentiment': color_sentiment,
                    'composition_sentiment': composition_sentiment,
                    'image_size': image.size
                }
            )
            
        except Exception as e:
            logger.error(f"Error analyzing visual sentiment: {e}")
            raise
    
    async def analyze_audio_sentiment(self, audio_path: str, source: str = "unknown") -> SentimentResult:
        """Analyze sentiment of audio content"""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.config.sampling_rate)
            
            # Transcribe audio to text
            transcription = self.whisper_model.transcribe(audio_path)
            text = transcription['text']
            
            # Analyze transcribed text sentiment
            text_sentiment = await self.analyze_text_sentiment(text, source)
            
            # Analyze audio features
            audio_features = self._extract_audio_features(audio, sr)
            audio_sentiment = self._analyze_audio_features_sentiment(audio_features)
            
            # Combine text and audio sentiment
            combined_sentiment = (text_sentiment.sentiment_score + audio_sentiment) / 2
            combined_confidence = (text_sentiment.confidence + 0.7) / 2  # Audio confidence estimation
            
            return SentimentResult(
                content=text,
                sentiment_type=SentimentType.AUDIO,
                sentiment_score=combined_sentiment,
                confidence=combined_confidence,
                emotions=text_sentiment.emotions,
                keywords=text_sentiment.keywords,
                timestamp=datetime.now(),
                source=source,
                metadata={
                    'transcription': text,
                    'audio_sentiment': audio_sentiment,
                    'audio_features': audio_features,
                    'duration': len(audio) / sr
                }
            )
            
        except Exception as e:
            logger.error(f"Error analyzing audio sentiment: {e}")
            raise
    
    async def analyze_brand_sentiment(self, brand_name: str, time_window: int = 24) -> BrandSentimentProfile:
        """Analyze comprehensive brand sentiment"""
        try:
            # Get brand mentions from various sources
            brand_mentions = await self._collect_brand_mentions(brand_name, time_window)
            
            # Analyze sentiment for each mention
            sentiment_results = []
            for mention in brand_mentions:
                if mention['type'] == 'text':
                    result = await self.analyze_text_sentiment(mention['content'], mention['source'])
                elif mention['type'] == 'image':
                    result = await self.analyze_visual_sentiment(mention['content'], mention['source'])
                elif mention['type'] == 'audio':
                    result = await self.analyze_audio_sentiment(mention['content'], mention['source'])
                else:
                    continue
                
                sentiment_results.append(result)
            
            # Calculate overall sentiment
            if sentiment_results:
                overall_sentiment = np.mean([r.sentiment_score for r in sentiment_results])
                sentiment_trend = [r.sentiment_score for r in sentiment_results]
                
                # Calculate emotion distribution
                emotion_distribution = defaultdict(list)
                for result in sentiment_results:
                    for emotion, score in result.emotions.items():
                        emotion_distribution[emotion].append(score)
                
                emotion_dist = {emotion: np.mean(scores) for emotion, scores in emotion_distribution.items()}
                
                # Extract key topics
                all_keywords = []
                for result in sentiment_results:
                    all_keywords.extend(result.keywords)
                
                keyword_counts = Counter(all_keywords)
                key_topics = [(keyword, count) for keyword, count in keyword_counts.most_common(10)]
                
                # Competitor comparison
                competitor_comparison = await self._analyze_competitor_sentiment(brand_name, time_window)
                
                # Crisis indicators
                crisis_indicators = self._detect_crisis_indicators(sentiment_results)
                
                # Generate recommendations
                recommendations = self._generate_sentiment_recommendations(
                    overall_sentiment, emotion_dist, crisis_indicators
                )
                
                return BrandSentimentProfile(
                    brand_name=brand_name,
                    overall_sentiment=overall_sentiment,
                    sentiment_trend=sentiment_trend,
                    emotion_distribution=emotion_dist,
                    key_topics=key_topics,
                    competitor_comparison=competitor_comparison,
                    crisis_indicators=crisis_indicators,
                    recommendations=recommendations,
                    last_updated=datetime.now()
                )
            else:
                return BrandSentimentProfile(
                    brand_name=brand_name,
                    overall_sentiment=0.0,
                    sentiment_trend=[],
                    emotion_distribution={},
                    key_topics=[],
                    competitor_comparison={},
                    crisis_indicators=[],
                    recommendations=["No recent mentions found"],
                    last_updated=datetime.now()
                )
                
        except Exception as e:
            logger.error(f"Error analyzing brand sentiment: {e}")
            raise
    
    async def real_time_sentiment_monitoring(self, brand_name: str, callback_func=None):
        """Real-time sentiment monitoring for a brand"""
        try:
            while True:
                # Get latest sentiment
                brand_profile = await self.analyze_brand_sentiment(brand_name, 1)  # Last hour
                
                # Check for alerts
                if brand_profile.overall_sentiment < self.config.alert_threshold:
                    alert = {
                        'brand': brand_name,
                        'sentiment': brand_profile.overall_sentiment,
                        'timestamp': datetime.now(),
                        'crisis_indicators': brand_profile.crisis_indicators,
                        'recommendations': brand_profile.recommendations
                    }
                    
                    # Store alert
                    await self._store_sentiment_alert(alert)
                    
                    # Call callback if provided
                    if callback_func:
                        await callback_func(alert)
                
                # Store sentiment data
                await self._store_sentiment_data(brand_profile)
                
                # Wait for next update
                await asyncio.sleep(self.config.update_interval)
                
        except Exception as e:
            logger.error(f"Error in real-time monitoring: {e}")
            raise
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for sentiment analysis"""
        # Remove URLs, mentions, hashtags
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Convert to lowercase
        text = text.lower()
        
        return text
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        doc = self.nlp(text)
        keywords = []
        
        for token in doc:
            if (token.pos_ in ['NOUN', 'ADJ', 'VERB'] and 
                not token.is_stop and 
                not token.is_punct and 
                len(token.text) > 2):
                keywords.append(token.lemma_)
        
        return keywords[:10]  # Top 10 keywords
    
    def _extract_visual_sentiment(self, features: torch.Tensor) -> float:
        """Extract sentiment from visual features (simplified)"""
        # This is a simplified approach - in practice, you'd need trained models
        # that map visual features to sentiment scores
        feature_mean = features.mean().item()
        return np.tanh(feature_mean)  # Normalize to [-1, 1]
    
    def _analyze_color_sentiment(self, image: Image.Image) -> float:
        """Analyze sentiment based on color palette"""
        # Convert to numpy array
        img_array = np.array(image)
        
        # Calculate dominant colors
        pixels = img_array.reshape(-1, 3)
        
        # Simple color sentiment mapping
        # Bright, warm colors = positive sentiment
        # Dark, cool colors = negative sentiment
        brightness = np.mean(pixels)
        saturation = np.std(pixels)
        
        # Normalize to [-1, 1]
        sentiment = (brightness - 128) / 128
        return np.clip(sentiment, -1, 1)
    
    def _analyze_composition_sentiment(self, image: Image.Image) -> float:
        """Analyze sentiment based on image composition"""
        # Simple composition analysis
        # Symmetrical, balanced compositions = positive sentiment
        # Chaotic, unbalanced compositions = negative sentiment
        
        # This is a simplified approach
        # In practice, you'd use more sophisticated computer vision techniques
        return 0.0  # Neutral for now
    
    def _extract_visual_keywords(self, image: Image.Image) -> List[str]:
        """Extract keywords from visual content"""
        # This would typically use object detection and scene understanding
        # For now, return basic visual descriptors
        return ['image', 'visual', 'content']
    
    def _extract_audio_features(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract audio features for sentiment analysis"""
        features = {}
        
        # Spectral features
        features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
        features['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
        features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(audio))
        
        # Rhythm features
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
        features['tempo'] = tempo
        
        # Energy features
        features['rms_energy'] = np.mean(librosa.feature.rms(y=audio))
        features['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))
        
        return features
    
    def _analyze_audio_features_sentiment(self, features: Dict[str, float]) -> float:
        """Analyze sentiment from audio features"""
        # Simple mapping of audio features to sentiment
        # Higher tempo, energy = positive sentiment
        # Lower tempo, energy = negative sentiment
        
        tempo_sentiment = (features['tempo'] - 120) / 120  # Normalize around 120 BPM
        energy_sentiment = features['rms_energy'] * 10  # Scale energy
        
        combined_sentiment = (tempo_sentiment + energy_sentiment) / 2
        return np.clip(combined_sentiment, -1, 1)
    
    async def _collect_brand_mentions(self, brand_name: str, time_window: int) -> List[Dict]:
        """Collect brand mentions from various sources"""
        mentions = []
        
        # This would integrate with social media APIs
        # For now, return mock data
        mock_mentions = [
            {
                'content': f"Great product from {brand_name}!",
                'type': 'text',
                'source': 'twitter',
                'timestamp': datetime.now()
            },
            {
                'content': f"Not impressed with {brand_name}",
                'type': 'text',
                'source': 'facebook',
                'timestamp': datetime.now()
            }
        ]
        
        return mock_mentions
    
    async def _analyze_competitor_sentiment(self, brand_name: str, time_window: int) -> Dict[str, float]:
        """Analyze competitor sentiment for comparison"""
        # This would analyze competitor brands
        # For now, return mock data
        return {
            'competitor1': 0.2,
            'competitor2': -0.1,
            'competitor3': 0.3
        }
    
    def _detect_crisis_indicators(self, sentiment_results: List[SentimentResult]) -> List[str]:
        """Detect crisis indicators from sentiment results"""
        indicators = []
        
        # Check for negative sentiment spikes
        negative_count = sum(1 for r in sentiment_results if r.sentiment_score < -0.5)
        if negative_count > len(sentiment_results) * 0.3:
            indicators.append("High negative sentiment volume")
        
        # Check for anger emotion spikes
        anger_scores = [r.emotions.get('anger', 0) for r in sentiment_results if 'anger' in r.emotions]
        if anger_scores and np.mean(anger_scores) > 0.7:
            indicators.append("High anger emotion detected")
        
        # Check for crisis keywords
        crisis_keywords = ['crisis', 'scandal', 'boycott', 'lawsuit', 'recall']
        for result in sentiment_results:
            for keyword in crisis_keywords:
                if keyword in result.content.lower():
                    indicators.append(f"Crisis keyword detected: {keyword}")
        
        return indicators
    
    def _generate_sentiment_recommendations(self, sentiment: float, emotions: Dict, indicators: List[str]) -> List[str]:
        """Generate recommendations based on sentiment analysis"""
        recommendations = []
        
        if sentiment < -0.3:
            recommendations.append("Consider immediate reputation management response")
            recommendations.append("Monitor social media channels closely")
            recommendations.append("Prepare crisis communication plan")
        
        if 'anger' in emotions and emotions['anger'] > 0.6:
            recommendations.append("Address customer complaints promptly")
            recommendations.append("Consider public apology or explanation")
        
        if 'sadness' in emotions and emotions['sadness'] > 0.6:
            recommendations.append("Show empathy in communications")
            recommendations.append("Highlight positive brand stories")
        
        if sentiment > 0.3:
            recommendations.append("Leverage positive sentiment for marketing")
            recommendations.append("Engage with satisfied customers")
        
        return recommendations
    
    async def _store_sentiment_alert(self, alert: Dict):
        """Store sentiment alert in database"""
        try:
            # Store in Redis for real-time access
            alert_key = f"sentiment_alert:{alert['brand']}:{alert['timestamp'].isoformat()}"
            await self.redis_client.setex(alert_key, 86400, json.dumps(alert, default=str))
            
            # Store in SQLite for persistence
            # Implementation would go here
            
        except Exception as e:
            logger.error(f"Error storing sentiment alert: {e}")
    
    async def _store_sentiment_data(self, profile: BrandSentimentProfile):
        """Store sentiment data in database"""
        try:
            # Store in Redis
            profile_key = f"brand_sentiment:{profile.brand_name}"
            profile_data = {
                'overall_sentiment': profile.overall_sentiment,
                'emotion_distribution': {k.value: v for k, v in profile.emotion_distribution.items()},
                'key_topics': profile.key_topics,
                'last_updated': profile.last_updated.isoformat()
            }
            await self.redis_client.setex(profile_key, 3600, json.dumps(profile_data))
            
        except Exception as e:
            logger.error(f"Error storing sentiment data: {e}")

class SentimentAnalyticsDashboard:
    """Dashboard for sentiment analytics visualization"""
    
    def __init__(self, analyzer: AdvancedSentimentAnalyzer):
        self.analyzer = analyzer
    
    def create_sentiment_trend_chart(self, brand_name: str, days: int = 30) -> go.Figure:
        """Create sentiment trend visualization"""
        # This would fetch real data from the database
        # For now, create mock data
        dates = pd.date_range(start=datetime.now() - timedelta(days=days), end=datetime.now(), freq='D')
        sentiment_scores = np.random.normal(0.1, 0.3, len(dates))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=sentiment_scores,
            mode='lines+markers',
            name='Sentiment Score',
            line=dict(color='blue', width=2)
        ))
        
        fig.update_layout(
            title=f'Sentiment Trend for {brand_name}',
            xaxis_title='Date',
            yaxis_title='Sentiment Score',
            hovermode='x unified'
        )
        
        return fig
    
    def create_emotion_distribution_chart(self, emotions: Dict[EmotionType, float]) -> go.Figure:
        """Create emotion distribution visualization"""
        emotion_names = [e.value for e in emotions.keys()]
        emotion_values = list(emotions.values())
        
        fig = go.Figure(data=[
            go.Bar(x=emotion_names, y=emotion_values, marker_color='lightblue')
        ])
        
        fig.update_layout(
            title='Emotion Distribution',
            xaxis_title='Emotion',
            yaxis_title='Score'
        )
        
        return fig
    
    def create_competitor_comparison_chart(self, comparison_data: Dict[str, float]) -> go.Figure:
        """Create competitor comparison visualization"""
        brands = list(comparison_data.keys())
        sentiments = list(comparison_data.values())
        
        colors = ['green' if s > 0 else 'red' for s in sentiments]
        
        fig = go.Figure(data=[
            go.Bar(x=brands, y=sentiments, marker_color=colors)
        ])
        
        fig.update_layout(
            title='Competitor Sentiment Comparison',
            xaxis_title='Brand',
            yaxis_title='Sentiment Score'
        )
        
        return fig

# Example usage and testing
async def main():
    """Example usage of the sentiment analysis system"""
    try:
        # Initialize configuration
        config = SentimentConfig()
        
        # Initialize analyzer
        analyzer = AdvancedSentimentAnalyzer(config)
        await analyzer.initialize_models()
        
        # Analyze text sentiment
        text = "I love this brand! The products are amazing and the customer service is excellent."
        result = await analyzer.analyze_text_sentiment(text, "test")
        print(f"Text Sentiment: {result.sentiment_score:.3f} (confidence: {result.confidence:.3f})")
        print(f"Emotions: {result.emotions}")
        print(f"Keywords: {result.keywords}")
        
        # Analyze brand sentiment
        brand_profile = await analyzer.analyze_brand_sentiment("TestBrand", 24)
        print(f"\nBrand Sentiment: {brand_profile.overall_sentiment:.3f}")
        print(f"Key Topics: {brand_profile.key_topics}")
        print(f"Recommendations: {brand_profile.recommendations}")
        
        # Create dashboard
        dashboard = SentimentAnalyticsDashboard(analyzer)
        trend_chart = dashboard.create_sentiment_trend_chart("TestBrand", 30)
        print(f"\nCreated sentiment trend chart")
        
        logger.info("Sentiment analysis system test completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
























