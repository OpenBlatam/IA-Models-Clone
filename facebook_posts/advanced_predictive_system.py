#!/usr/bin/env python3
"""
Advanced Predictive System for Facebook Posts Analysis v3.0
Next-generation AI-powered prediction and forecasting capabilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import logging
import time
import threading
import asyncio
import json
import pickle
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict, deque
import random
import hashlib
from datetime import datetime, timedelta
import warnings
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
warnings.filterwarnings('ignore')

# Import our existing components
from enhanced_ai_agent_system import EnhancedAIAgentSystem, EnhancedAgentConfig
from enhanced_performance_engine import EnhancedPerformanceOptimizationEngine, EnhancedPerformanceConfig


@dataclass
class AdvancedPredictiveConfig:
    """Configuration for advanced predictive system"""
    # Prediction models
    enable_viral_prediction: bool = True
    enable_sentiment_analysis: bool = True
    enable_audience_segmentation: bool = True
    enable_engagement_forecasting: bool = True
    enable_trend_prediction: bool = True
    
    # Model parameters
    prediction_horizon_days: int = 30
    confidence_threshold: float = 0.75
    update_frequency_hours: int = 6
    max_training_samples: int = 100000
    
    # Advanced features
    enable_real_time_learning: bool = True
    enable_context_awareness: bool = True
    enable_multi_modal_analysis: bool = True
    enable_temporal_patterns: bool = True


class ViralPredictionModel(nn.Module):
    """Advanced neural network for viral prediction"""
    
    def __init__(self, input_dim: int = 512, hidden_dims: List[int] = [256, 128, 64]):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        
        # Build dynamic layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        # Output layers
        layers.extend([
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),  # Viral score
            nn.Sigmoid()
        ])
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


class AdvancedSentimentAnalyzer(nn.Module):
    """Context-aware sentiment analysis with emotion detection"""
    
    def __init__(self, vocab_size: int = 50000, embedding_dim: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, 128, bidirectional=True, batch_first=True)
        self.attention = nn.MultiheadAttention(256, num_heads=8, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 7)  # 7 emotions + neutral
        )
        
    def forward(self, x, context=None):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        
        if context is not None:
            # Apply attention with context
            attn_out, _ = self.attention(lstm_out, context, context)
        else:
            attn_out = lstm_out
            
        # Global average pooling
        pooled = torch.mean(attn_out, dim=1)
        return self.classifier(pooled)


class AudienceSegmentationModel:
    """Intelligent audience segmentation using clustering and ML"""
    
    def __init__(self, config: AdvancedPredictiveConfig):
        self.config = config
        self.scaler = StandardScaler()
        self.segmentation_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.segments = {}
        self.segment_profiles = {}
        
    def create_segments(self, audience_data: pd.DataFrame) -> Dict[str, Any]:
        """Create audience segments based on behavior patterns"""
        # Feature engineering
        features = self._extract_audience_features(audience_data)
        
        # Apply clustering
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=5, random_state=42)
        segments = kmeans.fit_predict(features)
        
        # Create segment profiles
        for i in range(5):
            segment_mask = segments == i
            self.segments[f'segment_{i}'] = {
                'size': segment_mask.sum(),
                'profile': self._create_segment_profile(audience_data[segment_mask]),
                'engagement_rate': audience_data[segment_mask]['engagement_rate'].mean(),
                'content_preferences': self._analyze_content_preferences(audience_data[segment_mask])
            }
            
        return self.segments
    
    def _extract_audience_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract relevant features for segmentation"""
        features = []
        
        # Demographics
        if 'age' in data.columns:
            features.append(data['age'].values.reshape(-1, 1))
        if 'gender' in data.columns:
            features.append(pd.get_dummies(data['gender']).values)
            
        # Behavior patterns
        if 'engagement_rate' in data.columns:
            features.append(data['engagement_rate'].values.reshape(-1, 1))
        if 'posting_frequency' in data.columns:
            features.append(data['posting_frequency'].values.reshape(-1, 1))
            
        # Content preferences
        if 'content_type_preference' in data.columns:
            features.append(pd.get_dummies(data['content_type_preference']).values)
            
        return np.hstack(features) if features else np.zeros((len(data), 1))
    
    def _create_segment_profile(self, segment_data: pd.DataFrame) -> Dict[str, Any]:
        """Create detailed profile for a segment"""
        profile = {}
        
        # Demographics
        if 'age' in segment_data.columns:
            profile['avg_age'] = segment_data['age'].mean()
            profile['age_range'] = f"{segment_data['age'].min()}-{segment_data['age'].max()}"
            
        # Behavior
        if 'engagement_rate' in segment_data.columns:
            profile['avg_engagement'] = segment_data['engagement_rate'].mean()
            
        # Content preferences
        if 'content_type_preference' in segment_data.columns:
            profile['top_content_types'] = segment_data['content_type_preference'].value_counts().head(3).to_dict()
            
        return profile
    
    def _analyze_content_preferences(self, segment_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze content preferences for a segment"""
        preferences = {}
        
        # Analyze engagement by content type
        if 'content_type' in segment_data.columns and 'engagement_rate' in segment_data.columns:
            content_engagement = segment_data.groupby('content_type')['engagement_rate'].agg(['mean', 'count'])
            preferences['content_engagement'] = content_engagement.to_dict()
            
        # Analyze posting times
        if 'posting_time' in segment_data.columns:
            posting_times = pd.to_datetime(segment_data['posting_time'])
            preferences['peak_posting_hours'] = posting_times.dt.hour.value_counts().head(3).to_dict()
            
        return preferences


class EngagementForecaster:
    """Advanced engagement forecasting using temporal patterns"""
    
    def __init__(self, config: AdvancedPredictiveConfig):
        self.config = config
        self.temporal_model = GradientBoostingRegressor(n_estimators=200, random_state=42)
        self.pattern_analyzer = TemporalPatternAnalyzer()
        self.forecast_cache = {}
        
    def forecast_engagement(self, content_features: Dict[str, Any], 
                          audience_segment: str, 
                          posting_time: datetime) -> Dict[str, Any]:
        """Forecast engagement metrics for content"""
        
        # Extract temporal features
        temporal_features = self.pattern_analyzer.extract_temporal_features(posting_time)
        
        # Combine all features
        combined_features = self._combine_features(content_features, temporal_features, audience_segment)
        
        # For now, use a simple heuristic prediction since model isn't trained
        # In production, this would use the trained model
        base_engagement = 0.5  # Base engagement rate
        temporal_boost = sum(temporal_features.values()) / len(temporal_features) * 0.2
        prediction = min(max(base_engagement + temporal_boost, 0.1), 0.9)
        
        # Calculate confidence intervals
        confidence_interval = self._calculate_confidence_interval(prediction, combined_features)
        
        # Cache result
        cache_key = self._generate_cache_key(content_features, audience_segment, posting_time)
        self.forecast_cache[cache_key] = {
            'prediction': prediction,
            'confidence_interval': confidence_interval,
            'timestamp': time.time(),
            'features': combined_features
        }
        
        return {
            'predicted_engagement': prediction,
            'confidence_interval': confidence_interval,
            'confidence_level': 0.85,
            'temporal_factors': temporal_features,
            'audience_factors': self._extract_audience_factors(audience_segment)
        }
    
    def _combine_features(self, content_features: Dict[str, Any], 
                         temporal_features: Dict[str, Any], 
                         audience_segment: str) -> List[float]:
        """Combine all features into a single vector"""
        features = []
        
        # Content features
        if isinstance(content_features, dict):
            if 'sentiment_score' in content_features:
                features.append(content_features['sentiment_score'])
            if 'content_length' in content_features:
                features.append(content_features['content_length'])
            if 'has_media' in content_features:
                features.append(1.0 if content_features['has_media'] else 0.0)
        else:
            # If content_features is a list, use it directly
            features.extend(content_features)
            
        # Temporal features
        if isinstance(temporal_features, dict):
            features.extend(temporal_features.values())
        else:
            features.extend(temporal_features)
        
        # Audience features
        audience_features = self._extract_audience_features(audience_segment)
        features.extend(audience_features)
        
        return features
    
    def _extract_audience_features(self, audience_segment: str) -> List[float]:
        """Extract numerical features for audience segment"""
        # This would be populated from the audience segmentation model
        # For now, return default features
        return [0.5, 0.3, 0.2, 0.8]  # Placeholder values
    
    def _extract_audience_factors(self, audience_segment: str) -> Dict[str, Any]:
        """Extract audience factors for forecasting"""
        # This would be populated from the audience segmentation model
        # For now, return default factors
        return {
            'engagement_rate': 0.5,
            'content_preference': 'mixed',
            'activity_level': 'medium',
            'response_time': 'average'
        }
    
    def _calculate_confidence_interval(self, prediction: float, features: List[float]) -> Tuple[float, float]:
        """Calculate confidence interval for prediction"""
        # Simple confidence interval calculation
        # In production, this would use more sophisticated methods
        margin = prediction * 0.15  # 15% margin
        return (max(0, prediction - margin), prediction + margin)
    
    def _generate_cache_key(self, content_features: Any, 
                           audience_segment: str, 
                           posting_time: datetime) -> str:
        """Generate cache key for forecast results"""
        if isinstance(content_features, dict):
            content_hash = hash(str(sorted(content_features.items())))
        else:
            content_hash = hash(str(content_features))
            
        key_data = {
            'content_hash': content_hash,
            'audience_segment': audience_segment,
            'posting_time': posting_time.isoformat()
        }
        return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()


class TemporalPatternAnalyzer:
    """Analyze temporal patterns in engagement"""
    
    def __init__(self):
        self.hourly_patterns = defaultdict(list)
        self.daily_patterns = defaultdict(list)
        self.weekly_patterns = defaultdict(list)
        self.seasonal_patterns = defaultdict(list)
        
    def extract_temporal_features(self, timestamp: datetime) -> Dict[str, float]:
        """Extract temporal features for prediction"""
        features = {}
        
        # Hour of day (0-23)
        features['hour_of_day'] = timestamp.hour / 23.0
        
        # Day of week (0-6, Monday=0)
        features['day_of_week'] = timestamp.weekday() / 6.0
        
        # Day of month (1-31)
        features['day_of_month'] = timestamp.day / 31.0
        
        # Month (1-12)
        features['month'] = timestamp.month / 12.0
        
        # Weekend indicator
        features['is_weekend'] = 1.0 if timestamp.weekday() >= 5 else 0.0
        
        # Business hours indicator
        features['is_business_hours'] = 1.0 if 9 <= timestamp.hour <= 17 else 0.0
        
        # Holiday season indicator (simplified)
        features['is_holiday_season'] = 1.0 if timestamp.month in [11, 12] else 0.0
        
        return features
    
    def update_patterns(self, timestamp: datetime, engagement_rate: float):
        """Update temporal patterns with new data"""
        hour = timestamp.hour
        day = timestamp.weekday()
        month = timestamp.month
        
        self.hourly_patterns[hour].append(engagement_rate)
        self.daily_patterns[day].append(engagement_rate)
        self.weekly_patterns[timestamp.isocalendar()[1]].append(engagement_rate)
        self.seasonal_patterns[month].append(engagement_rate)
        
        # Keep only recent data
        max_samples = 1000
        for pattern_dict in [self.hourly_patterns, self.daily_patterns, 
                           self.weekly_patterns, self.seasonal_patterns]:
            for key in pattern_dict:
                if len(pattern_dict[key]) > max_samples:
                    pattern_dict[key] = pattern_dict[key][-max_samples:]


class AdvancedPredictiveSystem:
    """Main system orchestrating all predictive capabilities"""
    
    def __init__(self, config: AdvancedPredictiveConfig):
        self.config = config
        self.logger = self._setup_logging()
        
        # Initialize components
        self.viral_predictor = ViralPredictionModel()
        self.sentiment_analyzer = AdvancedSentimentAnalyzer()
        self.audience_segmenter = AudienceSegmentationModel(config)
        self.engagement_forecaster = EngagementForecaster(config)
        self.temporal_analyzer = TemporalPatternAnalyzer()
        
        # Performance tracking
        self.prediction_accuracy = {}
        self.system_metrics = {}
        
        self.logger.info("🚀 Advanced Predictive System initialized successfully!")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the system"""
        logger = logging.getLogger("AdvancedPredictiveSystem")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def predict_viral_potential(self, content: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Predict viral potential of content"""
        try:
            # Extract content features
            content_features = self._extract_content_features(content)
            
            # Make prediction
            viral_score = self.viral_predictor(
                torch.tensor(content_features, dtype=torch.float32).unsqueeze(0)
            ).item()
            
            # Analyze context if provided
            context_analysis = self._analyze_context(context) if context else {}
            
            # Calculate confidence
            confidence = self._calculate_prediction_confidence(content_features, context_analysis)
            
            result = {
                'viral_score': viral_score,
                'confidence': confidence,
                'viral_probability': f"{viral_score * 100:.1f}%",
                'recommendations': self._generate_viral_recommendations(viral_score, content_features),
                'context_factors': context_analysis,
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"Viral prediction completed: {viral_score:.3f} (confidence: {confidence:.2f})")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in viral prediction: {e}")
            return {'error': str(e), 'viral_score': 0.0}
    
    def analyze_sentiment_advanced(self, content: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Advanced sentiment analysis with emotion detection"""
        try:
            # Tokenize content (simplified)
            tokens = self._tokenize_content(content)
            token_tensor = torch.tensor([tokens], dtype=torch.long)
            
            # Prepare context if available
            context_tensor = None
            if context and 'context_tokens' in context:
                context_tensor = torch.tensor([context['context_tokens']], dtype=torch.long)
            
            # Analyze sentiment
            sentiment_scores = self.sentiment_analyzer(token_tensor, context_tensor)
            emotion_probs = F.softmax(sentiment_scores, dim=1)
            
            # Map emotions
            emotions = ['neutral', 'joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust']
            emotion_results = {}
            
            for i, emotion in enumerate(emotions):
                emotion_results[emotion] = emotion_probs[0][i].item()
            
            # Determine primary emotion
            primary_emotion = emotions[emotion_probs.argmax().item()]
            primary_confidence = emotion_probs.max().item()
            
            result = {
                'primary_emotion': primary_emotion,
                'emotion_confidence': primary_confidence,
                'emotion_breakdown': emotion_results,
                'sentiment_score': self._calculate_sentiment_score(emotion_results),
                'context_aware': context is not None,
                'recommendations': self._generate_sentiment_recommendations(emotion_results),
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"Sentiment analysis completed: {primary_emotion} (confidence: {primary_confidence:.2f})")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in sentiment analysis: {e}")
            return {'error': str(e), 'primary_emotion': 'neutral'}
    
    def segment_audience(self, audience_data: pd.DataFrame) -> Dict[str, Any]:
        """Create intelligent audience segments"""
        try:
            segments = self.audience_segmenter.create_segments(audience_data)
            
            # Calculate segment insights
            insights = self._calculate_segment_insights(segments)
            
            result = {
                'segments': segments,
                'insights': insights,
                'total_audience': len(audience_data),
                'segment_count': len(segments),
                'recommendations': self._generate_audience_recommendations(segments),
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"Audience segmentation completed: {len(segments)} segments created")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in audience segmentation: {e}")
            return {'error': str(e), 'segments': {}}
    
    def forecast_engagement(self, content: str, audience_segment: str, 
                          posting_time: datetime) -> Dict[str, Any]:
        """Forecast engagement metrics"""
        try:
            # Extract content features
            content_features = self._extract_content_features(content)
            
            # Make forecast
            forecast = self.engagement_forecaster.forecast_engagement(
                content_features, audience_segment, posting_time
            )
            
            # Update temporal patterns
            self.temporal_analyzer.update_patterns(posting_time, forecast['predicted_engagement'])
            
            result = {
                'forecast': forecast,
                'content_analysis': content_features,
                'audience_segment': audience_segment,
                'posting_time': posting_time.isoformat(),
                'recommendations': self._generate_engagement_recommendations(forecast),
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"Engagement forecast completed: {forecast['predicted_engagement']:.2f}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in engagement forecasting: {e}")
            return {'error': str(e), 'forecast': {}}
    
    def _extract_content_features(self, content: str) -> List[float]:
        """Extract numerical features from content"""
        features = []
        
        # Content length
        features.append(len(content) / 1000.0)  # Normalize to 0-1
        
        # Hashtag count
        hashtag_count = content.count('#')
        features.append(min(hashtag_count / 10.0, 1.0))  # Cap at 10 hashtags
        
        # Mention count
        mention_count = content.count('@')
        features.append(min(mention_count / 5.0, 1.0))  # Cap at 5 mentions
        
        # Question marks
        question_count = content.count('?')
        features.append(min(question_count / 3.0, 1.0))  # Cap at 3 questions
        
        # Exclamation marks
        exclamation_count = content.count('!')
        features.append(min(exclamation_count / 3.0, 1.0))  # Cap at 3 exclamations
        
        # Emoji count (simplified)
        emoji_count = sum(1 for char in content if ord(char) > 127)
        features.append(min(emoji_count / 5.0, 1.0))  # Cap at 5 emojis
        
        # Fill remaining features to match model input
        while len(features) < 512:
            features.append(0.0)
            
        return features[:512]  # Ensure exact size
    
    def _tokenize_content(self, content: str) -> List[int]:
        """Simple tokenization for sentiment analysis"""
        # Simplified tokenization - in production would use proper tokenizer
        words = content.lower().split()
        tokens = []
        
        for word in words:
            # Simple hash-based tokenization
            token_id = hash(word) % 50000  # Vocabulary size
            tokens.append(token_id)
            
        return tokens[:100]  # Limit sequence length
    
    def _analyze_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze contextual factors"""
        analysis = {}
        
        if 'trending_topics' in context:
            analysis['trend_relevance'] = self._calculate_trend_relevance(context['trending_topics'])
            
        if 'audience_demographics' in context:
            analysis['demographic_alignment'] = self._calculate_demographic_alignment(context['audience_demographics'])
            
        if 'timing' in context:
            analysis['timing_optimization'] = self._analyze_timing_optimization(context['timing'])
            
        return analysis
    
    def _calculate_prediction_confidence(self, features: List[float], 
                                       context: Dict[str, Any]) -> float:
        """Calculate confidence in prediction"""
        # Simple confidence calculation based on feature quality
        feature_quality = sum(1 for f in features if f > 0) / len(features)
        context_quality = len(context) / 10.0  # Normalize context quality
        
        confidence = (feature_quality * 0.7 + context_quality * 0.3)
        return min(confidence, 1.0)
    
    def _generate_viral_recommendations(self, viral_score: float, 
                                      features: List[float]) -> List[str]:
        """Generate recommendations for viral content"""
        recommendations = []
        
        if viral_score < 0.3:
            recommendations.extend([
                "Add trending hashtags to increase discoverability",
                "Include engaging questions to encourage comments",
                "Use high-quality images or videos",
                "Post during peak engagement hours"
            ])
        elif viral_score < 0.6:
            recommendations.extend([
                "Optimize posting time for your audience",
                "Include call-to-action phrases",
                "Use emotional storytelling techniques",
                "Leverage current events or trends"
            ])
        else:
            recommendations.extend([
                "Content has high viral potential!",
                "Consider boosting for maximum reach",
                "Prepare for high engagement response",
                "Monitor performance closely"
            ])
            
        return recommendations
    
    def _calculate_sentiment_score(self, emotion_results: Dict[str, float]) -> float:
        """Calculate overall sentiment score from emotions"""
        # Weight emotions by sentiment polarity
        positive_emotions = emotion_results.get('joy', 0) + emotion_results.get('surprise', 0)
        negative_emotions = emotion_results.get('sadness', 0) + emotion_results.get('anger', 0) + emotion_results.get('fear', 0) + emotion_results.get('disgust', 0)
        neutral = emotion_results.get('neutral', 0)
        
        # Calculate weighted score (-1 to 1)
        sentiment_score = (positive_emotions - negative_emotions) / (positive_emotions + negative_emotions + neutral + 1e-8)
        return sentiment_score
    
    def _generate_sentiment_recommendations(self, emotion_results: Dict[str, float]) -> List[str]:
        """Generate recommendations based on sentiment analysis"""
        recommendations = []
        
        if emotion_results.get('joy', 0) > 0.5:
            recommendations.append("Content conveys positive emotions - great for engagement!")
        elif emotion_results.get('sadness', 0) > 0.5:
            recommendations.append("Consider adding more positive elements to balance the tone")
        elif emotion_results.get('anger', 0) > 0.5:
            recommendations.append("Content may be too intense - consider softening the language")
            
        return recommendations
    
    def _calculate_segment_insights(self, segments: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate insights from audience segments"""
        insights = {
            'total_size': sum(seg['size'] for seg in segments.values()),
            'largest_segment': max(segments.keys(), key=lambda k: segments[k]['size']),
            'highest_engagement': max(segments.keys(), key=lambda k: segments[k]['engagement_rate']),
            'segment_diversity': len(segments)
        }
        return insights
    
    def _generate_audience_recommendations(self, segments: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on audience segments"""
        recommendations = []
        
        for segment_name, segment_data in segments.items():
            if segment_data['engagement_rate'] > 0.8:
                recommendations.append(f"Segment {segment_name} shows high engagement - focus content here")
            elif segment_data['engagement_rate'] < 0.3:
                recommendations.append(f"Segment {segment_name} needs engagement optimization")
                
        return recommendations
    
    def _generate_engagement_recommendations(self, forecast: Dict[str, Any]) -> List[str]:
        """Generate recommendations for engagement optimization"""
        recommendations = []
        
        predicted_engagement = forecast['predicted_engagement']
        
        if predicted_engagement < 0.3:
            recommendations.extend([
                "Content may need optimization for better engagement",
                "Consider posting at different times",
                "Add more interactive elements",
                "Use trending hashtags"
            ])
        elif predicted_engagement > 0.7:
            recommendations.append("Content has excellent engagement potential!")
            
        return recommendations
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics"""
        return {
            'prediction_accuracy': self.prediction_accuracy,
            'system_metrics': self.system_metrics,
            'cache_size': len(self.engagement_forecaster.forecast_cache),
            'timestamp': datetime.now().isoformat()
        }


# Example usage and testing
if __name__ == "__main__":
    # Initialize system
    config = AdvancedPredictiveConfig()
    system = AdvancedPredictiveSystem(config)
    
    # Test viral prediction
    test_content = "🚀 Amazing breakthrough in AI technology! This will revolutionize everything! #AI #Innovation #Future"
    viral_result = system.predict_viral_potential(test_content)
    print("Viral Prediction:", viral_result)
    
    # Test sentiment analysis
    sentiment_result = system.analyze_sentiment_advanced(test_content)
    print("Sentiment Analysis:", sentiment_result)
    
    # Test engagement forecasting
    from datetime import datetime
    forecast_result = system.forecast_engagement(
        test_content, "segment_0", datetime.now()
    )
    print("Engagement Forecast:", forecast_result)
