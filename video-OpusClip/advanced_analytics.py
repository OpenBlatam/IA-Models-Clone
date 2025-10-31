"""
Advanced Analytics with Machine Learning for Ultimate Opus Clip

Comprehensive analytics system with ML-powered insights,
predictive modeling, and intelligent recommendations.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Tuple, Union
import asyncio
import time
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from enum import Enum
import structlog
from pathlib import Path
import json
import pickle
from datetime import datetime, timedelta
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logger = structlog.get_logger("advanced_analytics")

class MetricType(Enum):
    """Types of metrics for analytics."""
    ENGAGEMENT = "engagement"
    PERFORMANCE = "performance"
    QUALITY = "quality"
    VIRAL_POTENTIAL = "viral_potential"
    USER_BEHAVIOR = "user_behavior"
    CONTENT_ANALYSIS = "content_analysis"

class PredictionType(Enum):
    """Types of predictions."""
    VIRAL_SCORE = "viral_score"
    ENGAGEMENT_RATE = "engagement_rate"
    COMPLETION_RATE = "completion_rate"
    SHARE_PROBABILITY = "share_probability"
    TREND_PREDICTION = "trend_prediction"

@dataclass
class AnalyticsDataPoint:
    """A single data point for analytics."""
    timestamp: float
    metric_type: MetricType
    value: float
    metadata: Dict[str, Any]
    user_id: Optional[str] = None
    content_id: Optional[str] = None
    platform: Optional[str] = None

@dataclass
class PredictionResult:
    """Result of a prediction."""
    prediction_type: PredictionType
    predicted_value: float
    confidence: float
    factors: Dict[str, float]
    recommendations: List[str]
    timestamp: float

@dataclass
class TrendAnalysis:
    """Analysis of trends in data."""
    trend_direction: str  # "increasing", "decreasing", "stable"
    trend_strength: float  # 0-1
    change_rate: float
    significance: float
    period: str
    forecast: List[float]

@dataclass
class UserSegment:
    """User segment for analytics."""
    segment_id: str
    name: str
    characteristics: Dict[str, Any]
    size: int
    engagement_rate: float
    content_preferences: List[str]
    behavior_patterns: Dict[str, Any]

class MLPredictor:
    """Machine learning predictor for various metrics."""
    
    def __init__(self):
        self.models: Dict[PredictionType, Any] = {}
        self.scalers: Dict[PredictionType, StandardScaler] = {}
        self.feature_importance: Dict[PredictionType, Dict[str, float]] = {}
        self.training_data: Dict[PredictionType, pd.DataFrame] = {}
        
        logger.info("ML Predictor initialized")
    
    def train_viral_score_model(self, data: pd.DataFrame):
        """Train model to predict viral scores."""
        try:
            # Prepare features
            feature_columns = [
                'duration', 'quality_score', 'engagement_early', 'sentiment_score',
                'topic_relevance', 'creator_followers', 'upload_time_hour',
                'content_type_score', 'thumbnail_attractiveness', 'title_length'
            ]
            
            # Filter available features
            available_features = [col for col in feature_columns if col in data.columns]
            X = data[available_features].fillna(0)
            y = data['viral_score'].fillna(0)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Store model
            self.models[PredictionType.VIRAL_SCORE] = model
            self.scalers[PredictionType.VIRAL_SCORE] = scaler
            self.training_data[PredictionType.VIRAL_SCORE] = data
            
            # Calculate feature importance
            feature_importance = dict(zip(available_features, model.feature_importances_))
            self.feature_importance[PredictionType.VIRAL_SCORE] = feature_importance
            
            logger.info(f"Viral score model trained - MSE: {mse:.4f}, R2: {r2:.4f}")
            
        except Exception as e:
            logger.error(f"Error training viral score model: {e}")
    
    def train_engagement_model(self, data: pd.DataFrame):
        """Train model to predict engagement rates."""
        try:
            # Prepare features
            feature_columns = [
                'duration', 'quality_score', 'content_type', 'upload_day',
                'creator_engagement_rate', 'hashtag_count', 'mention_count',
                'title_sentiment', 'description_length', 'thumbnail_score'
            ]
            
            available_features = [col for col in feature_columns if col in data.columns]
            X = data[available_features].fillna(0)
            y = data['engagement_rate'].fillna(0)
            
            # Split and scale
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Store model
            self.models[PredictionType.ENGAGEMENT_RATE] = model
            self.scalers[PredictionType.ENGAGEMENT_RATE] = scaler
            self.training_data[PredictionType.ENGAGEMENT_RATE] = data
            
            # Feature importance
            feature_importance = dict(zip(available_features, model.feature_importances_))
            self.feature_importance[PredictionType.ENGAGEMENT_RATE] = feature_importance
            
            logger.info(f"Engagement model trained - MSE: {mse:.4f}, R2: {r2:.4f}")
            
        except Exception as e:
            logger.error(f"Error training engagement model: {e}")
    
    def predict(self, prediction_type: PredictionType, features: Dict[str, float]) -> PredictionResult:
        """Make a prediction."""
        try:
            if prediction_type not in self.models:
                raise ValueError(f"Model not trained for {prediction_type}")
            
            model = self.models[prediction_type]
            scaler = self.scalers[prediction_type]
            feature_importance = self.feature_importance[prediction_type]
            
            # Prepare features
            feature_vector = []
            for feature_name in feature_importance.keys():
                feature_vector.append(features.get(feature_name, 0.0))
            
            # Scale features
            feature_vector_scaled = scaler.transform([feature_vector])
            
            # Make prediction
            predicted_value = model.predict(feature_vector_scaled)[0]
            
            # Calculate confidence (simplified)
            confidence = min(0.95, max(0.1, abs(predicted_value) / 10))
            
            # Generate recommendations
            recommendations = self._generate_recommendations(prediction_type, features, feature_importance)
            
            return PredictionResult(
                prediction_type=prediction_type,
                predicted_value=predicted_value,
                confidence=confidence,
                factors=feature_importance,
                recommendations=recommendations,
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return PredictionResult(
                prediction_type=prediction_type,
                predicted_value=0.0,
                confidence=0.0,
                factors={},
                recommendations=["Unable to make prediction"],
                timestamp=time.time()
            )
    
    def _generate_recommendations(self, prediction_type: PredictionType, features: Dict[str, float], importance: Dict[str, float]) -> List[str]:
        """Generate recommendations based on prediction."""
        recommendations = []
        
        if prediction_type == PredictionType.VIRAL_SCORE:
            if features.get('duration', 0) > 60:
                recommendations.append("Consider shortening video duration for better viral potential")
            if features.get('quality_score', 0) < 0.7:
                recommendations.append("Improve video quality to increase viral potential")
            if features.get('engagement_early', 0) < 0.3:
                recommendations.append("Focus on strong opening to boost early engagement")
        
        elif prediction_type == PredictionType.ENGAGEMENT_RATE:
            if features.get('hashtag_count', 0) < 3:
                recommendations.append("Add more relevant hashtags to increase discoverability")
            if features.get('title_length', 0) < 10:
                recommendations.append("Use more descriptive titles to improve engagement")
            if features.get('creator_engagement_rate', 0) < 0.05:
                recommendations.append("Increase creator engagement with audience")
        
        return recommendations

class TrendAnalyzer:
    """Analyze trends in data over time."""
    
    def __init__(self):
        self.trend_cache: Dict[str, TrendAnalysis] = {}
        logger.info("Trend Analyzer initialized")
    
    def analyze_trend(self, data: List[AnalyticsDataPoint], metric_type: MetricType, period: str = "daily") -> TrendAnalysis:
        """Analyze trend in data."""
        try:
            # Filter data by metric type
            filtered_data = [dp for dp in data if dp.metric_type == metric_type]
            
            if len(filtered_data) < 10:
                return self._create_default_trend()
            
            # Sort by timestamp
            filtered_data.sort(key=lambda x: x.timestamp)
            
            # Extract values and timestamps
            values = [dp.value for dp in filtered_data]
            timestamps = [dp.timestamp for dp in filtered_data]
            
            # Calculate trend
            x = np.arange(len(values))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
            
            # Determine trend direction
            if slope > 0.01:
                direction = "increasing"
            elif slope < -0.01:
                direction = "decreasing"
            else:
                direction = "stable"
            
            # Calculate trend strength
            trend_strength = abs(r_value)
            
            # Calculate change rate
            if len(values) > 1:
                change_rate = (values[-1] - values[0]) / values[0] if values[0] != 0 else 0
            else:
                change_rate = 0
            
            # Generate forecast
            forecast = self._generate_forecast(values, slope, intercept, 7)
            
            trend_analysis = TrendAnalysis(
                trend_direction=direction,
                trend_strength=trend_strength,
                change_rate=change_rate,
                significance=1 - p_value,
                period=period,
                forecast=forecast
            )
            
            # Cache result
            cache_key = f"{metric_type.value}_{period}"
            self.trend_cache[cache_key] = trend_analysis
            
            return trend_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing trend: {e}")
            return self._create_default_trend()
    
    def _create_default_trend(self) -> TrendAnalysis:
        """Create default trend analysis."""
        return TrendAnalysis(
            trend_direction="stable",
            trend_strength=0.0,
            change_rate=0.0,
            significance=0.0,
            period="daily",
            forecast=[0.0] * 7
        )
    
    def _generate_forecast(self, values: List[float], slope: float, intercept: float, days: int) -> List[float]:
        """Generate forecast for future values."""
        forecast = []
        last_index = len(values) - 1
        
        for i in range(1, days + 1):
            predicted_value = slope * (last_index + i) + intercept
            forecast.append(max(0, predicted_value))  # Ensure non-negative values
        
        return forecast

class UserSegmentation:
    """User segmentation using clustering."""
    
    def __init__(self):
        self.segments: Dict[str, UserSegment] = {}
        self.clustering_model = None
        self.scaler = StandardScaler()
        logger.info("User Segmentation initialized")
    
    def create_segments(self, user_data: pd.DataFrame) -> List[UserSegment]:
        """Create user segments using clustering."""
        try:
            # Prepare features for clustering
            feature_columns = [
                'engagement_rate', 'content_creation_frequency', 'preferred_duration',
                'quality_preference', 'platform_usage', 'upload_time_preference',
                'hashtag_usage', 'interaction_rate', 'content_type_preference'
            ]
            
            available_features = [col for col in feature_columns if col in user_data.columns]
            X = user_data[available_features].fillna(0)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Determine optimal number of clusters
            optimal_k = self._find_optimal_clusters(X_scaled)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=optimal_k, random_state=42)
            cluster_labels = kmeans.fit_predict(X_scaled)
            
            # Create segments
            segments = []
            for i in range(optimal_k):
                cluster_mask = cluster_labels == i
                cluster_data = user_data[cluster_mask]
                
                if len(cluster_data) == 0:
                    continue
                
                # Calculate segment characteristics
                characteristics = {
                    'avg_engagement_rate': cluster_data['engagement_rate'].mean(),
                    'avg_content_frequency': cluster_data['content_creation_frequency'].mean(),
                    'preferred_duration': cluster_data['preferred_duration'].mean(),
                    'quality_preference': cluster_data['quality_preference'].mean()
                }
                
                # Determine content preferences
                content_preferences = self._analyze_content_preferences(cluster_data)
                
                # Analyze behavior patterns
                behavior_patterns = self._analyze_behavior_patterns(cluster_data)
                
                segment = UserSegment(
                    segment_id=f"segment_{i}",
                    name=f"Segment {i+1}",
                    characteristics=characteristics,
                    size=len(cluster_data),
                    engagement_rate=cluster_data['engagement_rate'].mean(),
                    content_preferences=content_preferences,
                    behavior_patterns=behavior_patterns
                )
                
                segments.append(segment)
                self.segments[segment.segment_id] = segment
            
            self.clustering_model = kmeans
            logger.info(f"Created {len(segments)} user segments")
            
            return segments
            
        except Exception as e:
            logger.error(f"Error creating user segments: {e}")
            return []
    
    def _find_optimal_clusters(self, X_scaled: np.ndarray) -> int:
        """Find optimal number of clusters using elbow method."""
        try:
            max_k = min(10, len(X_scaled) // 10)
            if max_k < 2:
                return 2
            
            inertias = []
            k_range = range(2, max_k + 1)
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(X_scaled)
                inertias.append(kmeans.inertia_)
            
            # Find elbow point
            if len(inertias) < 3:
                return 2
            
            # Simple elbow detection
            diffs = np.diff(inertias)
            second_diffs = np.diff(diffs)
            elbow_idx = np.argmax(second_diffs) + 2
            
            return k_range[elbow_idx]
            
        except Exception:
            return 3  # Default fallback
    
    def _analyze_content_preferences(self, cluster_data: pd.DataFrame) -> List[str]:
        """Analyze content preferences for a cluster."""
        # This is a simplified analysis
        preferences = []
        
        if 'content_type_preference' in cluster_data.columns:
            avg_preference = cluster_data['content_type_preference'].mean()
            if avg_preference > 0.7:
                preferences.append("high_quality_content")
            elif avg_preference > 0.4:
                preferences.append("balanced_content")
            else:
                preferences.append("casual_content")
        
        return preferences
    
    def _analyze_behavior_patterns(self, cluster_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze behavior patterns for a cluster."""
        patterns = {}
        
        if 'upload_time_preference' in cluster_data.columns:
            avg_upload_time = cluster_data['upload_time_preference'].mean()
            if avg_upload_time < 12:
                patterns['upload_pattern'] = 'morning_uploader'
            elif avg_upload_time < 18:
                patterns['upload_pattern'] = 'afternoon_uploader'
            else:
                patterns['upload_pattern'] = 'evening_uploader'
        
        return patterns

class AdvancedAnalytics:
    """Main advanced analytics orchestrator."""
    
    def __init__(self):
        self.ml_predictor = MLPredictor()
        self.trend_analyzer = TrendAnalyzer()
        self.user_segmentation = UserSegmentation()
        self.data_points: List[AnalyticsDataPoint] = []
        
        logger.info("Advanced Analytics initialized")
    
    def add_data_point(self, data_point: AnalyticsDataPoint):
        """Add a data point for analysis."""
        self.data_points.append(data_point)
        
        # Keep only recent data (last 30 days)
        cutoff_time = time.time() - (30 * 24 * 60 * 60)
        self.data_points = [dp for dp in self.data_points if dp.timestamp > cutoff_time]
    
    def get_insights(self) -> Dict[str, Any]:
        """Get comprehensive insights."""
        try:
            insights = {
                "summary": self._generate_summary(),
                "trends": self._analyze_all_trends(),
                "predictions": self._generate_predictions(),
                "user_segments": self._get_user_segments(),
                "recommendations": self._generate_recommendations(),
                "timestamp": time.time()
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return {"error": str(e)}
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        if not self.data_points:
            return {"message": "No data available"}
        
        # Group by metric type
        metric_summaries = {}
        for metric_type in MetricType:
            metric_data = [dp for dp in self.data_points if dp.metric_type == metric_type]
            
            if metric_data:
                values = [dp.value for dp in metric_data]
                metric_summaries[metric_type.value] = {
                    "count": len(values),
                    "mean": np.mean(values),
                    "median": np.median(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values)
                }
        
        return {
            "total_data_points": len(self.data_points),
            "metric_summaries": metric_summaries,
            "data_span_days": self._calculate_data_span_days()
        }
    
    def _analyze_all_trends(self) -> Dict[str, TrendAnalysis]:
        """Analyze trends for all metric types."""
        trends = {}
        
        for metric_type in MetricType:
            trend = self.trend_analyzer.analyze_trend(self.data_points, metric_type)
            trends[metric_type.value] = asdict(trend)
        
        return trends
    
    def _generate_predictions(self) -> Dict[str, Any]:
        """Generate predictions for various metrics."""
        predictions = {}
        
        # This would use actual feature data in production
        sample_features = {
            'duration': 30.0,
            'quality_score': 0.8,
            'engagement_early': 0.6,
            'sentiment_score': 0.7,
            'topic_relevance': 0.8,
            'creator_followers': 10000,
            'upload_time_hour': 14,
            'content_type_score': 0.7,
            'thumbnail_attractiveness': 0.8,
            'title_length': 50
        }
        
        for prediction_type in PredictionType:
            if prediction_type in self.ml_predictor.models:
                prediction = self.ml_predictor.predict(prediction_type, sample_features)
                predictions[prediction_type.value] = asdict(prediction)
        
        return predictions
    
    def _get_user_segments(self) -> List[Dict[str, Any]]:
        """Get user segments."""
        if not self.user_segmentation.segments:
            return []
        
        return [asdict(segment) for segment in self.user_segmentation.segments.values()]
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Analyze trends and generate recommendations
        trends = self._analyze_all_trends()
        
        for metric_type, trend_data in trends.items():
            if trend_data['trend_direction'] == 'decreasing' and trend_data['trend_strength'] > 0.7:
                recommendations.append(f"Consider improving {metric_type} - showing declining trend")
            elif trend_data['trend_direction'] == 'increasing' and trend_data['trend_strength'] > 0.7:
                recommendations.append(f"Great job on {metric_type} - showing positive trend")
        
        return recommendations
    
    def _calculate_data_span_days(self) -> float:
        """Calculate span of data in days."""
        if not self.data_points:
            return 0.0
        
        timestamps = [dp.timestamp for dp in self.data_points]
        return (max(timestamps) - min(timestamps)) / (24 * 60 * 60)
    
    def train_models(self, training_data: Dict[str, pd.DataFrame]):
        """Train ML models with provided data."""
        try:
            if 'viral_scores' in training_data:
                self.ml_predictor.train_viral_score_model(training_data['viral_scores'])
            
            if 'engagement_data' in training_data:
                self.ml_predictor.train_engagement_model(training_data['engagement_data'])
            
            if 'user_data' in training_data:
                self.user_segmentation.create_segments(training_data['user_data'])
            
            logger.info("ML models trained successfully")
            
        except Exception as e:
            logger.error(f"Error training models: {e}")

# Global analytics instance
_global_analytics: Optional[AdvancedAnalytics] = None

def get_advanced_analytics() -> AdvancedAnalytics:
    """Get the global advanced analytics instance."""
    global _global_analytics
    if _global_analytics is None:
        _global_analytics = AdvancedAnalytics()
    return _global_analytics

def add_analytics_data_point(metric_type: MetricType, value: float, metadata: Dict[str, Any] = None):
    """Add a data point to analytics."""
    analytics = get_advanced_analytics()
    
    data_point = AnalyticsDataPoint(
        timestamp=time.time(),
        metric_type=metric_type,
        value=value,
        metadata=metadata or {}
    )
    
    analytics.add_data_point(data_point)


