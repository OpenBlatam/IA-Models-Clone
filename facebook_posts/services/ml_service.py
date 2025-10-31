"""
Advanced Machine Learning Service for Facebook Posts API
Predictive analytics, content scoring, and intelligent recommendations
"""

import asyncio
import json
import time
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import structlog

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

from ..core.config import get_settings
from ..core.models import FacebookPost, PostStatus, ContentType, AudienceType
from ..infrastructure.cache import get_cache_manager
from ..infrastructure.database import get_db_manager, PostRepository
from ..infrastructure.monitoring import get_monitor, timed
from ..services.analytics_service import get_analytics_service

logger = structlog.get_logger(__name__)


@dataclass
class PredictionResult:
    """ML prediction result"""
    predicted_value: float
    confidence: float
    model_used: str
    features_used: List[str]
    processing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContentScore:
    """Content quality score"""
    overall_score: float
    engagement_score: float
    reach_score: float
    quality_score: float
    virality_score: float
    factors: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class TrendPrediction:
    """Trend prediction result"""
    topic: str
    predicted_engagement: float
    confidence: float
    time_to_peak: int  # hours
    peak_engagement: float
    duration: int  # hours
    factors: List[str] = field(default_factory=list)


class FeatureExtractor:
    """Extract features from content for ML models"""
    
    def __init__(self):
        self.cache_manager = get_cache_manager()
    
    def extract_content_features(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Extract features from content"""
        features = {}
        
        # Basic text features
        features["length"] = len(content)
        features["word_count"] = len(content.split())
        features["sentence_count"] = len([s for s in content.split(".") if s.strip()])
        features["avg_word_length"] = sum(len(word) for word in content.split()) / max(len(content.split()), 1)
        
        # Engagement features
        features["question_count"] = content.count("?")
        features["exclamation_count"] = content.count("!")
        features["hashtag_count"] = content.count("#")
        features["mention_count"] = content.count("@")
        features["emoji_count"] = sum(1 for char in content if ord(char) > 127)
        
        # Readability features
        features["avg_sentence_length"] = features["word_count"] / max(features["sentence_count"], 1)
        features["complex_word_ratio"] = self._calculate_complex_word_ratio(content)
        
        # Sentiment features (simplified)
        features["positive_word_count"] = self._count_positive_words(content)
        features["negative_word_count"] = self._count_negative_words(content)
        features["sentiment_score"] = (features["positive_word_count"] - features["negative_word_count"]) / max(features["word_count"], 1)
        
        # Content type features
        if metadata:
            features["content_type_educational"] = 1.0 if metadata.get("content_type") == "educational" else 0.0
            features["content_type_entertainment"] = 1.0 if metadata.get("content_type") == "entertainment" else 0.0
            features["content_type_promotional"] = 1.0 if metadata.get("content_type") == "promotional" else 0.0
            
            features["audience_professionals"] = 1.0 if metadata.get("audience_type") == "professionals" else 0.0
            features["audience_general"] = 1.0 if metadata.get("audience_type") == "general" else 0.0
            features["audience_students"] = 1.0 if metadata.get("audience_type") == "students" else 0.0
        
        # Time features
        now = datetime.now()
        features["hour_of_day"] = now.hour
        features["day_of_week"] = now.weekday()
        features["is_weekend"] = 1.0 if now.weekday() >= 5 else 0.0
        
        return features
    
    def _calculate_complex_word_ratio(self, content: str) -> float:
        """Calculate ratio of complex words (simplified)"""
        words = content.split()
        if not words:
            return 0.0
        
        complex_words = sum(1 for word in words if len(word) > 6)
        return complex_words / len(words)
    
    def _count_positive_words(self, content: str) -> int:
        """Count positive words (simplified)"""
        positive_words = ["good", "great", "amazing", "excellent", "fantastic", "wonderful", "awesome", "brilliant", "outstanding", "perfect"]
        content_lower = content.lower()
        return sum(1 for word in positive_words if word in content_lower)
    
    def _count_negative_words(self, content: str) -> int:
        """Count negative words (simplified)"""
        negative_words = ["bad", "terrible", "awful", "horrible", "disappointing", "poor", "worst", "hate", "dislike", "annoying"]
        content_lower = content.lower()
        return sum(1 for word in negative_words if word in content_lower)


class EngagementPredictor:
    """Predict engagement metrics using ML models"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_extractor = FeatureExtractor()
        self.cache_manager = get_cache_manager()
        self.monitor = get_monitor()
        self._initialized = False
    
    async def initialize(self):
        """Initialize ML models"""
        if not SKLEARN_AVAILABLE:
            logger.warning("Scikit-learn not available, using mock predictions")
            return
        
        try:
            # Load or train models
            await self._load_or_train_models()
            self._initialized = True
            logger.info("ML models initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize ML models", error=str(e))
            self._initialized = False
    
    async def _load_or_train_models(self):
        """Load existing models or train new ones"""
        try:
            # Try to load existing models
            models_data = await self.cache_manager.cache.get("ml_models")
            if models_data:
                self.models = models_data["models"]
                self.scalers = models_data["scalers"]
                logger.info("Loaded existing ML models from cache")
                return
        except Exception as e:
            logger.warning("Failed to load models from cache", error=str(e))
        
        # Train new models with mock data
        await self._train_models_with_mock_data()
    
    async def _train_models_with_mock_data(self):
        """Train models with mock data"""
        try:
            # Generate mock training data
            n_samples = 1000
            features = self._generate_mock_features(n_samples)
            
            # Mock target variables
            engagement_rates = np.random.beta(2, 5, n_samples)  # Beta distribution for engagement rates
            reach_scores = np.random.normal(0.5, 0.2, n_samples)
            click_rates = np.random.exponential(0.1, n_samples)
            
            # Train models
            self._train_engagement_model(features, engagement_rates)
            self._train_reach_model(features, reach_scores)
            self._train_click_model(features, click_rates)
            
            # Cache models
            models_data = {
                "models": self.models,
                "scalers": self.scalers
            }
            await self.cache_manager.cache.set("ml_models", models_data, ttl=86400)
            
            logger.info("Trained new ML models with mock data")
            
        except Exception as e:
            logger.error("Failed to train models", error=str(e))
            raise
    
    def _generate_mock_features(self, n_samples: int) -> np.ndarray:
        """Generate mock feature data"""
        features = []
        
        for _ in range(n_samples):
            feature_dict = {
                "length": np.random.randint(50, 500),
                "word_count": np.random.randint(10, 100),
                "sentence_count": np.random.randint(1, 10),
                "avg_word_length": np.random.uniform(3, 8),
                "question_count": np.random.randint(0, 3),
                "exclamation_count": np.random.randint(0, 2),
                "hashtag_count": np.random.randint(0, 5),
                "mention_count": np.random.randint(0, 3),
                "emoji_count": np.random.randint(0, 5),
                "avg_sentence_length": np.random.uniform(5, 25),
                "complex_word_ratio": np.random.uniform(0, 0.5),
                "positive_word_count": np.random.randint(0, 5),
                "negative_word_count": np.random.randint(0, 3),
                "sentiment_score": np.random.uniform(-0.5, 0.5),
                "content_type_educational": np.random.choice([0, 1]),
                "content_type_entertainment": np.random.choice([0, 1]),
                "content_type_promotional": np.random.choice([0, 1]),
                "audience_professionals": np.random.choice([0, 1]),
                "audience_general": np.random.choice([0, 1]),
                "audience_students": np.random.choice([0, 1]),
                "hour_of_day": np.random.randint(0, 24),
                "day_of_week": np.random.randint(0, 7),
                "is_weekend": np.random.choice([0, 1])
            }
            features.append(list(feature_dict.values()))
        
        return np.array(features)
    
    def _train_engagement_model(self, features: np.ndarray, targets: np.ndarray):
        """Train engagement prediction model"""
        if not SKLEARN_AVAILABLE:
            return
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        self.models["engagement"] = model
        self.scalers["engagement"] = scaler
        
        logger.info(f"Engagement model trained - MSE: {mse:.4f}, R²: {r2:.4f}")
    
    def _train_reach_model(self, features: np.ndarray, targets: np.ndarray):
        """Train reach prediction model"""
        if not SKLEARN_AVAILABLE:
            return
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        self.models["reach"] = model
        self.scalers["reach"] = scaler
        
        logger.info(f"Reach model trained - MSE: {mse:.4f}, R²: {r2:.4f}")
    
    def _train_click_model(self, features: np.ndarray, targets: np.ndarray):
        """Train click prediction model"""
        if not SKLEARN_AVAILABLE:
            return
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        self.models["clicks"] = model
        self.scalers["clicks"] = scaler
        
        logger.info(f"Click model trained - MSE: {mse:.4f}, R²: {r2:.4f}")
    
    @timed("ml_prediction")
    async def predict_engagement(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> PredictionResult:
        """Predict engagement rate for content"""
        start_time = time.time()
        
        try:
            if not self._initialized:
                await self.initialize()
            
            # Extract features
            features_dict = self.feature_extractor.extract_content_features(content, metadata)
            features = np.array(list(features_dict.values())).reshape(1, -1)
            
            if not SKLEARN_AVAILABLE or "engagement" not in self.models:
                # Mock prediction
                predicted_value = np.random.beta(2, 5)
                confidence = 0.7
                model_used = "mock"
            else:
                # Scale features
                features_scaled = self.scalers["engagement"].transform(features)
                
                # Predict
                predicted_value = self.models["engagement"].predict(features_scaled)[0]
                confidence = 0.8
                model_used = "random_forest"
            
            processing_time = time.time() - start_time
            
            return PredictionResult(
                predicted_value=max(0, min(1, predicted_value)),  # Clamp to [0, 1]
                confidence=confidence,
                model_used=model_used,
                features_used=list(features_dict.keys()),
                processing_time=processing_time,
                metadata={
                    "content_length": len(content),
                    "feature_count": len(features_dict)
                }
            )
            
        except Exception as e:
            logger.error("Engagement prediction failed", error=str(e))
            return PredictionResult(
                predicted_value=0.5,
                confidence=0.3,
                model_used="error_fallback",
                features_used=[],
                processing_time=time.time() - start_time,
                metadata={"error": str(e)}
            )
    
    @timed("ml_prediction")
    async def predict_reach(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> PredictionResult:
        """Predict reach for content"""
        start_time = time.time()
        
        try:
            if not self._initialized:
                await self.initialize()
            
            # Extract features
            features_dict = self.feature_extractor.extract_content_features(content, metadata)
            features = np.array(list(features_dict.values())).reshape(1, -1)
            
            if not SKLEARN_AVAILABLE or "reach" not in self.models:
                # Mock prediction
                predicted_value = np.random.normal(0.5, 0.2)
                confidence = 0.7
                model_used = "mock"
            else:
                # Scale features
                features_scaled = self.scalers["reach"].transform(features)
                
                # Predict
                predicted_value = self.models["reach"].predict(features_scaled)[0]
                confidence = 0.8
                model_used = "gradient_boosting"
            
            processing_time = time.time() - start_time
            
            return PredictionResult(
                predicted_value=max(0, min(1, predicted_value)),  # Clamp to [0, 1]
                confidence=confidence,
                model_used=model_used,
                features_used=list(features_dict.keys()),
                processing_time=processing_time,
                metadata={
                    "content_length": len(content),
                    "feature_count": len(features_dict)
                }
            )
            
        except Exception as e:
            logger.error("Reach prediction failed", error=str(e))
            return PredictionResult(
                predicted_value=0.5,
                confidence=0.3,
                model_used="error_fallback",
                features_used=[],
                processing_time=time.time() - start_time,
                metadata={"error": str(e)}
            )
    
    @timed("ml_prediction")
    async def predict_clicks(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> PredictionResult:
        """Predict click rate for content"""
        start_time = time.time()
        
        try:
            if not self._initialized:
                await self.initialize()
            
            # Extract features
            features_dict = self.feature_extractor.extract_content_features(content, metadata)
            features = np.array(list(features_dict.values())).reshape(1, -1)
            
            if not SKLEARN_AVAILABLE or "clicks" not in self.models:
                # Mock prediction
                predicted_value = np.random.exponential(0.1)
                confidence = 0.7
                model_used = "mock"
            else:
                # Scale features
                features_scaled = self.scalers["clicks"].transform(features)
                
                # Predict
                predicted_value = self.models["clicks"].predict(features_scaled)[0]
                confidence = 0.8
                model_used = "linear_regression"
            
            processing_time = time.time() - start_time
            
            return PredictionResult(
                predicted_value=max(0, min(1, predicted_value)),  # Clamp to [0, 1]
                confidence=confidence,
                model_used=model_used,
                features_used=list(features_dict.keys()),
                processing_time=processing_time,
                metadata={
                    "content_length": len(content),
                    "feature_count": len(features_dict)
                }
            )
            
        except Exception as e:
            logger.error("Click prediction failed", error=str(e))
            return PredictionResult(
                predicted_value=0.1,
                confidence=0.3,
                model_used="error_fallback",
                features_used=[],
                processing_time=time.time() - start_time,
                metadata={"error": str(e)}
            )


class ContentScorer:
    """Score content quality using ML models"""
    
    def __init__(self):
        self.engagement_predictor = EngagementPredictor()
        self.cache_manager = get_cache_manager()
    
    @timed("content_scoring")
    async def score_content(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> ContentScore:
        """Score content quality"""
        try:
            # Get predictions
            engagement_pred = await self.engagement_predictor.predict_engagement(content, metadata)
            reach_pred = await self.engagement_predictor.predict_reach(content, metadata)
            click_pred = await self.engagement_predictor.predict_clicks(content, metadata)
            
            # Calculate scores
            engagement_score = engagement_pred.predicted_value
            reach_score = reach_pred.predicted_value
            quality_score = (engagement_score + reach_score) / 2
            virality_score = engagement_score * reach_score * click_pred.predicted_value
            
            # Calculate overall score
            overall_score = (engagement_score * 0.4 + reach_score * 0.3 + quality_score * 0.2 + virality_score * 0.1)
            
            # Generate factors
            factors = {
                "engagement_potential": engagement_score,
                "reach_potential": reach_score,
                "click_potential": click_pred.predicted_value,
                "content_quality": quality_score,
                "virality_potential": virality_score
            }
            
            # Generate recommendations
            recommendations = self._generate_recommendations(content, factors)
            
            return ContentScore(
                overall_score=overall_score,
                engagement_score=engagement_score,
                reach_score=reach_score,
                quality_score=quality_score,
                virality_score=virality_score,
                factors=factors,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error("Content scoring failed", error=str(e))
            return ContentScore(
                overall_score=0.5,
                engagement_score=0.5,
                reach_score=0.5,
                quality_score=0.5,
                virality_score=0.5,
                factors={},
                recommendations=["Unable to analyze content"]
            )
    
    def _generate_recommendations(self, content: str, factors: Dict[str, float]) -> List[str]:
        """Generate content improvement recommendations"""
        recommendations = []
        
        if factors.get("engagement_potential", 0) < 0.5:
            recommendations.append("Add questions or call-to-action to increase engagement")
        
        if factors.get("reach_potential", 0) < 0.5:
            recommendations.append("Include trending hashtags to improve reach")
        
        if factors.get("click_potential", 0) < 0.3:
            recommendations.append("Add compelling links or call-to-action buttons")
        
        if len(content) < 100:
            recommendations.append("Consider adding more detail to make the post more informative")
        elif len(content) > 300:
            recommendations.append("Consider shortening the post for better engagement")
        
        if not any(char in content for char in ["?", "!"]):
            recommendations.append("Add punctuation to make the post more engaging")
        
        if not recommendations:
            recommendations.append("Content looks good! Consider A/B testing different variations")
        
        return recommendations


class TrendPredictor:
    """Predict content trends and virality"""
    
    def __init__(self):
        self.cache_manager = get_cache_manager()
        self.analytics_service = get_analytics_service()
    
    @timed("trend_prediction")
    async def predict_trend(self, topic: str, content_type: str) -> TrendPrediction:
        """Predict trend for a topic"""
        try:
            # Mock trend prediction (in real implementation, this would use historical data)
            base_engagement = np.random.uniform(0.3, 0.8)
            
            # Adjust based on content type
            type_multipliers = {
                "educational": 1.2,
                "entertainment": 1.5,
                "promotional": 0.8,
                "news": 1.3
            }
            
            predicted_engagement = base_engagement * type_multipliers.get(content_type, 1.0)
            
            # Predict timing
            time_to_peak = np.random.randint(2, 24)  # 2-24 hours
            peak_engagement = predicted_engagement * np.random.uniform(1.2, 2.0)
            duration = np.random.randint(24, 168)  # 1-7 days
            
            # Generate factors
            factors = [
                "trending_topic",
                "optimal_timing",
                "content_quality",
                "audience_interest"
            ]
            
            return TrendPrediction(
                topic=topic,
                predicted_engagement=predicted_engagement,
                confidence=0.75,
                time_to_peak=time_to_peak,
                peak_engagement=peak_engagement,
                duration=duration,
                factors=factors
            )
            
        except Exception as e:
            logger.error("Trend prediction failed", error=str(e))
            return TrendPrediction(
                topic=topic,
                predicted_engagement=0.5,
                confidence=0.3,
                time_to_peak=12,
                peak_engagement=0.6,
                duration=48,
                factors=["error"]
            )


class MLService:
    """Main ML service orchestrator"""
    
    def __init__(self):
        self.engagement_predictor = EngagementPredictor()
        self.content_scorer = ContentScorer()
        self.trend_predictor = TrendPredictor()
        self.cache_manager = get_cache_manager()
        self.monitor = get_monitor()
    
    async def initialize(self):
        """Initialize ML services"""
        await self.engagement_predictor.initialize()
        logger.info("ML service initialized successfully")
    
    async def predict_engagement(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> PredictionResult:
        """Predict engagement for content"""
        return await self.engagement_predictor.predict_engagement(content, metadata)
    
    async def predict_reach(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> PredictionResult:
        """Predict reach for content"""
        return await self.engagement_predictor.predict_reach(content, metadata)
    
    async def predict_clicks(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> PredictionResult:
        """Predict clicks for content"""
        return await self.engagement_predictor.predict_clicks(content, metadata)
    
    async def score_content(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> ContentScore:
        """Score content quality"""
        return await self.content_scorer.score_content(content, metadata)
    
    async def predict_trend(self, topic: str, content_type: str) -> TrendPrediction:
        """Predict trend for topic"""
        return await self.trend_predictor.predict_trend(topic, content_type)


# Global ML service instance
_ml_service: Optional[MLService] = None


def get_ml_service() -> MLService:
    """Get global ML service instance"""
    global _ml_service
    
    if _ml_service is None:
        _ml_service = MLService()
    
    return _ml_service


async def initialize_ml_service():
    """Initialize global ML service"""
    ml_service = get_ml_service()
    await ml_service.initialize()


# Export all classes and functions
__all__ = [
    # Data classes
    'PredictionResult',
    'ContentScore',
    'TrendPrediction',
    
    # ML services
    'FeatureExtractor',
    'EngagementPredictor',
    'ContentScorer',
    'TrendPredictor',
    'MLService',
    
    # Utility functions
    'get_ml_service',
    'initialize_ml_service',
]






























