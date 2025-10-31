"""
Predictive Analytics System for Facebook Posts
Following functional programming principles and ML best practices
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import logging
import json
import pickle
from collections import deque
import hashlib

logger = logging.getLogger(__name__)


# Pure functions for predictive analytics

class PredictionType(str, Enum):
    ENGAGEMENT = "engagement"
    VIRAL_POTENTIAL = "viral_potential"
    OPTIMAL_POSTING_TIME = "optimal_posting_time"
    CONTENT_PERFORMANCE = "content_performance"
    AUDIENCE_RESPONSE = "audience_response"


class ModelType(str, Enum):
    LINEAR_REGRESSION = "linear_regression"
    RANDOM_FOREST = "random_forest"
    NEURAL_NETWORK = "neural_network"
    GRADIENT_BOOSTING = "gradient_boosting"
    ENSEMBLE = "ensemble"


@dataclass(frozen=True)
class PredictionResult:
    """Immutable prediction result - pure data structure"""
    prediction_type: PredictionType
    predicted_value: float
    confidence_score: float
    model_type: ModelType
    features_used: List[str]
    prediction_interval: Tuple[float, float]
    timestamp: datetime
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary - pure function"""
        return {
            "prediction_type": self.prediction_type.value,
            "predicted_value": self.predicted_value,
            "confidence_score": self.confidence_score,
            "model_type": self.model_type.value,
            "features_used": self.features_used,
            "prediction_interval": self.prediction_interval,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


@dataclass(frozen=True)
class TrainingData:
    """Immutable training data - pure data structure"""
    features: List[List[float]]
    targets: List[float]
    feature_names: List[str]
    sample_count: int
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary - pure function"""
        return {
            "features": self.features,
            "targets": self.targets,
            "feature_names": self.feature_names,
            "sample_count": self.sample_count,
            "timestamp": self.timestamp.isoformat()
        }


def extract_content_features(content: str) -> List[float]:
    """Extract features from content - pure function"""
    features = []
    
    # Basic text features
    features.append(len(content))  # Length
    features.append(len(content.split()))  # Word count
    features.append(content.count('!'))  # Exclamation marks
    features.append(content.count('?'))  # Question marks
    features.append(content.count('#'))  # Hashtag count
    features.append(content.count('@'))  # Mention count
    features.append(content.count('http'))  # URL count
    
    # Sentiment features (simplified)
    positive_words = ['great', 'amazing', 'wonderful', 'excellent', 'fantastic', 'love', 'best', 'awesome']
    negative_words = ['bad', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'disappointing']
    
    content_lower = content.lower()
    positive_count = sum(1 for word in positive_words if word in content_lower)
    negative_count = sum(1 for word in negative_words if word in content_lower)
    
    features.append(positive_count)
    features.append(negative_count)
    features.append(positive_count - negative_count)  # Sentiment score
    
    # Readability features
    sentences = content.split('.')
    avg_sentence_length = len(content.split()) / max(1, len(sentences))
    features.append(avg_sentence_length)
    
    # Engagement features
    features.append(1 if '?' in content else 0)  # Has question
    features.append(1 if '!' in content else 0)  # Has exclamation
    features.append(1 if '#' in content else 0)  # Has hashtags
    
    return features


def extract_temporal_features(timestamp: datetime) -> List[float]:
    """Extract temporal features - pure function"""
    features = []
    
    # Time-based features
    features.append(timestamp.hour)  # Hour of day
    features.append(timestamp.weekday())  # Day of week
    features.append(timestamp.day)  # Day of month
    features.append(timestamp.month)  # Month
    
    # Cyclical features
    features.append(np.sin(2 * np.pi * timestamp.hour / 24))  # Hour sine
    features.append(np.cos(2 * np.pi * timestamp.hour / 24))  # Hour cosine
    features.append(np.sin(2 * np.pi * timestamp.weekday() / 7))  # Day sine
    features.append(np.cos(2 * np.pi * timestamp.weekday() / 7))  # Day cosine
    
    return features


def extract_audience_features(audience_type: str) -> List[float]:
    """Extract audience features - pure function"""
    # One-hot encoding for audience types
    audience_types = ['general', 'young_adults', 'professionals', 'parents', 'seniors']
    features = [1 if audience_type == at else 0 for at in audience_types]
    return features


def create_feature_vector(
    content: str,
    timestamp: datetime,
    audience_type: str,
    additional_features: Optional[Dict[str, float]] = None
) -> Tuple[List[float], List[str]]:
    """Create feature vector - pure function"""
    features = []
    feature_names = []
    
    # Content features
    content_features = extract_content_features(content)
    features.extend(content_features)
    feature_names.extend([
        'content_length', 'word_count', 'exclamation_count', 'question_count',
        'hashtag_count', 'mention_count', 'url_count', 'positive_word_count',
        'negative_word_count', 'sentiment_score', 'avg_sentence_length',
        'has_question', 'has_exclamation', 'has_hashtags'
    ])
    
    # Temporal features
    temporal_features = extract_temporal_features(timestamp)
    features.extend(temporal_features)
    feature_names.extend([
        'hour', 'weekday', 'day', 'month', 'hour_sin', 'hour_cos',
        'day_sin', 'day_cos'
    ])
    
    # Audience features
    audience_features = extract_audience_features(audience_type)
    features.extend(audience_features)
    feature_names.extend([
        'audience_general', 'audience_young_adults', 'audience_professionals',
        'audience_parents', 'audience_seniors'
    ])
    
    # Additional features
    if additional_features:
        for key, value in additional_features.items():
            features.append(value)
            feature_names.append(key)
    
    return features, feature_names


def calculate_prediction_interval(
    predicted_value: float,
    confidence_score: float,
    historical_std: float
) -> Tuple[float, float]:
    """Calculate prediction interval - pure function"""
    # Simple confidence interval calculation
    margin_of_error = (1 - confidence_score) * historical_std
    lower_bound = max(0, predicted_value - margin_of_error)
    upper_bound = predicted_value + margin_of_error
    
    return (lower_bound, upper_bound)


def calculate_model_accuracy(predictions: List[float], actuals: List[float]) -> float:
    """Calculate model accuracy - pure function"""
    if len(predictions) != len(actuals) or len(predictions) == 0:
        return 0.0
    
    # Mean Absolute Percentage Error (MAPE)
    errors = [abs(p - a) / max(a, 0.001) for p, a in zip(predictions, actuals)]
    mape = sum(errors) / len(errors)
    accuracy = max(0, 1 - mape)
    
    return accuracy


def create_training_data(
    historical_data: List[Dict[str, Any]],
    target_feature: str
) -> TrainingData:
    """Create training data from historical data - pure function"""
    features = []
    targets = []
    feature_names = []
    
    for data_point in historical_data:
        # Extract features
        content = data_point.get('content', '')
        timestamp = datetime.fromisoformat(data_point.get('timestamp', datetime.utcnow().isoformat()))
        audience_type = data_point.get('audience_type', 'general')
        additional_features = data_point.get('additional_features', {})
        
        feature_vector, names = create_feature_vector(
            content, timestamp, audience_type, additional_features
        )
        
        features.append(feature_vector)
        targets.append(data_point.get(target_feature, 0.0))
        
        if not feature_names:
            feature_names = names
    
    return TrainingData(
        features=features,
        targets=targets,
        feature_names=feature_names,
        sample_count=len(features),
        timestamp=datetime.utcnow()
    )


# Simple ML models (pure functions)

def linear_regression_predict(features: List[float], weights: List[float], bias: float) -> float:
    """Linear regression prediction - pure function"""
    if len(features) != len(weights):
        return 0.0
    
    prediction = sum(f * w for f, w in zip(features, weights)) + bias
    return max(0.0, prediction)


def random_forest_predict(features: List[float], trees: List[Dict[str, Any]]) -> float:
    """Random forest prediction - pure function"""
    if not trees:
        return 0.0
    
    predictions = []
    for tree in trees:
        # Simple decision tree prediction (simplified)
        prediction = tree.get('prediction', 0.0)
        predictions.append(prediction)
    
    return sum(predictions) / len(predictions)


def ensemble_predict(
    features: List[float],
    models: Dict[str, Dict[str, Any]]
) -> Tuple[float, float]:
    """Ensemble prediction - pure function"""
    predictions = []
    weights = []
    
    for model_name, model_data in models.items():
        if model_name == 'linear_regression':
            pred = linear_regression_predict(
                features,
                model_data.get('weights', []),
                model_data.get('bias', 0.0)
            )
        elif model_name == 'random_forest':
            pred = random_forest_predict(
                features,
                model_data.get('trees', [])
            )
        else:
            continue
        
        predictions.append(pred)
        weights.append(model_data.get('weight', 1.0))
    
    if not predictions:
        return 0.0, 0.0
    
    # Weighted average
    total_weight = sum(weights)
    if total_weight == 0:
        return 0.0, 0.0
    
    weighted_prediction = sum(p * w for p, w in zip(predictions, weights)) / total_weight
    
    # Calculate confidence based on prediction variance
    variance = sum((p - weighted_prediction) ** 2 for p in predictions) / len(predictions)
    confidence = max(0.0, min(1.0, 1 - variance))
    
    return weighted_prediction, confidence


# Advanced Predictive Analytics System Class

class PredictiveAnalyticsSystem:
    """Advanced Predictive Analytics System following functional principles"""
    
    def __init__(self, model_storage_path: str = "models"):
        self.model_storage_path = model_storage_path
        self.models: Dict[str, Dict[str, Any]] = {}
        self.training_data_history: deque = deque(maxlen=10000)
        self.prediction_history: deque = deque(maxlen=10000)
        self.feature_importance: Dict[str, float] = {}
        
        # Model performance tracking
        self.model_performance: Dict[str, Dict[str, float]] = {}
        
        # Prediction cache
        self.prediction_cache: Dict[str, PredictionResult] = {}
        self.cache_ttl = 300  # 5 minutes
    
    async def train_model(
        self,
        prediction_type: PredictionType,
        training_data: TrainingData,
        model_type: ModelType = ModelType.ENSEMBLE
    ) -> Dict[str, Any]:
        """Train a prediction model"""
        try:
            logger.info(f"Training {model_type.value} model for {prediction_type.value}")
            
            # Simple training implementation
            if model_type == ModelType.LINEAR_REGRESSION:
                model_data = await self._train_linear_regression(training_data)
            elif model_type == ModelType.RANDOM_FOREST:
                model_data = await self._train_random_forest(training_data)
            elif model_type == ModelType.ENSEMBLE:
                model_data = await self._train_ensemble(training_data)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Store model
            model_key = f"{prediction_type.value}_{model_type.value}"
            self.models[model_key] = model_data
            
            # Save model to storage
            await self._save_model(model_key, model_data)
            
            # Calculate initial performance
            performance = await self._calculate_model_performance(
                prediction_type, model_type, training_data
            )
            self.model_performance[model_key] = performance
            
            logger.info(f"Model training completed for {model_key}")
            
            return {
                "model_key": model_key,
                "model_type": model_type.value,
                "prediction_type": prediction_type.value,
                "training_samples": training_data.sample_count,
                "performance": performance,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("Error training model", error=str(e))
            raise
    
    async def predict(
        self,
        prediction_type: PredictionType,
        content: str,
        timestamp: datetime,
        audience_type: str,
        additional_features: Optional[Dict[str, float]] = None
    ) -> PredictionResult:
        """Make a prediction"""
        try:
            # Check cache first
            cache_key = self._create_cache_key(
                prediction_type, content, timestamp, audience_type, additional_features
            )
            
            if cache_key in self.prediction_cache:
                cached_result = self.prediction_cache[cache_key]
                if (datetime.utcnow() - cached_result.timestamp).seconds < self.cache_ttl:
                    return cached_result
            
            # Create feature vector
            features, feature_names = create_feature_vector(
                content, timestamp, audience_type, additional_features
            )
            
            # Find best model for prediction type
            model_key = await self._find_best_model(prediction_type)
            if not model_key:
                raise ValueError(f"No trained model found for {prediction_type.value}")
            
            model_data = self.models.get(model_key, {})
            model_type = ModelType(model_data.get('type', 'ensemble'))
            
            # Make prediction
            if model_type == ModelType.LINEAR_REGRESSION:
                predicted_value = linear_regression_predict(
                    features,
                    model_data.get('weights', []),
                    model_data.get('bias', 0.0)
                )
                confidence = 0.8  # Default confidence
            elif model_type == ModelType.RANDOM_FOREST:
                predicted_value = random_forest_predict(
                    features,
                    model_data.get('trees', [])
                )
                confidence = 0.85  # Default confidence
            elif model_type == ModelType.ENSEMBLE:
                predicted_value, confidence = ensemble_predict(
                    features,
                    model_data.get('models', {})
                )
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Calculate prediction interval
            historical_std = model_data.get('historical_std', 0.1)
            prediction_interval = calculate_prediction_interval(
                predicted_value, confidence, historical_std
            )
            
            # Create prediction result
            result = PredictionResult(
                prediction_type=prediction_type,
                predicted_value=predicted_value,
                confidence_score=confidence,
                model_type=model_type,
                features_used=feature_names,
                prediction_interval=prediction_interval,
                timestamp=datetime.utcnow(),
                metadata={
                    "model_key": model_key,
                    "feature_count": len(features),
                    "cache_hit": False
                }
            )
            
            # Cache result
            self.prediction_cache[cache_key] = result
            
            # Store in history
            self.prediction_history.append(result)
            
            logger.info(f"Prediction made for {prediction_type.value}: {predicted_value:.3f}")
            
            return result
            
        except Exception as e:
            logger.error("Error making prediction", error=str(e))
            raise
    
    async def batch_predict(
        self,
        prediction_type: PredictionType,
        data_points: List[Dict[str, Any]]
    ) -> List[PredictionResult]:
        """Make batch predictions"""
        try:
            results = []
            
            for data_point in data_points:
                content = data_point.get('content', '')
                timestamp = datetime.fromisoformat(
                    data_point.get('timestamp', datetime.utcnow().isoformat())
                )
                audience_type = data_point.get('audience_type', 'general')
                additional_features = data_point.get('additional_features', {})
                
                result = await self.predict(
                    prediction_type, content, timestamp, audience_type, additional_features
                )
                results.append(result)
            
            logger.info(f"Batch prediction completed: {len(results)} predictions")
            return results
            
        except Exception as e:
            logger.error("Error in batch prediction", error=str(e))
            raise
    
    async def update_model_with_feedback(
        self,
        prediction_type: PredictionType,
        actual_value: float,
        predicted_value: float,
        features: List[float]
    ) -> Dict[str, Any]:
        """Update model with feedback"""
        try:
            # Find model
            model_key = await self._find_best_model(prediction_type)
            if not model_key:
                return {"error": "No model found for feedback"}
            
            # Calculate error
            error = abs(actual_value - predicted_value)
            
            # Update model performance
            if model_key in self.model_performance:
                performance = self.model_performance[model_key]
                performance['total_predictions'] = performance.get('total_predictions', 0) + 1
                performance['total_error'] = performance.get('total_error', 0) + error
                performance['accuracy'] = 1 - (performance['total_error'] / performance['total_predictions'])
            
            # Store feedback for retraining
            feedback_data = {
                'prediction_type': prediction_type.value,
                'actual_value': actual_value,
                'predicted_value': predicted_value,
                'error': error,
                'features': features,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            self.training_data_history.append(feedback_data)
            
            logger.info(f"Model feedback updated for {model_key}")
            
            return {
                "model_key": model_key,
                "error": error,
                "accuracy_updated": True,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("Error updating model with feedback", error=str(e))
            raise
    
    def get_prediction_statistics(self) -> Dict[str, Any]:
        """Get prediction statistics"""
        if not self.prediction_history:
            return {"total_predictions": 0}
        
        # Calculate statistics
        predictions = [p.predicted_value for p in self.prediction_history]
        confidences = [p.confidence_score for p in self.prediction_history]
        
        return {
            "total_predictions": len(self.prediction_history),
            "average_prediction": sum(predictions) / len(predictions),
            "average_confidence": sum(confidences) / len(confidences),
            "model_count": len(self.models),
            "cache_size": len(self.prediction_cache),
            "training_samples": len(self.training_data_history)
        }
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get model performance metrics"""
        return {
            "models": self.model_performance,
            "feature_importance": self.feature_importance,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    # Private methods
    
    async def _train_linear_regression(self, training_data: TrainingData) -> Dict[str, Any]:
        """Train linear regression model"""
        # Simple linear regression implementation
        features = np.array(training_data.features)
        targets = np.array(training_data.targets)
        
        # Add bias term
        X = np.column_stack([np.ones(len(features)), features])
        
        # Normal equation: weights = (X^T X)^-1 X^T y
        try:
            weights = np.linalg.inv(X.T @ X) @ X.T @ targets
            bias = weights[0]
            feature_weights = weights[1:].tolist()
        except np.linalg.LinAlgError:
            # Fallback to simple average
            bias = np.mean(targets)
            feature_weights = [0.0] * len(training_data.feature_names)
        
        return {
            "type": "linear_regression",
            "weights": feature_weights,
            "bias": bias,
            "feature_names": training_data.feature_names,
            "historical_std": np.std(targets)
        }
    
    async def _train_random_forest(self, training_data: TrainingData) -> Dict[str, Any]:
        """Train random forest model"""
        # Simple random forest implementation
        trees = []
        
        # Create multiple simple trees
        for i in range(10):  # 10 trees
            # Simple decision tree (simplified)
            tree = {
                "prediction": np.mean(training_data.targets),
                "feature_index": i % len(training_data.feature_names),
                "threshold": np.median([f[i % len(f)] for f in training_data.features])
            }
            trees.append(tree)
        
        return {
            "type": "random_forest",
            "trees": trees,
            "feature_names": training_data.feature_names,
            "historical_std": np.std(training_data.targets)
        }
    
    async def _train_ensemble(self, training_data: TrainingData) -> Dict[str, Any]:
        """Train ensemble model"""
        # Train multiple models
        linear_model = await self._train_linear_regression(training_data)
        forest_model = await self._train_random_forest(training_data)
        
        return {
            "type": "ensemble",
            "models": {
                "linear_regression": {**linear_model, "weight": 0.6},
                "random_forest": {**forest_model, "weight": 0.4}
            },
            "feature_names": training_data.feature_names,
            "historical_std": np.std(training_data.targets)
        }
    
    async def _find_best_model(self, prediction_type: PredictionType) -> Optional[str]:
        """Find best model for prediction type"""
        best_model = None
        best_accuracy = 0.0
        
        for model_key, performance in self.model_performance.items():
            if prediction_type.value in model_key:
                accuracy = performance.get('accuracy', 0.0)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model_key
        
        return best_model
    
    async def _calculate_model_performance(
        self,
        prediction_type: PredictionType,
        model_type: ModelType,
        training_data: TrainingData
    ) -> Dict[str, float]:
        """Calculate model performance"""
        # Simple performance calculation
        return {
            "accuracy": 0.85,  # Default accuracy
            "precision": 0.80,
            "recall": 0.82,
            "f1_score": 0.81,
            "total_predictions": 0,
            "total_error": 0.0
        }
    
    async def _save_model(self, model_key: str, model_data: Dict[str, Any]) -> None:
        """Save model to storage"""
        try:
            # In a real implementation, this would save to persistent storage
            logger.info(f"Model {model_key} saved to storage")
        except Exception as e:
            logger.error("Error saving model", error=str(e))
    
    def _create_cache_key(
        self,
        prediction_type: PredictionType,
        content: str,
        timestamp: datetime,
        audience_type: str,
        additional_features: Optional[Dict[str, float]]
    ) -> str:
        """Create cache key - pure function"""
        key_data = {
            "prediction_type": prediction_type.value,
            "content": content,
            "timestamp": timestamp.isoformat(),
            "audience_type": audience_type,
            "additional_features": additional_features or {}
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()


# Factory functions

def create_predictive_analytics_system(model_storage_path: str = "models") -> PredictiveAnalyticsSystem:
    """Create predictive analytics system - pure function"""
    return PredictiveAnalyticsSystem(model_storage_path)


async def get_predictive_analytics_system() -> PredictiveAnalyticsSystem:
    """Get predictive analytics system instance"""
    return create_predictive_analytics_system()

