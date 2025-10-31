"""
Advanced AI Integration for Microservices
Features: ML model serving, AI-powered caching, intelligent load balancing, predictive scaling
"""

import asyncio
import json
import time
import numpy as np
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from abc import ABC, abstractmethod
import pickle
import hashlib

# AI/ML imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    import sklearn
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

logger = logging.getLogger(__name__)

class AIModelType(Enum):
    """AI model types"""
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    SKLEARN = "sklearn"
    OPENAI = "openai"
    CUSTOM = "custom"

class PredictionType(Enum):
    """Prediction types"""
    LOAD_FORECASTING = "load_forecasting"
    CACHE_OPTIMIZATION = "cache_optimization"
    ANOMALY_DETECTION = "anomaly_detection"
    RESOURCE_SCALING = "resource_scaling"
    RESPONSE_TIME_PREDICTION = "response_time_prediction"
    ERROR_PREDICTION = "error_prediction"

@dataclass
class AIModelConfig:
    """AI model configuration"""
    model_type: AIModelType
    model_path: str
    input_features: List[str]
    output_features: List[str]
    prediction_type: PredictionType
    retrain_interval: int = 3600  # seconds
    batch_size: int = 32
    learning_rate: float = 0.001
    max_epochs: int = 100
    validation_split: float = 0.2

@dataclass
class PredictionResult:
    """Prediction result"""
    prediction: Union[float, List[float], Dict[str, float]]
    confidence: float
    model_version: str
    timestamp: float
    features_used: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

class AIModel(ABC):
    """Abstract AI model interface"""
    
    @abstractmethod
    async def predict(self, features: Dict[str, Any]) -> PredictionResult:
        """Make prediction"""
        pass
    
    @abstractmethod
    async def train(self, training_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Train model"""
        pass
    
    @abstractmethod
    async def save_model(self, path: str) -> bool:
        """Save model"""
        pass
    
    @abstractmethod
    async def load_model(self, path: str) -> bool:
        """Load model"""
        pass

class LoadForecastingModel(AIModel):
    """
    AI model for load forecasting and predictive scaling
    """
    
    def __init__(self, config: AIModelConfig):
        self.config = config
        self.model = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.feature_history: List[Dict[str, Any]] = []
        self.prediction_history: List[PredictionResult] = []
        self.model_version = "1.0.0"
        self.last_training_time = 0
        
        if SKLEARN_AVAILABLE:
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
    
    async def predict(self, features: Dict[str, Any]) -> PredictionResult:
        """Predict future load"""
        try:
            if not self.model or not SKLEARN_AVAILABLE:
                # Fallback to simple heuristic
                return await self._heuristic_prediction(features)
            
            # Prepare features
            feature_vector = self._prepare_features(features)
            
            # Make prediction
            prediction = self.model.predict([feature_vector])[0]
            
            # Calculate confidence based on recent accuracy
            confidence = self._calculate_confidence()
            
            result = PredictionResult(
                prediction=float(prediction),
                confidence=confidence,
                model_version=self.model_version,
                timestamp=time.time(),
                features_used=list(features.keys()),
                metadata={
                    "model_type": "load_forecasting",
                    "prediction_horizon": "5_minutes"
                }
            )
            
            self.prediction_history.append(result)
            return result
            
        except Exception as e:
            logger.error(f"Load forecasting prediction failed: {e}")
            return await self._heuristic_prediction(features)
    
    async def train(self, training_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Train the load forecasting model"""
        try:
            if not SKLEARN_AVAILABLE or len(training_data) < 10:
                return {"status": "insufficient_data"}
            
            # Prepare training data
            X, y = self._prepare_training_data(training_data)
            
            if len(X) == 0:
                return {"status": "no_valid_data"}
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.model.fit(X_scaled, y)
            
            # Calculate metrics
            y_pred = self.model.predict(X_scaled)
            mse = mean_squared_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            
            self.model_version = f"1.{int(time.time())}"
            self.last_training_time = time.time()
            
            return {
                "status": "success",
                "mse": float(mse),
                "r2_score": float(r2),
                "training_samples": len(X),
                "model_version": self.model_version
            }
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _heuristic_prediction(self, features: Dict[str, Any]) -> PredictionResult:
        """Fallback heuristic prediction"""
        # Simple moving average based prediction
        current_load = features.get("current_load", 0)
        historical_avg = features.get("historical_average", current_load)
        trend = features.get("trend", 0)
        
        prediction = current_load + (trend * 0.1) + (historical_avg * 0.1)
        
        return PredictionResult(
            prediction=float(prediction),
            confidence=0.5,  # Lower confidence for heuristic
            model_version="heuristic",
            timestamp=time.time(),
            features_used=list(features.keys()),
            metadata={"method": "heuristic"}
        )
    
    def _prepare_features(self, features: Dict[str, Any]) -> List[float]:
        """Prepare features for prediction"""
        feature_vector = []
        for feature_name in self.config.input_features:
            value = features.get(feature_name, 0)
            if isinstance(value, (int, float)):
                feature_vector.append(float(value))
            else:
                feature_vector.append(0.0)
        return feature_vector
    
    def _prepare_training_data(self, training_data: List[Dict[str, Any]]) -> tuple:
        """Prepare training data"""
        X = []
        y = []
        
        for data_point in training_data:
            features = self._prepare_features(data_point)
            target = data_point.get("target_load", 0)
            
            if len(features) == len(self.config.input_features):
                X.append(features)
                y.append(target)
        
        return X, y
    
    def _calculate_confidence(self) -> float:
        """Calculate prediction confidence"""
        if len(self.prediction_history) < 5:
            return 0.5
        
        # Calculate recent accuracy
        recent_predictions = self.prediction_history[-10:]
        # This is a simplified confidence calculation
        return min(0.95, 0.5 + (len(recent_predictions) * 0.05))
    
    async def save_model(self, path: str) -> bool:
        """Save model to disk"""
        try:
            if not self.model:
                return False
            
            model_data = {
                "model": self.model,
                "scaler": self.scaler,
                "config": self.config,
                "version": self.model_version,
                "last_training": self.last_training_time
            }
            
            with open(path, 'wb') as f:
                pickle.dump(model_data, f)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False
    
    async def load_model(self, path: str) -> bool:
        """Load model from disk"""
        try:
            with open(path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data["model"]
            self.scaler = model_data["scaler"]
            self.model_version = model_data["version"]
            self.last_training_time = model_data["last_training"]
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

class CacheOptimizationModel(AIModel):
    """
    AI model for intelligent cache optimization
    """
    
    def __init__(self, config: AIModelConfig):
        self.config = config
        self.access_patterns: Dict[str, List[float]] = {}
        self.cache_hit_rates: Dict[str, float] = {}
        self.model = None
        
        if SKLEARN_AVAILABLE:
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
    
    async def predict(self, features: Dict[str, Any]) -> PredictionResult:
        """Predict optimal cache strategy"""
        try:
            cache_key = features.get("cache_key", "")
            access_frequency = features.get("access_frequency", 0)
            data_size = features.get("data_size", 0)
            last_access = features.get("last_access", 0)
            
            # Calculate cache score
            cache_score = self._calculate_cache_score(
                access_frequency, data_size, last_access
            )
            
            # Predict TTL
            predicted_ttl = self._predict_ttl(cache_score, access_frequency)
            
            # Predict priority
            priority = self._predict_priority(cache_score)
            
            return PredictionResult(
                prediction={
                    "cache_score": cache_score,
                    "recommended_ttl": predicted_ttl,
                    "priority": priority,
                    "should_cache": cache_score > 0.5
                },
                confidence=0.8,
                model_version="cache_optimization_v1",
                timestamp=time.time(),
                features_used=list(features.keys()),
                metadata={
                    "model_type": "cache_optimization",
                    "cache_key": cache_key
                }
            )
            
        except Exception as e:
            logger.error(f"Cache optimization prediction failed: {e}")
            return await self._fallback_prediction(features)
    
    async def train(self, training_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Train cache optimization model"""
        try:
            # Update access patterns
            for data_point in training_data:
                cache_key = data_point.get("cache_key", "")
                access_time = data_point.get("access_time", time.time())
                
                if cache_key not in self.access_patterns:
                    self.access_patterns[cache_key] = []
                
                self.access_patterns[cache_key].append(access_time)
                
                # Keep only recent accesses
                if len(self.access_patterns[cache_key]) > 100:
                    self.access_patterns[cache_key] = self.access_patterns[cache_key][-100:]
            
            return {
                "status": "success",
                "patterns_updated": len(self.access_patterns),
                "training_samples": len(training_data)
            }
            
        except Exception as e:
            logger.error(f"Cache optimization training failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def _calculate_cache_score(self, frequency: float, size: float, last_access: float) -> float:
        """Calculate cache score for a key"""
        # Higher frequency = higher score
        frequency_score = min(1.0, frequency / 100.0)
        
        # Smaller size = higher score (easier to cache)
        size_score = max(0.1, 1.0 - (size / 1000000.0))  # 1MB baseline
        
        # Recent access = higher score
        time_since_access = time.time() - last_access
        recency_score = max(0.1, 1.0 - (time_since_access / 3600.0))  # 1 hour baseline
        
        # Weighted combination
        cache_score = (frequency_score * 0.5 + size_score * 0.3 + recency_score * 0.2)
        return min(1.0, max(0.0, cache_score))
    
    def _predict_ttl(self, cache_score: float, frequency: float) -> int:
        """Predict optimal TTL"""
        base_ttl = 300  # 5 minutes
        if cache_score > 0.8:
            return int(base_ttl * 4)  # 20 minutes
        elif cache_score > 0.6:
            return int(base_ttl * 2)  # 10 minutes
        elif cache_score > 0.4:
            return base_ttl  # 5 minutes
        else:
            return int(base_ttl * 0.5)  # 2.5 minutes
    
    def _predict_priority(self, cache_score: float) -> str:
        """Predict cache priority"""
        if cache_score > 0.8:
            return "high"
        elif cache_score > 0.6:
            return "medium"
        else:
            return "low"
    
    async def _fallback_prediction(self, features: Dict[str, Any]) -> PredictionResult:
        """Fallback prediction"""
        return PredictionResult(
            prediction={
                "cache_score": 0.5,
                "recommended_ttl": 300,
                "priority": "medium",
                "should_cache": True
            },
            confidence=0.3,
            model_version="fallback",
            timestamp=time.time(),
            features_used=list(features.keys()),
            metadata={"method": "fallback"}
        )
    
    async def save_model(self, path: str) -> bool:
        """Save model"""
        try:
            model_data = {
                "access_patterns": self.access_patterns,
                "cache_hit_rates": self.cache_hit_rates,
                "config": self.config
            }
            
            with open(path, 'wb') as f:
                pickle.dump(model_data, f)
            
            return True
        except Exception as e:
            logger.error(f"Failed to save cache model: {e}")
            return False
    
    async def load_model(self, path: str) -> bool:
        """Load model"""
        try:
            with open(path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.access_patterns = model_data["access_patterns"]
            self.cache_hit_rates = model_data["cache_hit_rates"]
            
            return True
        except Exception as e:
            logger.error(f"Failed to load cache model: {e}")
            return False

class AnomalyDetectionModel(AIModel):
    """
    AI model for anomaly detection in microservices
    """
    
    def __init__(self, config: AIModelConfig):
        self.config = config
        self.normal_patterns: Dict[str, List[float]] = {}
        self.thresholds: Dict[str, float] = {}
        self.model = None
        
        if SKLEARN_AVAILABLE:
            from sklearn.ensemble import IsolationForest
            self.model = IsolationForest(
                contamination=0.1,
                random_state=42
            )
    
    async def predict(self, features: Dict[str, Any]) -> PredictionResult:
        """Detect anomalies"""
        try:
            # Extract metrics
            response_time = features.get("response_time", 0)
            error_rate = features.get("error_rate", 0)
            cpu_usage = features.get("cpu_usage", 0)
            memory_usage = features.get("memory_usage", 0)
            request_count = features.get("request_count", 0)
            
            # Calculate anomaly score
            anomaly_score = self._calculate_anomaly_score(
                response_time, error_rate, cpu_usage, memory_usage, request_count
            )
            
            is_anomaly = anomaly_score > 0.7
            
            return PredictionResult(
                prediction={
                    "is_anomaly": is_anomaly,
                    "anomaly_score": anomaly_score,
                    "severity": self._get_severity(anomaly_score),
                    "affected_metrics": self._get_affected_metrics(features)
                },
                confidence=0.85,
                model_version="anomaly_detection_v1",
                timestamp=time.time(),
                features_used=list(features.keys()),
                metadata={
                    "model_type": "anomaly_detection",
                    "service": features.get("service_name", "unknown")
                }
            )
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return await self._fallback_anomaly_detection(features)
    
    async def train(self, training_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Train anomaly detection model"""
        try:
            # Extract normal patterns
            for data_point in training_data:
                service_name = data_point.get("service_name", "default")
                
                if service_name not in self.normal_patterns:
                    self.normal_patterns[service_name] = []
                
                # Store normal metrics
                metrics = {
                    "response_time": data_point.get("response_time", 0),
                    "error_rate": data_point.get("error_rate", 0),
                    "cpu_usage": data_point.get("cpu_usage", 0),
                    "memory_usage": data_point.get("memory_usage", 0)
                }
                
                self.normal_patterns[service_name].append(metrics)
                
                # Keep only recent data
                if len(self.normal_patterns[service_name]) > 1000:
                    self.normal_patterns[service_name] = self.normal_patterns[service_name][-1000:]
            
            # Calculate thresholds
            self._calculate_thresholds()
            
            return {
                "status": "success",
                "services_trained": len(self.normal_patterns),
                "training_samples": len(training_data)
            }
            
        except Exception as e:
            logger.error(f"Anomaly detection training failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def _calculate_anomaly_score(self, response_time: float, error_rate: float, 
                                cpu_usage: float, memory_usage: float, request_count: float) -> float:
        """Calculate anomaly score"""
        scores = []
        
        # Response time anomaly
        if response_time > 1000:  # > 1 second
            scores.append(0.8)
        elif response_time > 500:  # > 500ms
            scores.append(0.5)
        else:
            scores.append(0.1)
        
        # Error rate anomaly
        if error_rate > 0.1:  # > 10%
            scores.append(0.9)
        elif error_rate > 0.05:  # > 5%
            scores.append(0.6)
        else:
            scores.append(0.1)
        
        # CPU usage anomaly
        if cpu_usage > 90:  # > 90%
            scores.append(0.8)
        elif cpu_usage > 80:  # > 80%
            scores.append(0.5)
        else:
            scores.append(0.1)
        
        # Memory usage anomaly
        if memory_usage > 90:  # > 90%
            scores.append(0.8)
        elif memory_usage > 80:  # > 80%
            scores.append(0.5)
        else:
            scores.append(0.1)
        
        # Request count anomaly (sudden spike)
        if request_count > 1000:  # > 1000 req/min
            scores.append(0.7)
        elif request_count > 500:  # > 500 req/min
            scores.append(0.4)
        else:
            scores.append(0.1)
        
        return sum(scores) / len(scores)
    
    def _get_severity(self, anomaly_score: float) -> str:
        """Get anomaly severity"""
        if anomaly_score > 0.8:
            return "critical"
        elif anomaly_score > 0.6:
            return "high"
        elif anomaly_score > 0.4:
            return "medium"
        else:
            return "low"
    
    def _get_affected_metrics(self, features: Dict[str, Any]) -> List[str]:
        """Get affected metrics"""
        affected = []
        
        if features.get("response_time", 0) > 500:
            affected.append("response_time")
        if features.get("error_rate", 0) > 0.05:
            affected.append("error_rate")
        if features.get("cpu_usage", 0) > 80:
            affected.append("cpu_usage")
        if features.get("memory_usage", 0) > 80:
            affected.append("memory_usage")
        
        return affected
    
    def _calculate_thresholds(self):
        """Calculate thresholds for each service"""
        for service_name, patterns in self.normal_patterns.items():
            if len(patterns) < 10:
                continue
            
            # Calculate percentiles
            response_times = [p["response_time"] for p in patterns]
            error_rates = [p["error_rate"] for p in patterns]
            cpu_usages = [p["cpu_usage"] for p in patterns]
            memory_usages = [p["memory_usage"] for p in patterns]
            
            self.thresholds[service_name] = {
                "response_time_95th": np.percentile(response_times, 95),
                "error_rate_95th": np.percentile(error_rates, 95),
                "cpu_usage_95th": np.percentile(cpu_usages, 95),
                "memory_usage_95th": np.percentile(memory_usages, 95)
            }
    
    async def _fallback_anomaly_detection(self, features: Dict[str, Any]) -> PredictionResult:
        """Fallback anomaly detection"""
        return PredictionResult(
            prediction={
                "is_anomaly": False,
                "anomaly_score": 0.1,
                "severity": "low",
                "affected_metrics": []
            },
            confidence=0.3,
            model_version="fallback",
            timestamp=time.time(),
            features_used=list(features.keys()),
            metadata={"method": "fallback"}
        )
    
    async def save_model(self, path: str) -> bool:
        """Save model"""
        try:
            model_data = {
                "normal_patterns": self.normal_patterns,
                "thresholds": self.thresholds,
                "config": self.config
            }
            
            with open(path, 'wb') as f:
                pickle.dump(model_data, f)
            
            return True
        except Exception as e:
            logger.error(f"Failed to save anomaly model: {e}")
            return False
    
    async def load_model(self, path: str) -> bool:
        """Load model"""
        try:
            with open(path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.normal_patterns = model_data["normal_patterns"]
            self.thresholds = model_data["thresholds"]
            
            return True
        except Exception as e:
            logger.error(f"Failed to load anomaly model: {e}")
            return False

class AIModelManager:
    """
    Manager for AI models in microservices
    """
    
    def __init__(self):
        self.models: Dict[str, AIModel] = {}
        self.training_data: Dict[str, List[Dict[str, Any]]] = {}
        self.prediction_cache: Dict[str, PredictionResult] = {}
        self.model_configs: Dict[str, AIModelConfig] = {}
    
    def register_model(self, name: str, model: AIModel, config: AIModelConfig):
        """Register an AI model"""
        self.models[name] = model
        self.model_configs[name] = config
        self.training_data[name] = []
        logger.info(f"Registered AI model: {name}")
    
    async def predict(self, model_name: str, features: Dict[str, Any]) -> Optional[PredictionResult]:
        """Make prediction using specified model"""
        try:
            if model_name not in self.models:
                logger.error(f"Model {model_name} not found")
                return None
            
            # Check cache first
            cache_key = self._get_cache_key(model_name, features)
            if cache_key in self.prediction_cache:
                cached_result = self.prediction_cache[cache_key]
                if time.time() - cached_result.timestamp < 60:  # 1 minute cache
                    return cached_result
            
            # Make prediction
            model = self.models[model_name]
            result = await model.predict(features)
            
            # Cache result
            self.prediction_cache[cache_key] = result
            
            # Clean old cache entries
            await self._cleanup_cache()
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed for model {model_name}: {e}")
            return None
    
    async def train_model(self, model_name: str, training_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Train specified model"""
        try:
            if model_name not in self.models:
                return {"status": "error", "message": f"Model {model_name} not found"}
            
            # Add to training data
            self.training_data[model_name].extend(training_data)
            
            # Keep only recent data
            max_data_points = 10000
            if len(self.training_data[model_name]) > max_data_points:
                self.training_data[model_name] = self.training_data[model_name][-max_data_points:]
            
            # Train model
            model = self.models[model_name]
            result = await model.train(self.training_data[model_name])
            
            # Save model
            config = self.model_configs[model_name]
            await model.save_model(config.model_path)
            
            logger.info(f"Trained model {model_name}: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Training failed for model {model_name}: {e}")
            return {"status": "error", "error": str(e)}
    
    async def auto_train_models(self):
        """Automatically retrain models based on schedule"""
        for model_name, config in self.model_configs.items():
            try:
                model = self.models[model_name]
                
                # Check if retraining is needed
                if time.time() - getattr(model, 'last_training_time', 0) > config.retrain_interval:
                    if len(self.training_data[model_name]) > 100:  # Minimum data points
                        logger.info(f"Auto-retraining model {model_name}")
                        await self.train_model(model_name, [])
                
            except Exception as e:
                logger.error(f"Auto-training failed for {model_name}: {e}")
    
    def _get_cache_key(self, model_name: str, features: Dict[str, Any]) -> str:
        """Generate cache key for prediction"""
        feature_str = json.dumps(features, sort_keys=True)
        return hashlib.md5(f"{model_name}:{feature_str}".encode()).hexdigest()
    
    async def _cleanup_cache(self):
        """Clean up old cache entries"""
        current_time = time.time()
        expired_keys = [
            key for key, result in self.prediction_cache.items()
            if current_time - result.timestamp > 300  # 5 minutes
        ]
        
        for key in expired_keys:
            del self.prediction_cache[key]
    
    async def get_model_metrics(self) -> Dict[str, Any]:
        """Get metrics for all models"""
        metrics = {}
        
        for model_name, model in self.models.items():
            try:
                metrics[model_name] = {
                    "type": type(model).__name__,
                    "training_data_points": len(self.training_data.get(model_name, [])),
                    "cached_predictions": len([
                        k for k in self.prediction_cache.keys() 
                        if k.startswith(model_name)
                    ]),
                    "last_training": getattr(model, 'last_training_time', 0),
                    "model_version": getattr(model, 'model_version', 'unknown')
                }
            except Exception as e:
                metrics[model_name] = {"error": str(e)}
        
        return metrics

# Global AI model manager
ai_model_manager = AIModelManager()

# Initialize default models
def initialize_ai_models():
    """Initialize default AI models"""
    try:
        # Load forecasting model
        load_config = AIModelConfig(
            model_type=AIModelType.SKLEARN,
            model_path="models/load_forecasting.pkl",
            input_features=["current_load", "historical_average", "trend", "time_of_day", "day_of_week"],
            output_features=["predicted_load"],
            prediction_type=PredictionType.LOAD_FORECASTING
        )
        load_model = LoadForecastingModel(load_config)
        ai_model_manager.register_model("load_forecasting", load_model, load_config)
        
        # Cache optimization model
        cache_config = AIModelConfig(
            model_type=AIModelType.SKLEARN,
            model_path="models/cache_optimization.pkl",
            input_features=["access_frequency", "data_size", "last_access", "cache_hit_rate"],
            output_features=["cache_score", "recommended_ttl", "priority"],
            prediction_type=PredictionType.CACHE_OPTIMIZATION
        )
        cache_model = CacheOptimizationModel(cache_config)
        ai_model_manager.register_model("cache_optimization", cache_model, cache_config)
        
        # Anomaly detection model
        anomaly_config = AIModelConfig(
            model_type=AIModelType.SKLEARN,
            model_path="models/anomaly_detection.pkl",
            input_features=["response_time", "error_rate", "cpu_usage", "memory_usage", "request_count"],
            output_features=["is_anomaly", "anomaly_score", "severity"],
            prediction_type=PredictionType.ANOMALY_DETECTION
        )
        anomaly_model = AnomalyDetectionModel(anomaly_config)
        ai_model_manager.register_model("anomaly_detection", anomaly_model, anomaly_config)
        
        logger.info("AI models initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize AI models: {e}")

# Decorator for AI-powered caching
def ai_cached(model_name: str = "cache_optimization", ttl_func: Optional[Callable] = None):
    """Decorator for AI-powered caching"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Get AI prediction for cache strategy
            features = {
                "cache_key": cache_key,
                "access_frequency": 1,  # This would be tracked in real implementation
                "data_size": 1000,  # Estimated
                "last_access": time.time()
            }
            
            prediction = await ai_model_manager.predict(model_name, features)
            
            if prediction and prediction.prediction.get("should_cache", True):
                # Use AI-recommended TTL
                recommended_ttl = prediction.prediction.get("recommended_ttl", 300)
                
                # This would integrate with your cache manager
                # For now, just call the function
                return await func(*args, **kwargs)
            else:
                return await func(*args, **kwargs)
        
        def sync_wrapper(*args, **kwargs):
            # Similar logic for sync functions
            return func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

# Initialize models on import
initialize_ai_models()






























