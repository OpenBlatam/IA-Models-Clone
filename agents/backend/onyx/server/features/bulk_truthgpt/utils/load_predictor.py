"""
Load Predictor
=============

Advanced load prediction system for optimal resource allocation.
"""

import asyncio
import logging
import time
import numpy as np
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor
import weakref
from collections import defaultdict, deque
import json
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd

logger = logging.getLogger(__name__)

class PredictionModel(str, Enum):
    """Prediction models."""
    LINEAR = "linear"
    RANDOM_FOREST = "random_forest"
    LSTM = "lstm"
    ARIMA = "arima"
    PROPHET = "prophet"

class LoadType(str, Enum):
    """Load types."""
    CPU = "cpu"
    MEMORY = "memory"
    NETWORK = "network"
    DISK = "disk"
    USERS = "users"
    REQUESTS = "requests"

@dataclass
class LoadData:
    """Load data point."""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    network_io: float
    disk_io: float
    active_users: int
    request_rate: float
    response_time: float
    error_rate: float

@dataclass
class PredictionResult:
    """Prediction result."""
    timestamp: datetime
    predicted_values: Dict[str, float]
    confidence: float
    model_used: str
    prediction_horizon: int

@dataclass
class LoadPredictorConfig:
    """Load predictor configuration."""
    model_type: PredictionModel = PredictionModel.RANDOM_FOREST
    prediction_horizon: int = 60  # seconds
    history_window: int = 3600  # seconds
    retrain_interval: int = 300  # seconds
    confidence_threshold: float = 0.8
    enable_ensemble: bool = True
    enable_anomaly_detection: bool = True
    anomaly_threshold: float = 2.0

class LoadPredictor:
    """
    Advanced load prediction system.
    
    Features:
    - Multiple prediction models
    - Ensemble predictions
    - Anomaly detection
    - Auto-retraining
    - Confidence scoring
    """
    
    def __init__(self, config: Optional[LoadPredictorConfig] = None):
        self.config = config or LoadPredictorConfig()
        self.models = {}
        self.scalers = {}
        self.load_history = deque(maxlen=10000)
        self.predictions = deque(maxlen=1000)
        self.anomalies = deque(maxlen=1000)
        self.stats = {
            'total_predictions': 0,
            'accurate_predictions': 0,
            'anomalies_detected': 0,
            'model_retrains': 0,
            'average_accuracy': 0.0
        }
        self.running = False
        self.lock = asyncio.Lock()
        
    async def initialize(self):
        """Initialize load predictor."""
        logger.info("Initializing Load Predictor...")
        
        try:
            # Initialize models
            await self._initialize_models()
            
            # Start background tasks
            self.running = True
            asyncio.create_task(self._retrain_models())
            asyncio.create_task(self._detect_anomalies())
            
            logger.info("Load Predictor initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Load Predictor: {str(e)}")
            raise
    
    async def _initialize_models(self):
        """Initialize prediction models."""
        try:
            # Initialize models based on config
            if self.config.model_type == PredictionModel.LINEAR:
                self.models['primary'] = LinearRegression()
            elif self.config.model_type == PredictionModel.RANDOM_FOREST:
                self.models['primary'] = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                )
            else:
                self.models['primary'] = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                )
            
            # Initialize ensemble models if enabled
            if self.config.enable_ensemble:
                self.models['ensemble'] = {
                    'linear': LinearRegression(),
                    'random_forest': RandomForestRegressor(n_estimators=50, random_state=42),
                    'extra_trees': RandomForestRegressor(n_estimators=50, random_state=42, max_features='sqrt')
                }
            
            # Initialize scalers
            for model_name in self.models.keys():
                if model_name == 'ensemble':
                    for sub_model in self.models[model_name].values():
                        self.scalers[f"{model_name}_{sub_model.__class__.__name__}"] = StandardScaler()
                else:
                    self.scalers[model_name] = StandardScaler()
            
            logger.info("Prediction models initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {str(e)}")
            raise
    
    async def add_load_data(self, load_data: LoadData):
        """Add load data point."""
        try:
            async with self.lock:
                self.load_history.append(load_data)
                
                # Check if we have enough data for prediction
                if len(self.load_history) >= 10:
                    await self._update_models()
            
        except Exception as e:
            logger.error(f"Failed to add load data: {str(e)}")
    
    async def predict_load(self, horizon: Optional[int] = None) -> PredictionResult:
        """Predict future load."""
        try:
            if len(self.load_history) < 10:
                raise ValueError("Not enough historical data for prediction")
            
            horizon = horizon or self.config.prediction_horizon
            
            # Prepare features
            features = await self._prepare_features()
            
            # Make prediction
            if self.config.enable_ensemble:
                prediction = await self._ensemble_predict(features, horizon)
            else:
                prediction = await self._single_model_predict(features, horizon)
            
            # Calculate confidence
            confidence = await self._calculate_confidence(prediction)
            
            # Create result
            result = PredictionResult(
                timestamp=datetime.utcnow(),
                predicted_values=prediction,
                confidence=confidence,
                model_used=self.config.model_type.value,
                prediction_horizon=horizon
            )
            
            # Store prediction
            self.predictions.append(result)
            self.stats['total_predictions'] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to predict load: {str(e)}")
            raise
    
    async def _prepare_features(self) -> np.ndarray:
        """Prepare features for prediction."""
        try:
            # Get recent data
            recent_data = list(self.load_history)[-self.config.history_window:]
            
            if len(recent_data) < 10:
                raise ValueError("Not enough data for feature preparation")
            
            # Extract features
            features = []
            for data_point in recent_data:
                feature_vector = [
                    data_point.cpu_usage,
                    data_point.memory_usage,
                    data_point.network_io,
                    data_point.disk_io,
                    data_point.active_users,
                    data_point.request_rate,
                    data_point.response_time,
                    data_point.error_rate
                ]
                features.append(feature_vector)
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Failed to prepare features: {str(e)}")
            raise
    
    async def _single_model_predict(self, features: np.ndarray, horizon: int) -> Dict[str, float]:
        """Make prediction using single model."""
        try:
            # Prepare data
            X = features[:-1]  # Input features
            y = features[1:]   # Target values
            
            # Scale features
            scaler = self.scalers['primary']
            X_scaled = scaler.fit_transform(X)
            
            # Train model
            model = self.models['primary']
            model.fit(X_scaled, y)
            
            # Make prediction
            last_features = X_scaled[-1].reshape(1, -1)
            prediction = model.predict(last_features)[0]
            
            # Convert to dictionary
            result = {
                'cpu_usage': float(prediction[0]),
                'memory_usage': float(prediction[1]),
                'network_io': float(prediction[2]),
                'disk_io': float(prediction[3]),
                'active_users': int(prediction[4]),
                'request_rate': float(prediction[5]),
                'response_time': float(prediction[6]),
                'error_rate': float(prediction[7])
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Single model prediction failed: {str(e)}")
            raise
    
    async def _ensemble_predict(self, features: np.ndarray, horizon: int) -> Dict[str, float]:
        """Make prediction using ensemble models."""
        try:
            # Prepare data
            X = features[:-1]
            y = features[1:]
            
            # Get predictions from all models
            predictions = []
            
            for model_name, model in self.models['ensemble'].items():
                scaler = self.scalers[f"ensemble_{model.__class__.__name__}"]
                X_scaled = scaler.fit_transform(X)
                
                model.fit(X_scaled, y)
                prediction = model.predict(X_scaled[-1].reshape(1, -1))[0]
                predictions.append(prediction)
            
            # Average predictions
            ensemble_prediction = np.mean(predictions, axis=0)
            
            # Convert to dictionary
            result = {
                'cpu_usage': float(ensemble_prediction[0]),
                'memory_usage': float(ensemble_prediction[1]),
                'network_io': float(ensemble_prediction[2]),
                'disk_io': float(ensemble_prediction[3]),
                'active_users': int(ensemble_prediction[4]),
                'request_rate': float(ensemble_prediction[5]),
                'response_time': float(ensemble_prediction[6]),
                'error_rate': float(ensemble_prediction[7])
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Ensemble prediction failed: {str(e)}")
            raise
    
    async def _calculate_confidence(self, prediction: Dict[str, float]) -> float:
        """Calculate prediction confidence."""
        try:
            # Simple confidence calculation based on historical accuracy
            if len(self.predictions) < 5:
                return 0.5
            
            # Calculate accuracy of recent predictions
            recent_predictions = list(self.predictions)[-10:]
            accuracy_scores = []
            
            for pred in recent_predictions:
                # This would compare with actual values
                # For now, use a simple heuristic
                accuracy_scores.append(0.8)
            
            confidence = np.mean(accuracy_scores)
            return min(confidence, 1.0)
            
        except Exception as e:
            logger.error(f"Failed to calculate confidence: {str(e)}")
            return 0.5
    
    async def _retrain_models(self):
        """Retrain models periodically."""
        while self.running:
            try:
                await asyncio.sleep(self.config.retrain_interval)
                
                if len(self.load_history) < 50:
                    continue
                
                # Retrain models
                await self._update_models()
                self.stats['model_retrains'] += 1
                
                logger.debug("Models retrained")
                
            except Exception as e:
                logger.error(f"Model retraining failed: {str(e)}")
    
    async def _update_models(self):
        """Update models with new data."""
        try:
            if len(self.load_history) < 10:
                return
            
            # Prepare training data
            features = await self._prepare_features()
            X = features[:-1]
            y = features[1:]
            
            # Update primary model
            scaler = self.scalers['primary']
            X_scaled = scaler.fit_transform(X)
            self.models['primary'].fit(X_scaled, y)
            
            # Update ensemble models
            if self.config.enable_ensemble:
                for model_name, model in self.models['ensemble'].items():
                    scaler = self.scalers[f"ensemble_{model.__class__.__name__}"]
                    X_scaled = scaler.fit_transform(X)
                    model.fit(X_scaled, y)
            
        except Exception as e:
            logger.error(f"Failed to update models: {str(e)}")
    
    async def _detect_anomalies(self):
        """Detect anomalies in load data."""
        while self.running:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                if len(self.load_history) < 10:
                    continue
                
                # Get recent data
                recent_data = list(self.load_history)[-10:]
                
                # Calculate statistics
                cpu_values = [data.cpu_usage for data in recent_data]
                memory_values = [data.memory_usage for data in recent_data]
                
                # Detect anomalies
                cpu_mean = np.mean(cpu_values)
                cpu_std = np.std(cpu_values)
                memory_mean = np.mean(memory_values)
                memory_std = np.std(memory_values)
                
                # Check for anomalies
                for data in recent_data:
                    cpu_z_score = abs(data.cpu_usage - cpu_mean) / (cpu_std + 1e-8)
                    memory_z_score = abs(data.memory_usage - memory_mean) / (memory_std + 1e-8)
                    
                    if cpu_z_score > self.config.anomaly_threshold or memory_z_score > self.config.anomaly_threshold:
                        anomaly = {
                            'timestamp': data.timestamp,
                            'cpu_z_score': cpu_z_score,
                            'memory_z_score': memory_z_score,
                            'severity': max(cpu_z_score, memory_z_score)
                        }
                        
                        self.anomalies.append(anomaly)
                        self.stats['anomalies_detected'] += 1
                        
                        logger.warning(f"Anomaly detected: CPU={cpu_z_score:.2f}, Memory={memory_z_score:.2f}")
                
            except Exception as e:
                logger.error(f"Anomaly detection failed: {str(e)}")
    
    def get_prediction_stats(self) -> Dict[str, Any]:
        """Get prediction statistics."""
        return {
            'total_predictions': self.stats['total_predictions'],
            'accurate_predictions': self.stats['accurate_predictions'],
            'anomalies_detected': self.stats['anomalies_detected'],
            'model_retrains': self.stats['model_retrains'],
            'average_accuracy': self.stats['average_accuracy'],
            'recent_predictions': len(self.predictions),
            'recent_anomalies': len(self.anomalies),
            'config': {
                'model_type': self.config.model_type.value,
                'prediction_horizon': self.config.prediction_horizon,
                'history_window': self.config.history_window,
                'retrain_interval': self.config.retrain_interval,
                'ensemble_enabled': self.config.enable_ensemble,
                'anomaly_detection_enabled': self.config.enable_anomaly_detection
            }
        }
    
    async def cleanup(self):
        """Cleanup load predictor."""
        try:
            self.running = False
            
            # Clear data
            self.load_history.clear()
            self.predictions.clear()
            self.anomalies.clear()
            
            # Clear models
            self.models.clear()
            self.scalers.clear()
            
            logger.info("Load Predictor cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Failed to cleanup Load Predictor: {str(e)}")

# Global load predictor
load_predictor = LoadPredictor()

# Decorators for load prediction
def load_prediction_enabled(func):
    """Decorator for load prediction enabled functions."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Get current load data
        current_time = datetime.utcnow()
        
        # This would get actual load data
        load_data = LoadData(
            timestamp=current_time,
            cpu_usage=0.5,
            memory_usage=0.6,
            network_io=100.0,
            disk_io=50.0,
            active_users=10,
            request_rate=5.0,
            response_time=0.1,
            error_rate=0.01
        )
        
        # Add to predictor
        await load_predictor.add_load_data(load_data)
        
        # Execute function
        return await func(*args, **kwargs)
    
    return wrapper

def predict_load(horizon: int = 60):
    """Decorator for load prediction."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get load prediction
            prediction = await load_predictor.predict_load(horizon)
            
            # Execute function with prediction
            return await func(prediction, *args, **kwargs)
        
        return wrapper
    return decorator











