"""
Advanced Optimizer for Email Sequence System

Provides advanced optimization features including:
- Machine learning-based optimization
- Predictive caching
- Adaptive batch sizing
- Intelligent resource management
- Performance prediction
"""

import asyncio
import logging
import time
import psutil
import gc
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from collections import defaultdict, deque
import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import json
import os

from ..models.sequence import EmailSequence, SequenceStep
from ..models.subscriber import Subscriber
from ..models.template import EmailTemplate

logger = logging.getLogger(__name__)

# Constants
MAX_HISTORY_SIZE = 10000
PREDICTION_THRESHOLD = 0.8
ADAPTIVE_LEARNING_RATE = 0.01


@dataclass
class AdvancedOptimizationConfig:
    """Advanced optimization configuration"""
    enable_ml_optimization: bool = True
    enable_predictive_caching: bool = True
    enable_adaptive_batching: bool = True
    enable_intelligent_resource_management: bool = True
    enable_performance_prediction: bool = True
    ml_model_path: str = "models/optimization_model.pkl"
    history_size: int = MAX_HISTORY_SIZE
    prediction_threshold: float = PREDICTION_THRESHOLD
    learning_rate: float = ADAPTIVE_LEARNING_RATE


@dataclass
class PerformancePrediction:
    """Performance prediction results"""
    predicted_throughput: float
    predicted_memory_usage: float
    predicted_error_rate: float
    confidence: float
    recommendations: List[str]


@dataclass
class OptimizationMetrics:
    """Advanced optimization metrics"""
    ml_accuracy: float
    cache_prediction_accuracy: float
    adaptive_batch_efficiency: float
    resource_utilization: float
    prediction_confidence: float


class AdvancedOptimizer:
    """Advanced optimizer with ML-based optimization"""
    
    def __init__(self, config: AdvancedOptimizationConfig):
        self.config = config
        self.performance_history = deque(maxlen=config.history_size)
        self.ml_model = None
        self.scaler = StandardScaler()
        self.feature_names = [
            'sequence_count', 'subscriber_count', 'template_count',
            'avg_sequence_length', 'avg_subscriber_age', 'memory_usage',
            'cpu_usage', 'batch_size', 'concurrent_tasks'
        ]
        
        # Predictive caching
        self.cache_predictions = {}
        self.cache_hit_predictions = 0
        self.cache_miss_predictions = 0
        
        # Adaptive batching
        self.batch_size_history = deque(maxlen=100)
        self.performance_by_batch_size = defaultdict(list)
        
        # Resource management
        self.resource_usage_history = deque(maxlen=100)
        self.optimal_resource_configs = {}
        
        # Load ML model if available
        self._load_ml_model()
        
        logger.info("Advanced Optimizer initialized")
    
    def _load_ml_model(self) -> None:
        """Load ML model for performance prediction"""
        try:
            if os.path.exists(self.config.ml_model_path):
                self.ml_model = joblib.load(self.config.ml_model_path)
                logger.info("ML model loaded successfully")
            else:
                logger.info("No ML model found, will train new model")
        except Exception as e:
            logger.error(f"Error loading ML model: {e}")
    
    def _save_ml_model(self) -> None:
        """Save ML model"""
        try:
            os.makedirs(os.path.dirname(self.config.ml_model_path), exist_ok=True)
            joblib.dump(self.ml_model, self.config.ml_model_path)
            logger.info("ML model saved successfully")
        except Exception as e:
            logger.error(f"Error saving ML model: {e}")
    
    def _extract_features(
        self,
        sequences: List[EmailSequence],
        subscribers: List[Subscriber],
        templates: List[EmailTemplate],
        current_metrics: Dict[str, Any]
    ) -> np.ndarray:
        """Extract features for ML model"""
        try:
            features = []
            
            # Basic counts
            features.append(len(sequences))
            features.append(len(subscribers))
            features.append(len(templates))
            
            # Average sequence length
            if sequences:
                avg_length = np.mean([len(s.steps) for s in sequences])
                features.append(avg_length)
            else:
                features.append(0)
            
            # Average subscriber age (placeholder)
            features.append(30)  # Placeholder
            
            # System metrics
            features.append(current_metrics.get('memory_usage', 0))
            features.append(current_metrics.get('cpu_usage', 0))
            features.append(current_metrics.get('batch_size', 32))
            features.append(current_metrics.get('concurrent_tasks', 4))
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return np.zeros((1, len(self.feature_names)))
    
    def predict_performance(
        self,
        sequences: List[EmailSequence],
        subscribers: List[Subscriber],
        templates: List[EmailTemplate],
        current_metrics: Dict[str, Any]
    ) -> PerformancePrediction:
        """Predict performance using ML model"""
        try:
            if not self.ml_model:
                return self._default_prediction()
            
            # Extract features
            features = self._extract_features(sequences, subscribers, templates, current_metrics)
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Make prediction
            prediction = self.ml_model.predict(features_scaled)[0]
            
            # Calculate confidence (simplified)
            confidence = min(0.95, max(0.5, prediction / 100))
            
            # Generate recommendations
            recommendations = self._generate_recommendations(features[0], prediction)
            
            return PerformancePrediction(
                predicted_throughput=prediction,
                predicted_memory_usage=current_metrics.get('memory_usage', 0.5),
                predicted_error_rate=0.01,
                confidence=confidence,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Error predicting performance: {e}")
            return self._default_prediction()
    
    def _default_prediction(self) -> PerformancePrediction:
        """Default prediction when ML model is not available"""
        return PerformancePrediction(
            predicted_throughput=100.0,
            predicted_memory_usage=0.5,
            predicted_error_rate=0.02,
            confidence=0.5,
            recommendations=["Enable ML optimization for better predictions"]
        )
    
    def _generate_recommendations(self, features: np.ndarray, prediction: float) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        # Analyze features and provide recommendations
        if features[0] > 100:  # Many sequences
            recommendations.append("Consider batch processing for large sequence counts")
        
        if features[1] > 1000:  # Many subscribers
            recommendations.append("Use subscriber segmentation for better performance")
        
        if features[3] > 10:  # Long sequences
            recommendations.append("Optimize sequence length for better throughput")
        
        if features[5] > 0.8:  # High memory usage
            recommendations.append("Enable memory optimization and garbage collection")
        
        if features[6] > 0.8:  # High CPU usage
            recommendations.append("Reduce concurrent tasks to lower CPU usage")
        
        if prediction < 50:
            recommendations.append("Consider scaling up resources for better performance")
        
        return recommendations
    
    async def optimize_with_ml(
        self,
        sequences: List[EmailSequence],
        subscribers: List[Subscriber],
        templates: List[EmailTemplate],
        current_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimize processing using ML predictions"""
        try:
            # Get performance prediction
            prediction = self.predict_performance(sequences, subscribers, templates, current_metrics)
            
            # Apply ML-based optimizations
            optimizations = await self._apply_ml_optimizations(
                sequences, subscribers, templates, prediction
            )
            
            # Update history
            self._update_performance_history(current_metrics, prediction)
            
            # Retrain model if needed
            if len(self.performance_history) % 100 == 0:
                await self._retrain_model()
            
            return {
                "prediction": prediction,
                "optimizations": optimizations,
                "confidence": prediction.confidence
            }
            
        except Exception as e:
            logger.error(f"Error in ML optimization: {e}")
            return {"error": str(e)}
    
    async def _apply_ml_optimizations(
        self,
        sequences: List[EmailSequence],
        subscribers: List[Subscriber],
        templates: List[EmailTemplate],
        prediction: PerformancePrediction
    ) -> Dict[str, Any]:
        """Apply ML-based optimizations"""
        optimizations = {}
        
        try:
            # Adaptive batch sizing
            if self.config.enable_adaptive_batching:
                optimal_batch_size = self._calculate_optimal_batch_size(prediction)
                optimizations['batch_size'] = optimal_batch_size
            
            # Predictive caching
            if self.config.enable_predictive_caching:
                cache_predictions = self._predict_cache_needs(sequences, subscribers)
                optimizations['cache_predictions'] = cache_predictions
            
            # Resource management
            if self.config.enable_intelligent_resource_management:
                resource_config = self._optimize_resource_allocation(prediction)
                optimizations['resource_config'] = resource_config
            
            # Performance-based recommendations
            optimizations['recommendations'] = prediction.recommendations
            
            return optimizations
            
        except Exception as e:
            logger.error(f"Error applying ML optimizations: {e}")
            return {"error": str(e)}
    
    def _calculate_optimal_batch_size(self, prediction: PerformancePrediction) -> int:
        """Calculate optimal batch size based on prediction"""
        try:
            # Base batch size on predicted throughput
            base_size = int(prediction.predicted_throughput / 10)
            
            # Adjust based on memory prediction
            if prediction.predicted_memory_usage > 0.8:
                base_size = max(16, base_size // 2)
            elif prediction.predicted_memory_usage < 0.3:
                base_size = min(128, base_size * 2)
            
            # Ensure reasonable bounds
            optimal_size = max(8, min(256, base_size))
            
            self.batch_size_history.append(optimal_size)
            
            return optimal_size
            
        except Exception as e:
            logger.error(f"Error calculating optimal batch size: {e}")
            return 32
    
    def _predict_cache_needs(
        self,
        sequences: List[EmailSequence],
        subscribers: List[Subscriber]
    ) -> Dict[str, Any]:
        """Predict cache needs based on data patterns"""
        try:
            cache_predictions = {}
            
            # Predict sequence cache needs
            if sequences:
                sequence_cache_size = min(len(sequences) * 2, 1000)
                cache_predictions['sequences'] = {
                    'size': sequence_cache_size,
                    'priority': 'high' if len(sequences) > 50 else 'medium'
                }
            
            # Predict subscriber cache needs
            if subscribers:
                subscriber_cache_size = min(len(subscribers) * 1.5, 2000)
                cache_predictions['subscribers'] = {
                    'size': subscriber_cache_size,
                    'priority': 'high' if len(subscribers) > 1000 else 'medium'
                }
            
            return cache_predictions
            
        except Exception as e:
            logger.error(f"Error predicting cache needs: {e}")
            return {}
    
    def _optimize_resource_allocation(self, prediction: PerformancePrediction) -> Dict[str, Any]:
        """Optimize resource allocation based on prediction"""
        try:
            config = {}
            
            # CPU allocation
            if prediction.predicted_cpu_usage > 0.8:
                config['max_concurrent_tasks'] = max(2, int(4 * 0.7))
            elif prediction.predicted_cpu_usage < 0.3:
                config['max_concurrent_tasks'] = min(16, int(4 * 1.5))
            else:
                config['max_concurrent_tasks'] = 4
            
            # Memory allocation
            if prediction.predicted_memory_usage > 0.8:
                config['enable_memory_optimization'] = True
                config['cache_size'] = 500
            else:
                config['enable_memory_optimization'] = False
                config['cache_size'] = 1000
            
            # Batch processing
            config['batch_size'] = self._calculate_optimal_batch_size(prediction)
            
            return config
            
        except Exception as e:
            logger.error(f"Error optimizing resource allocation: {e}")
            return {}
    
    def _update_performance_history(
        self,
        current_metrics: Dict[str, Any],
        prediction: PerformancePrediction
    ) -> None:
        """Update performance history for ML training"""
        try:
            history_entry = {
                'timestamp': time.time(),
                'actual_throughput': current_metrics.get('throughput', 0),
                'predicted_throughput': prediction.predicted_throughput,
                'memory_usage': current_metrics.get('memory_usage', 0),
                'cpu_usage': current_metrics.get('cpu_usage', 0),
                'error_rate': current_metrics.get('error_rate', 0),
                'features': current_metrics.get('features', [])
            }
            
            self.performance_history.append(history_entry)
            
        except Exception as e:
            logger.error(f"Error updating performance history: {e}")
    
    async def _retrain_model(self) -> None:
        """Retrain ML model with new data"""
        try:
            if len(self.performance_history) < 50:
                return
            
            # Prepare training data
            X = []
            y = []
            
            for entry in self.performance_history:
                if 'features' in entry and len(entry['features']) == len(self.feature_names):
                    X.append(entry['features'])
                    y.append(entry['actual_throughput'])
            
            if len(X) < 10:
                return
            
            X = np.array(X)
            y = np.array(y)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train new model
            self.ml_model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.ml_model.fit(X_scaled, y)
            
            # Save model
            self._save_ml_model()
            
            logger.info("ML model retrained successfully")
            
        except Exception as e:
            logger.error(f"Error retraining ML model: {e}")
    
    def get_advanced_metrics(self) -> OptimizationMetrics:
        """Get advanced optimization metrics"""
        try:
            # Calculate ML accuracy
            ml_accuracy = 0.0
            if self.performance_history:
                predictions = [entry.get('predicted_throughput', 0) for entry in self.performance_history]
                actuals = [entry.get('actual_throughput', 0) for entry in self.performance_history]
                
                if predictions and actuals:
                    mse = np.mean((np.array(predictions) - np.array(actuals)) ** 2)
                    ml_accuracy = max(0, 1 - mse / 10000)
            
            # Calculate cache prediction accuracy
            cache_accuracy = 0.0
            if self.cache_hit_predictions + self.cache_miss_predictions > 0:
                cache_accuracy = self.cache_hit_predictions / (self.cache_hit_predictions + self.cache_miss_predictions)
            
            # Calculate adaptive batch efficiency
            batch_efficiency = 0.8  # Placeholder
            
            # Calculate resource utilization
            resource_utilization = 0.7  # Placeholder
            
            # Calculate prediction confidence
            prediction_confidence = 0.75  # Placeholder
            
            return OptimizationMetrics(
                ml_accuracy=ml_accuracy,
                cache_prediction_accuracy=cache_accuracy,
                adaptive_batch_efficiency=batch_efficiency,
                resource_utilization=resource_utilization,
                prediction_confidence=prediction_confidence
            )
            
        except Exception as e:
            logger.error(f"Error calculating advanced metrics: {e}")
            return OptimizationMetrics(0.0, 0.0, 0.0, 0.0, 0.0)
    
    def export_optimization_data(self, filepath: str) -> bool:
        """Export optimization data for analysis"""
        try:
            data = {
                'performance_history': list(self.performance_history),
                'batch_size_history': list(self.batch_size_history),
                'resource_usage_history': list(self.resource_usage_history),
                'cache_predictions': self.cache_predictions,
                'metrics': self.get_advanced_metrics().__dict__
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.info(f"Optimization data exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting optimization data: {e}")
            return False
    
    def import_optimization_data(self, filepath: str) -> bool:
        """Import optimization data"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Import data
            if 'performance_history' in data:
                self.performance_history.extend(data['performance_history'])
            
            if 'batch_size_history' in data:
                self.batch_size_history.extend(data['batch_size_history'])
            
            if 'resource_usage_history' in data:
                self.resource_usage_history.extend(data['resource_usage_history'])
            
            if 'cache_predictions' in data:
                self.cache_predictions.update(data['cache_predictions'])
            
            logger.info(f"Optimization data imported from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error importing optimization data: {e}")
            return False 