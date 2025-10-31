"""
Machine Learning Predictor for AI Model Performance
==================================================

This module provides advanced machine learning capabilities for predicting
AI model performance, optimizing model selection, and forecasting trends.

Features:
- Performance prediction models
- Model selection optimization
- Trend forecasting with ML
- Anomaly detection using ML
- Performance optimization recommendations
- Automated model training and validation
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import pickle
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.cluster import DBSCAN
from sklearn.anomaly_detection import IsolationForest
import joblib

# Import our AI history analyzer components
from .ai_history_analyzer import (
    AIHistoryAnalyzer, ModelType, PerformanceMetric,
    get_ai_history_analyzer
)
from .config import get_ai_history_config

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Result of a machine learning prediction"""
    model_name: str
    metric: str
    predicted_value: float
    confidence: float
    prediction_date: datetime
    features_used: List[str]
    model_accuracy: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class OptimizationRecommendation:
    """ML-based optimization recommendation"""
    recommendation_type: str  # "model_selection", "parameter_tuning", "resource_allocation"
    model_name: str
    current_performance: float
    predicted_improvement: float
    confidence: float
    reasoning: str
    implementation_cost: float
    expected_roi: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class AnomalyDetectionResult:
    """Result of anomaly detection"""
    timestamp: datetime
    model_name: str
    metric: str
    value: float
    anomaly_score: float
    is_anomaly: bool
    severity: str  # "low", "medium", "high", "critical"
    explanation: str
    recommended_action: str


class MLPredictor:
    """Machine learning predictor for AI model performance"""
    
    def __init__(self, model_storage_path: str = "ml_models"):
        self.model_storage_path = model_storage_path
        self.analyzer = get_ai_history_analyzer()
        self.config = get_ai_history_config()
        
        # ML Models
        self.performance_models: Dict[str, Any] = {}
        self.anomaly_detectors: Dict[str, Any] = {}
        self.optimization_models: Dict[str, Any] = {}
        
        # Preprocessing
        self.scalers: Dict[str, StandardScaler] = {}
        self.label_encoders: Dict[str, LabelEncoder] = {}
        
        # Model metadata
        self.model_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Ensure model storage directory exists
        os.makedirs(model_storage_path, exist_ok=True)
    
    async def train_performance_prediction_model(self, 
                                               model_name: str,
                                               metric: PerformanceMetric,
                                               algorithm: str = "random_forest") -> Dict[str, Any]:
        """Train a machine learning model to predict performance"""
        try:
            # Get historical data
            performance_data = self.analyzer.get_model_performance(model_name, metric, days=365)
            
            if len(performance_data) < 50:  # Need sufficient data
                raise ValueError(f"Insufficient data for training: {len(performance_data)} samples")
            
            # Prepare features and target
            features, target = self._prepare_training_data(performance_data, model_name, metric)
            
            if len(features) < 30:
                raise ValueError(f"Insufficient features for training: {len(features)} samples")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features, target, test_size=0.2, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = self._create_model(algorithm)
            model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
            
            # Store model and scaler
            model_key = f"{model_name}_{metric.value}"
            self.performance_models[model_key] = model
            self.scalers[model_key] = scaler
            
            # Store metadata
            self.model_metadata[model_key] = {
                "model_name": model_name,
                "metric": metric.value,
                "algorithm": algorithm,
                "training_samples": len(features),
                "test_samples": len(X_test),
                "mse": mse,
                "r2_score": r2,
                "mae": mae,
                "cv_mean": cv_scores.mean(),
                "cv_std": cv_scores.std(),
                "feature_names": self._get_feature_names(),
                "trained_at": datetime.now().isoformat()
            }
            
            # Save model
            await self._save_model(model_key, model, scaler)
            
            logger.info(f"Trained {algorithm} model for {model_name} - {metric.value}")
            logger.info(f"R² Score: {r2:.3f}, MAE: {mae:.3f}")
            
            return {
                "success": True,
                "model_key": model_key,
                "algorithm": algorithm,
                "r2_score": r2,
                "mae": mae,
                "cv_score": cv_scores.mean(),
                "training_samples": len(features),
                "feature_importance": self._get_feature_importance(model, algorithm)
            }
            
        except Exception as e:
            logger.error(f"Error training performance prediction model: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _prepare_training_data(self, 
                              performance_data: List,
                              model_name: str,
                              metric: PerformanceMetric) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data with features and target"""
        try:
            features = []
            target = []
            
            for i, perf in enumerate(performance_data):
                # Target: current performance value
                target.append(perf.value)
                
                # Features: time-based and context features
                feature_vector = []
                
                # Time features
                feature_vector.append(perf.timestamp.hour)
                feature_vector.append(perf.timestamp.weekday())
                feature_vector.append(perf.timestamp.day)
                feature_vector.append(perf.timestamp.month)
                
                # Historical performance features (if available)
                if i > 0:
                    feature_vector.append(performance_data[i-1].value)  # Previous value
                else:
                    feature_vector.append(perf.value)  # Use current value
                
                if i > 1:
                    feature_vector.append(performance_data[i-2].value)  # Two steps back
                else:
                    feature_vector.append(perf.value)
                
                # Rolling averages (if enough data)
                if i >= 7:
                    recent_values = [p.value for p in performance_data[i-7:i]]
                    feature_vector.append(np.mean(recent_values))  # 7-day average
                    feature_vector.append(np.std(recent_values))   # 7-day std
                else:
                    feature_vector.extend([perf.value, 0.0])
                
                if i >= 30:
                    recent_values = [p.value for p in performance_data[i-30:i]]
                    feature_vector.append(np.mean(recent_values))  # 30-day average
                else:
                    feature_vector.append(perf.value)
                
                # Context features
                if perf.context:
                    feature_vector.append(len(perf.context))  # Context complexity
                else:
                    feature_vector.append(0)
                
                # Model-specific features
                model_def = self.config.get_model(model_name)
                if model_def:
                    feature_vector.append(model_def.context_length / 1000)  # Normalized context length
                    feature_vector.append(model_def.cost_per_1k_tokens * 1000)  # Cost feature
                else:
                    feature_vector.extend([0.0, 0.0])
                
                features.append(feature_vector)
            
            return np.array(features), np.array(target)
            
        except Exception as e:
            logger.error(f"Error preparing training data: {str(e)}")
            raise
    
    def _create_model(self, algorithm: str):
        """Create ML model based on algorithm name"""
        if algorithm == "random_forest":
            return RandomForestRegressor(n_estimators=100, random_state=42)
        elif algorithm == "gradient_boosting":
            return GradientBoostingRegressor(n_estimators=100, random_state=42)
        elif algorithm == "linear_regression":
            return LinearRegression()
        elif algorithm == "ridge":
            return Ridge(alpha=1.0)
        else:
            return RandomForestRegressor(n_estimators=100, random_state=42)
    
    def _get_feature_names(self) -> List[str]:
        """Get feature names for the model"""
        return [
            "hour", "weekday", "day", "month",
            "prev_value", "prev_value_2", "avg_7d", "std_7d", "avg_30d",
            "context_complexity", "context_length", "cost_per_1k"
        ]
    
    def _get_feature_importance(self, model, algorithm: str) -> Dict[str, float]:
        """Get feature importance from trained model"""
        try:
            if hasattr(model, 'feature_importances_'):
                feature_names = self._get_feature_names()
                importance_dict = dict(zip(feature_names, model.feature_importances_))
                return importance_dict
            else:
                return {}
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            return {}
    
    async def predict_performance(self, 
                                model_name: str,
                                metric: PerformanceMetric,
                                prediction_date: datetime = None) -> Optional[PredictionResult]:
        """Predict performance for a model and metric"""
        try:
            if prediction_date is None:
                prediction_date = datetime.now()
            
            model_key = f"{model_name}_{metric.value}"
            
            # Check if model exists
            if model_key not in self.performance_models:
                # Try to load model
                await self._load_model(model_key)
                
                if model_key not in self.performance_models:
                    logger.warning(f"No trained model found for {model_key}")
                    return None
            
            # Get recent performance data for features
            recent_data = self.analyzer.get_model_performance(model_name, metric, days=30)
            
            if not recent_data:
                logger.warning(f"No recent data for {model_name} - {metric.value}")
                return None
            
            # Prepare features for prediction
            features = self._prepare_prediction_features(recent_data, model_name, prediction_date)
            
            # Scale features
            scaler = self.scalers[model_key]
            features_scaled = scaler.transform([features])
            
            # Make prediction
            model = self.performance_models[model_key]
            predicted_value = model.predict(features_scaled)[0]
            
            # Calculate confidence based on model metadata
            metadata = self.model_metadata.get(model_key, {})
            confidence = metadata.get("r2_score", 0.5)
            
            # Get model accuracy
            model_accuracy = metadata.get("cv_mean", 0.5)
            
            return PredictionResult(
                model_name=model_name,
                metric=metric.value,
                predicted_value=predicted_value,
                confidence=confidence,
                prediction_date=prediction_date,
                features_used=self._get_feature_names(),
                model_accuracy=model_accuracy,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error predicting performance: {str(e)}")
            return None
    
    def _prepare_prediction_features(self, 
                                   recent_data: List,
                                   model_name: str,
                                   prediction_date: datetime) -> List[float]:
        """Prepare features for prediction"""
        try:
            feature_vector = []
            
            # Time features
            feature_vector.append(prediction_date.hour)
            feature_vector.append(prediction_date.weekday())
            feature_vector.append(prediction_date.day)
            feature_vector.append(prediction_date.month)
            
            # Historical performance features
            if recent_data:
                feature_vector.append(recent_data[-1].value)  # Latest value
                if len(recent_data) > 1:
                    feature_vector.append(recent_data[-2].value)  # Previous value
                else:
                    feature_vector.append(recent_data[-1].value)
                
                # Rolling averages
                if len(recent_data) >= 7:
                    recent_values = [p.value for p in recent_data[-7:]]
                    feature_vector.append(np.mean(recent_values))
                    feature_vector.append(np.std(recent_values))
                else:
                    recent_values = [p.value for p in recent_data]
                    feature_vector.append(np.mean(recent_values))
                    feature_vector.append(np.std(recent_values))
                
                if len(recent_data) >= 30:
                    recent_values = [p.value for p in recent_data[-30:]]
                    feature_vector.append(np.mean(recent_values))
                else:
                    recent_values = [p.value for p in recent_data]
                    feature_vector.append(np.mean(recent_values))
            else:
                # Default values if no recent data
                feature_vector.extend([0.5, 0.5, 0.5, 0.0, 0.5])
            
            # Context features
            feature_vector.append(0)  # Context complexity (default)
            
            # Model-specific features
            model_def = self.config.get_model(model_name)
            if model_def:
                feature_vector.append(model_def.context_length / 1000)
                feature_vector.append(model_def.cost_per_1k_tokens * 1000)
            else:
                feature_vector.extend([0.0, 0.0])
            
            return feature_vector
            
        except Exception as e:
            logger.error(f"Error preparing prediction features: {str(e)}")
            return [0.0] * 12  # Default feature vector
    
    async def detect_anomalies(self, 
                             model_name: str,
                             metric: PerformanceMetric,
                             days: int = 30) -> List[AnomalyDetectionResult]:
        """Detect anomalies in model performance using ML"""
        try:
            # Get performance data
            performance_data = self.analyzer.get_model_performance(model_name, metric, days)
            
            if len(performance_data) < 10:
                return []
            
            # Prepare data for anomaly detection
            values = np.array([p.value for p in performance_data]).reshape(-1, 1)
            
            # Train anomaly detector
            anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
            anomaly_scores = anomaly_detector.fit_predict(values)
            anomaly_scores_float = anomaly_detector.decision_function(values)
            
            # Process results
            anomalies = []
            for i, (perf, score, is_anomaly) in enumerate(zip(performance_data, anomaly_scores_float, anomaly_scores)):
                if is_anomaly == -1:  # Anomaly detected
                    severity = self._determine_anomaly_severity(score, perf.value, metric)
                    
                    anomaly = AnomalyDetectionResult(
                        timestamp=perf.timestamp,
                        model_name=model_name,
                        metric=metric.value,
                        value=perf.value,
                        anomaly_score=abs(score),
                        is_anomaly=True,
                        severity=severity,
                        explanation=self._explain_anomaly(perf.value, metric, severity),
                        recommended_action=self._recommend_anomaly_action(severity, metric)
                    )
                    anomalies.append(anomaly)
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {str(e)}")
            return []
    
    def _determine_anomaly_severity(self, score: float, value: float, metric: PerformanceMetric) -> str:
        """Determine anomaly severity based on score and value"""
        try:
            # Get metric configuration
            metric_config = self.config.get_metric(metric.value)
            if not metric_config:
                return "medium"
            
            # Calculate how far the value is from optimal range
            optimal_min, optimal_max = metric_config.optimal_range
            
            if optimal_min <= value <= optimal_max:
                return "low"
            elif abs(score) > 0.5:
                return "high"
            else:
                return "medium"
                
        except Exception as e:
            logger.error(f"Error determining anomaly severity: {str(e)}")
            return "medium"
    
    def _explain_anomaly(self, value: float, metric: PerformanceMetric, severity: str) -> str:
        """Generate explanation for anomaly"""
        try:
            metric_config = self.config.get_metric(metric.value)
            if not metric_config:
                return f"Unusual {metric.value} value detected"
            
            optimal_min, optimal_max = metric_config.optimal_range
            
            if value < optimal_min:
                return f"{metric.value} is significantly below optimal range ({value:.3f} < {optimal_min:.3f})"
            elif value > optimal_max:
                return f"{metric.value} is significantly above optimal range ({value:.3f} > {optimal_max:.3f})"
            else:
                return f"Unusual {metric.value} pattern detected (value: {value:.3f})"
                
        except Exception as e:
            logger.error(f"Error explaining anomaly: {str(e)}")
            return f"Anomaly detected in {metric.value}"
    
    def _recommend_anomaly_action(self, severity: str, metric: PerformanceMetric) -> str:
        """Recommend action for anomaly"""
        if severity == "high":
            return f"Immediate investigation required for {metric.value} anomaly"
        elif severity == "medium":
            return f"Monitor {metric.value} closely and investigate if pattern continues"
        else:
            return f"Continue monitoring {metric.value} for any trends"
    
    async def generate_optimization_recommendations(self, 
                                                  model_name: str,
                                                  target_metric: PerformanceMetric) -> List[OptimizationRecommendation]:
        """Generate ML-based optimization recommendations"""
        try:
            recommendations = []
            
            # Get current performance
            current_summary = self.analyzer.get_performance_summary(model_name, days=30)
            if not current_summary or "metrics" not in current_summary:
                return recommendations
            
            current_value = current_summary["metrics"].get(target_metric.value, {}).get("mean", 0.0)
            
            # Model selection recommendation
            model_selection_rec = await self._generate_model_selection_recommendation(
                model_name, target_metric, current_value
            )
            if model_selection_rec:
                recommendations.append(model_selection_rec)
            
            # Parameter tuning recommendation
            param_tuning_rec = await self._generate_parameter_tuning_recommendation(
                model_name, target_metric, current_value
            )
            if param_tuning_rec:
                recommendations.append(param_tuning_rec)
            
            # Resource allocation recommendation
            resource_rec = await self._generate_resource_allocation_recommendation(
                model_name, target_metric, current_value
            )
            if resource_rec:
                recommendations.append(resource_rec)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating optimization recommendations: {str(e)}")
            return []
    
    async def _generate_model_selection_recommendation(self, 
                                                     model_name: str,
                                                     metric: PerformanceMetric,
                                                     current_value: float) -> Optional[OptimizationRecommendation]:
        """Generate model selection recommendation"""
        try:
            # Get rankings for the metric
            rankings = self.analyzer.get_model_rankings(metric, days=30)
            
            if not rankings or len(rankings) < 2:
                return None
            
            # Find current model ranking
            current_rank = None
            for ranking in rankings:
                if ranking["model_name"] == model_name:
                    current_rank = ranking["rank"]
                    break
            
            if not current_rank or current_rank <= 2:
                return None  # Already performing well
            
            # Find better performing model
            better_model = rankings[0]  # Top performer
            improvement = better_model["mean_value"] - current_value
            
            if improvement < 0.05:  # Less than 5% improvement
                return None
            
            return OptimizationRecommendation(
                recommendation_type="model_selection",
                model_name=model_name,
                current_performance=current_value,
                predicted_improvement=improvement,
                confidence=better_model["confidence"],
                reasoning=f"Switch to {better_model['model_name']} for {improvement:.1%} improvement",
                implementation_cost=0.1,  # Low cost for model switch
                expected_roi=improvement / 0.1
            )
            
        except Exception as e:
            logger.error(f"Error generating model selection recommendation: {str(e)}")
            return None
    
    async def _generate_parameter_tuning_recommendation(self, 
                                                      model_name: str,
                                                      metric: PerformanceMetric,
                                                      current_value: float) -> Optional[OptimizationRecommendation]:
        """Generate parameter tuning recommendation"""
        try:
            # Analyze trends to see if tuning might help
            trend_analysis = self.analyzer.analyze_trends(model_name, metric, days=30)
            
            if not trend_analysis or trend_analysis.trend_direction != "declining":
                return None
            
            # Estimate potential improvement from tuning
            estimated_improvement = 0.1  # 10% improvement estimate
            confidence = 0.6  # Moderate confidence
            
            return OptimizationRecommendation(
                recommendation_type="parameter_tuning",
                model_name=model_name,
                current_performance=current_value,
                predicted_improvement=estimated_improvement,
                confidence=confidence,
                reasoning=f"Parameter tuning recommended due to declining {metric.value} trend",
                implementation_cost=0.3,  # Medium cost for tuning
                expected_roi=estimated_improvement / 0.3
            )
            
        except Exception as e:
            logger.error(f"Error generating parameter tuning recommendation: {str(e)}")
            return None
    
    async def _generate_resource_allocation_recommendation(self, 
                                                         model_name: str,
                                                         metric: PerformanceMetric,
                                                         current_value: float) -> Optional[OptimizationRecommendation]:
        """Generate resource allocation recommendation"""
        try:
            # Check if model is resource-constrained
            model_def = self.config.get_model(model_name)
            if not model_def:
                return None
            
            # Simple heuristic: if response time is high, recommend more resources
            if metric == PerformanceMetric.RESPONSE_TIME and current_value > 5.0:
                estimated_improvement = 0.2  # 20% improvement
                confidence = 0.7
                
                return OptimizationRecommendation(
                    recommendation_type="resource_allocation",
                    model_name=model_name,
                    current_performance=current_value,
                    predicted_improvement=estimated_improvement,
                    confidence=confidence,
                    reasoning="Increase computational resources to improve response time",
                    implementation_cost=0.5,  # High cost for resource upgrade
                    expected_roi=estimated_improvement / 0.5
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating resource allocation recommendation: {str(e)}")
            return None
    
    async def _save_model(self, model_key: str, model: Any, scaler: StandardScaler):
        """Save trained model and scaler"""
        try:
            model_path = os.path.join(self.model_storage_path, f"{model_key}_model.pkl")
            scaler_path = os.path.join(self.model_storage_path, f"{model_key}_scaler.pkl")
            metadata_path = os.path.join(self.model_storage_path, f"{model_key}_metadata.json")
            
            # Save model
            joblib.dump(model, model_path)
            
            # Save scaler
            joblib.dump(scaler, scaler_path)
            
            # Save metadata
            with open(metadata_path, 'w') as f:
                json.dump(self.model_metadata[model_key], f, indent=2)
            
            logger.info(f"Saved model {model_key} to {self.model_storage_path}")
            
        except Exception as e:
            logger.error(f"Error saving model {model_key}: {str(e)}")
    
    async def _load_model(self, model_key: str):
        """Load trained model and scaler"""
        try:
            model_path = os.path.join(self.model_storage_path, f"{model_key}_model.pkl")
            scaler_path = os.path.join(self.model_storage_path, f"{model_key}_scaler.pkl")
            metadata_path = os.path.join(self.model_storage_path, f"{model_key}_metadata.json")
            
            if not all(os.path.exists(p) for p in [model_path, scaler_path, metadata_path]):
                return False
            
            # Load model
            model = joblib.load(model_path)
            self.performance_models[model_key] = model
            
            # Load scaler
            scaler = joblib.load(scaler_path)
            self.scalers[model_key] = scaler
            
            # Load metadata
            with open(metadata_path, 'r') as f:
                self.model_metadata[model_key] = json.load(f)
            
            logger.info(f"Loaded model {model_key} from {self.model_storage_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model {model_key}: {str(e)}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about trained models"""
        return {
            "trained_models": list(self.performance_models.keys()),
            "model_metadata": self.model_metadata,
            "storage_path": self.model_storage_path
        }


# Global ML predictor instance
_ml_predictor: Optional[MLPredictor] = None


def get_ml_predictor(model_storage_path: str = "ml_models") -> MLPredictor:
    """Get or create global ML predictor"""
    global _ml_predictor
    if _ml_predictor is None:
        _ml_predictor = MLPredictor(model_storage_path)
    return _ml_predictor


# Example usage and testing
async def main():
    """Example usage of the ML predictor"""
    predictor = get_ml_predictor()
    
    # Train a performance prediction model
    training_result = await predictor.train_performance_prediction_model(
        model_name="gpt-4",
        metric=PerformanceMetric.QUALITY_SCORE,
        algorithm="random_forest"
    )
    
    print(f"Training result: {training_result}")
    
    # Make a prediction
    prediction = await predictor.predict_performance(
        model_name="gpt-4",
        metric=PerformanceMetric.QUALITY_SCORE
    )
    
    if prediction:
        print(f"Predicted quality score: {prediction.predicted_value:.3f}")
        print(f"Confidence: {prediction.confidence:.3f}")
    
    # Detect anomalies
    anomalies = await predictor.detect_anomalies(
        model_name="gpt-4",
        metric=PerformanceMetric.QUALITY_SCORE,
        days=30
    )
    
    print(f"Detected {len(anomalies)} anomalies")
    
    # Generate optimization recommendations
    recommendations = await predictor.generate_optimization_recommendations(
        model_name="gpt-4",
        target_metric=PerformanceMetric.QUALITY_SCORE
    )
    
    print(f"Generated {len(recommendations)} optimization recommendations")


if __name__ == "__main__":
    asyncio.run(main())

























