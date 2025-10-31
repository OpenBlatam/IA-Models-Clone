"""
AI Performance Predictor
========================

Advanced AI performance prediction system with machine learning models
for forecasting AI model performance, resource usage, and optimization opportunities.
"""

import asyncio
import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import hashlib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import joblib
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class PredictionType(str, Enum):
    """Types of predictions"""
    PERFORMANCE = "performance"
    RESOURCE_USAGE = "resource_usage"
    COST = "cost"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    QUALITY = "quality"
    SCALABILITY = "scalability"
    RELIABILITY = "reliability"


class PredictionHorizon(str, Enum):
    """Prediction horizons"""
    SHORT_TERM = "short_term"  # 1-7 days
    MEDIUM_TERM = "medium_term"  # 1-4 weeks
    LONG_TERM = "long_term"  # 1-12 months
    CUSTOM = "custom"


class ModelComplexity(str, Enum):
    """Model complexity levels"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"


@dataclass
class PredictionResult:
    """Result of AI performance prediction"""
    prediction_id: str
    model_name: str
    prediction_type: PredictionType
    horizon: PredictionHorizon
    predicted_values: List[Dict[str, Any]]
    confidence_intervals: List[Dict[str, Any]]
    feature_importance: Dict[str, float]
    model_accuracy: float
    prediction_confidence: float
    created_at: datetime
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class PerformanceForecast:
    """Performance forecast with multiple metrics"""
    forecast_id: str
    model_name: str
    forecast_period: str
    predictions: Dict[str, List[Dict[str, Any]]]
    trends: Dict[str, str]
    recommendations: List[str]
    risk_factors: List[str]
    opportunities: List[str]
    confidence_score: float
    created_at: datetime


@dataclass
class ResourcePrediction:
    """Resource usage prediction"""
    resource_type: str
    current_usage: float
    predicted_usage: float
    usage_trend: str
    scaling_recommendation: str
    cost_impact: float
    confidence: float


class AIPerformancePredictor:
    """Advanced AI performance predictor with machine learning capabilities"""
    
    def __init__(self, max_history_days: int = 365):
        self.max_history_days = max_history_days
        self.performance_data: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.prediction_models: Dict[str, Any] = {}
        self.prediction_results: List[PredictionResult] = []
        self.performance_forecasts: List[PerformanceForecast] = []
        
        # Model configurations
        self.model_configs = {
            "performance": {
                "algorithms": ["RandomForestRegressor", "GradientBoostingRegressor", "MLPRegressor"],
                "features": ["timestamp", "model_version", "data_size", "complexity", "hardware_specs"],
                "target": "performance_score"
            },
            "resource_usage": {
                "algorithms": ["RandomForestRegressor", "SVR", "LinearRegression"],
                "features": ["timestamp", "workload", "model_size", "batch_size", "concurrency"],
                "target": "resource_usage"
            },
            "cost": {
                "algorithms": ["GradientBoostingRegressor", "RandomForestRegressor", "Ridge"],
                "features": ["timestamp", "resource_usage", "model_complexity", "data_volume", "processing_time"],
                "target": "cost"
            }
        }
        
        # Feature engineering
        self.feature_encoders = {}
        self.scalers = {}
        
        # Cache for predictions
        self.prediction_cache = {}
        self.cache_ttl = 1800  # 30 minutes
    
    async def record_performance_data(self, 
                                    model_name: str,
                                    performance_metrics: Dict[str, float],
                                    context: Dict[str, Any] = None,
                                    metadata: Dict[str, Any] = None) -> bool:
        """Record performance data for prediction models"""
        try:
            data_point = {
                "timestamp": datetime.now(),
                "model_name": model_name,
                "performance_metrics": performance_metrics,
                "context": context or {},
                "metadata": metadata or {}
            }
            
            self.performance_data[model_name].append(data_point)
            
            # Clean old data
            await self._cleanup_old_data()
            
            # Invalidate cache
            self._invalidate_cache()
            
            logger.debug(f"Recorded performance data for {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error recording performance data: {str(e)}")
            return False
    
    async def predict_performance(self, 
                                model_name: str,
                                prediction_type: PredictionType,
                                horizon: PredictionHorizon = PredictionHorizon.SHORT_TERM,
                                custom_horizon_days: int = None) -> PredictionResult:
        """Predict AI model performance"""
        try:
            prediction_id = hashlib.md5(f"{model_name}_{prediction_type}_{horizon}_{datetime.now()}".encode()).hexdigest()
            
            # Get historical data
            historical_data = self.performance_data.get(model_name, [])
            if len(historical_data) < 50:  # Need sufficient data
                raise ValueError(f"Insufficient data for prediction: {len(historical_data)} data points")
            
            # Prepare data for prediction
            df = await self._prepare_prediction_data(historical_data, prediction_type)
            
            # Train prediction model
            prediction_model = await self._train_prediction_model(df, prediction_type)
            if not prediction_model:
                raise ValueError("Failed to train prediction model")
            
            # Generate predictions
            predicted_values, confidence_intervals = await self._generate_predictions(
                prediction_model, df, horizon, custom_horizon_days
            )
            
            # Calculate feature importance
            feature_importance = await self._calculate_feature_importance(prediction_model, df)
            
            # Calculate model accuracy
            model_accuracy = await self._calculate_model_accuracy(prediction_model, df)
            
            # Calculate prediction confidence
            prediction_confidence = await self._calculate_prediction_confidence(
                predicted_values, confidence_intervals, model_accuracy
            )
            
            # Create prediction result
            result = PredictionResult(
                prediction_id=prediction_id,
                model_name=model_name,
                prediction_type=prediction_type,
                horizon=horizon,
                predicted_values=predicted_values,
                confidence_intervals=confidence_intervals,
                feature_importance=feature_importance,
                model_accuracy=model_accuracy,
                prediction_confidence=prediction_confidence,
                created_at=datetime.now(),
                metadata={
                    "data_points_used": len(historical_data),
                    "prediction_horizon_days": custom_horizon_days or self._get_horizon_days(horizon),
                    "model_algorithm": prediction_model.get("algorithm", "unknown")
                }
            )
            
            # Store result
            self.prediction_results.append(result)
            
            logger.info(f"Generated performance prediction for {model_name}: {prediction_type.value}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error predicting performance: {str(e)}")
            raise e
    
    async def generate_comprehensive_forecast(self, 
                                            model_name: str,
                                            forecast_period: str = "30_days") -> PerformanceForecast:
        """Generate comprehensive performance forecast"""
        try:
            forecast_id = hashlib.md5(f"{model_name}_{forecast_period}_{datetime.now()}".encode()).hexdigest()
            
            # Generate predictions for different metrics
            predictions = {}
            trends = {}
            recommendations = []
            risk_factors = []
            opportunities = []
            
            # Performance prediction
            try:
                perf_result = await self.predict_performance(
                    model_name, PredictionType.PERFORMANCE, PredictionHorizon.MEDIUM_TERM
                )
                predictions["performance"] = perf_result.predicted_values
                trends["performance"] = await self._analyze_trend(perf_result.predicted_values)
            except Exception as e:
                logger.warning(f"Failed to predict performance: {str(e)}")
            
            # Resource usage prediction
            try:
                resource_result = await self.predict_performance(
                    model_name, PredictionType.RESOURCE_USAGE, PredictionHorizon.MEDIUM_TERM
                )
                predictions["resource_usage"] = resource_result.predicted_values
                trends["resource_usage"] = await self._analyze_trend(resource_result.predicted_values)
            except Exception as e:
                logger.warning(f"Failed to predict resource usage: {str(e)}")
            
            # Cost prediction
            try:
                cost_result = await self.predict_performance(
                    model_name, PredictionType.COST, PredictionHorizon.MEDIUM_TERM
                )
                predictions["cost"] = cost_result.predicted_values
                trends["cost"] = await self._analyze_trend(cost_result.predicted_values)
            except Exception as e:
                logger.warning(f"Failed to predict cost: {str(e)}")
            
            # Generate recommendations
            recommendations = await self._generate_forecast_recommendations(predictions, trends)
            
            # Identify risk factors
            risk_factors = await self._identify_risk_factors(predictions, trends)
            
            # Identify opportunities
            opportunities = await self._identify_opportunities(predictions, trends)
            
            # Calculate confidence score
            confidence_score = await self._calculate_forecast_confidence(predictions)
            
            # Create performance forecast
            forecast = PerformanceForecast(
                forecast_id=forecast_id,
                model_name=model_name,
                forecast_period=forecast_period,
                predictions=predictions,
                trends=trends,
                recommendations=recommendations,
                risk_factors=risk_factors,
                opportunities=opportunities,
                confidence_score=confidence_score,
                created_at=datetime.now()
            )
            
            # Store forecast
            self.performance_forecasts.append(forecast)
            
            logger.info(f"Generated comprehensive forecast for {model_name}")
            
            return forecast
            
        except Exception as e:
            logger.error(f"Error generating comprehensive forecast: {str(e)}")
            raise e
    
    async def predict_resource_scaling(self, 
                                     model_name: str,
                                     target_performance: float,
                                     current_resources: Dict[str, float]) -> List[ResourcePrediction]:
        """Predict resource scaling requirements"""
        try:
            # Get current performance data
            historical_data = self.performance_data.get(model_name, [])
            if not historical_data:
                raise ValueError(f"No historical data for {model_name}")
            
            # Analyze current resource usage vs performance
            resource_predictions = []
            
            for resource_type, current_usage in current_resources.items():
                # Predict resource usage for target performance
                predicted_usage = await self._predict_resource_for_performance(
                    historical_data, resource_type, target_performance
                )
                
                # Analyze usage trend
                usage_trend = await self._analyze_resource_trend(historical_data, resource_type)
                
                # Generate scaling recommendation
                scaling_recommendation = await self._generate_scaling_recommendation(
                    current_usage, predicted_usage, usage_trend
                )
                
                # Calculate cost impact
                cost_impact = await self._calculate_cost_impact(current_usage, predicted_usage, resource_type)
                
                # Calculate confidence
                confidence = await self._calculate_resource_prediction_confidence(
                    historical_data, resource_type
                )
                
                resource_prediction = ResourcePrediction(
                    resource_type=resource_type,
                    current_usage=current_usage,
                    predicted_usage=predicted_usage,
                    usage_trend=usage_trend,
                    scaling_recommendation=scaling_recommendation,
                    cost_impact=cost_impact,
                    confidence=confidence
                )
                
                resource_predictions.append(resource_prediction)
            
            return resource_predictions
            
        except Exception as e:
            logger.error(f"Error predicting resource scaling: {str(e)}")
            return []
    
    async def get_prediction_analytics(self, model_name: str = None) -> Dict[str, Any]:
        """Get prediction analytics and insights"""
        try:
            if model_name:
                # Get analytics for specific model
                model_predictions = [p for p in self.prediction_results if p.model_name == model_name]
                model_forecasts = [f for f in self.performance_forecasts if f.model_name == model_name]
                
                analytics = {
                    "model_name": model_name,
                    "total_predictions": len(model_predictions),
                    "total_forecasts": len(model_forecasts),
                    "prediction_accuracy": {},
                    "trend_analysis": {},
                    "recommendations": []
                }
                
                # Analyze prediction accuracy
                for prediction in model_predictions:
                    pred_type = prediction.prediction_type.value
                    if pred_type not in analytics["prediction_accuracy"]:
                        analytics["prediction_accuracy"][pred_type] = []
                    analytics["prediction_accuracy"][pred_type].append(prediction.model_accuracy)
                
                # Calculate average accuracy
                for pred_type, accuracies in analytics["prediction_accuracy"].items():
                    analytics["prediction_accuracy"][pred_type] = {
                        "average": np.mean(accuracies),
                        "best": np.max(accuracies),
                        "worst": np.min(accuracies),
                        "count": len(accuracies)
                    }
                
                # Analyze trends
                for forecast in model_forecasts:
                    analytics["trend_analysis"].update(forecast.trends)
                
                # Generate recommendations
                analytics["recommendations"] = await self._generate_analytics_recommendations(
                    model_predictions, model_forecasts
                )
                
            else:
                # Get global analytics
                analytics = {
                    "total_models": len(set(p.model_name for p in self.prediction_results)),
                    "total_predictions": len(self.prediction_results),
                    "total_forecasts": len(self.performance_forecasts),
                    "prediction_types": {},
                    "model_performance": {},
                    "insights": []
                }
                
                # Analyze prediction types
                for prediction in self.prediction_results:
                    pred_type = prediction.prediction_type.value
                    if pred_type not in analytics["prediction_types"]:
                        analytics["prediction_types"][pred_type] = 0
                    analytics["prediction_types"][pred_type] += 1
                
                # Analyze model performance
                for prediction in self.prediction_results:
                    model_name = prediction.model_name
                    if model_name not in analytics["model_performance"]:
                        analytics["model_performance"][model_name] = []
                    analytics["model_performance"][model_name].append(prediction.model_accuracy)
                
                # Calculate average performance per model
                for model_name, accuracies in analytics["model_performance"].items():
                    analytics["model_performance"][model_name] = {
                        "average_accuracy": np.mean(accuracies),
                        "prediction_count": len(accuracies)
                    }
                
                # Generate insights
                analytics["insights"] = await self._generate_global_insights(analytics)
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error getting prediction analytics: {str(e)}")
            return {"error": str(e)}
    
    # Private helper methods
    async def _prepare_prediction_data(self, 
                                     historical_data: List[Dict[str, Any]], 
                                     prediction_type: PredictionType) -> pd.DataFrame:
        """Prepare data for prediction model training"""
        try:
            # Convert to DataFrame
            data_points = []
            for item in historical_data:
                data_point = {
                    "timestamp": item["timestamp"],
                    "model_name": item["model_name"]
                }
                
                # Add performance metrics
                data_point.update(item["performance_metrics"])
                
                # Add context features
                context = item.get("context", {})
                data_point.update(context)
                
                # Add metadata features
                metadata = item.get("metadata", {})
                data_point.update(metadata)
                
                data_points.append(data_point)
            
            df = pd.DataFrame(data_points)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            # Feature engineering
            df = await self._engineer_features(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error preparing prediction data: {str(e)}")
            return pd.DataFrame()
    
    async def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for prediction"""
        try:
            # Time-based features
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['day_of_month'] = df['timestamp'].dt.day
            df['month'] = df['timestamp'].dt.month
            df['quarter'] = df['timestamp'].dt.quarter
            
            # Lag features
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if col not in ['hour', 'day_of_week', 'day_of_month', 'month', 'quarter']:
                    df[f'{col}_lag_1'] = df[col].shift(1)
                    df[f'{col}_lag_7'] = df[col].shift(7)
                    df[f'{col}_lag_30'] = df[col].shift(30)
            
            # Rolling statistics
            for col in numeric_columns:
                if col not in ['hour', 'day_of_week', 'day_of_month', 'month', 'quarter']:
                    df[f'{col}_rolling_mean_7'] = df[col].rolling(window=7).mean()
                    df[f'{col}_rolling_std_7'] = df[col].rolling(window=7).std()
                    df[f'{col}_rolling_mean_30'] = df[col].rolling(window=30).mean()
                    df[f'{col}_rolling_std_30'] = df[col].rolling(window=30).std()
            
            # Trend features
            for col in numeric_columns:
                if col not in ['hour', 'day_of_week', 'day_of_month', 'month', 'quarter']:
                    df[f'{col}_trend_7'] = df[col].rolling(window=7).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0)
                    df[f'{col}_trend_30'] = df[col].rolling(window=30).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0)
            
            # Fill NaN values
            df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            return df
            
        except Exception as e:
            logger.error(f"Error engineering features: {str(e)}")
            return df
    
    async def _train_prediction_model(self, 
                                    df: pd.DataFrame, 
                                    prediction_type: PredictionType) -> Optional[Dict[str, Any]]:
        """Train prediction model"""
        try:
            config = self.model_configs.get(prediction_type.value, {})
            if not config:
                return None
            
            # Prepare features and target
            feature_columns = [col for col in df.columns if col not in ['timestamp', 'model_name', config['target']]]
            X = df[feature_columns].fillna(0)
            y = df[config['target']].fillna(0)
            
            if len(X) < 20:  # Need sufficient data
                return None
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train multiple models and select best
            best_model = None
            best_score = -float('inf')
            best_algorithm = ""
            
            for algorithm in config['algorithms']:
                try:
                    model = await self._create_model(algorithm)
                    model.fit(X_train_scaled, y_train)
                    
                    # Evaluate model
                    y_pred = model.predict(X_test_scaled)
                    score = r2_score(y_test, y_pred)
                    
                    if score > best_score:
                        best_score = score
                        best_model = model
                        best_algorithm = algorithm
                        
                except Exception as e:
                    logger.warning(f"Error training {algorithm}: {str(e)}")
                    continue
            
            if best_model is None:
                return None
            
            # Store model and scaler
            model_key = f"{prediction_type.value}_{best_algorithm}"
            self.prediction_models[model_key] = {
                "model": best_model,
                "algorithm": best_algorithm,
                "scaler": scaler,
                "feature_columns": feature_columns,
                "target": config['target'],
                "accuracy": best_score
            }
            
            return self.prediction_models[model_key]
            
        except Exception as e:
            logger.error(f"Error training prediction model: {str(e)}")
            return None
    
    async def _create_model(self, algorithm: str) -> Any:
        """Create model instance"""
        try:
            if algorithm == "RandomForestRegressor":
                return RandomForestRegressor(n_estimators=100, random_state=42)
            elif algorithm == "GradientBoostingRegressor":
                return GradientBoostingRegressor(n_estimators=100, random_state=42)
            elif algorithm == "AdaBoostRegressor":
                return AdaBoostRegressor(n_estimators=100, random_state=42)
            elif algorithm == "LinearRegression":
                return LinearRegression()
            elif algorithm == "Ridge":
                return Ridge(alpha=1.0, random_state=42)
            elif algorithm == "Lasso":
                return Lasso(alpha=1.0, random_state=42)
            elif algorithm == "ElasticNet":
                return ElasticNet(alpha=1.0, random_state=42)
            elif algorithm == "SVR":
                return SVR(kernel='rbf')
            elif algorithm == "MLPRegressor":
                return MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
                
        except Exception as e:
            logger.error(f"Error creating model: {str(e)}")
            raise e
    
    async def _generate_predictions(self, 
                                  prediction_model: Dict[str, Any], 
                                  df: pd.DataFrame, 
                                  horizon: PredictionHorizon,
                                  custom_horizon_days: int = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Generate predictions for future time periods"""
        try:
            model = prediction_model["model"]
            scaler = prediction_model["scaler"]
            feature_columns = prediction_model["feature_columns"]
            target = prediction_model["target"]
            
            # Determine prediction horizon
            horizon_days = custom_horizon_days or self._get_horizon_days(horizon)
            
            # Get last data point
            last_row = df.iloc[-1]
            last_timestamp = last_row['timestamp']
            
            predicted_values = []
            confidence_intervals = []
            
            # Generate predictions
            for i in range(1, horizon_days + 1):
                future_timestamp = last_timestamp + timedelta(days=i)
                
                # Prepare features for prediction
                feature_values = []
                for col in feature_columns:
                    if col in ['hour', 'day_of_week', 'day_of_month', 'month', 'quarter']:
                        if col == 'hour':
                            feature_values.append(future_timestamp.hour)
                        elif col == 'day_of_week':
                            feature_values.append(future_timestamp.weekday())
                        elif col == 'day_of_month':
                            feature_values.append(future_timestamp.day)
                        elif col == 'month':
                            feature_values.append(future_timestamp.month)
                        elif col == 'quarter':
                            feature_values.append((future_timestamp.month - 1) // 3 + 1)
                    else:
                        # Use recent average for other features
                        recent_values = df[col].tail(7).mean()
                        feature_values.append(recent_values)
                
                # Scale features
                scaled_features = scaler.transform([feature_values])
                
                # Make prediction
                prediction = model.predict(scaled_features)[0]
                
                # Calculate confidence interval (simplified)
                confidence = 0.8  # Base confidence
                std_error = 0.1  # Simplified standard error
                margin = 1.96 * std_error  # 95% confidence interval
                
                predicted_values.append({
                    "timestamp": future_timestamp.isoformat(),
                    "predicted_value": float(prediction),
                    "day_ahead": i
                })
                
                confidence_intervals.append({
                    "timestamp": future_timestamp.isoformat(),
                    "lower_bound": float(prediction - margin),
                    "upper_bound": float(prediction + margin),
                    "confidence_level": 0.95
                })
            
            return predicted_values, confidence_intervals
            
        except Exception as e:
            logger.error(f"Error generating predictions: {str(e)}")
            return [], []
    
    async def _calculate_feature_importance(self, 
                                          prediction_model: Dict[str, Any], 
                                          df: pd.DataFrame) -> Dict[str, float]:
        """Calculate feature importance"""
        try:
            model = prediction_model["model"]
            feature_columns = prediction_model["feature_columns"]
            
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_)
            else:
                # Use mutual information as fallback
                X = df[feature_columns].fillna(0)
                y = df[prediction_model["target"]].fillna(0)
                importances = mutual_info_regression(X, y)
            
            # Normalize importances
            if len(importances) > 0:
                importances = importances / np.sum(importances)
            
            feature_importance = {
                feature: float(importance) 
                for feature, importance in zip(feature_columns, importances)
            }
            
            return feature_importance
            
        except Exception as e:
            logger.error(f"Error calculating feature importance: {str(e)}")
            return {}
    
    async def _calculate_model_accuracy(self, 
                                      prediction_model: Dict[str, Any], 
                                      df: pd.DataFrame) -> float:
        """Calculate model accuracy"""
        try:
            return prediction_model.get("accuracy", 0.0)
            
        except Exception as e:
            logger.error(f"Error calculating model accuracy: {str(e)}")
            return 0.0
    
    async def _calculate_prediction_confidence(self, 
                                             predicted_values: List[Dict[str, Any]], 
                                             confidence_intervals: List[Dict[str, Any]], 
                                             model_accuracy: float) -> float:
        """Calculate prediction confidence"""
        try:
            # Base confidence on model accuracy
            base_confidence = model_accuracy
            
            # Adjust based on prediction horizon (longer horizon = lower confidence)
            horizon_factor = 1.0 / (1.0 + len(predicted_values) * 0.01)
            
            # Adjust based on confidence interval width
            if confidence_intervals:
                avg_width = np.mean([
                    ci["upper_bound"] - ci["lower_bound"] 
                    for ci in confidence_intervals
                ])
                width_factor = 1.0 / (1.0 + avg_width)
            else:
                width_factor = 1.0
            
            confidence = base_confidence * horizon_factor * width_factor
            
            return min(1.0, max(0.0, confidence))
            
        except Exception as e:
            logger.error(f"Error calculating prediction confidence: {str(e)}")
            return 0.5
    
    def _get_horizon_days(self, horizon: PredictionHorizon) -> int:
        """Get number of days for prediction horizon"""
        horizon_mapping = {
            PredictionHorizon.SHORT_TERM: 7,
            PredictionHorizon.MEDIUM_TERM: 30,
            PredictionHorizon.LONG_TERM: 90,
            PredictionHorizon.CUSTOM: 30
        }
        return horizon_mapping.get(horizon, 30)
    
    async def _analyze_trend(self, predicted_values: List[Dict[str, Any]]) -> str:
        """Analyze trend in predicted values"""
        try:
            if len(predicted_values) < 2:
                return "insufficient_data"
            
            values = [pv["predicted_value"] for pv in predicted_values]
            
            # Calculate trend using linear regression
            x = np.arange(len(values))
            slope = np.polyfit(x, values, 1)[0]
            
            if slope > 0.01:
                return "improving"
            elif slope < -0.01:
                return "declining"
            else:
                return "stable"
                
        except Exception as e:
            logger.error(f"Error analyzing trend: {str(e)}")
            return "unknown"
    
    async def _generate_forecast_recommendations(self, 
                                               predictions: Dict[str, List[Dict[str, Any]]], 
                                               trends: Dict[str, str]) -> List[str]:
        """Generate recommendations based on forecast"""
        try:
            recommendations = []
            
            # Performance recommendations
            if "performance" in trends:
                if trends["performance"] == "declining":
                    recommendations.append("Performance is declining - consider model retraining or optimization")
                elif trends["performance"] == "improving":
                    recommendations.append("Performance is improving - consider scaling up usage")
                else:
                    recommendations.append("Performance is stable - maintain current monitoring")
            
            # Resource recommendations
            if "resource_usage" in trends:
                if trends["resource_usage"] == "improving":
                    recommendations.append("Resource usage is increasing - plan for scaling")
                elif trends["resource_usage"] == "declining":
                    recommendations.append("Resource usage is decreasing - consider cost optimization")
            
            # Cost recommendations
            if "cost" in trends:
                if trends["cost"] == "improving":
                    recommendations.append("Costs are increasing - review resource allocation")
                elif trends["cost"] == "declining":
                    recommendations.append("Costs are decreasing - good optimization")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating forecast recommendations: {str(e)}")
            return []
    
    async def _identify_risk_factors(self, 
                                   predictions: Dict[str, List[Dict[str, Any]]], 
                                   trends: Dict[str, str]) -> List[str]:
        """Identify risk factors from predictions"""
        try:
            risk_factors = []
            
            # Performance risks
            if "performance" in predictions:
                perf_values = [p["predicted_value"] for p in predictions["performance"]]
                if min(perf_values) < 0.7:  # Performance threshold
                    risk_factors.append("Performance may drop below acceptable threshold")
            
            # Resource risks
            if "resource_usage" in predictions:
                resource_values = [p["predicted_value"] for p in predictions["resource_usage"]]
                if max(resource_values) > 0.9:  # Resource threshold
                    risk_factors.append("Resource usage may exceed capacity")
            
            # Cost risks
            if "cost" in predictions:
                cost_values = [p["predicted_value"] for p in predictions["cost"]]
                if max(cost_values) > np.mean(cost_values) * 1.5:  # 50% increase
                    risk_factors.append("Costs may increase significantly")
            
            return risk_factors
            
        except Exception as e:
            logger.error(f"Error identifying risk factors: {str(e)}")
            return []
    
    async def _identify_opportunities(self, 
                                    predictions: Dict[str, List[Dict[str, Any]]], 
                                    trends: Dict[str, str]) -> List[str]:
        """Identify opportunities from predictions"""
        try:
            opportunities = []
            
            # Performance opportunities
            if "performance" in trends and trends["performance"] == "improving":
                opportunities.append("Performance improvement trend - consider expanding usage")
            
            # Cost opportunities
            if "cost" in trends and trends["cost"] == "declining":
                opportunities.append("Cost reduction trend - consider increasing capacity")
            
            # Resource opportunities
            if "resource_usage" in trends and trends["resource_usage"] == "declining":
                opportunities.append("Resource efficiency improving - consider optimization")
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Error identifying opportunities: {str(e)}")
            return []
    
    async def _calculate_forecast_confidence(self, predictions: Dict[str, List[Dict[str, Any]]]) -> float:
        """Calculate overall forecast confidence"""
        try:
            if not predictions:
                return 0.0
            
            # Calculate confidence based on number of successful predictions
            successful_predictions = len(predictions)
            total_attempted = 3  # performance, resource_usage, cost
            
            base_confidence = successful_predictions / total_attempted
            
            # Adjust based on prediction consistency
            if len(predictions) > 1:
                # Check if trends are consistent
                trends = []
                for pred_type, pred_values in predictions.items():
                    trend = await self._analyze_trend(pred_values)
                    trends.append(trend)
                
                # Consistency factor (simplified)
                consistency_factor = 1.0 if len(set(trends)) == 1 else 0.8
            else:
                consistency_factor = 1.0
            
            confidence = base_confidence * consistency_factor
            
            return min(1.0, max(0.0, confidence))
            
        except Exception as e:
            logger.error(f"Error calculating forecast confidence: {str(e)}")
            return 0.5
    
    async def _predict_resource_for_performance(self, 
                                              historical_data: List[Dict[str, Any]], 
                                              resource_type: str, 
                                              target_performance: float) -> float:
        """Predict resource usage for target performance"""
        try:
            # Simple linear relationship (in practice, use more sophisticated models)
            if not historical_data:
                return 0.0
            
            # Extract performance and resource data
            performances = []
            resources = []
            
            for data in historical_data:
                metrics = data.get("performance_metrics", {})
                if "performance_score" in metrics and resource_type in metrics:
                    performances.append(metrics["performance_score"])
                    resources.append(metrics[resource_type])
            
            if len(performances) < 2:
                return 0.0
            
            # Simple linear regression
            slope, intercept = np.polyfit(performances, resources, 1)
            predicted_resource = slope * target_performance + intercept
            
            return max(0.0, predicted_resource)
            
        except Exception as e:
            logger.error(f"Error predicting resource for performance: {str(e)}")
            return 0.0
    
    async def _analyze_resource_trend(self, 
                                    historical_data: List[Dict[str, Any]], 
                                    resource_type: str) -> str:
        """Analyze resource usage trend"""
        try:
            if len(historical_data) < 5:
                return "insufficient_data"
            
            # Extract resource values
            resources = []
            for data in historical_data[-10:]:  # Last 10 data points
                metrics = data.get("performance_metrics", {})
                if resource_type in metrics:
                    resources.append(metrics[resource_type])
            
            if len(resources) < 3:
                return "insufficient_data"
            
            # Calculate trend
            x = np.arange(len(resources))
            slope = np.polyfit(x, resources, 1)[0]
            
            if slope > 0.01:
                return "increasing"
            elif slope < -0.01:
                return "decreasing"
            else:
                return "stable"
                
        except Exception as e:
            logger.error(f"Error analyzing resource trend: {str(e)}")
            return "unknown"
    
    async def _generate_scaling_recommendation(self, 
                                             current_usage: float, 
                                             predicted_usage: float, 
                                             usage_trend: str) -> str:
        """Generate scaling recommendation"""
        try:
            usage_ratio = predicted_usage / current_usage if current_usage > 0 else 1.0
            
            if usage_ratio > 1.5:
                return "Scale up significantly - predicted usage is 50%+ higher"
            elif usage_ratio > 1.2:
                return "Scale up moderately - predicted usage is 20%+ higher"
            elif usage_ratio < 0.8:
                return "Scale down - predicted usage is 20%+ lower"
            elif usage_ratio < 0.9:
                return "Consider scaling down - predicted usage is 10%+ lower"
            else:
                return "Maintain current scaling - predicted usage is similar"
                
        except Exception as e:
            logger.error(f"Error generating scaling recommendation: {str(e)}")
            return "Unable to determine scaling recommendation"
    
    async def _calculate_cost_impact(self, 
                                   current_usage: float, 
                                   predicted_usage: float, 
                                   resource_type: str) -> float:
        """Calculate cost impact of scaling"""
        try:
            # Simplified cost calculation (in practice, use actual pricing)
            cost_per_unit = {
                "cpu": 0.1,
                "memory": 0.05,
                "storage": 0.02,
                "network": 0.01
            }.get(resource_type, 0.1)
            
            usage_difference = predicted_usage - current_usage
            cost_impact = usage_difference * cost_per_unit
            
            return cost_impact
            
        except Exception as e:
            logger.error(f"Error calculating cost impact: {str(e)}")
            return 0.0
    
    async def _calculate_resource_prediction_confidence(self, 
                                                      historical_data: List[Dict[str, Any]], 
                                                      resource_type: str) -> float:
        """Calculate confidence for resource prediction"""
        try:
            # Base confidence on amount of historical data
            data_points = len(historical_data)
            base_confidence = min(1.0, data_points / 50)  # Max confidence at 50 data points
            
            # Adjust based on data consistency
            if data_points > 5:
                resources = []
                for data in historical_data[-10:]:
                    metrics = data.get("performance_metrics", {})
                    if resource_type in metrics:
                        resources.append(metrics[resource_type])
                
                if len(resources) > 2:
                    std_dev = np.std(resources)
                    mean_val = np.mean(resources)
                    consistency_factor = 1.0 / (1.0 + std_dev / mean_val) if mean_val > 0 else 0.5
                else:
                    consistency_factor = 0.5
            else:
                consistency_factor = 0.5
            
            confidence = base_confidence * consistency_factor
            
            return min(1.0, max(0.0, confidence))
            
        except Exception as e:
            logger.error(f"Error calculating resource prediction confidence: {str(e)}")
            return 0.5
    
    async def _generate_analytics_recommendations(self, 
                                                predictions: List[PredictionResult], 
                                                forecasts: List[PerformanceForecast]) -> List[str]:
        """Generate recommendations based on analytics"""
        try:
            recommendations = []
            
            # Analyze prediction accuracy
            if predictions:
                accuracies = [p.model_accuracy for p in predictions]
                avg_accuracy = np.mean(accuracies)
                
                if avg_accuracy < 0.7:
                    recommendations.append("Low prediction accuracy - consider improving data quality or model selection")
                elif avg_accuracy > 0.9:
                    recommendations.append("High prediction accuracy - predictions are reliable")
            
            # Analyze forecast trends
            if forecasts:
                for forecast in forecasts:
                    if forecast.confidence_score < 0.6:
                        recommendations.append("Low forecast confidence - consider collecting more data")
                    
                    if forecast.risk_factors:
                        recommendations.append(f"Risk factors identified: {', '.join(forecast.risk_factors)}")
                    
                    if forecast.opportunities:
                        recommendations.append(f"Opportunities identified: {', '.join(forecast.opportunities)}")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating analytics recommendations: {str(e)}")
            return []
    
    async def _generate_global_insights(self, analytics: Dict[str, Any]) -> List[str]:
        """Generate global insights from analytics"""
        try:
            insights = []
            
            # Model performance insights
            model_performance = analytics.get("model_performance", {})
            if model_performance:
                best_model = max(model_performance.items(), key=lambda x: x[1]["average_accuracy"])
                insights.append(f"Best performing model: {best_model[0]} with {best_model[1]['average_accuracy']:.3f} accuracy")
            
            # Prediction type insights
            prediction_types = analytics.get("prediction_types", {})
            if prediction_types:
                most_common = max(prediction_types.items(), key=lambda x: x[1])
                insights.append(f"Most common prediction type: {most_common[0]} ({most_common[1]} predictions)")
            
            # Overall insights
            total_predictions = analytics.get("total_predictions", 0)
            total_models = analytics.get("total_models", 0)
            insights.append(f"Analyzing {total_models} models with {total_predictions} total predictions")
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating global insights: {str(e)}")
            return []
    
    async def _cleanup_old_data(self):
        """Clean up old data"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.max_history_days)
            
            for model_name in list(self.performance_data.keys()):
                self.performance_data[model_name] = [
                    data for data in self.performance_data[model_name]
                    if data["timestamp"] >= cutoff_date
                ]
                
                if not self.performance_data[model_name]:
                    del self.performance_data[model_name]
            
            # Clean up old predictions (keep last 1000)
            if len(self.prediction_results) > 1000:
                self.prediction_results = self.prediction_results[-1000:]
            
            # Clean up old forecasts (keep last 100)
            if len(self.performance_forecasts) > 100:
                self.performance_forecasts = self.performance_forecasts[-100:]
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {str(e)}")
    
    def _invalidate_cache(self):
        """Invalidate prediction cache"""
        self.prediction_cache.clear()


# Global predictor instance
_predictor: Optional[AIPerformancePredictor] = None


def get_ai_performance_predictor(max_history_days: int = 365) -> AIPerformancePredictor:
    """Get or create global AI performance predictor instance"""
    global _predictor
    if _predictor is None:
        _predictor = AIPerformancePredictor(max_history_days)
    return _predictor


# Example usage
async def main():
    """Example usage of the AI performance predictor"""
    predictor = get_ai_performance_predictor()
    
    # Record sample performance data
    models = ["gpt-4", "claude-3", "gemini-pro"]
    
    # Simulate performance data over time
    for i in range(100):
        for model in models:
            performance_metrics = {
                "performance_score": 0.7 + (i * 0.001) + np.random.normal(0, 0.05),
                "resource_usage": 0.6 + (i * 0.002) + np.random.normal(0, 0.03),
                "cost": 0.5 + (i * 0.001) + np.random.normal(0, 0.02),
                "latency": 2.0 - (i * 0.01) + np.random.normal(0, 0.2),
                "throughput": 100 + (i * 0.5) + np.random.normal(0, 5)
            }
            
            await predictor.record_performance_data(
                model_name=model,
                performance_metrics=performance_metrics,
                context={"iteration": i, "test_type": "benchmark"},
                metadata={"version": "1.0", "environment": "production"}
            )
    
    # Generate predictions
    for model in models:
        try:
            # Performance prediction
            perf_prediction = await predictor.predict_performance(
                model, PredictionType.PERFORMANCE, PredictionHorizon.SHORT_TERM
            )
            print(f"Performance prediction for {model}: {perf_prediction.model_accuracy:.3f} accuracy")
            
            # Comprehensive forecast
            forecast = await predictor.generate_comprehensive_forecast(model)
            print(f"Comprehensive forecast for {model}: {forecast.confidence_score:.3f} confidence")
            
        except Exception as e:
            print(f"Error predicting for {model}: {str(e)}")
    
    # Get analytics
    analytics = await predictor.get_prediction_analytics()
    print(f"Analytics: {analytics.get('total_models', 0)} models, {analytics.get('total_predictions', 0)} predictions")


if __name__ == "__main__":
    asyncio.run(main())



























