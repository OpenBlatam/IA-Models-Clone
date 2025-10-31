"""
Advanced Predictive Analytics for AI Model Performance
====================================================

This module provides advanced predictive analytics capabilities including:
- Time series forecasting with multiple algorithms
- Ensemble prediction models
- Automated feature engineering
- Performance optimization recommendations
- Cost-benefit analysis
- Risk assessment and mitigation
- Automated model retraining
- A/B testing framework
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
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, 
    VotingRegressor, StackingRegressor
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import joblib
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Import our AI history analyzer components
from .ai_history_analyzer import (
    AIHistoryAnalyzer, ModelType, PerformanceMetric,
    get_ai_history_analyzer
)
from .config import get_ai_history_config
from .ml_predictor import get_ml_predictor, MLPredictor

logger = logging.getLogger(__name__)


@dataclass
class TimeSeriesForecast:
    """Time series forecast result"""
    model_name: str
    metric: str
    forecast_values: List[float]
    forecast_dates: List[datetime]
    confidence_intervals: List[Tuple[float, float]]
    model_accuracy: float
    trend_direction: str
    seasonality_detected: bool
    anomaly_probability: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class EnsemblePrediction:
    """Ensemble prediction result"""
    model_name: str
    metric: str
    individual_predictions: Dict[str, float]
    ensemble_prediction: float
    confidence_score: float
    model_weights: Dict[str, float]
    prediction_variance: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class OptimizationRecommendation:
    """Advanced optimization recommendation"""
    recommendation_type: str
    model_name: str
    current_performance: float
    predicted_improvement: float
    confidence: float
    implementation_cost: float
    expected_roi: float
    risk_level: str
    time_to_implement: str
    prerequisites: List[str]
    alternative_options: List[str]
    success_probability: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ABTestResult:
    """A/B test result"""
    test_id: str
    model_a: str
    model_b: str
    metric: str
    sample_size_a: int
    sample_size_b: int
    mean_a: float
    mean_b: float
    statistical_significance: bool
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    recommendation: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class AdvancedPredictiveAnalytics:
    """Advanced predictive analytics for AI model performance"""
    
    def __init__(self, model_storage_path: str = "advanced_ml_models"):
        self.model_storage_path = model_storage_path
        self.analyzer = get_ai_history_analyzer()
        self.config = get_ai_history_config()
        self.ml_predictor = get_ml_predictor()
        
        # Advanced models
        self.ensemble_models: Dict[str, Any] = {}
        self.time_series_models: Dict[str, Any] = {}
        self.feature_engineering_models: Dict[str, Any] = {}
        self.optimization_models: Dict[str, Any] = {}
        
        # Preprocessing
        self.scalers: Dict[str, Any] = {}
        self.feature_selectors: Dict[str, Any] = {}
        
        # Model metadata
        self.model_metadata: Dict[str, Dict[str, Any]] = {}
        
        # A/B testing
        self.ab_tests: Dict[str, ABTestResult] = {}
        
        # Ensure model storage directory exists
        os.makedirs(model_storage_path, exist_ok=True)
    
    async def create_ensemble_model(self, 
                                  model_name: str,
                                  metric: PerformanceMetric,
                                  algorithms: List[str] = None) -> Dict[str, Any]:
        """Create an ensemble model for better predictions"""
        try:
            if algorithms is None:
                algorithms = ["random_forest", "gradient_boosting", "linear_regression", "svr"]
            
            # Get historical data
            performance_data = self.analyzer.get_model_performance(model_name, metric, days=365)
            
            if len(performance_data) < 100:
                raise ValueError(f"Insufficient data for ensemble: {len(performance_data)} samples")
            
            # Prepare features and target
            features, target = self._prepare_advanced_features(performance_data, model_name, metric)
            
            # Split data with time series split
            tscv = TimeSeriesSplit(n_splits=5)
            X, y = np.array(features), np.array(target)
            
            # Create individual models
            individual_models = {}
            for algorithm in algorithms:
                model = self._create_advanced_model(algorithm)
                individual_models[algorithm] = model
            
            # Create ensemble model
            ensemble_model = VotingRegressor(
                [(name, model) for name, model in individual_models.items()]
            )
            
            # Train ensemble model
            ensemble_model.fit(X, y)
            
            # Evaluate ensemble model
            scores = []
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                ensemble_model.fit(X_train, y_train)
                y_pred = ensemble_model.predict(X_test)
                score = r2_score(y_test, y_pred)
                scores.append(score)
            
            ensemble_accuracy = np.mean(scores)
            
            # Store ensemble model
            model_key = f"{model_name}_{metric.value}_ensemble"
            self.ensemble_models[model_key] = ensemble_model
            
            # Store metadata
            self.model_metadata[model_key] = {
                "model_name": model_name,
                "metric": metric.value,
                "type": "ensemble",
                "algorithms": algorithms,
                "accuracy": ensemble_accuracy,
                "individual_models": list(individual_models.keys()),
                "training_samples": len(features),
                "trained_at": datetime.now().isoformat()
            }
            
            # Save model
            await self._save_advanced_model(model_key, ensemble_model)
            
            logger.info(f"Created ensemble model for {model_name} - {metric.value}")
            logger.info(f"Ensemble accuracy: {ensemble_accuracy:.3f}")
            
            return {
                "success": True,
                "model_key": model_key,
                "type": "ensemble",
                "algorithms": algorithms,
                "accuracy": ensemble_accuracy,
                "individual_models": list(individual_models.keys())
            }
            
        except Exception as e:
            logger.error(f"Error creating ensemble model: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _create_advanced_model(self, algorithm: str):
        """Create advanced ML model"""
        if algorithm == "random_forest":
            return RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
        elif algorithm == "gradient_boosting":
            return GradientBoostingRegressor(n_estimators=200, max_depth=6, random_state=42)
        elif algorithm == "linear_regression":
            return LinearRegression()
        elif algorithm == "ridge":
            return Ridge(alpha=1.0)
        elif algorithm == "lasso":
            return Lasso(alpha=0.1)
        elif algorithm == "elastic_net":
            return ElasticNet(alpha=0.1, l1_ratio=0.5)
        elif algorithm == "svr":
            return SVR(kernel='rbf', C=1.0, gamma='scale')
        elif algorithm == "mlp":
            return MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        else:
            return RandomForestRegressor(n_estimators=100, random_state=42)
    
    def _prepare_advanced_features(self, 
                                 performance_data: List,
                                 model_name: str,
                                 metric: PerformanceMetric) -> Tuple[List[List[float]], List[float]]:
        """Prepare advanced features for ML models"""
        try:
            features = []
            target = []
            
            for i, perf in enumerate(performance_data):
                # Target: current performance value
                target.append(perf.value)
                
                # Advanced feature vector
                feature_vector = []
                
                # Time-based features
                feature_vector.extend([
                    perf.timestamp.hour,
                    perf.timestamp.weekday(),
                    perf.timestamp.day,
                    perf.timestamp.month,
                    perf.timestamp.isocalendar()[1],  # Week of year
                    perf.timestamp.timetuple().tm_yday  # Day of year
                ])
                
                # Historical performance features
                if i > 0:
                    feature_vector.append(performance_data[i-1].value)
                else:
                    feature_vector.append(perf.value)
                
                if i > 1:
                    feature_vector.append(performance_data[i-2].value)
                else:
                    feature_vector.append(perf.value)
                
                # Rolling statistics (multiple windows)
                for window in [3, 7, 14, 30]:
                    if i >= window:
                        recent_values = [p.value for p in performance_data[i-window:i]]
                        feature_vector.extend([
                            np.mean(recent_values),
                            np.std(recent_values),
                            np.min(recent_values),
                            np.max(recent_values),
                            np.median(recent_values)
                        ])
                    else:
                        feature_vector.extend([perf.value, 0.0, perf.value, perf.value, perf.value])
                
                # Trend features
                if i >= 7:
                    recent_values = [p.value for p in performance_data[i-7:i]]
                    # Linear trend
                    x = np.arange(len(recent_values))
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, recent_values)
                    feature_vector.extend([slope, r_value, p_value])
                else:
                    feature_vector.extend([0.0, 0.0, 1.0])
                
                # Volatility features
                if i >= 14:
                    recent_values = [p.value for p in performance_data[i-14:i]]
                    # Calculate volatility
                    returns = np.diff(recent_values) / np.array(recent_values[:-1])
                    volatility = np.std(returns) if len(returns) > 0 else 0.0
                    feature_vector.append(volatility)
                else:
                    feature_vector.append(0.0)
                
                # Context features
                if perf.context:
                    feature_vector.extend([
                        len(perf.context),
                        len(perf.context.split()) if isinstance(perf.context, str) else 0,
                        perf.context.count('.') if isinstance(perf.context, str) else 0
                    ])
                else:
                    feature_vector.extend([0, 0, 0])
                
                # Model-specific features
                model_def = self.config.get_model(model_name)
                if model_def:
                    feature_vector.extend([
                        model_def.context_length / 1000,
                        model_def.cost_per_1k_tokens * 1000,
                        model_def.max_tokens_per_minute / 1000
                    ])
                else:
                    feature_vector.extend([0.0, 0.0, 0.0])
                
                # Interaction features
                if len(feature_vector) >= 2:
                    feature_vector.append(feature_vector[0] * feature_vector[1])  # Hour * Weekday
                
                features.append(feature_vector)
            
            return features, target
            
        except Exception as e:
            logger.error(f"Error preparing advanced features: {str(e)}")
            raise
    
    async def generate_time_series_forecast(self, 
                                          model_name: str,
                                          metric: PerformanceMetric,
                                          forecast_days: int = 30) -> Optional[TimeSeriesForecast]:
        """Generate time series forecast for model performance"""
        try:
            # Get historical data
            performance_data = self.analyzer.get_model_performance(model_name, metric, days=365)
            
            if len(performance_data) < 30:
                logger.warning(f"Insufficient data for time series forecast: {len(performance_data)} samples")
                return None
            
            # Prepare time series data
            dates = [p.timestamp for p in performance_data]
            values = [p.value for p in performance_data]
            
            # Create time series DataFrame
            df = pd.DataFrame({
                'date': dates,
                'value': values
            })
            df.set_index('date', inplace=True)
            df = df.resample('D').mean().fillna(method='ffill')
            
            # Detect seasonality
            seasonality_detected = self._detect_seasonality(df['value'])
            
            # Generate forecast using multiple methods
            forecast_values = []
            confidence_intervals = []
            
            # Simple moving average forecast
            ma_forecast = self._moving_average_forecast(df['value'], forecast_days)
            forecast_values.extend(ma_forecast)
            
            # Linear trend forecast
            trend_forecast = self._linear_trend_forecast(df['value'], forecast_days)
            
            # Exponential smoothing forecast
            exp_forecast = self._exponential_smoothing_forecast(df['value'], forecast_days)
            
            # Combine forecasts (ensemble)
            combined_forecast = []
            for i in range(forecast_days):
                combined_value = (ma_forecast[i] + trend_forecast[i] + exp_forecast[i]) / 3
                combined_forecast.append(combined_value)
            
            # Calculate confidence intervals
            historical_std = df['value'].std()
            for value in combined_forecast:
                ci_lower = value - 1.96 * historical_std
                ci_upper = value + 1.96 * historical_std
                confidence_intervals.append((ci_lower, ci_upper))
            
            # Generate forecast dates
            last_date = df.index[-1]
            forecast_dates = [last_date + timedelta(days=i+1) for i in range(forecast_days)]
            
            # Calculate trend direction
            recent_trend = self._calculate_trend(df['value'].tail(30))
            trend_direction = "improving" if recent_trend > 0.01 else "declining" if recent_trend < -0.01 else "stable"
            
            # Calculate anomaly probability
            anomaly_probability = self._calculate_anomaly_probability(df['value'], combined_forecast)
            
            # Calculate model accuracy (using historical data)
            model_accuracy = self._calculate_forecast_accuracy(df['value'])
            
            return TimeSeriesForecast(
                model_name=model_name,
                metric=metric.value,
                forecast_values=combined_forecast,
                forecast_dates=forecast_dates,
                confidence_intervals=confidence_intervals,
                model_accuracy=model_accuracy,
                trend_direction=trend_direction,
                seasonality_detected=seasonality_detected,
                anomaly_probability=anomaly_probability,
                metadata={
                    "forecast_methods": ["moving_average", "linear_trend", "exponential_smoothing"],
                    "historical_samples": len(performance_data),
                    "forecast_days": forecast_days
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating time series forecast: {str(e)}")
            return None
    
    def _detect_seasonality(self, series: pd.Series) -> bool:
        """Detect seasonality in time series"""
        try:
            # Simple seasonality detection using autocorrelation
            if len(series) < 14:
                return False
            
            # Calculate autocorrelation for different lags
            autocorr_7 = series.autocorr(lag=7)
            autocorr_30 = series.autocorr(lag=30)
            
            # Consider seasonal if autocorrelation is significant
            return abs(autocorr_7) > 0.3 or abs(autocorr_30) > 0.3
            
        except Exception:
            return False
    
    def _moving_average_forecast(self, series: pd.Series, periods: int) -> List[float]:
        """Generate moving average forecast"""
        try:
            window = min(14, len(series) // 4)
            ma = series.rolling(window=window).mean()
            last_ma = ma.iloc[-1]
            
            # Simple forecast: use last moving average value
            return [last_ma] * periods
            
        except Exception:
            return [series.mean()] * periods
    
    def _linear_trend_forecast(self, series: pd.Series, periods: int) -> List[float]:
        """Generate linear trend forecast"""
        try:
            if len(series) < 2:
                return [series.mean()] * periods
            
            # Fit linear trend
            x = np.arange(len(series))
            slope, intercept, _, _, _ = stats.linregress(x, series.values)
            
            # Generate forecast
            forecast = []
            for i in range(1, periods + 1):
                forecast_value = intercept + slope * (len(series) + i)
                forecast.append(forecast_value)
            
            return forecast
            
        except Exception:
            return [series.mean()] * periods
    
    def _exponential_smoothing_forecast(self, series: pd.Series, periods: int) -> List[float]:
        """Generate exponential smoothing forecast"""
        try:
            if len(series) < 2:
                return [series.mean()] * periods
            
            # Simple exponential smoothing
            alpha = 0.3
            forecast = []
            last_value = series.iloc[-1]
            
            for i in range(periods):
                forecast_value = alpha * last_value + (1 - alpha) * last_value
                forecast.append(forecast_value)
                last_value = forecast_value
            
            return forecast
            
        except Exception:
            return [series.mean()] * periods
    
    def _calculate_trend(self, series: pd.Series) -> float:
        """Calculate trend direction and strength"""
        try:
            if len(series) < 2:
                return 0.0
            
            x = np.arange(len(series))
            slope, _, _, _, _ = stats.linregress(x, series.values)
            return slope
            
        except Exception:
            return 0.0
    
    def _calculate_anomaly_probability(self, historical: pd.Series, forecast: List[float]) -> float:
        """Calculate probability of anomalies in forecast"""
        try:
            historical_mean = historical.mean()
            historical_std = historical.std()
            
            if historical_std == 0:
                return 0.0
            
            # Calculate how many forecast values are outside 2 standard deviations
            anomalies = 0
            for value in forecast:
                if abs(value - historical_mean) > 2 * historical_std:
                    anomalies += 1
            
            return anomalies / len(forecast)
            
        except Exception:
            return 0.0
    
    def _calculate_forecast_accuracy(self, series: pd.Series) -> float:
        """Calculate forecast accuracy using historical data"""
        try:
            if len(series) < 20:
                return 0.5
            
            # Use last 20% of data for validation
            split_point = int(len(series) * 0.8)
            train_data = series[:split_point]
            test_data = series[split_point:]
            
            # Generate forecast for test period
            forecast = self._moving_average_forecast(train_data, len(test_data))
            
            # Calculate accuracy
            mae = mean_absolute_error(test_data, forecast)
            mean_value = test_data.mean()
            accuracy = max(0, 1 - (mae / mean_value)) if mean_value != 0 else 0.5
            
            return accuracy
            
        except Exception:
            return 0.5
    
    async def generate_ensemble_prediction(self, 
                                         model_name: str,
                                         metric: PerformanceMetric) -> Optional[EnsemblePrediction]:
        """Generate ensemble prediction using multiple models"""
        try:
            model_key = f"{model_name}_{metric.value}_ensemble"
            
            # Check if ensemble model exists
            if model_key not in self.ensemble_models:
                # Try to load model
                await self._load_advanced_model(model_key)
                
                if model_key not in self.ensemble_models:
                    logger.warning(f"No ensemble model found for {model_key}")
                    return None
            
            # Get recent performance data for features
            recent_data = self.analyzer.get_model_performance(model_name, metric, days=30)
            
            if not recent_data:
                logger.warning(f"No recent data for {model_name} - {metric.value}")
                return None
            
            # Prepare features for prediction
            features, _ = self._prepare_advanced_features(recent_data, model_name, metric)
            
            if not features:
                return None
            
            # Get latest features
            latest_features = features[-1]
            
            # Get ensemble model
            ensemble_model = self.ensemble_models[model_key]
            
            # Get individual model predictions
            individual_predictions = {}
            model_weights = {}
            
            # Get individual models from ensemble
            for name, model in ensemble_model.named_estimators_.items():
                try:
                    pred = model.predict([latest_features])[0]
                    individual_predictions[name] = pred
                    model_weights[name] = 1.0 / len(ensemble_model.named_estimators_)
                except Exception as e:
                    logger.warning(f"Error getting prediction from {name}: {str(e)}")
                    individual_predictions[name] = 0.0
                    model_weights[name] = 0.0
            
            # Get ensemble prediction
            ensemble_prediction = ensemble_model.predict([latest_features])[0]
            
            # Calculate confidence score
            predictions = list(individual_predictions.values())
            prediction_variance = np.var(predictions) if len(predictions) > 1 else 0.0
            confidence_score = max(0, 1 - prediction_variance)
            
            return EnsemblePrediction(
                model_name=model_name,
                metric=metric.value,
                individual_predictions=individual_predictions,
                ensemble_prediction=ensemble_prediction,
                confidence_score=confidence_score,
                model_weights=model_weights,
                prediction_variance=prediction_variance,
                metadata=self.model_metadata.get(model_key, {})
            )
            
        except Exception as e:
            logger.error(f"Error generating ensemble prediction: {str(e)}")
            return None
    
    async def run_ab_test(self, 
                         model_a: str,
                         model_b: str,
                         metric: PerformanceMetric,
                         test_duration_days: int = 30) -> ABTestResult:
        """Run A/B test between two models"""
        try:
            # Get performance data for both models
            data_a = self.analyzer.get_model_performance(model_a, metric, days=test_duration_days)
            data_b = self.analyzer.get_model_performance(model_b, metric, days=test_duration_days)
            
            if len(data_a) < 10 or len(data_b) < 10:
                raise ValueError("Insufficient data for A/B test")
            
            # Extract values
            values_a = [p.value for p in data_a]
            values_b = [p.value for p in data_b]
            
            # Calculate statistics
            mean_a = np.mean(values_a)
            mean_b = np.mean(values_b)
            std_a = np.std(values_a)
            std_b = np.std(values_b)
            
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(values_a, values_b)
            
            # Calculate effect size (Cohen's d)
            pooled_std = np.sqrt(((len(values_a) - 1) * std_a**2 + (len(values_b) - 1) * std_b**2) / 
                                (len(values_a) + len(values_b) - 2))
            effect_size = (mean_a - mean_b) / pooled_std if pooled_std != 0 else 0
            
            # Calculate confidence interval for difference
            se_diff = pooled_std * np.sqrt(1/len(values_a) + 1/len(values_b))
            margin_error = 1.96 * se_diff
            diff = mean_a - mean_b
            ci_lower = diff - margin_error
            ci_upper = diff + margin_error
            
            # Determine recommendation
            statistical_significance = p_value < 0.05
            if statistical_significance:
                if mean_a > mean_b:
                    recommendation = f"Model A ({model_a}) performs significantly better"
                else:
                    recommendation = f"Model B ({model_b}) performs significantly better"
            else:
                recommendation = "No significant difference between models"
            
            # Create test result
            test_id = f"ab_test_{model_a}_{model_b}_{metric.value}_{int(datetime.now().timestamp())}"
            result = ABTestResult(
                test_id=test_id,
                model_a=model_a,
                model_b=model_b,
                metric=metric.value,
                sample_size_a=len(values_a),
                sample_size_b=len(values_b),
                mean_a=mean_a,
                mean_b=mean_b,
                statistical_significance=statistical_significance,
                p_value=p_value,
                effect_size=effect_size,
                confidence_interval=(ci_lower, ci_upper),
                recommendation=recommendation,
                metadata={
                    "test_duration_days": test_duration_days,
                    "t_statistic": t_stat,
                    "pooled_std": pooled_std
                }
            )
            
            # Store result
            self.ab_tests[test_id] = result
            
            logger.info(f"A/B test completed: {recommendation}")
            logger.info(f"P-value: {p_value:.4f}, Effect size: {effect_size:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error running A/B test: {str(e)}")
            raise
    
    async def generate_advanced_optimization_recommendations(self, 
                                                           model_name: str,
                                                           target_metric: PerformanceMetric) -> List[OptimizationRecommendation]:
        """Generate advanced optimization recommendations"""
        try:
            recommendations = []
            
            # Get current performance
            current_summary = self.analyzer.get_performance_summary(model_name, days=30)
            if not current_summary or "metrics" not in current_summary:
                return recommendations
            
            current_value = current_summary["metrics"].get(target_metric.value, {}).get("mean", 0.0)
            
            # Generate time series forecast
            forecast = await self.generate_time_series_forecast(model_name, target_metric, 30)
            
            if forecast:
                # Trend-based optimization
                if forecast.trend_direction == "declining":
                    trend_rec = OptimizationRecommendation(
                        recommendation_type="trend_optimization",
                        model_name=model_name,
                        current_performance=current_value,
                        predicted_improvement=0.15,
                        confidence=0.8,
                        implementation_cost=0.4,
                        expected_roi=0.375,
                        risk_level="medium",
                        time_to_implement="2-4 weeks",
                        prerequisites=["Performance analysis", "Model retraining"],
                        alternative_options=["Switch to alternative model", "Parameter tuning"],
                        success_probability=0.75,
                        metadata={"forecast_accuracy": forecast.model_accuracy}
                    )
                    recommendations.append(trend_rec)
                
                # Anomaly-based optimization
                if forecast.anomaly_probability > 0.3:
                    anomaly_rec = OptimizationRecommendation(
                        recommendation_type="anomaly_mitigation",
                        model_name=model_name,
                        current_performance=current_value,
                        predicted_improvement=0.1,
                        confidence=0.7,
                        implementation_cost=0.2,
                        expected_roi=0.5,
                        risk_level="low",
                        time_to_implement="1-2 weeks",
                        prerequisites=["Anomaly detection setup", "Monitoring enhancement"],
                        alternative_options=["Model validation", "Data quality check"],
                        success_probability=0.85,
                        metadata={"anomaly_probability": forecast.anomaly_probability}
                    )
                    recommendations.append(anomaly_rec)
            
            # Ensemble-based optimization
            ensemble_pred = await self.generate_ensemble_prediction(model_name, target_metric)
            
            if ensemble_pred and ensemble_pred.confidence_score > 0.8:
                # High confidence prediction optimization
                if ensemble_pred.ensemble_prediction < current_value * 0.9:
                    ensemble_rec = OptimizationRecommendation(
                        recommendation_type="ensemble_optimization",
                        model_name=model_name,
                        current_performance=current_value,
                        predicted_improvement=0.2,
                        confidence=ensemble_pred.confidence_score,
                        implementation_cost=0.6,
                        expected_roi=0.33,
                        risk_level="high",
                        time_to_implement="4-6 weeks",
                        prerequisites=["Model ensemble training", "Feature engineering"],
                        alternative_options=["Model replacement", "Architecture change"],
                        success_probability=0.9,
                        metadata={"ensemble_confidence": ensemble_pred.confidence_score}
                    )
                    recommendations.append(ensemble_rec)
            
            # Cost-benefit optimization
            cost_rec = await self._generate_cost_benefit_recommendation(model_name, target_metric, current_value)
            if cost_rec:
                recommendations.append(cost_rec)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating advanced optimization recommendations: {str(e)}")
            return []
    
    async def _generate_cost_benefit_recommendation(self, 
                                                  model_name: str,
                                                  metric: PerformanceMetric,
                                                  current_value: float) -> Optional[OptimizationRecommendation]:
        """Generate cost-benefit optimization recommendation"""
        try:
            # Get model definition
            model_def = self.config.get_model(model_name)
            if not model_def:
                return None
            
            # Calculate cost efficiency
            cost_per_1k = model_def.cost_per_1k_tokens
            current_efficiency = current_value / cost_per_1k if cost_per_1k > 0 else 0
            
            # Find more cost-effective alternatives
            all_models = self.config.get_all_models()
            better_alternatives = []
            
            for alt_model in all_models:
                if alt_model.name != model_name:
                    alt_cost = alt_model.cost_per_1k_tokens
                    if alt_cost < cost_per_1k:
                        # Estimate performance for alternative model
                        alt_performance = current_value * 0.95  # Assume 5% performance drop
                        alt_efficiency = alt_performance / alt_cost
                        
                        if alt_efficiency > current_efficiency:
                            better_alternatives.append({
                                "model": alt_model.name,
                                "cost_savings": (cost_per_1k - alt_cost) / cost_per_1k,
                                "efficiency_gain": (alt_efficiency - current_efficiency) / current_efficiency
                            })
            
            if better_alternatives:
                best_alternative = max(better_alternatives, key=lambda x: x["efficiency_gain"])
                
                return OptimizationRecommendation(
                    recommendation_type="cost_optimization",
                    model_name=model_name,
                    current_performance=current_value,
                    predicted_improvement=best_alternative["efficiency_gain"],
                    confidence=0.6,
                    implementation_cost=0.1,
                    expected_roi=best_alternative["cost_savings"] / 0.1,
                    risk_level="low",
                    time_to_implement="1 week",
                    prerequisites=["Model comparison", "Performance validation"],
                    alternative_options=[alt["model"] for alt in better_alternatives[:3]],
                    success_probability=0.8,
                    metadata={
                        "cost_savings": best_alternative["cost_savings"],
                        "alternative_model": best_alternative["model"]
                    }
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating cost-benefit recommendation: {str(e)}")
            return None
    
    async def _save_advanced_model(self, model_key: str, model: Any):
        """Save advanced ML model"""
        try:
            model_path = os.path.join(self.model_storage_path, f"{model_key}_model.pkl")
            metadata_path = os.path.join(self.model_storage_path, f"{model_key}_metadata.json")
            
            # Save model
            joblib.dump(model, model_path)
            
            # Save metadata
            with open(metadata_path, 'w') as f:
                json.dump(self.model_metadata[model_key], f, indent=2)
            
            logger.info(f"Saved advanced model {model_key}")
            
        except Exception as e:
            logger.error(f"Error saving advanced model {model_key}: {str(e)}")
    
    async def _load_advanced_model(self, model_key: str):
        """Load advanced ML model"""
        try:
            model_path = os.path.join(self.model_storage_path, f"{model_key}_model.pkl")
            metadata_path = os.path.join(self.model_storage_path, f"{model_key}_metadata.json")
            
            if not all(os.path.exists(p) for p in [model_path, metadata_path]):
                return False
            
            # Load model
            model = joblib.load(model_path)
            self.ensemble_models[model_key] = model
            
            # Load metadata
            with open(metadata_path, 'r') as f:
                self.model_metadata[model_key] = json.load(f)
            
            logger.info(f"Loaded advanced model {model_key}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading advanced model {model_key}: {str(e)}")
            return False
    
    def get_ab_test_results(self) -> List[ABTestResult]:
        """Get all A/B test results"""
        return list(self.ab_tests.values())
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about advanced models"""
        return {
            "ensemble_models": list(self.ensemble_models.keys()),
            "time_series_models": list(self.time_series_models.keys()),
            "ab_tests": len(self.ab_tests),
            "model_metadata": self.model_metadata,
            "storage_path": self.model_storage_path
        }


# Global advanced analytics instance
_advanced_analytics: Optional[AdvancedPredictiveAnalytics] = None


def get_advanced_predictive_analytics(model_storage_path: str = "advanced_ml_models") -> AdvancedPredictiveAnalytics:
    """Get or create global advanced predictive analytics"""
    global _advanced_analytics
    if _advanced_analytics is None:
        _advanced_analytics = AdvancedPredictiveAnalytics(model_storage_path)
    return _advanced_analytics


# Example usage
async def main():
    """Example usage of advanced predictive analytics"""
    analytics = get_advanced_predictive_analytics()
    
    # Create ensemble model
    ensemble_result = await analytics.create_ensemble_model(
        model_name="gpt-4",
        metric=PerformanceMetric.QUALITY_SCORE,
        algorithms=["random_forest", "gradient_boosting", "linear_regression", "svr"]
    )
    
    print(f"Ensemble model result: {ensemble_result}")
    
    # Generate time series forecast
    forecast = await analytics.generate_time_series_forecast(
        model_name="gpt-4",
        metric=PerformanceMetric.QUALITY_SCORE,
        forecast_days=30
    )
    
    if forecast:
        print(f"Forecast trend: {forecast.trend_direction}")
        print(f"Anomaly probability: {forecast.anomaly_probability:.3f}")
    
    # Generate ensemble prediction
    ensemble_pred = await analytics.generate_ensemble_prediction(
        model_name="gpt-4",
        metric=PerformanceMetric.QUALITY_SCORE
    )
    
    if ensemble_pred:
        print(f"Ensemble prediction: {ensemble_pred.ensemble_prediction:.3f}")
        print(f"Confidence: {ensemble_pred.confidence_score:.3f}")
    
    # Run A/B test
    ab_result = await analytics.run_ab_test(
        model_a="gpt-4",
        model_b="claude-3-sonnet",
        metric=PerformanceMetric.QUALITY_SCORE,
        test_duration_days=30
    )
    
    print(f"A/B test recommendation: {ab_result.recommendation}")
    
    # Generate optimization recommendations
    recommendations = await analytics.generate_advanced_optimization_recommendations(
        model_name="gpt-4",
        target_metric=PerformanceMetric.QUALITY_SCORE
    )
    
    print(f"Generated {len(recommendations)} optimization recommendations")


if __name__ == "__main__":
    asyncio.run(main())

























