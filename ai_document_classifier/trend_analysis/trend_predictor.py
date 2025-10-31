"""
Advanced Trend Analysis and Prediction System
============================================

Comprehensive trend analysis system with time series forecasting,
pattern recognition, and predictive analytics for document classification.
"""

import logging
import asyncio
import json
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import statistics
import math

logger = logging.getLogger(__name__)

class TrendType(Enum):
    """Trend types"""
    RISING = "rising"
    FALLING = "falling"
    STABLE = "stable"
    VOLATILE = "volatile"
    SEASONAL = "seasonal"
    CYCLICAL = "cyclical"
    EXPONENTIAL = "exponential"
    LOGARITHMIC = "logarithmic"

class PredictionConfidence(Enum):
    """Prediction confidence levels"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

class TimeGranularity(Enum):
    """Time granularity for analysis"""
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"

@dataclass
class TrendDataPoint:
    """Single trend data point"""
    timestamp: datetime
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TrendAnalysis:
    """Trend analysis result"""
    id: str
    metric_name: str
    time_series: List[TrendDataPoint]
    trend_type: TrendType
    trend_strength: float  # 0-1
    trend_direction: float  # -1 to 1
    volatility: float
    seasonality: float
    periodicity: Optional[int]
    confidence: PredictionConfidence
    analysis_period: Tuple[datetime, datetime]
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Prediction:
    """Prediction result"""
    id: str
    trend_analysis_id: str
    predicted_value: float
    confidence_interval: Tuple[float, float]
    prediction_date: datetime
    confidence: PredictionConfidence
    model_used: str
    accuracy_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TrendInsight:
    """Trend insight"""
    id: str
    insight_type: str
    description: str
    impact_score: float  # 0-1
    confidence: PredictionConfidence
    recommendations: List[str]
    related_trends: List[str]
    created_at: datetime

class AdvancedTrendPredictor:
    """
    Advanced trend analysis and prediction system
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize trend predictor
        
        Args:
            data_dir: Directory for trend data storage
        """
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).parent / "data"
        self.data_dir.mkdir(exist_ok=True)
        
        # Trend data storage
        self.trend_analyses: Dict[str, TrendAnalysis] = {}
        self.predictions: Dict[str, Prediction] = {}
        self.insights: Dict[str, TrendInsight] = {}
        
        # Time series data
        self.time_series_data: Dict[str, List[TrendDataPoint]] = defaultdict(list)
        
        # Analysis parameters
        self.min_data_points = 10
        self.max_prediction_horizon = 365  # days
        
        # Pattern recognition
        self.pattern_templates = self._initialize_pattern_templates()
        
        # Statistical models
        self.statistical_models = self._initialize_statistical_models()
    
    def _initialize_pattern_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize pattern templates for trend recognition"""
        return {
            "linear_rising": {
                "slope_range": (0.01, float('inf')),
                "r_squared_min": 0.7,
                "volatility_max": 0.3
            },
            "linear_falling": {
                "slope_range": (-float('inf'), -0.01),
                "r_squared_min": 0.7,
                "volatility_max": 0.3
            },
            "exponential_growth": {
                "growth_rate_min": 0.05,
                "r_squared_min": 0.8,
                "acceleration_positive": True
            },
            "exponential_decay": {
                "decay_rate_max": -0.05,
                "r_squared_min": 0.8,
                "acceleration_negative": True
            },
            "seasonal": {
                "periodicity_detected": True,
                "seasonal_strength_min": 0.3,
                "multiple_cycles": True
            },
            "cyclical": {
                "cycle_length_min": 7,
                "cycle_strength_min": 0.2,
                "irregular_cycles": True
            },
            "volatile": {
                "volatility_min": 0.5,
                "trend_strength_max": 0.3,
                "high_frequency_changes": True
            },
            "stable": {
                "volatility_max": 0.1,
                "trend_strength_max": 0.2,
                "low_variance": True
            }
        }
    
    def _initialize_statistical_models(self) -> Dict[str, Dict[str, Any]]:
        """Initialize statistical models for prediction"""
        return {
            "linear_regression": {
                "type": "linear",
                "complexity": "low",
                "best_for": ["linear_trends", "short_term"]
            },
            "polynomial_regression": {
                "type": "polynomial",
                "complexity": "medium",
                "best_for": ["curved_trends", "medium_term"]
            },
            "exponential_smoothing": {
                "type": "smoothing",
                "complexity": "medium",
                "best_for": ["seasonal", "trending"]
            },
            "arima": {
                "type": "time_series",
                "complexity": "high",
                "best_for": ["complex_patterns", "long_term"]
            },
            "moving_average": {
                "type": "smoothing",
                "complexity": "low",
                "best_for": ["noisy_data", "short_term"]
            },
            "seasonal_decomposition": {
                "type": "decomposition",
                "complexity": "high",
                "best_for": ["seasonal", "cyclical"]
            }
        }
    
    async def add_data_point(self, metric_name: str, value: float, timestamp: Optional[datetime] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a data point to time series
        
        Args:
            metric_name: Name of the metric
            value: Value of the data point
            timestamp: Timestamp (defaults to now)
            metadata: Additional metadata
            
        Returns:
            Data point ID
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        if metadata is None:
            metadata = {}
        
        data_point = TrendDataPoint(
            timestamp=timestamp,
            value=value,
            metadata=metadata
        )
        
        self.time_series_data[metric_name].append(data_point)
        
        # Sort by timestamp
        self.time_series_data[metric_name].sort(key=lambda x: x.timestamp)
        
        logger.info(f"Added data point for {metric_name}: {value} at {timestamp}")
        
        return str(uuid.uuid4())
    
    async def analyze_trend(self, metric_name: str, analysis_period: Optional[Tuple[datetime, datetime]] = None) -> TrendAnalysis:
        """
        Analyze trend for a metric
        
        Args:
            metric_name: Name of the metric to analyze
            analysis_period: Optional time period for analysis
            
        Returns:
            Trend analysis result
        """
        if metric_name not in self.time_series_data:
            raise ValueError(f"No data found for metric: {metric_name}")
        
        data_points = self.time_series_data[metric_name]
        
        if len(data_points) < self.min_data_points:
            raise ValueError(f"Insufficient data points. Need at least {self.min_data_points}, got {len(data_points)}")
        
        # Filter by analysis period if provided
        if analysis_period:
            start_date, end_date = analysis_period
            data_points = [
                dp for dp in data_points
                if start_date <= dp.timestamp <= end_date
            ]
        
        if len(data_points) < self.min_data_points:
            raise ValueError(f"Insufficient data points in analysis period")
        
        # Perform trend analysis
        trend_type = self._identify_trend_type(data_points)
        trend_strength = self._calculate_trend_strength(data_points)
        trend_direction = self._calculate_trend_direction(data_points)
        volatility = self._calculate_volatility(data_points)
        seasonality = self._calculate_seasonality(data_points)
        periodicity = self._detect_periodicity(data_points)
        confidence = self._calculate_analysis_confidence(data_points, trend_strength)
        
        # Create trend analysis
        analysis = TrendAnalysis(
            id=str(uuid.uuid4()),
            metric_name=metric_name,
            time_series=data_points,
            trend_type=trend_type,
            trend_strength=trend_strength,
            trend_direction=trend_direction,
            volatility=volatility,
            seasonality=seasonality,
            periodicity=periodicity,
            confidence=confidence,
            analysis_period=(data_points[0].timestamp, data_points[-1].timestamp),
            created_at=datetime.now(),
            metadata={
                "data_points_count": len(data_points),
                "analysis_method": "advanced_statistical"
            }
        )
        
        self.trend_analyses[analysis.id] = analysis
        
        logger.info(f"Trend analysis completed for {metric_name}: {trend_type.value}")
        
        return analysis
    
    def _identify_trend_type(self, data_points: List[TrendDataPoint]) -> TrendType:
        """Identify the type of trend in data"""
        values = [dp.value for dp in data_points]
        timestamps = [dp.timestamp for dp in data_points]
        
        # Convert timestamps to numeric values
        time_numeric = [(ts - timestamps[0]).total_seconds() for ts in timestamps]
        
        # Calculate basic statistics
        slope = self._calculate_slope(time_numeric, values)
        r_squared = self._calculate_r_squared(time_numeric, values)
        volatility = self._calculate_volatility(data_points)
        
        # Check for exponential patterns
        log_values = [math.log(max(v, 0.001)) for v in values]
        log_slope = self._calculate_slope(time_numeric, log_values)
        log_r_squared = self._calculate_r_squared(time_numeric, log_values)
        
        # Check for seasonality
        seasonality = self._calculate_seasonality(data_points)
        periodicity = self._detect_periodicity(data_points)
        
        # Apply pattern templates
        if volatility > 0.5:
            return TrendType.VOLATILE
        elif volatility < 0.1 and abs(slope) < 0.01:
            return TrendType.STABLE
        elif seasonality > 0.3 and periodicity:
            return TrendType.SEASONAL
        elif periodicity and seasonality < 0.3:
            return TrendType.CYCLICAL
        elif log_r_squared > 0.8 and log_slope > 0.05:
            return TrendType.EXPONENTIAL
        elif log_r_squared > 0.8 and log_slope < -0.05:
            return TrendType.LOGARITHMIC
        elif slope > 0.01 and r_squared > 0.7:
            return TrendType.RISING
        elif slope < -0.01 and r_squared > 0.7:
            return TrendType.FALLING
        else:
            return TrendType.STABLE
    
    def _calculate_slope(self, x_values: List[float], y_values: List[float]) -> float:
        """Calculate slope of linear regression"""
        if len(x_values) != len(y_values) or len(x_values) < 2:
            return 0.0
        
        n = len(x_values)
        sum_x = sum(x_values)
        sum_y = sum(y_values)
        sum_xy = sum(x * y for x, y in zip(x_values, y_values))
        sum_x2 = sum(x * x for x in x_values)
        
        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return slope
    
    def _calculate_r_squared(self, x_values: List[float], y_values: List[float]) -> float:
        """Calculate R-squared value"""
        if len(x_values) != len(y_values) or len(x_values) < 2:
            return 0.0
        
        slope = self._calculate_slope(x_values, y_values)
        y_mean = statistics.mean(y_values)
        
        # Calculate predicted values
        predicted = [slope * x + (y_mean - slope * statistics.mean(x_values)) for x in x_values]
        
        # Calculate R-squared
        ss_res = sum((y - p) ** 2 for y, p in zip(y_values, predicted))
        ss_tot = sum((y - y_mean) ** 2 for y in y_values)
        
        if ss_tot == 0:
            return 1.0
        
        r_squared = 1 - (ss_res / ss_tot)
        return max(0, min(1, r_squared))
    
    def _calculate_trend_strength(self, data_points: List[TrendDataPoint]) -> float:
        """Calculate trend strength (0-1)"""
        values = [dp.value for dp in data_points]
        timestamps = [dp.timestamp for dp in data_points]
        
        if len(values) < 2:
            return 0.0
        
        # Convert timestamps to numeric
        time_numeric = [(ts - timestamps[0]).total_seconds() for ts in timestamps]
        
        # Calculate R-squared as trend strength
        r_squared = self._calculate_r_squared(time_numeric, values)
        
        return r_squared
    
    def _calculate_trend_direction(self, data_points: List[TrendDataPoint]) -> float:
        """Calculate trend direction (-1 to 1)"""
        values = [dp.value for dp in data_points]
        timestamps = [dp.timestamp for dp in data_points]
        
        if len(values) < 2:
            return 0.0
        
        # Convert timestamps to numeric
        time_numeric = [(ts - timestamps[0]).total_seconds() for ts in timestamps]
        
        # Calculate slope
        slope = self._calculate_slope(time_numeric, values)
        
        # Normalize slope to -1 to 1 range
        max_slope = max(abs(slope), 0.1)  # Avoid division by zero
        normalized_slope = slope / max_slope
        
        return max(-1, min(1, normalized_slope))
    
    def _calculate_volatility(self, data_points: List[TrendDataPoint]) -> float:
        """Calculate volatility (0-1)"""
        values = [dp.value for dp in data_points]
        
        if len(values) < 2:
            return 0.0
        
        # Calculate coefficient of variation
        mean_value = statistics.mean(values)
        if mean_value == 0:
            return 0.0
        
        std_dev = statistics.stdev(values)
        volatility = std_dev / abs(mean_value)
        
        # Normalize to 0-1 range
        return min(1.0, volatility)
    
    def _calculate_seasonality(self, data_points: List[TrendDataPoint]) -> float:
        """Calculate seasonality strength (0-1)"""
        if len(data_points) < 12:  # Need at least 12 points for seasonality
            return 0.0
        
        values = [dp.value for dp in data_points]
        
        # Simple seasonality detection using autocorrelation
        # This is a simplified version - in practice, you'd use more sophisticated methods
        
        # Calculate autocorrelation for different lags
        max_lag = min(len(values) // 4, 12)
        autocorrelations = []
        
        for lag in range(1, max_lag + 1):
            if len(values) > lag:
                correlation = self._calculate_autocorrelation(values, lag)
                autocorrelations.append(abs(correlation))
        
        if not autocorrelations:
            return 0.0
        
        # Seasonality strength is the maximum autocorrelation
        seasonality = max(autocorrelations)
        
        return min(1.0, seasonality)
    
    def _calculate_autocorrelation(self, values: List[float], lag: int) -> float:
        """Calculate autocorrelation for given lag"""
        if len(values) <= lag:
            return 0.0
        
        # Calculate mean
        mean_value = statistics.mean(values)
        
        # Calculate numerator and denominator
        numerator = sum((values[i] - mean_value) * (values[i + lag] - mean_value) 
                       for i in range(len(values) - lag))
        
        denominator = sum((values[i] - mean_value) ** 2 for i in range(len(values)))
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def _detect_periodicity(self, data_points: List[TrendDataPoint]) -> Optional[int]:
        """Detect periodicity in data"""
        if len(data_points) < 12:
            return None
        
        values = [dp.value for dp in data_points]
        
        # Find peaks in autocorrelation
        max_lag = min(len(values) // 4, 24)
        autocorrelations = []
        
        for lag in range(1, max_lag + 1):
            correlation = self._calculate_autocorrelation(values, lag)
            autocorrelations.append((lag, abs(correlation)))
        
        # Find significant peaks
        if not autocorrelations:
            return None
        
        # Sort by correlation strength
        autocorrelations.sort(key=lambda x: x[1], reverse=True)
        
        # Return the lag with highest correlation if it's significant
        best_lag, best_correlation = autocorrelations[0]
        
        if best_correlation > 0.3:  # Threshold for significance
            return best_lag
        
        return None
    
    def _calculate_analysis_confidence(self, data_points: List[TrendDataPoint], trend_strength: float) -> PredictionConfidence:
        """Calculate confidence in trend analysis"""
        data_count = len(data_points)
        
        # Base confidence on data points and trend strength
        if data_count >= 100 and trend_strength >= 0.8:
            return PredictionConfidence.VERY_HIGH
        elif data_count >= 50 and trend_strength >= 0.6:
            return PredictionConfidence.HIGH
        elif data_count >= 20 and trend_strength >= 0.4:
            return PredictionConfidence.MEDIUM
        elif data_count >= 10 and trend_strength >= 0.2:
            return PredictionConfidence.LOW
        else:
            return PredictionConfidence.VERY_LOW
    
    async def predict_future(self, trend_analysis_id: str, prediction_days: int, model_type: str = "auto") -> Prediction:
        """
        Predict future values based on trend analysis
        
        Args:
            trend_analysis_id: ID of trend analysis to use
            prediction_days: Number of days to predict ahead
            model_type: Type of prediction model to use
            
        Returns:
            Prediction result
        """
        if trend_analysis_id not in self.trend_analyses:
            raise ValueError(f"Trend analysis not found: {trend_analysis_id}")
        
        if prediction_days > self.max_prediction_horizon:
            raise ValueError(f"Prediction horizon too long. Max: {self.max_prediction_horizon} days")
        
        analysis = self.trend_analyses[trend_analysis_id]
        data_points = analysis.time_series
        
        if len(data_points) < self.min_data_points:
            raise ValueError("Insufficient data for prediction")
        
        # Select appropriate model
        if model_type == "auto":
            model_type = self._select_best_model(analysis)
        
        # Make prediction
        predicted_value, confidence_interval = self._make_prediction(
            data_points, prediction_days, model_type
        )
        
        # Calculate confidence
        confidence = self._calculate_prediction_confidence(analysis, prediction_days)
        
        # Create prediction
        prediction = Prediction(
            id=str(uuid.uuid4()),
            trend_analysis_id=trend_analysis_id,
            predicted_value=predicted_value,
            confidence_interval=confidence_interval,
            prediction_date=datetime.now() + timedelta(days=prediction_days),
            confidence=confidence,
            model_used=model_type,
            metadata={
                "prediction_horizon_days": prediction_days,
                "data_points_used": len(data_points),
                "trend_type": analysis.trend_type.value
            }
        )
        
        self.predictions[prediction.id] = prediction
        
        logger.info(f"Prediction made for {analysis.metric_name}: {predicted_value:.2f}")
        
        return prediction
    
    def _select_best_model(self, analysis: TrendAnalysis) -> str:
        """Select best prediction model based on trend characteristics"""
        trend_type = analysis.trend_type
        volatility = analysis.volatility
        seasonality = analysis.seasonality
        
        if trend_type == TrendType.SEASONAL or seasonality > 0.3:
            return "seasonal_decomposition"
        elif trend_type == TrendType.EXPONENTIAL:
            return "exponential_smoothing"
        elif volatility > 0.5:
            return "moving_average"
        elif trend_type in [TrendType.RISING, TrendType.FALLING]:
            return "linear_regression"
        else:
            return "arima"
    
    def _make_prediction(self, data_points: List[TrendDataPoint], prediction_days: int, model_type: str) -> Tuple[float, Tuple[float, float]]:
        """Make prediction using specified model"""
        values = [dp.value for dp in data_points]
        timestamps = [dp.timestamp for dp in data_points]
        
        # Convert timestamps to numeric
        time_numeric = [(ts - timestamps[0]).total_seconds() for ts in timestamps]
        
        if model_type == "linear_regression":
            return self._linear_regression_prediction(time_numeric, values, prediction_days)
        elif model_type == "moving_average":
            return self._moving_average_prediction(values, prediction_days)
        elif model_type == "exponential_smoothing":
            return self._exponential_smoothing_prediction(values, prediction_days)
        else:
            # Default to linear regression
            return self._linear_regression_prediction(time_numeric, values, prediction_days)
    
    def _linear_regression_prediction(self, x_values: List[float], y_values: List[float], prediction_days: int) -> Tuple[float, Tuple[float, float]]:
        """Make prediction using linear regression"""
        slope = self._calculate_slope(x_values, y_values)
        y_mean = statistics.mean(y_values)
        x_mean = statistics.mean(x_values)
        
        # Calculate intercept
        intercept = y_mean - slope * x_mean
        
        # Predict future value
        last_x = x_values[-1]
        future_x = last_x + (prediction_days * 24 * 3600)  # Convert days to seconds
        predicted_value = slope * future_x + intercept
        
        # Calculate confidence interval (simplified)
        std_error = statistics.stdev(y_values) if len(y_values) > 1 else 0
        margin_error = 1.96 * std_error  # 95% confidence interval
        
        confidence_interval = (
            predicted_value - margin_error,
            predicted_value + margin_error
        )
        
        return predicted_value, confidence_interval
    
    def _moving_average_prediction(self, values: List[float], prediction_days: int) -> Tuple[float, Tuple[float, float]]:
        """Make prediction using moving average"""
        # Use last 7 values for moving average
        window_size = min(7, len(values))
        recent_values = values[-window_size:]
        
        predicted_value = statistics.mean(recent_values)
        
        # Calculate confidence interval
        std_error = statistics.stdev(recent_values) if len(recent_values) > 1 else 0
        margin_error = 1.96 * std_error
        
        confidence_interval = (
            predicted_value - margin_error,
            predicted_value + margin_error
        )
        
        return predicted_value, confidence_interval
    
    def _exponential_smoothing_prediction(self, values: List[float], prediction_days: int) -> Tuple[float, Tuple[float, float]]:
        """Make prediction using exponential smoothing"""
        if len(values) < 2:
            return values[0] if values else 0, (0, 0)
        
        # Simple exponential smoothing with alpha = 0.3
        alpha = 0.3
        smoothed = [values[0]]
        
        for i in range(1, len(values)):
            smoothed_value = alpha * values[i] + (1 - alpha) * smoothed[i-1]
            smoothed.append(smoothed_value)
        
        predicted_value = smoothed[-1]
        
        # Calculate confidence interval
        std_error = statistics.stdev(values) if len(values) > 1 else 0
        margin_error = 1.96 * std_error
        
        confidence_interval = (
            predicted_value - margin_error,
            predicted_value + margin_error
        )
        
        return predicted_value, confidence_interval
    
    def _calculate_prediction_confidence(self, analysis: TrendAnalysis, prediction_days: int) -> PredictionConfidence:
        """Calculate confidence in prediction"""
        # Base confidence on trend strength and prediction horizon
        trend_strength = analysis.trend_strength
        volatility = analysis.volatility
        
        # Adjust confidence based on prediction horizon
        horizon_factor = max(0.1, 1 - (prediction_days / self.max_prediction_horizon))
        
        # Calculate overall confidence
        confidence_score = trend_strength * horizon_factor * (1 - volatility)
        
        if confidence_score >= 0.8:
            return PredictionConfidence.VERY_HIGH
        elif confidence_score >= 0.6:
            return PredictionConfidence.HIGH
        elif confidence_score >= 0.4:
            return PredictionConfidence.MEDIUM
        elif confidence_score >= 0.2:
            return PredictionConfidence.LOW
        else:
            return PredictionConfidence.VERY_LOW
    
    async def generate_insights(self, metric_name: str) -> List[TrendInsight]:
        """
        Generate insights from trend analysis
        
        Args:
            metric_name: Name of the metric to analyze
            
        Returns:
            List of trend insights
        """
        insights = []
        
        # Find relevant trend analyses
        relevant_analyses = [
            analysis for analysis in self.trend_analyses.values()
            if analysis.metric_name == metric_name
        ]
        
        if not relevant_analyses:
            return insights
        
        # Get latest analysis
        latest_analysis = max(relevant_analyses, key=lambda x: x.created_at)
        
        # Generate insights based on trend characteristics
        if latest_analysis.trend_type == TrendType.RISING:
            insights.append(self._create_insight(
                "positive_trend",
                f"{metric_name} is showing a strong upward trend with {latest_analysis.trend_strength:.1%} strength",
                0.8,
                ["Continue monitoring the trend", "Consider scaling resources", "Prepare for increased demand"]
            ))
        
        elif latest_analysis.trend_type == TrendType.FALLING:
            insights.append(self._create_insight(
                "negative_trend",
                f"{metric_name} is declining with {latest_analysis.trend_strength:.1%} strength",
                0.7,
                ["Investigate the cause of decline", "Consider corrective actions", "Monitor closely"]
            ))
        
        if latest_analysis.volatility > 0.5:
            insights.append(self._create_insight(
                "high_volatility",
                f"{metric_name} shows high volatility ({latest_analysis.volatility:.1%})",
                0.6,
                ["Implement risk management strategies", "Consider smoothing techniques", "Monitor for patterns"]
            ))
        
        if latest_analysis.seasonality > 0.3:
            insights.append(self._create_insight(
                "seasonal_pattern",
                f"{metric_name} exhibits seasonal patterns with {latest_analysis.seasonality:.1%} strength",
                0.7,
                ["Plan for seasonal variations", "Adjust strategies accordingly", "Use seasonal forecasting"]
            ))
        
        # Store insights
        for insight in insights:
            self.insights[insight.id] = insight
        
        return insights
    
    def _create_insight(self, insight_type: str, description: str, impact_score: float, recommendations: List[str]) -> TrendInsight:
        """Create a trend insight"""
        return TrendInsight(
            id=str(uuid.uuid4()),
            insight_type=insight_type,
            description=description,
            impact_score=impact_score,
            confidence=PredictionConfidence.MEDIUM,
            recommendations=recommendations,
            related_trends=[],
            created_at=datetime.now()
        )
    
    def get_trend_analysis(self, analysis_id: str) -> Optional[TrendAnalysis]:
        """Get trend analysis by ID"""
        return self.trend_analyses.get(analysis_id)
    
    def get_prediction(self, prediction_id: str) -> Optional[Prediction]:
        """Get prediction by ID"""
        return self.predictions.get(prediction_id)
    
    def get_insights(self, metric_name: str) -> List[TrendInsight]:
        """Get insights for a metric"""
        return [
            insight for insight in self.insights.values()
            if metric_name in insight.description
        ]
    
    def get_predictor_statistics(self) -> Dict[str, Any]:
        """Get predictor statistics"""
        total_analyses = len(self.trend_analyses)
        total_predictions = len(self.predictions)
        total_insights = len(self.insights)
        
        # Count by trend type
        trend_types = Counter(analysis.trend_type for analysis in self.trend_analyses.values())
        
        # Count by confidence level
        confidence_levels = Counter(prediction.confidence for prediction in self.predictions.values())
        
        return {
            "total_analyses": total_analyses,
            "total_predictions": total_predictions,
            "total_insights": total_insights,
            "trend_types": {trend_type.value: count for trend_type, count in trend_types.items()},
            "confidence_distribution": {confidence.value: count for confidence, count in confidence_levels.items()},
            "metrics_tracked": len(self.time_series_data),
            "total_data_points": sum(len(points) for points in self.time_series_data.values())
        }

# Example usage
if __name__ == "__main__":
    # Initialize trend predictor
    predictor = AdvancedTrendPredictor()
    
    # Add sample data
    base_date = datetime.now() - timedelta(days=30)
    for i in range(30):
        # Simulate rising trend with some noise
        value = 100 + i * 2 + np.random.normal(0, 5)
        timestamp = base_date + timedelta(days=i)
        await predictor.add_data_point("document_classifications", value, timestamp)
    
    # Analyze trend
    analysis = await predictor.analyze_trend("document_classifications")
    
    print("Trend Analysis Results:")
    print(f"Trend Type: {analysis.trend_type.value}")
    print(f"Trend Strength: {analysis.trend_strength:.3f}")
    print(f"Trend Direction: {analysis.trend_direction:.3f}")
    print(f"Volatility: {analysis.volatility:.3f}")
    print(f"Seasonality: {analysis.seasonality:.3f}")
    print(f"Confidence: {analysis.confidence.value}")
    
    # Make prediction
    prediction = await predictor.predict_future(analysis.id, 7)
    
    print(f"\nPrediction Results:")
    print(f"Predicted Value: {prediction.predicted_value:.2f}")
    print(f"Confidence Interval: {prediction.confidence_interval}")
    print(f"Prediction Confidence: {prediction.confidence.value}")
    print(f"Model Used: {prediction.model_used}")
    
    # Generate insights
    insights = await predictor.generate_insights("document_classifications")
    
    print(f"\nInsights Generated: {len(insights)}")
    for insight in insights:
        print(f"- {insight.description}")
        print(f"  Recommendations: {', '.join(insight.recommendations[:2])}")
    
    # Get statistics
    stats = predictor.get_predictor_statistics()
    print(f"\nPredictor Statistics:")
    print(f"Total Analyses: {stats['total_analyses']}")
    print(f"Total Predictions: {stats['total_predictions']}")
    print(f"Metrics Tracked: {stats['metrics_tracked']}")
    
    print("\nAdvanced Trend Predictor initialized successfully")
