"""
NLP Trends and Temporal Analysis System
======================================

Sistema de análisis temporal y de tendencias para el sistema NLP.
Incluye análisis de patrones, predicciones y detección de anomalías.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import statistics
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import threading

logger = logging.getLogger(__name__)

class TrendDirection(str, Enum):
    """Direcciones de tendencia."""
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    VOLATILE = "volatile"

class AnomalyType(str, Enum):
    """Tipos de anomalías."""
    SPIKE = "spike"
    DROP = "drop"
    PATTERN_BREAK = "pattern_break"
    OUTLIER = "outlier"

@dataclass
class TrendPoint:
    """Punto de tendencia."""
    timestamp: datetime
    value: float
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TrendAnalysis:
    """Análisis de tendencia."""
    direction: TrendDirection
    strength: float  # 0-1
    confidence: float  # 0-1
    start_time: datetime
    end_time: datetime
    data_points: List[TrendPoint]
    slope: float = 0.0
    r_squared: float = 0.0

@dataclass
class Anomaly:
    """Anomalía detectada."""
    id: str
    timestamp: datetime
    value: float
    expected_value: float
    anomaly_type: AnomalyType
    severity: float  # 0-1
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Prediction:
    """Predicción temporal."""
    timestamp: datetime
    predicted_value: float
    confidence_interval: Tuple[float, float]
    confidence: float
    model_used: str

class TemporalAnalyzer:
    """Analizador temporal para datos NLP."""
    
    def __init__(self, window_size: int = 100, min_points: int = 10):
        """Initialize temporal analyzer."""
        self.window_size = window_size
        self.min_points = min_points
        self.data_points: deque = deque(maxlen=window_size)
        self._lock = threading.RLock()
        
        # Anomaly detection
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        self.anomaly_threshold = 0.1
        
        # Trend detection
        self.trend_window = 20
        self.min_trend_points = 5
        
    def add_data_point(self, timestamp: datetime, value: float, metadata: Dict[str, Any] = None):
        """Add a data point for analysis."""
        with self._lock:
            point = TrendPoint(
                timestamp=timestamp,
                value=value,
                metadata=metadata or {}
            )
            self.data_points.append(point)
    
    def get_trend_analysis(self, metric_name: str, hours: int = 24) -> Optional[TrendAnalysis]:
        """Analyze trends for a specific metric."""
        with self._lock:
            if len(self.data_points) < self.min_points:
                return None
            
            # Filter data by time window
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_points = [
                p for p in self.data_points
                if p.timestamp >= cutoff_time and metric_name in p.metadata.get('metric', '')
            ]
            
            if len(recent_points) < self.min_trend_points:
                return None
            
            # Extract values and timestamps
            values = [p.value for p in recent_points]
            timestamps = [p.timestamp for p in recent_points]
            
            # Calculate trend
            return self._calculate_trend(recent_points, values, timestamps)
    
    def _calculate_trend(self, points: List[TrendPoint], values: List[float], timestamps: List[datetime]) -> TrendAnalysis:
        """Calculate trend from data points."""
        # Linear regression
        x = np.arange(len(values))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
        
        # Determine direction
        if abs(slope) < 0.01:
            direction = TrendDirection.STABLE
        elif slope > 0:
            direction = TrendDirection.INCREASING
        else:
            direction = TrendDirection.DECREASING
        
        # Calculate strength (absolute slope normalized)
        max_value = max(values) if values else 1
        min_value = min(values) if values else 0
        value_range = max_value - min_value if max_value != min_value else 1
        strength = min(abs(slope) / (value_range / len(values)), 1.0)
        
        # Calculate confidence based on R-squared
        confidence = max(0, min(1, r_value ** 2))
        
        return TrendAnalysis(
            direction=direction,
            strength=strength,
            confidence=confidence,
            start_time=timestamps[0],
            end_time=timestamps[-1],
            data_points=points,
            slope=slope,
            r_squared=r_value ** 2
        )
    
    def detect_anomalies(self, metric_name: str, hours: int = 24) -> List[Anomaly]:
        """Detect anomalies in metric data."""
        with self._lock:
            if len(self.data_points) < self.min_points:
                return []
            
            # Filter data by time window
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_points = [
                p for p in self.data_points
                if p.timestamp >= cutoff_time and metric_name in p.metadata.get('metric', '')
            ]
            
            if len(recent_points) < self.min_points:
                return []
            
            # Prepare data for anomaly detection
            values = np.array([p.value for p in recent_points]).reshape(-1, 1)
            
            # Standardize values
            values_scaled = self.scaler.fit_transform(values)
            
            # Fit isolation forest
            self.isolation_forest.fit(values_scaled)
            
            # Predict anomalies
            anomaly_scores = self.isolation_forest.decision_function(values_scaled)
            predictions = self.isolation_forest.predict(values_scaled)
            
            anomalies = []
            for i, (point, score, prediction) in enumerate(zip(recent_points, anomaly_scores, predictions)):
                if prediction == -1 or score < -self.anomaly_threshold:
                    # Calculate expected value (moving average)
                    window_start = max(0, i - 5)
                    window_end = min(len(values), i + 5)
                    expected_value = np.mean(values[window_start:window_end])
                    
                    # Determine anomaly type
                    if point.value > expected_value * 1.5:
                        anomaly_type = AnomalyType.SPIKE
                    elif point.value < expected_value * 0.5:
                        anomaly_type = AnomalyType.DROP
                    else:
                        anomaly_type = AnomalyType.OUTLIER
                    
                    # Calculate severity
                    severity = min(1.0, abs(point.value - expected_value) / expected_value if expected_value != 0 else 0)
                    
                    anomaly = Anomaly(
                        id=f"{metric_name}_{point.timestamp.isoformat()}",
                        timestamp=point.timestamp,
                        value=point.value,
                        expected_value=expected_value,
                        anomaly_type=anomaly_type,
                        severity=severity,
                        description=f"{anomaly_type.value} in {metric_name}",
                        metadata=point.metadata
                    )
                    anomalies.append(anomaly)
            
            return anomalies
    
    def predict_future_values(self, metric_name: str, hours_ahead: int = 1) -> List[Prediction]:
        """Predict future values for a metric."""
        with self._lock:
            if len(self.data_points) < self.min_points:
                return []
            
            # Filter data by metric
            metric_points = [
                p for p in self.data_points
                if metric_name in p.metadata.get('metric', '')
            ]
            
            if len(metric_points) < self.min_points:
                return []
            
            # Simple linear regression prediction
            values = [p.value for p in metric_points]
            x = np.arange(len(values))
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
            
            predictions = []
            for i in range(1, hours_ahead + 1):
                future_x = len(values) + i
                predicted_value = slope * future_x + intercept
                
                # Calculate confidence interval (simplified)
                confidence_std = std_err * np.sqrt(1 + 1/len(values) + (future_x - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
                confidence_interval = (
                    predicted_value - 1.96 * confidence_std,
                    predicted_value + 1.96 * confidence_std
                )
                
                prediction = Prediction(
                    timestamp=datetime.now() + timedelta(hours=i),
                    predicted_value=predicted_value,
                    confidence_interval=confidence_interval,
                    confidence=max(0, min(1, r_value ** 2)),
                    model_used="linear_regression"
                )
                predictions.append(prediction)
            
            return predictions

class NLPTrendAnalyzer:
    """Analizador de tendencias para el sistema NLP."""
    
    def __init__(self):
        """Initialize NLP trend analyzer."""
        self.analyzers: Dict[str, TemporalAnalyzer] = {}
        self.trend_history: List[TrendAnalysis] = []
        self.anomaly_history: List[Anomaly] = []
        self.prediction_history: List[Prediction] = []
        
        # Initialize analyzers for different metrics
        self.metrics = [
            'processing_time',
            'sentiment_accuracy',
            'entity_precision',
            'entity_recall',
            'keyword_relevance',
            'readability_score',
            'error_rate',
            'throughput',
            'cache_hit_rate',
            'memory_usage',
            'cpu_usage'
        ]
        
        for metric in self.metrics:
            self.analyzers[metric] = TemporalAnalyzer()
    
    async def record_metric_value(
        self,
        metric_name: str,
        value: float,
        timestamp: datetime = None,
        metadata: Dict[str, Any] = None
    ):
        """Record a metric value for trend analysis."""
        if metric_name not in self.analyzers:
            self.analyzers[metric_name] = TemporalAnalyzer()
        
        self.analyzers[metric_name].add_data_point(
            timestamp or datetime.now(),
            value,
            metadata or {}
        )
    
    async def analyze_trends(self, hours: int = 24) -> Dict[str, TrendAnalysis]:
        """Analyze trends for all metrics."""
        trends = {}
        
        for metric_name, analyzer in self.analyzers.items():
            trend = analyzer.get_trend_analysis(metric_name, hours)
            if trend:
                trends[metric_name] = trend
                self.trend_history.append(trend)
        
        return trends
    
    async def detect_anomalies(self, hours: int = 24) -> Dict[str, List[Anomaly]]:
        """Detect anomalies for all metrics."""
        all_anomalies = {}
        
        for metric_name, analyzer in self.analyzers.items():
            anomalies = analyzer.detect_anomalies(metric_name, hours)
            if anomalies:
                all_anomalies[metric_name] = anomalies
                self.anomaly_history.extend(anomalies)
        
        return all_anomalies
    
    async def generate_predictions(self, hours_ahead: int = 24) -> Dict[str, List[Prediction]]:
        """Generate predictions for all metrics."""
        predictions = {}
        
        for metric_name, analyzer in self.analyzers.items():
            metric_predictions = analyzer.predict_future_values(metric_name, hours_ahead)
            if metric_predictions:
                predictions[metric_name] = metric_predictions
                self.prediction_history.extend(metric_predictions)
        
        return predictions
    
    def get_trend_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of trends."""
        trends = {}
        
        for metric_name, analyzer in self.analyzers.items():
            trend = analyzer.get_trend_analysis(metric_name, hours)
            if trend:
                trends[metric_name] = {
                    'direction': trend.direction.value,
                    'strength': trend.strength,
                    'confidence': trend.confidence,
                    'slope': trend.slope,
                    'r_squared': trend.r_squared,
                    'data_points': len(trend.data_points)
                }
        
        return trends
    
    def get_anomaly_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of anomalies."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_anomalies = [
            a for a in self.anomaly_history
            if a.timestamp >= cutoff_time
        ]
        
        if not recent_anomalies:
            return {'total_anomalies': 0, 'by_type': {}, 'by_severity': {}}
        
        # Group by type
        by_type = defaultdict(int)
        for anomaly in recent_anomalies:
            by_type[anomaly.anomaly_type.value] += 1
        
        # Group by severity
        by_severity = {
            'low': len([a for a in recent_anomalies if a.severity < 0.3]),
            'medium': len([a for a in recent_anomalies if 0.3 <= a.severity < 0.7]),
            'high': len([a for a in recent_anomalies if a.severity >= 0.7])
        }
        
        return {
            'total_anomalies': len(recent_anomalies),
            'by_type': dict(by_type),
            'by_severity': by_severity,
            'most_common_type': max(by_type.items(), key=lambda x: x[1])[0] if by_type else None,
            'average_severity': statistics.mean([a.severity for a in recent_anomalies])
        }
    
    def get_prediction_summary(self, hours_ahead: int = 24) -> Dict[str, Any]:
        """Get summary of predictions."""
        future_time = datetime.now() + timedelta(hours=hours_ahead)
        
        relevant_predictions = [
            p for p in self.prediction_history
            if p.timestamp <= future_time
        ]
        
        if not relevant_predictions:
            return {'total_predictions': 0, 'average_confidence': 0}
        
        return {
            'total_predictions': len(relevant_predictions),
            'average_confidence': statistics.mean([p.confidence for p in relevant_predictions]),
            'models_used': list(set([p.model_used for p in relevant_predictions])),
            'prediction_range': {
                'min': min([p.predicted_value for p in relevant_predictions]),
                'max': max([p.predicted_value for p in relevant_predictions]),
                'avg': statistics.mean([p.predicted_value for p in relevant_predictions])
            }
        }
    
    def get_insights(self, hours: int = 24) -> List[str]:
        """Generate insights from trend analysis."""
        insights = []
        
        # Analyze trends
        trends = {}
        for metric_name, analyzer in self.analyzers.items():
            trend = analyzer.get_trend_analysis(metric_name, hours)
            if trend:
                trends[metric_name] = trend
        
        # Generate insights
        for metric_name, trend in trends.items():
            if trend.direction == TrendDirection.INCREASING and trend.strength > 0.7:
                insights.append(f"{metric_name} is showing a strong upward trend (strength: {trend.strength:.2f})")
            elif trend.direction == TrendDirection.DECREASING and trend.strength > 0.7:
                insights.append(f"{metric_name} is showing a strong downward trend (strength: {trend.strength:.2f})")
            elif trend.direction == TrendDirection.STABLE and trend.confidence > 0.8:
                insights.append(f"{metric_name} is stable with high confidence ({trend.confidence:.2f})")
        
        # Analyze anomalies
        anomalies = {}
        for metric_name, analyzer in self.analyzers.items():
            metric_anomalies = analyzer.detect_anomalies(metric_name, hours)
            if metric_anomalies:
                anomalies[metric_name] = metric_anomalies
        
        for metric_name, metric_anomalies in anomalies.items():
            if len(metric_anomalies) > 3:
                insights.append(f"Multiple anomalies detected in {metric_name} ({len(metric_anomalies)} anomalies)")
            
            high_severity = [a for a in metric_anomalies if a.severity > 0.7]
            if high_severity:
                insights.append(f"High severity anomalies in {metric_name} ({len(high_severity)} anomalies)")
        
        return insights

# Global trend analyzer instance
nlp_trend_analyzer = NLPTrendAnalyzer()












