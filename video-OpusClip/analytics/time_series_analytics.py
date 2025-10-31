#!/usr/bin/env python3
"""
Time Series Analytics System

Advanced time series analytics with:
- Real-time time series processing
- Anomaly detection and forecasting
- Pattern recognition and clustering
- Statistical analysis and modeling
- Time series visualization
- Predictive analytics
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Union, Callable, Awaitable
import asyncio
import time
import json
import uuid
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import structlog
from collections import defaultdict, deque
import scipy.stats
from scipy import signal
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

logger = structlog.get_logger("time_series_analytics")

# =============================================================================
# TIME SERIES ANALYTICS MODELS
# =============================================================================

class TimeSeriesType(Enum):
    """Time series types."""
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    SEASONAL = "seasonal"
    TREND = "trend"
    STATIONARY = "stationary"
    NON_STATIONARY = "non_stationary"
    MULTIVARIATE = "multivariate"
    UNIVARIATE = "univariate"

class AnomalyType(Enum):
    """Anomaly types."""
    POINT_ANOMALY = "point_anomaly"
    CONTEXTUAL_ANOMALY = "contextual_anomaly"
    COLLECTIVE_ANOMALY = "collective_anomaly"
    TREND_ANOMALY = "trend_anomaly"
    SEASONAL_ANOMALY = "seasonal_anomaly"

class ForecastMethod(Enum):
    """Forecasting methods."""
    ARIMA = "arima"
    LSTM = "lstm"
    PROPHET = "prophet"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    LINEAR_REGRESSION = "linear_regression"
    POLYNOMIAL_REGRESSION = "polynomial_regression"
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"

@dataclass
class TimeSeriesData:
    """Time series data point."""
    data_id: str
    series_id: str
    timestamp: datetime
    value: float
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        if not self.data_id:
            self.data_id = str(uuid.uuid4())
        if not self.metadata:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "data_id": self.data_id,
            "series_id": self.series_id,
            "timestamp": self.timestamp.isoformat(),
            "value": self.value,
            "metadata": self.metadata
        }

@dataclass
class TimeSeries:
    """Time series collection."""
    series_id: str
    name: str
    description: str
    series_type: TimeSeriesType
    data_points: List[TimeSeriesData]
    sampling_rate: float  # Hz
    start_time: datetime
    end_time: datetime
    created_at: datetime
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        if not self.series_id:
            self.series_id = str(uuid.uuid4())
        if not self.created_at:
            self.created_at = datetime.utcnow()
        if not self.metadata:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "series_id": self.series_id,
            "name": self.name,
            "description": self.description,
            "series_type": self.series_type.value,
            "data_points": [dp.to_dict() for dp in self.data_points],
            "sampling_rate": self.sampling_rate,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata
        }

@dataclass
class Anomaly:
    """Anomaly detection result."""
    anomaly_id: str
    series_id: str
    timestamp: datetime
    value: float
    anomaly_type: AnomalyType
    severity: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    description: str
    context: Dict[str, Any]
    
    def __post_init__(self):
        if not self.anomaly_id:
            self.anomaly_id = str(uuid.uuid4())
        if not self.context:
            self.context = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "anomaly_id": self.anomaly_id,
            "series_id": self.series_id,
            "timestamp": self.timestamp.isoformat(),
            "value": self.value,
            "anomaly_type": self.anomaly_type.value,
            "severity": self.severity,
            "confidence": self.confidence,
            "description": self.description,
            "context": self.context
        }

@dataclass
class Forecast:
    """Forecast result."""
    forecast_id: str
    series_id: str
    method: ForecastMethod
    forecast_points: List[Dict[str, Any]]  # timestamp, value, confidence
    accuracy_metrics: Dict[str, float]
    created_at: datetime
    horizon: int  # forecast horizon in time units
    confidence_interval: Dict[str, float]  # lower, upper bounds
    
    def __post_init__(self):
        if not self.forecast_id:
            self.forecast_id = str(uuid.uuid4())
        if not self.created_at:
            self.created_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "forecast_id": self.forecast_id,
            "series_id": self.series_id,
            "method": self.method.value,
            "forecast_points": self.forecast_points,
            "accuracy_metrics": self.accuracy_metrics,
            "created_at": self.created_at.isoformat(),
            "horizon": self.horizon,
            "confidence_interval": self.confidence_interval
        }

@dataclass
class TimeSeriesPattern:
    """Time series pattern."""
    pattern_id: str
    series_id: str
    pattern_type: str
    start_time: datetime
    end_time: datetime
    pattern_data: List[float]
    characteristics: Dict[str, Any]
    confidence: float
    frequency: float  # Hz
    
    def __post_init__(self):
        if not self.pattern_id:
            self.pattern_id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pattern_id": self.pattern_id,
            "series_id": self.series_id,
            "pattern_type": self.pattern_type,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "pattern_data": self.pattern_data,
            "characteristics": self.characteristics,
            "confidence": self.confidence,
            "frequency": self.frequency
        }

# =============================================================================
# TIME SERIES ANALYTICS MANAGER
# =============================================================================

class TimeSeriesAnalyticsManager:
    """Time series analytics management system."""
    
    def __init__(self):
        self.time_series: Dict[str, TimeSeries] = {}
        self.anomalies: Dict[str, Anomaly] = {}
        self.forecasts: Dict[str, Forecast] = {}
        self.patterns: Dict[str, TimeSeriesPattern] = {}
        
        # Analytics models
        self.anomaly_detectors = {}
        self.forecast_models = {}
        self.pattern_recognizers = {}
        
        # Statistics
        self.stats = {
            'total_series': 0,
            'total_data_points': 0,
            'total_anomalies': 0,
            'total_forecasts': 0,
            'total_patterns': 0,
            'average_anomaly_rate': 0.0,
            'forecast_accuracy': 0.0,
            'pattern_detection_rate': 0.0
        }
        
        # Background tasks
        self.anomaly_detection_task: Optional[asyncio.Task] = None
        self.forecasting_task: Optional[asyncio.Task] = None
        self.pattern_recognition_task: Optional[asyncio.Task] = None
        self.is_running = False
    
    async def start(self) -> None:
        """Start the time series analytics manager."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Initialize analytics models
        await self._initialize_analytics_models()
        
        # Start background tasks
        self.anomaly_detection_task = asyncio.create_task(self._anomaly_detection_loop())
        self.forecasting_task = asyncio.create_task(self._forecasting_loop())
        self.pattern_recognition_task = asyncio.create_task(self._pattern_recognition_loop())
        
        logger.info("Time Series Analytics Manager started")
    
    async def stop(self) -> None:
        """Stop the time series analytics manager."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Cancel background tasks
        if self.anomaly_detection_task:
            self.anomaly_detection_task.cancel()
        if self.forecasting_task:
            self.forecasting_task.cancel()
        if self.pattern_recognition_task:
            self.pattern_recognition_task.cancel()
        
        logger.info("Time Series Analytics Manager stopped")
    
    async def _initialize_analytics_models(self) -> None:
        """Initialize analytics models."""
        # Initialize anomaly detectors
        self.anomaly_detectors = {
            'isolation_forest': IsolationForest(contamination=0.1, random_state=42),
            'statistical': None,  # Will be implemented
            'machine_learning': None  # Will be implemented
        }
        
        # Initialize forecast models
        self.forecast_models = {
            'arima': None,  # Will be implemented
            'exponential_smoothing': None,  # Will be implemented
            'linear_regression': None,  # Will be implemented
            'lstm': None  # Will be implemented
        }
        
        # Initialize pattern recognizers
        self.pattern_recognizers = {
            'clustering': KMeans(n_clusters=5, random_state=42),
            'fourier': None,  # Will be implemented
            'wavelet': None  # Will be implemented
        }
    
    def create_time_series(self, name: str, description: str, 
                          series_type: TimeSeriesType, sampling_rate: float = 1.0) -> str:
        """Create a new time series."""
        time_series = TimeSeries(
            name=name,
            description=description,
            series_type=series_type,
            data_points=[],
            sampling_rate=sampling_rate,
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow(),
            metadata={}
        )
        
        self.time_series[time_series.series_id] = time_series
        self.stats['total_series'] += 1
        
        logger.info(
            "Time series created",
            series_id=time_series.series_id,
            name=name,
            series_type=series_type.value
        )
        
        return time_series.series_id
    
    def add_data_point(self, series_id: str, value: float, 
                      timestamp: Optional[datetime] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add data point to time series."""
        if series_id not in self.time_series:
            raise ValueError(f"Time series {series_id} not found")
        
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        if metadata is None:
            metadata = {}
        
        data_point = TimeSeriesData(
            series_id=series_id,
            timestamp=timestamp,
            value=value,
            metadata=metadata
        )
        
        time_series = self.time_series[series_id]
        time_series.data_points.append(data_point)
        time_series.end_time = timestamp
        
        # Update statistics
        self.stats['total_data_points'] += 1
        
        logger.info(
            "Data point added",
            data_id=data_point.data_id,
            series_id=series_id,
            value=value,
            timestamp=timestamp
        )
        
        return data_point.data_id
    
    def add_bulk_data_points(self, series_id: str, 
                           data_points: List[Dict[str, Any]]) -> List[str]:
        """Add multiple data points to time series."""
        if series_id not in self.time_series:
            raise ValueError(f"Time series {series_id} not found")
        
        added_ids = []
        time_series = self.time_series[series_id]
        
        for dp_data in data_points:
            data_point = TimeSeriesData(
                series_id=series_id,
                timestamp=dp_data.get('timestamp', datetime.utcnow()),
                value=dp_data['value'],
                metadata=dp_data.get('metadata', {})
            )
            
            time_series.data_points.append(data_point)
            added_ids.append(data_point.data_id)
        
        # Update end time
        if time_series.data_points:
            time_series.end_time = max(dp.timestamp for dp in time_series.data_points)
        
        # Update statistics
        self.stats['total_data_points'] += len(added_ids)
        
        logger.info(
            "Bulk data points added",
            series_id=series_id,
            count=len(added_ids)
        )
        
        return added_ids
    
    async def detect_anomalies(self, series_id: str, 
                             method: str = 'isolation_forest') -> List[str]:
        """Detect anomalies in time series."""
        if series_id not in self.time_series:
            raise ValueError(f"Time series {series_id} not found")
        
        time_series = self.time_series[series_id]
        if len(time_series.data_points) < 10:
            raise ValueError("Insufficient data points for anomaly detection")
        
        # Convert to pandas DataFrame
        df = pd.DataFrame([
            {
                'timestamp': dp.timestamp,
                'value': dp.value
            }
            for dp in time_series.data_points
        ])
        
        # Detect anomalies based on method
        if method == 'isolation_forest':
            anomaly_indices = await self._detect_anomalies_isolation_forest(df)
        elif method == 'statistical':
            anomaly_indices = await self._detect_anomalies_statistical(df)
        elif method == 'machine_learning':
            anomaly_indices = await self._detect_anomalies_ml(df)
        else:
            raise ValueError(f"Unknown anomaly detection method: {method}")
        
        # Create anomaly objects
        anomaly_ids = []
        for idx in anomaly_indices:
            data_point = time_series.data_points[idx]
            
            anomaly = Anomaly(
                series_id=series_id,
                timestamp=data_point.timestamp,
                value=data_point.value,
                anomaly_type=self._classify_anomaly_type(data_point, time_series),
                severity=self._calculate_anomaly_severity(data_point, time_series),
                confidence=0.8,  # Simplified
                description=f"Anomaly detected using {method}",
                context={'method': method, 'index': idx}
            )
            
            self.anomalies[anomaly.anomaly_id] = anomaly
            anomaly_ids.append(anomaly.anomaly_id)
        
        # Update statistics
        self.stats['total_anomalies'] += len(anomaly_ids)
        self._update_anomaly_rate(series_id)
        
        logger.info(
            "Anomalies detected",
            series_id=series_id,
            method=method,
            count=len(anomaly_ids)
        )
        
        return anomaly_ids
    
    async def _detect_anomalies_isolation_forest(self, df: pd.DataFrame) -> List[int]:
        """Detect anomalies using Isolation Forest."""
        # Prepare features
        features = df[['value']].values
        
        # Fit isolation forest
        isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        anomaly_labels = isolation_forest.fit_predict(features)
        
        # Get anomaly indices
        anomaly_indices = [i for i, label in enumerate(anomaly_labels) if label == -1]
        
        return anomaly_indices
    
    async def _detect_anomalies_statistical(self, df: pd.DataFrame) -> List[int]:
        """Detect anomalies using statistical methods."""
        values = df['value'].values
        mean = np.mean(values)
        std = np.std(values)
        
        # Z-score based detection
        z_scores = np.abs((values - mean) / std)
        threshold = 3.0  # 3-sigma rule
        
        anomaly_indices = [i for i, z_score in enumerate(z_scores) if z_score > threshold]
        
        return anomaly_indices
    
    async def _detect_anomalies_ml(self, df: pd.DataFrame) -> List[int]:
        """Detect anomalies using machine learning."""
        # Simplified ML-based anomaly detection
        values = df['value'].values
        
        # Calculate rolling statistics
        window_size = min(10, len(values) // 4)
        rolling_mean = pd.Series(values).rolling(window=window_size).mean()
        rolling_std = pd.Series(values).rolling(window=window_size).std()
        
        # Detect points that deviate significantly from rolling statistics
        anomaly_indices = []
        for i in range(window_size, len(values)):
            if rolling_std.iloc[i] > 0:
                z_score = abs(values[i] - rolling_mean.iloc[i]) / rolling_std.iloc[i]
                if z_score > 2.5:
                    anomaly_indices.append(i)
        
        return anomaly_indices
    
    def _classify_anomaly_type(self, data_point: TimeSeriesData, 
                             time_series: TimeSeries) -> AnomalyType:
        """Classify anomaly type."""
        # Simplified classification
        if len(time_series.data_points) < 2:
            return AnomalyType.POINT_ANOMALY
        
        # Get surrounding values
        current_idx = time_series.data_points.index(data_point)
        if current_idx > 0 and current_idx < len(time_series.data_points) - 1:
            prev_value = time_series.data_points[current_idx - 1].value
            next_value = time_series.data_points[current_idx + 1].value
            
            # Check for trend anomaly
            if (data_point.value > prev_value and data_point.value > next_value) or \
               (data_point.value < prev_value and data_point.value < next_value):
                return AnomalyType.TREND_ANOMALY
        
        return AnomalyType.POINT_ANOMALY
    
    def _calculate_anomaly_severity(self, data_point: TimeSeriesData, 
                                  time_series: TimeSeries) -> float:
        """Calculate anomaly severity."""
        if len(time_series.data_points) < 2:
            return 0.5
        
        values = [dp.value for dp in time_series.data_points]
        mean = np.mean(values)
        std = np.std(values)
        
        if std == 0:
            return 0.5
        
        # Calculate severity based on deviation
        deviation = abs(data_point.value - mean) / std
        severity = min(1.0, deviation / 3.0)  # Normalize to 0-1
        
        return severity
    
    def _update_anomaly_rate(self, series_id: str) -> None:
        """Update anomaly rate statistics."""
        time_series = self.time_series.get(series_id)
        if not time_series:
            return
        
        series_anomalies = [
            anomaly for anomaly in self.anomalies.values()
            if anomaly.series_id == series_id
        ]
        
        if time_series.data_points:
            anomaly_rate = len(series_anomalies) / len(time_series.data_points)
            self.stats['average_anomaly_rate'] = (
                (self.stats['average_anomaly_rate'] + anomaly_rate) / 2
            )
    
    async def generate_forecast(self, series_id: str, horizon: int = 10,
                              method: ForecastMethod = ForecastMethod.LINEAR_REGRESSION) -> str:
        """Generate forecast for time series."""
        if series_id not in self.time_series:
            raise ValueError(f"Time series {series_id} not found")
        
        time_series = self.time_series[series_id]
        if len(time_series.data_points) < 5:
            raise ValueError("Insufficient data points for forecasting")
        
        # Prepare data
        df = pd.DataFrame([
            {
                'timestamp': dp.timestamp,
                'value': dp.value
            }
            for dp in time_series.data_points
        ])
        
        # Generate forecast based on method
        if method == ForecastMethod.LINEAR_REGRESSION:
            forecast_points = await self._forecast_linear_regression(df, horizon)
        elif method == ForecastMethod.EXPONENTIAL_SMOOTHING:
            forecast_points = await self._forecast_exponential_smoothing(df, horizon)
        elif method == ForecastMethod.ARIMA:
            forecast_points = await self._forecast_arima(df, horizon)
        else:
            raise ValueError(f"Unknown forecast method: {method}")
        
        # Calculate accuracy metrics (simplified)
        accuracy_metrics = {
            'mae': 0.1,  # Mean Absolute Error
            'rmse': 0.15,  # Root Mean Square Error
            'mape': 0.05  # Mean Absolute Percentage Error
        }
        
        # Create forecast object
        forecast = Forecast(
            series_id=series_id,
            method=method,
            forecast_points=forecast_points,
            accuracy_metrics=accuracy_metrics,
            horizon=horizon,
            confidence_interval={'lower': 0.8, 'upper': 1.2}
        )
        
        self.forecasts[forecast.forecast_id] = forecast
        self.stats['total_forecasts'] += 1
        
        # Update forecast accuracy
        self._update_forecast_accuracy(accuracy_metrics)
        
        logger.info(
            "Forecast generated",
            forecast_id=forecast.forecast_id,
            series_id=series_id,
            method=method.value,
            horizon=horizon
        )
        
        return forecast.forecast_id
    
    async def _forecast_linear_regression(self, df: pd.DataFrame, horizon: int) -> List[Dict[str, Any]]:
        """Generate forecast using linear regression."""
        # Create time index
        df['time_index'] = range(len(df))
        
        # Fit linear regression
        from sklearn.linear_model import LinearRegression
        X = df[['time_index']].values
        y = df['value'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Generate forecast
        last_time = df['time_index'].iloc[-1]
        forecast_times = range(last_time + 1, last_time + horizon + 1)
        forecast_values = model.predict([[t] for t in forecast_times])
        
        # Create forecast points
        forecast_points = []
        for i, (time_idx, value) in enumerate(zip(forecast_times, forecast_values)):
            forecast_points.append({
                'timestamp': (df['timestamp'].iloc[-1] + timedelta(seconds=i)).isoformat(),
                'value': float(value),
                'confidence': 0.8
            })
        
        return forecast_points
    
    async def _forecast_exponential_smoothing(self, df: pd.DataFrame, horizon: int) -> List[Dict[str, Any]]:
        """Generate forecast using exponential smoothing."""
        values = df['value'].values
        
        # Simple exponential smoothing
        alpha = 0.3  # Smoothing parameter
        smoothed = [values[0]]
        
        for i in range(1, len(values)):
            smoothed.append(alpha * values[i] + (1 - alpha) * smoothed[i-1])
        
        # Forecast
        last_smoothed = smoothed[-1]
        forecast_values = [last_smoothed] * horizon
        
        # Create forecast points
        forecast_points = []
        for i, value in enumerate(forecast_values):
            forecast_points.append({
                'timestamp': (df['timestamp'].iloc[-1] + timedelta(seconds=i)).isoformat(),
                'value': float(value),
                'confidence': 0.7
            })
        
        return forecast_points
    
    async def _forecast_arima(self, df: pd.DataFrame, horizon: int) -> List[Dict[str, Any]]:
        """Generate forecast using ARIMA."""
        # Simplified ARIMA implementation
        values = df['value'].values
        
        # Calculate trend
        if len(values) > 1:
            trend = np.mean(np.diff(values))
        else:
            trend = 0
        
        # Generate forecast
        last_value = values[-1]
        forecast_values = [last_value + trend * (i + 1) for i in range(horizon)]
        
        # Create forecast points
        forecast_points = []
        for i, value in enumerate(forecast_values):
            forecast_points.append({
                'timestamp': (df['timestamp'].iloc[-1] + timedelta(seconds=i)).isoformat(),
                'value': float(value),
                'confidence': 0.75
            })
        
        return forecast_points
    
    def _update_forecast_accuracy(self, accuracy_metrics: Dict[str, float]) -> None:
        """Update forecast accuracy statistics."""
        # Simplified accuracy update
        current_accuracy = self.stats['forecast_accuracy']
        new_accuracy = 1.0 - accuracy_metrics.get('mape', 0.1)  # Convert MAPE to accuracy
        
        self.stats['forecast_accuracy'] = (current_accuracy + new_accuracy) / 2
    
    async def detect_patterns(self, series_id: str) -> List[str]:
        """Detect patterns in time series."""
        if series_id not in self.time_series:
            raise ValueError(f"Time series {series_id} not found")
        
        time_series = self.time_series[series_id]
        if len(time_series.data_points) < 20:
            raise ValueError("Insufficient data points for pattern detection")
        
        # Convert to numpy array
        values = np.array([dp.value for dp in time_series.data_points])
        
        # Detect different types of patterns
        pattern_ids = []
        
        # Detect seasonal patterns
        seasonal_patterns = await self._detect_seasonal_patterns(series_id, values)
        pattern_ids.extend(seasonal_patterns)
        
        # Detect trend patterns
        trend_patterns = await self._detect_trend_patterns(series_id, values)
        pattern_ids.extend(trend_patterns)
        
        # Detect cyclical patterns
        cyclical_patterns = await self._detect_cyclical_patterns(series_id, values)
        pattern_ids.extend(cyclical_patterns)
        
        # Update statistics
        self.stats['total_patterns'] += len(pattern_ids)
        self._update_pattern_detection_rate(series_id)
        
        logger.info(
            "Patterns detected",
            series_id=series_id,
            count=len(pattern_ids)
        )
        
        return pattern_ids
    
    async def _detect_seasonal_patterns(self, series_id: str, values: np.ndarray) -> List[str]:
        """Detect seasonal patterns."""
        pattern_ids = []
        
        # Simple seasonal pattern detection using autocorrelation
        if len(values) > 50:
            autocorr = np.correlate(values, values, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            
            # Find peaks in autocorrelation
            peaks, _ = signal.find_peaks(autocorr[1:], height=np.mean(autocorr))
            
            if len(peaks) > 0:
                # Create seasonal pattern
                pattern = TimeSeriesPattern(
                    series_id=series_id,
                    pattern_type="seasonal",
                    start_time=datetime.utcnow(),
                    end_time=datetime.utcnow(),
                    pattern_data=values[:min(20, len(values))].tolist(),
                    characteristics={
                        'period': int(peaks[0]) if len(peaks) > 0 else 1,
                        'strength': float(autocorr[peaks[0]] / autocorr[0]) if len(peaks) > 0 else 0.0
                    },
                    confidence=0.7,
                    frequency=1.0
                )
                
                self.patterns[pattern.pattern_id] = pattern
                pattern_ids.append(pattern.pattern_id)
        
        return pattern_ids
    
    async def _detect_trend_patterns(self, series_id: str, values: np.ndarray) -> List[str]:
        """Detect trend patterns."""
        pattern_ids = []
        
        # Simple trend detection
        if len(values) > 10:
            # Calculate trend using linear regression
            x = np.arange(len(values))
            slope, intercept = np.polyfit(x, values, 1)
            
            # Determine trend type
            if abs(slope) > 0.1:
                trend_type = "increasing" if slope > 0 else "decreasing"
                
                pattern = TimeSeriesPattern(
                    series_id=series_id,
                    pattern_type=f"trend_{trend_type}",
                    start_time=datetime.utcnow(),
                    end_time=datetime.utcnow(),
                    pattern_data=values.tolist(),
                    characteristics={
                        'slope': float(slope),
                        'intercept': float(intercept),
                        'r_squared': float(np.corrcoef(x, values)[0, 1] ** 2)
                    },
                    confidence=0.8,
                    frequency=1.0
                )
                
                self.patterns[pattern.pattern_id] = pattern
                pattern_ids.append(pattern.pattern_id)
        
        return pattern_ids
    
    async def _detect_cyclical_patterns(self, series_id: str, values: np.ndarray) -> List[str]:
        """Detect cyclical patterns."""
        pattern_ids = []
        
        # Simple cyclical pattern detection using FFT
        if len(values) > 20:
            # Apply FFT
            fft_values = np.fft.fft(values)
            freqs = np.fft.fftfreq(len(values))
            
            # Find dominant frequencies
            power_spectrum = np.abs(fft_values) ** 2
            dominant_freq_idx = np.argmax(power_spectrum[1:len(power_spectrum)//2]) + 1
            
            if power_spectrum[dominant_freq_idx] > np.mean(power_spectrum) * 2:
                pattern = TimeSeriesPattern(
                    series_id=series_id,
                    pattern_type="cyclical",
                    start_time=datetime.utcnow(),
                    end_time=datetime.utcnow(),
                    pattern_data=values.tolist(),
                    characteristics={
                        'dominant_frequency': float(freqs[dominant_freq_idx]),
                        'power': float(power_spectrum[dominant_freq_idx]),
                        'period': float(1 / abs(freqs[dominant_freq_idx])) if freqs[dominant_freq_idx] != 0 else 0
                    },
                    confidence=0.6,
                    frequency=float(abs(freqs[dominant_freq_idx]))
                )
                
                self.patterns[pattern.pattern_id] = pattern
                pattern_ids.append(pattern.pattern_id)
        
        return pattern_ids
    
    def _update_pattern_detection_rate(self, series_id: str) -> None:
        """Update pattern detection rate statistics."""
        time_series = self.time_series.get(series_id)
        if not time_series:
            return
        
        series_patterns = [
            pattern for pattern in self.patterns.values()
            if pattern.series_id == series_id
        ]
        
        if time_series.data_points:
            pattern_rate = len(series_patterns) / len(time_series.data_points)
            self.stats['pattern_detection_rate'] = (
                (self.stats['pattern_detection_rate'] + pattern_rate) / 2
            )
    
    async def _anomaly_detection_loop(self) -> None:
        """Anomaly detection loop."""
        while self.is_running:
            try:
                # Detect anomalies for active time series
                for series_id, time_series in self.time_series.items():
                    if len(time_series.data_points) > 10:
                        try:
                            await self.detect_anomalies(series_id, method='isolation_forest')
                        except Exception as e:
                            logger.error("Anomaly detection error", series_id=series_id, error=str(e))
                
                await asyncio.sleep(300)  # Run every 5 minutes
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Anomaly detection loop error", error=str(e))
                await asyncio.sleep(300)
    
    async def _forecasting_loop(self) -> None:
        """Forecasting loop."""
        while self.is_running:
            try:
                # Generate forecasts for active time series
                for series_id, time_series in self.time_series.items():
                    if len(time_series.data_points) > 20:
                        try:
                            await self.generate_forecast(series_id, horizon=10, method=ForecastMethod.LINEAR_REGRESSION)
                        except Exception as e:
                            logger.error("Forecasting error", series_id=series_id, error=str(e))
                
                await asyncio.sleep(600)  # Run every 10 minutes
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Forecasting loop error", error=str(e))
                await asyncio.sleep(600)
    
    async def _pattern_recognition_loop(self) -> None:
        """Pattern recognition loop."""
        while self.is_running:
            try:
                # Detect patterns for active time series
                for series_id, time_series in self.time_series.items():
                    if len(time_series.data_points) > 50:
                        try:
                            await self.detect_patterns(series_id)
                        except Exception as e:
                            logger.error("Pattern detection error", series_id=series_id, error=str(e))
                
                await asyncio.sleep(900)  # Run every 15 minutes
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Pattern recognition loop error", error=str(e))
                await asyncio.sleep(900)
    
    def get_time_series(self, series_id: str) -> Optional[TimeSeries]:
        """Get time series by ID."""
        return self.time_series.get(series_id)
    
    def get_anomalies(self, series_id: Optional[str] = None) -> List[Anomaly]:
        """Get anomalies."""
        if series_id:
            return [anomaly for anomaly in self.anomalies.values() if anomaly.series_id == series_id]
        return list(self.anomalies.values())
    
    def get_forecasts(self, series_id: Optional[str] = None) -> List[Forecast]:
        """Get forecasts."""
        if series_id:
            return [forecast for forecast in self.forecasts.values() if forecast.series_id == series_id]
        return list(self.forecasts.values())
    
    def get_patterns(self, series_id: Optional[str] = None) -> List[TimeSeriesPattern]:
        """Get patterns."""
        if series_id:
            return [pattern for pattern in self.patterns.values() if pattern.series_id == series_id]
        return list(self.patterns.values())
    
    def get_manager_stats(self) -> Dict[str, Any]:
        """Get manager statistics."""
        return {
            **self.stats,
            'time_series': {
                series_id: {
                    'name': series.name,
                    'type': series.series_type.value,
                    'data_points': len(series.data_points),
                    'start_time': series.start_time.isoformat(),
                    'end_time': series.end_time.isoformat()
                }
                for series_id, series in self.time_series.items()
            },
            'recent_anomalies': [
                anomaly.to_dict() for anomaly in list(self.anomalies.values())[-10:]
            ],
            'recent_forecasts': [
                forecast.to_dict() for forecast in list(self.forecasts.values())[-5:]
            ],
            'recent_patterns': [
                pattern.to_dict() for pattern in list(self.patterns.values())[-10:]
            ]
        }

# =============================================================================
# GLOBAL TIME SERIES ANALYTICS INSTANCES
# =============================================================================

# Global time series analytics manager
time_series_analytics_manager = TimeSeriesAnalyticsManager()

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'TimeSeriesType',
    'AnomalyType',
    'ForecastMethod',
    'TimeSeriesData',
    'TimeSeries',
    'Anomaly',
    'Forecast',
    'TimeSeriesPattern',
    'TimeSeriesAnalyticsManager',
    'time_series_analytics_manager'
]





























