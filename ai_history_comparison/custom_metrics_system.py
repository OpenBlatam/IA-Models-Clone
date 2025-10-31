"""
Custom Metrics System
=====================

Advanced custom metrics system for AI model analysis with flexible
metric definitions, calculations, and real-time monitoring.
"""

import asyncio
import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import hashlib
import math
import statistics
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Metric types"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    CUSTOM = "custom"
    DERIVED = "derived"
    COMPOSITE = "composite"
    PREDICTIVE = "predictive"
    STATISTICAL = "statistical"
    BUSINESS = "business"


class MetricAggregation(str, Enum):
    """Metric aggregation types"""
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    MEDIAN = "median"
    STDDEV = "stddev"
    PERCENTILE = "percentile"
    RATE = "rate"
    DELTA = "delta"
    CUSTOM = "custom"


class MetricCalculationType(str, Enum):
    """Metric calculation types"""
    SIMPLE = "simple"
    WINDOWED = "windowed"
    ROLLING = "rolling"
    EXPONENTIAL = "exponential"
    CUMULATIVE = "cumulative"
    DIFFERENTIAL = "differential"
    RATIO = "ratio"
    CORRELATION = "correlation"
    REGRESSION = "regression"
    MACHINE_LEARNING = "machine_learning"


@dataclass
class MetricDefinition:
    """Custom metric definition"""
    metric_id: str
    name: str
    description: str
    metric_type: MetricType
    calculation_type: MetricCalculationType
    formula: str
    parameters: Dict[str, Any]
    aggregation: MetricAggregation
    window_size: int = 100
    update_frequency: int = 1
    tags: Dict[str, str] = None
    thresholds: Dict[str, float] = None
    enabled: bool = True
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.tags is None:
            self.tags = {}
        if self.thresholds is None:
            self.thresholds = {}


@dataclass
class MetricValue:
    """Metric value with metadata"""
    metric_id: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.labels is None:
            self.labels = {}
        if self.metadata is None:
            self.metadata = {}


@dataclass
class MetricCalculation:
    """Metric calculation result"""
    metric_id: str
    calculated_value: float
    raw_values: List[float]
    calculation_method: str
    confidence: float
    timestamp: datetime
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class MetricAlert:
    """Metric alert"""
    alert_id: str
    metric_id: str
    alert_type: str
    threshold: float
    current_value: float
    severity: str
    message: str
    timestamp: datetime
    resolved: bool = False
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class CustomMetricsSystem:
    """Advanced custom metrics system for AI model analysis"""
    
    def __init__(self, max_metrics: int = 1000, max_values_per_metric: int = 10000):
        self.max_metrics = max_metrics
        self.max_values_per_metric = max_values_per_metric
        
        self.metric_definitions: Dict[str, MetricDefinition] = {}
        self.metric_values: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_values_per_metric))
        self.metric_calculations: List[MetricCalculation] = []
        self.metric_alerts: List[MetricAlert] = []
        
        # Metric calculation functions
        self.calculation_functions = {
            MetricCalculationType.SIMPLE: self._calculate_simple_metric,
            MetricCalculationType.WINDOWED: self._calculate_windowed_metric,
            MetricCalculationType.ROLLING: self._calculate_rolling_metric,
            MetricCalculationType.EXPONENTIAL: self._calculate_exponential_metric,
            MetricCalculationType.CUMULATIVE: self._calculate_cumulative_metric,
            MetricCalculationType.DIFFERENTIAL: self._calculate_differential_metric,
            MetricCalculationType.RATIO: self._calculate_ratio_metric,
            MetricCalculationType.CORRELATION: self._calculate_correlation_metric,
            MetricCalculationType.REGRESSION: self._calculate_regression_metric,
            MetricCalculationType.MACHINE_LEARNING: self._calculate_ml_metric
        }
        
        # Aggregation functions
        self.aggregation_functions = {
            MetricAggregation.SUM: lambda x: sum(x),
            MetricAggregation.AVG: lambda x: statistics.mean(x) if x else 0,
            MetricAggregation.MIN: lambda x: min(x) if x else 0,
            MetricAggregation.MAX: lambda x: max(x) if x else 0,
            MetricAggregation.COUNT: lambda x: len(x),
            MetricAggregation.MEDIAN: lambda x: statistics.median(x) if x else 0,
            MetricAggregation.STDDEV: lambda x: statistics.stdev(x) if len(x) > 1 else 0,
            MetricAggregation.PERCENTILE: lambda x, p=95: np.percentile(x, p) if x else 0,
            MetricAggregation.RATE: self._calculate_rate,
            MetricAggregation.DELTA: self._calculate_delta
        }
    
    async def create_metric(self, 
                          name: str,
                          description: str,
                          metric_type: MetricType,
                          calculation_type: MetricCalculationType,
                          formula: str,
                          parameters: Dict[str, Any] = None,
                          aggregation: MetricAggregation = MetricAggregation.AVG,
                          window_size: int = 100,
                          update_frequency: int = 1,
                          tags: Dict[str, str] = None,
                          thresholds: Dict[str, float] = None) -> MetricDefinition:
        """Create custom metric"""
        try:
            metric_id = hashlib.md5(f"{name}_{metric_type}_{datetime.now()}".encode()).hexdigest()
            
            if parameters is None:
                parameters = {}
            if tags is None:
                tags = {}
            if thresholds is None:
                thresholds = {}
            
            metric = MetricDefinition(
                metric_id=metric_id,
                name=name,
                description=description,
                metric_type=metric_type,
                calculation_type=calculation_type,
                formula=formula,
                parameters=parameters,
                aggregation=aggregation,
                window_size=window_size,
                update_frequency=update_frequency,
                tags=tags,
                thresholds=thresholds
            )
            
            self.metric_definitions[metric_id] = metric
            
            logger.info(f"Created custom metric: {name}")
            
            return metric
            
        except Exception as e:
            logger.error(f"Error creating custom metric: {str(e)}")
            raise e
    
    async def record_metric_value(self, 
                                metric_id: str,
                                value: float,
                                labels: Dict[str, str] = None,
                                metadata: Dict[str, Any] = None) -> bool:
        """Record metric value"""
        try:
            if metric_id not in self.metric_definitions:
                raise ValueError(f"Metric {metric_id} not found")
            
            if labels is None:
                labels = {}
            if metadata is None:
                metadata = {}
            
            metric_value = MetricValue(
                metric_id=metric_id,
                value=value,
                timestamp=datetime.now(),
                labels=labels,
                metadata=metadata
            )
            
            self.metric_values[metric_id].append(metric_value)
            
            # Check thresholds and generate alerts
            await self._check_metric_thresholds(metric_id, value)
            
            logger.debug(f"Recorded metric value: {metric_id} = {value}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error recording metric value: {str(e)}")
            return False
    
    async def calculate_metric(self, 
                             metric_id: str,
                             calculation_params: Dict[str, Any] = None) -> MetricCalculation:
        """Calculate metric value"""
        try:
            if metric_id not in self.metric_definitions:
                raise ValueError(f"Metric {metric_id} not found")
            
            metric = self.metric_definitions[metric_id]
            values = [mv.value for mv in self.metric_values[metric_id]]
            
            if not values:
                return MetricCalculation(
                    metric_id=metric_id,
                    calculated_value=0.0,
                    raw_values=[],
                    calculation_method=metric.calculation_type.value,
                    confidence=0.0,
                    timestamp=datetime.now()
                )
            
            # Get calculation function
            calculation_func = self.calculation_functions.get(metric.calculation_type)
            if not calculation_func:
                raise ValueError(f"Unsupported calculation type: {metric.calculation_type}")
            
            # Calculate metric value
            calculated_value, confidence = await calculation_func(metric, values, calculation_params)
            
            # Create calculation result
            calculation = MetricCalculation(
                metric_id=metric_id,
                calculated_value=calculated_value,
                raw_values=values[-metric.window_size:] if len(values) > metric.window_size else values,
                calculation_method=metric.calculation_type.value,
                confidence=confidence,
                timestamp=datetime.now(),
                metadata={
                    "formula": metric.formula,
                    "parameters": metric.parameters,
                    "aggregation": metric.aggregation.value,
                    "window_size": metric.window_size
                }
            )
            
            self.metric_calculations.append(calculation)
            
            logger.info(f"Calculated metric {metric_id}: {calculated_value:.3f}")
            
            return calculation
            
        except Exception as e:
            logger.error(f"Error calculating metric: {str(e)}")
            raise e
    
    async def get_metric_values(self, 
                              metric_id: str,
                              time_range_hours: int = 24,
                              limit: int = 1000) -> List[MetricValue]:
        """Get metric values within time range"""
        try:
            if metric_id not in self.metric_values:
                return []
            
            cutoff_time = datetime.now() - timedelta(hours=time_range_hours)
            values = list(self.metric_values[metric_id])
            
            # Filter by time range
            filtered_values = [v for v in values if v.timestamp >= cutoff_time]
            
            # Limit results
            if limit > 0:
                filtered_values = filtered_values[-limit:]
            
            return filtered_values
            
        except Exception as e:
            logger.error(f"Error getting metric values: {str(e)}")
            return []
    
    async def get_metric_statistics(self, 
                                  metric_id: str,
                                  time_range_hours: int = 24) -> Dict[str, Any]:
        """Get metric statistics"""
        try:
            values = await self.get_metric_values(metric_id, time_range_hours)
            
            if not values:
                return {
                    "count": 0,
                    "mean": 0,
                    "std": 0,
                    "min": 0,
                    "max": 0,
                    "median": 0,
                    "percentiles": {}
                }
            
            numeric_values = [v.value for v in values]
            
            statistics = {
                "count": len(numeric_values),
                "mean": statistics.mean(numeric_values),
                "std": statistics.stdev(numeric_values) if len(numeric_values) > 1 else 0,
                "min": min(numeric_values),
                "max": max(numeric_values),
                "median": statistics.median(numeric_values),
                "percentiles": {
                    "25th": np.percentile(numeric_values, 25),
                    "75th": np.percentile(numeric_values, 75),
                    "90th": np.percentile(numeric_values, 90),
                    "95th": np.percentile(numeric_values, 95),
                    "99th": np.percentile(numeric_values, 99)
                }
            }
            
            return statistics
            
        except Exception as e:
            logger.error(f"Error getting metric statistics: {str(e)}")
            return {}
    
    async def create_performance_metrics(self, model_name: str) -> List[MetricDefinition]:
        """Create performance metrics for a model"""
        try:
            metrics = []
            
            # Accuracy metric
            accuracy_metric = await self.create_metric(
                name=f"{model_name}_accuracy",
                description=f"Accuracy metric for {model_name}",
                metric_type=MetricType.GAUGE,
                calculation_type=MetricCalculationType.ROLLING,
                formula="accuracy = correct_predictions / total_predictions",
                parameters={"window_size": 100},
                aggregation=MetricAggregation.AVG,
                tags={"model": model_name, "type": "performance"},
                thresholds={"warning": 0.8, "critical": 0.7}
            )
            metrics.append(accuracy_metric)
            
            # Response time metric
            response_time_metric = await self.create_metric(
                name=f"{model_name}_response_time",
                description=f"Response time metric for {model_name}",
                metric_type=MetricType.HISTOGRAM,
                calculation_type=MetricCalculationType.WINDOWED,
                formula="response_time = end_time - start_time",
                parameters={"window_size": 50},
                aggregation=MetricAggregation.PERCENTILE,
                tags={"model": model_name, "type": "performance"},
                thresholds={"warning": 5.0, "critical": 10.0}
            )
            metrics.append(response_time_metric)
            
            # Throughput metric
            throughput_metric = await self.create_metric(
                name=f"{model_name}_throughput",
                description=f"Throughput metric for {model_name}",
                metric_type=MetricType.COUNTER,
                calculation_type=MetricCalculationType.RATE,
                formula="throughput = requests_per_second",
                parameters={"time_window": 60},
                aggregation=MetricAggregation.RATE,
                tags={"model": model_name, "type": "performance"},
                thresholds={"warning": 50, "critical": 100}
            )
            metrics.append(throughput_metric)
            
            # Error rate metric
            error_rate_metric = await self.create_metric(
                name=f"{model_name}_error_rate",
                description=f"Error rate metric for {model_name}",
                metric_type=MetricType.GAUGE,
                calculation_type=MetricCalculationType.ROLLING,
                formula="error_rate = errors / total_requests",
                parameters={"window_size": 100},
                aggregation=MetricAggregation.AVG,
                tags={"model": model_name, "type": "reliability"},
                thresholds={"warning": 0.05, "critical": 0.1}
            )
            metrics.append(error_rate_metric)
            
            logger.info(f"Created {len(metrics)} performance metrics for {model_name}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error creating performance metrics: {str(e)}")
            return []
    
    async def create_business_metrics(self) -> List[MetricDefinition]:
        """Create business metrics"""
        try:
            metrics = []
            
            # Cost per prediction metric
            cost_metric = await self.create_metric(
                name="cost_per_prediction",
                description="Cost per prediction across all models",
                metric_type=MetricType.GAUGE,
                calculation_type=MetricCalculationType.ROLLING,
                formula="cost_per_prediction = total_cost / total_predictions",
                parameters={"window_size": 1000},
                aggregation=MetricAggregation.AVG,
                tags={"type": "business", "category": "cost"},
                thresholds={"warning": 0.01, "critical": 0.02}
            )
            metrics.append(cost_metric)
            
            # User satisfaction metric
            satisfaction_metric = await self.create_metric(
                name="user_satisfaction",
                description="User satisfaction score",
                metric_type=MetricType.GAUGE,
                calculation_type=MetricCalculationType.EXPONENTIAL,
                formula="satisfaction = weighted_average_of_ratings",
                parameters={"alpha": 0.1},
                aggregation=MetricAggregation.AVG,
                tags={"type": "business", "category": "satisfaction"},
                thresholds={"warning": 3.5, "critical": 3.0}
            )
            metrics.append(satisfaction_metric)
            
            # Revenue per model metric
            revenue_metric = await self.create_metric(
                name="revenue_per_model",
                description="Revenue generated per model",
                metric_type=MetricType.COUNTER,
                calculation_type=MetricCalculationType.CUMULATIVE,
                formula="revenue = sum(prediction_value * price)",
                parameters={},
                aggregation=MetricAggregation.SUM,
                tags={"type": "business", "category": "revenue"},
                thresholds={"warning": 1000, "critical": 500}
            )
            metrics.append(revenue_metric)
            
            logger.info(f"Created {len(metrics)} business metrics")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error creating business metrics: {str(e)}")
            return []
    
    async def get_metric_alerts(self, 
                              metric_id: str = None,
                              severity: str = None,
                              resolved: bool = None) -> List[MetricAlert]:
        """Get metric alerts"""
        try:
            alerts = self.metric_alerts.copy()
            
            # Apply filters
            if metric_id:
                alerts = [a for a in alerts if a.metric_id == metric_id]
            
            if severity:
                alerts = [a for a in alerts if a.severity == severity]
            
            if resolved is not None:
                alerts = [a for a in alerts if a.resolved == resolved]
            
            # Sort by timestamp (newest first)
            alerts.sort(key=lambda x: x.timestamp, reverse=True)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error getting metric alerts: {str(e)}")
            return []
    
    async def get_metrics_analytics(self, 
                                  time_range_hours: int = 24) -> Dict[str, Any]:
        """Get metrics analytics"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=time_range_hours)
            
            analytics = {
                "total_metrics": len(self.metric_definitions),
                "active_metrics": len([m for m in self.metric_definitions.values() if m.enabled]),
                "total_values": sum(len(values) for values in self.metric_values.values()),
                "total_calculations": len(self.metric_calculations),
                "total_alerts": len(self.metric_alerts),
                "active_alerts": len([a for a in self.metric_alerts if not a.resolved]),
                "metric_types": {},
                "calculation_types": {},
                "aggregation_types": {},
                "top_metrics": [],
                "alert_severity_distribution": {}
            }
            
            # Analyze metric types
            for metric in self.metric_definitions.values():
                metric_type = metric.metric_type.value
                if metric_type not in analytics["metric_types"]:
                    analytics["metric_types"][metric_type] = 0
                analytics["metric_types"][metric_type] += 1
            
            # Analyze calculation types
            for metric in self.metric_definitions.values():
                calc_type = metric.calculation_type.value
                if calc_type not in analytics["calculation_types"]:
                    analytics["calculation_types"][calc_type] = 0
                analytics["calculation_types"][calc_type] += 1
            
            # Analyze aggregation types
            for metric in self.metric_definitions.values():
                agg_type = metric.aggregation.value
                if agg_type not in analytics["aggregation_types"]:
                    analytics["aggregation_types"][agg_type] = 0
                analytics["aggregation_types"][agg_type] += 1
            
            # Find top metrics by value count
            metric_value_counts = [(metric_id, len(values)) for metric_id, values in self.metric_values.items()]
            metric_value_counts.sort(key=lambda x: x[1], reverse=True)
            
            for metric_id, count in metric_value_counts[:10]:
                metric = self.metric_definitions.get(metric_id)
                if metric:
                    analytics["top_metrics"].append({
                        "metric_id": metric_id,
                        "name": metric.name,
                        "value_count": count,
                        "type": metric.metric_type.value
                    })
            
            # Analyze alert severity distribution
            for alert in self.metric_alerts:
                severity = alert.severity
                if severity not in analytics["alert_severity_distribution"]:
                    analytics["alert_severity_distribution"][severity] = 0
                analytics["alert_severity_distribution"][severity] += 1
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error getting metrics analytics: {str(e)}")
            return {"error": str(e)}
    
    # Private helper methods
    async def _calculate_simple_metric(self, 
                                     metric: MetricDefinition, 
                                     values: List[float], 
                                     params: Dict[str, Any] = None) -> Tuple[float, float]:
        """Calculate simple metric"""
        try:
            if not values:
                return 0.0, 0.0
            
            # Apply aggregation
            agg_func = self.aggregation_functions.get(metric.aggregation)
            if not agg_func:
                agg_func = self.aggregation_functions[MetricAggregation.AVG]
            
            if metric.aggregation == MetricAggregation.PERCENTILE:
                percentile = metric.parameters.get("percentile", 95)
                calculated_value = agg_func(values, percentile)
            else:
                calculated_value = agg_func(values)
            
            # Calculate confidence based on data quality
            confidence = min(1.0, len(values) / 100.0)  # Simple confidence based on sample size
            
            return calculated_value, confidence
            
        except Exception as e:
            logger.error(f"Error calculating simple metric: {str(e)}")
            return 0.0, 0.0
    
    async def _calculate_windowed_metric(self, 
                                       metric: MetricDefinition, 
                                       values: List[float], 
                                       params: Dict[str, Any] = None) -> Tuple[float, float]:
        """Calculate windowed metric"""
        try:
            if not values:
                return 0.0, 0.0
            
            window_size = metric.window_size
            window_values = values[-window_size:] if len(values) > window_size else values
            
            return await self._calculate_simple_metric(metric, window_values, params)
            
        except Exception as e:
            logger.error(f"Error calculating windowed metric: {str(e)}")
            return 0.0, 0.0
    
    async def _calculate_rolling_metric(self, 
                                      metric: MetricDefinition, 
                                      values: List[float], 
                                      params: Dict[str, Any] = None) -> Tuple[float, float]:
        """Calculate rolling metric"""
        try:
            if not values:
                return 0.0, 0.0
            
            window_size = metric.window_size
            if len(values) < window_size:
                return await self._calculate_simple_metric(metric, values, params)
            
            # Calculate rolling average
            rolling_values = []
            for i in range(len(values) - window_size + 1):
                window = values[i:i + window_size]
                rolling_values.append(statistics.mean(window))
            
            return await self._calculate_simple_metric(metric, rolling_values, params)
            
        except Exception as e:
            logger.error(f"Error calculating rolling metric: {str(e)}")
            return 0.0, 0.0
    
    async def _calculate_exponential_metric(self, 
                                          metric: MetricDefinition, 
                                          values: List[float], 
                                          params: Dict[str, Any] = None) -> Tuple[float, float]:
        """Calculate exponential metric"""
        try:
            if not values:
                return 0.0, 0.0
            
            alpha = metric.parameters.get("alpha", 0.1)
            exponential_avg = values[0] if values else 0.0
            
            for value in values[1:]:
                exponential_avg = alpha * value + (1 - alpha) * exponential_avg
            
            confidence = min(1.0, len(values) / 50.0)
            
            return exponential_avg, confidence
            
        except Exception as e:
            logger.error(f"Error calculating exponential metric: {str(e)}")
            return 0.0, 0.0
    
    async def _calculate_cumulative_metric(self, 
                                         metric: MetricDefinition, 
                                         values: List[float], 
                                         params: Dict[str, Any] = None) -> Tuple[float, float]:
        """Calculate cumulative metric"""
        try:
            if not values:
                return 0.0, 0.0
            
            cumulative_value = sum(values)
            confidence = min(1.0, len(values) / 1000.0)
            
            return cumulative_value, confidence
            
        except Exception as e:
            logger.error(f"Error calculating cumulative metric: {str(e)}")
            return 0.0, 0.0
    
    async def _calculate_differential_metric(self, 
                                           metric: MetricDefinition, 
                                           values: List[float], 
                                           params: Dict[str, Any] = None) -> Tuple[float, float]:
        """Calculate differential metric"""
        try:
            if len(values) < 2:
                return 0.0, 0.0
            
            # Calculate difference between first and last value
            differential_value = values[-1] - values[0]
            confidence = min(1.0, len(values) / 100.0)
            
            return differential_value, confidence
            
        except Exception as e:
            logger.error(f"Error calculating differential metric: {str(e)}")
            return 0.0, 0.0
    
    async def _calculate_ratio_metric(self, 
                                    metric: MetricDefinition, 
                                    values: List[float], 
                                    params: Dict[str, Any] = None) -> Tuple[float, float]:
        """Calculate ratio metric"""
        try:
            if len(values) < 2:
                return 0.0, 0.0
            
            # Calculate ratio (simplified - in practice would need more complex logic)
            numerator = values[-1] if values else 0
            denominator = values[0] if values else 1
            
            if denominator == 0:
                return 0.0, 0.0
            
            ratio_value = numerator / denominator
            confidence = min(1.0, len(values) / 100.0)
            
            return ratio_value, confidence
            
        except Exception as e:
            logger.error(f"Error calculating ratio metric: {str(e)}")
            return 0.0, 0.0
    
    async def _calculate_correlation_metric(self, 
                                          metric: MetricDefinition, 
                                          values: List[float], 
                                          params: Dict[str, Any] = None) -> Tuple[float, float]:
        """Calculate correlation metric"""
        try:
            if len(values) < 3:
                return 0.0, 0.0
            
            # Calculate autocorrelation (simplified)
            x = list(range(len(values)))
            correlation, _ = stats.pearsonr(x, values)
            
            confidence = min(1.0, len(values) / 100.0)
            
            return correlation, confidence
            
        except Exception as e:
            logger.error(f"Error calculating correlation metric: {str(e)}")
            return 0.0, 0.0
    
    async def _calculate_regression_metric(self, 
                                         metric: MetricDefinition, 
                                         values: List[float], 
                                         params: Dict[str, Any] = None) -> Tuple[float, float]:
        """Calculate regression metric"""
        try:
            if len(values) < 3:
                return 0.0, 0.0
            
            # Simple linear regression slope
            x = np.array(range(len(values)))
            y = np.array(values)
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            confidence = r_value ** 2  # R-squared as confidence
            
            return slope, confidence
            
        except Exception as e:
            logger.error(f"Error calculating regression metric: {str(e)}")
            return 0.0, 0.0
    
    async def _calculate_ml_metric(self, 
                                 metric: MetricDefinition, 
                                 values: List[float], 
                                 params: Dict[str, Any] = None) -> Tuple[float, float]:
        """Calculate machine learning metric"""
        try:
            if len(values) < 10:
                return 0.0, 0.0
            
            # Simple trend prediction using linear regression
            x = np.array(range(len(values)))
            y = np.array(values)
            
            # Fit linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            # Predict next value
            next_x = len(values)
            predicted_value = slope * next_x + intercept
            
            confidence = r_value ** 2
            
            return predicted_value, confidence
            
        except Exception as e:
            logger.error(f"Error calculating ML metric: {str(e)}")
            return 0.0, 0.0
    
    def _calculate_rate(self, values: List[float]) -> float:
        """Calculate rate metric"""
        try:
            if len(values) < 2:
                return 0.0
            
            # Calculate rate of change per unit time
            time_diff = 1.0  # Assuming 1 second intervals
            value_diff = values[-1] - values[0]
            
            return value_diff / (len(values) * time_diff)
            
        except Exception as e:
            logger.error(f"Error calculating rate: {str(e)}")
            return 0.0
    
    def _calculate_delta(self, values: List[float]) -> float:
        """Calculate delta metric"""
        try:
            if len(values) < 2:
                return 0.0
            
            return values[-1] - values[-2]
            
        except Exception as e:
            logger.error(f"Error calculating delta: {str(e)}")
            return 0.0
    
    async def _check_metric_thresholds(self, metric_id: str, value: float) -> None:
        """Check metric thresholds and generate alerts"""
        try:
            metric = self.metric_definitions.get(metric_id)
            if not metric or not metric.thresholds:
                return
            
            for threshold_name, threshold_value in metric.thresholds.items():
                alert_generated = False
                severity = "warning"
                
                if threshold_name == "critical" and value <= threshold_value:
                    alert_generated = True
                    severity = "critical"
                elif threshold_name == "warning" and value <= threshold_value:
                    alert_generated = True
                    severity = "warning"
                
                if alert_generated:
                    # Check if alert already exists
                    existing_alert = next(
                        (a for a in self.metric_alerts 
                         if a.metric_id == metric_id and a.threshold == threshold_value and not a.resolved),
                        None
                    )
                    
                    if not existing_alert:
                        alert = MetricAlert(
                            alert_id=hashlib.md5(f"{metric_id}_{threshold_name}_{datetime.now()}".encode()).hexdigest(),
                            metric_id=metric_id,
                            alert_type=threshold_name,
                            threshold=threshold_value,
                            current_value=value,
                            severity=severity,
                            message=f"Metric {metric.name} {threshold_name} threshold exceeded: {value:.3f} <= {threshold_value:.3f}",
                            timestamp=datetime.now()
                        )
                        
                        self.metric_alerts.append(alert)
                        
                        logger.warning(f"Generated {severity} alert for metric {metric_id}: {alert.message}")
            
        except Exception as e:
            logger.error(f"Error checking metric thresholds: {str(e)}")


# Global custom metrics system instance
_custom_metrics_system: Optional[CustomMetricsSystem] = None


def get_custom_metrics_system(max_metrics: int = 1000, max_values_per_metric: int = 10000) -> CustomMetricsSystem:
    """Get or create global custom metrics system instance"""
    global _custom_metrics_system
    if _custom_metrics_system is None:
        _custom_metrics_system = CustomMetricsSystem(max_metrics, max_values_per_metric)
    return _custom_metrics_system


# Example usage
async def main():
    """Example usage of the custom metrics system"""
    system = get_custom_metrics_system()
    
    # Create performance metrics for a model
    performance_metrics = await system.create_performance_metrics("gpt-4")
    print(f"Created {len(performance_metrics)} performance metrics")
    
    # Create business metrics
    business_metrics = await system.create_business_metrics()
    print(f"Created {len(business_metrics)} business metrics")
    
    # Record some metric values
    for metric in performance_metrics:
        for i in range(10):
            value = 0.7 + np.random.normal(0, 0.1)
            await system.record_metric_value(metric.metric_id, value)
    
    # Calculate metrics
    for metric in performance_metrics:
        calculation = await system.calculate_metric(metric.metric_id)
        print(f"Calculated {metric.name}: {calculation.calculated_value:.3f}")
    
    # Get metric statistics
    for metric in performance_metrics:
        stats = await system.get_metric_statistics(metric.metric_id)
        print(f"Statistics for {metric.name}: {stats.get('mean', 0):.3f} ± {stats.get('std', 0):.3f}")
    
    # Get metric alerts
    alerts = await system.get_metric_alerts()
    print(f"Found {len(alerts)} metric alerts")
    
    # Get metrics analytics
    analytics = await system.get_metrics_analytics()
    print(f"Metrics analytics: {analytics.get('total_metrics', 0)} metrics, {analytics.get('total_values', 0)} values")


if __name__ == "__main__":
    asyncio.run(main())

























