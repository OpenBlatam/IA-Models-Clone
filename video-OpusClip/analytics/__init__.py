#!/usr/bin/env python3
"""
Analytics Package

Advanced analytics system for the Video-OpusClip API.
"""

from .analytics import (
    AnalyticsManager,
    RealTimeAnalytics,
    PredictiveAnalytics,
    AnomalyDetector,
    PerformanceAnalyzer,
    UserBehaviorAnalyzer,
    ContentAnalyzer,
    analytics_manager
)

from .time_series_analytics import (
    TimeSeriesType,
    AnomalyType,
    ForecastMethod,
    TimeSeriesData,
    TimeSeries,
    Anomaly,
    Forecast,
    TimeSeriesPattern,
    TimeSeriesAnalyticsManager,
    time_series_analytics_manager
)

from .multi_dimensional_analytics import (
    DimensionalityReductionMethod,
    ClusteringMethod,
    VisualizationMethod,
    MultiDimensionalDataset,
    DimensionalityReductionResult,
    ClusteringResult,
    CorrelationAnalysis,
    MultiDimensionalForecast,
    MultiDimensionalAnalyticsManager,
    multi_dimensional_analytics_manager
)

__all__ = [
    'AnalyticsManager',
    'RealTimeAnalytics',
    'PredictiveAnalytics',
    'AnomalyDetector',
    'PerformanceAnalyzer',
    'UserBehaviorAnalyzer',
    'ContentAnalyzer',
    'analytics_manager',
    'TimeSeriesType',
    'AnomalyType',
    'ForecastMethod',
    'TimeSeriesData',
    'TimeSeries',
    'Anomaly',
    'Forecast',
    'TimeSeriesPattern',
    'TimeSeriesAnalyticsManager',
    'time_series_analytics_manager',
    'DimensionalityReductionMethod',
    'ClusteringMethod',
    'VisualizationMethod',
    'MultiDimensionalDataset',
    'DimensionalityReductionResult',
    'ClusteringResult',
    'CorrelationAnalysis',
    'MultiDimensionalForecast',
    'MultiDimensionalAnalyticsManager',
    'multi_dimensional_analytics_manager'
]