"""
Advanced AI History Analyzer and Model Comparison System
======================================================

This module provides comprehensive analysis of AI model performance over time,
including historical trends, model comparisons, and predictive analytics.

Features:
- Historical performance tracking
- Model comparison and benchmarking
- Trend analysis and forecasting
- Performance degradation detection
- Model evolution tracking
- Predictive analytics
- Comprehensive reporting
"""

import asyncio
import json
import logging
import statistics
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import numpy as np
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Types of AI models"""
    TEXT_GENERATION = "text_generation"
    IMAGE_GENERATION = "image_generation"
    CODE_GENERATION = "code_generation"
    ANALYSIS = "analysis"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"


class PerformanceMetric(Enum):
    """Performance metrics for AI models"""
    QUALITY_SCORE = "quality_score"
    RESPONSE_TIME = "response_time"
    TOKEN_EFFICIENCY = "token_efficiency"
    COST_EFFICIENCY = "cost_efficiency"
    ACCURACY = "accuracy"
    COHERENCE = "coherence"
    RELEVANCE = "relevance"
    CREATIVITY = "creativity"


@dataclass
class ModelPerformance:
    """Represents performance data for a specific model"""
    model_name: str
    model_type: ModelType
    timestamp: datetime
    metric: PerformanceMetric
    value: float
    context: Dict[str, Any] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ModelComparison:
    """Represents a comparison between models"""
    model_a: str
    model_b: str
    metric: PerformanceMetric
    comparison_score: float  # -1 to 1, where 1 means A is better
    confidence: float
    sample_size: int
    timestamp: datetime
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}


@dataclass
class TrendAnalysis:
    """Represents trend analysis results"""
    model_name: str
    metric: PerformanceMetric
    trend_direction: str  # "improving", "declining", "stable"
    trend_strength: float  # 0 to 1
    confidence: float
    forecast: List[Tuple[datetime, float]] = None
    anomalies: List[Tuple[datetime, float]] = None
    
    def __post_init__(self):
        if self.forecast is None:
            self.forecast = []
        if self.anomalies is None:
            self.anomalies = []


class AIHistoryAnalyzer:
    """Advanced AI history analyzer with comprehensive tracking and analysis"""
    
    def __init__(self, max_history_days: int = 365):
        self.max_history_days = max_history_days
        self.performance_history: Dict[str, List[ModelPerformance]] = defaultdict(list)
        self.model_comparisons: List[ModelComparison] = []
        self.trend_analyses: Dict[str, List[TrendAnalysis]] = defaultdict(list)
        
        # Performance tracking
        self.performance_stats = {
            "total_measurements": 0,
            "models_tracked": set(),
            "metrics_tracked": set(),
            "last_analysis": None
        }
        
        # Cache for computed results
        self.analysis_cache = {}
        self.cache_ttl = 3600  # 1 hour
    
    def record_performance(self, 
                          model_name: str,
                          model_type: ModelType,
                          metric: PerformanceMetric,
                          value: float,
                          context: Dict[str, Any] = None,
                          metadata: Dict[str, Any] = None) -> bool:
        """Record performance data for a model"""
        try:
            performance = ModelPerformance(
                model_name=model_name,
                model_type=model_type,
                timestamp=datetime.now(),
                metric=metric,
                value=value,
                context=context or {},
                metadata=metadata or {}
            )
            
            # Store performance data
            key = f"{model_name}_{metric.value}"
            self.performance_history[key].append(performance)
            
            # Update stats
            self.performance_stats["total_measurements"] += 1
            self.performance_stats["models_tracked"].add(model_name)
            self.performance_stats["metrics_tracked"].add(metric.value)
            
            # Clean old data
            self._cleanup_old_data()
            
            # Invalidate cache
            self._invalidate_cache()
            
            logger.debug(f"Recorded performance: {model_name} - {metric.value}: {value}")
            return True
            
        except Exception as e:
            logger.error(f"Error recording performance: {str(e)}")
            return False
    
    def get_model_performance(self, 
                            model_name: str,
                            metric: PerformanceMetric,
                            days: int = 30) -> List[ModelPerformance]:
        """Get performance data for a specific model and metric"""
        try:
            key = f"{model_name}_{metric.value}"
            cutoff_date = datetime.now() - timedelta(days=days)
            
            return [
                p for p in self.performance_history[key]
                if p.timestamp >= cutoff_date
            ]
            
        except Exception as e:
            logger.error(f"Error getting model performance: {str(e)}")
            return []
    
    def compare_models(self, 
                      model_a: str,
                      model_b: str,
                      metric: PerformanceMetric,
                      days: int = 30) -> Optional[ModelComparison]:
        """Compare two models on a specific metric"""
        try:
            # Get performance data for both models
            performance_a = self.get_model_performance(model_a, metric, days)
            performance_b = self.get_model_performance(model_b, metric, days)
            
            if not performance_a or not performance_b:
                logger.warning(f"Insufficient data for comparison: {model_a} vs {model_b}")
                return None
            
            # Calculate comparison metrics
            values_a = [p.value for p in performance_a]
            values_b = [p.value for p in performance_b]
            
            mean_a = statistics.mean(values_a)
            mean_b = statistics.mean(values_b)
            
            # Calculate comparison score (-1 to 1)
            if mean_a == mean_b:
                comparison_score = 0.0
            else:
                comparison_score = (mean_a - mean_b) / max(mean_a, mean_b)
            
            # Calculate confidence based on sample size and variance
            sample_size = min(len(values_a), len(values_b))
            variance_a = statistics.variance(values_a) if len(values_a) > 1 else 0
            variance_b = statistics.variance(values_b) if len(values_b) > 1 else 0
            
            # Higher sample size and lower variance = higher confidence
            confidence = min(1.0, sample_size / 100) * (1 - min(1.0, (variance_a + variance_b) / 2))
            
            comparison = ModelComparison(
                model_a=model_a,
                model_b=model_b,
                metric=metric,
                comparison_score=comparison_score,
                confidence=confidence,
                sample_size=sample_size,
                timestamp=datetime.now(),
                details={
                    "mean_a": mean_a,
                    "mean_b": mean_b,
                    "variance_a": variance_a,
                    "variance_b": variance_b,
                    "sample_size_a": len(values_a),
                    "sample_size_b": len(values_b)
                }
            )
            
            # Store comparison
            self.model_comparisons.append(comparison)
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing models: {str(e)}")
            return None
    
    def analyze_trends(self, 
                      model_name: str,
                      metric: PerformanceMetric,
                      days: int = 90) -> Optional[TrendAnalysis]:
        """Analyze performance trends for a model"""
        try:
            # Get performance data
            performance_data = self.get_model_performance(model_name, metric, days)
            
            if len(performance_data) < 10:  # Need at least 10 data points
                logger.warning(f"Insufficient data for trend analysis: {model_name}")
                return None
            
            # Sort by timestamp
            performance_data.sort(key=lambda x: x.timestamp)
            
            # Extract values and timestamps
            timestamps = [p.timestamp for p in performance_data]
            values = [p.value for p in performance_data]
            
            # Calculate trend using linear regression
            trend_direction, trend_strength, confidence = self._calculate_trend(timestamps, values)
            
            # Detect anomalies
            anomalies = self._detect_anomalies(timestamps, values)
            
            # Generate forecast
            forecast = self._generate_forecast(timestamps, values, days=7)
            
            trend_analysis = TrendAnalysis(
                model_name=model_name,
                metric=metric,
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                confidence=confidence,
                forecast=forecast,
                anomalies=anomalies
            )
            
            # Store trend analysis
            key = f"{model_name}_{metric.value}"
            self.trend_analyses[key].append(trend_analysis)
            
            return trend_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing trends: {str(e)}")
            return None
    
    def _calculate_trend(self, timestamps: List[datetime], values: List[float]) -> Tuple[str, float, float]:
        """Calculate trend direction, strength, and confidence"""
        try:
            # Convert timestamps to numeric values (days since first timestamp)
            start_time = timestamps[0]
            x_values = [(t - start_time).total_seconds() / 86400 for t in timestamps]  # Days
            
            # Simple linear regression
            n = len(x_values)
            sum_x = sum(x_values)
            sum_y = sum(values)
            sum_xy = sum(x * y for x, y in zip(x_values, values))
            sum_x2 = sum(x * x for x in x_values)
            
            # Calculate slope
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            
            # Determine trend direction
            if slope > 0.01:
                trend_direction = "improving"
            elif slope < -0.01:
                trend_direction = "declining"
            else:
                trend_direction = "stable"
            
            # Calculate trend strength (0 to 1)
            trend_strength = min(1.0, abs(slope) * 100)  # Normalize slope
            
            # Calculate confidence based on R-squared
            y_mean = sum_y / n
            ss_tot = sum((y - y_mean) ** 2 for y in values)
            ss_res = sum((y - (slope * x + (sum_y - slope * sum_x) / n)) ** 2 
                        for x, y in zip(x_values, values))
            
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            confidence = max(0.0, min(1.0, r_squared))
            
            return trend_direction, trend_strength, confidence
            
        except Exception as e:
            logger.error(f"Error calculating trend: {str(e)}")
            return "stable", 0.0, 0.0
    
    def _detect_anomalies(self, timestamps: List[datetime], values: List[float]) -> List[Tuple[datetime, float]]:
        """Detect anomalies in performance data"""
        try:
            if len(values) < 5:
                return []
            
            # Calculate mean and standard deviation
            mean_val = statistics.mean(values)
            std_val = statistics.stdev(values) if len(values) > 1 else 0
            
            if std_val == 0:
                return []
            
            # Find values that are more than 2 standard deviations from mean
            anomalies = []
            for timestamp, value in zip(timestamps, values):
                z_score = abs(value - mean_val) / std_val
                if z_score > 2.0:  # 2-sigma rule
                    anomalies.append((timestamp, value))
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {str(e)}")
            return []
    
    def _generate_forecast(self, 
                          timestamps: List[datetime], 
                          values: List[float], 
                          days: int = 7) -> List[Tuple[datetime, float]]:
        """Generate simple forecast for future values"""
        try:
            if len(values) < 3:
                return []
            
            # Simple linear trend forecast
            start_time = timestamps[0]
            x_values = [(t - start_time).total_seconds() / 86400 for t in timestamps]
            
            # Calculate slope and intercept
            n = len(x_values)
            sum_x = sum(x_values)
            sum_y = sum(values)
            sum_xy = sum(x * y for x, y in zip(x_values, values))
            sum_x2 = sum(x * x for x in x_values)
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            intercept = (sum_y - slope * sum_x) / n
            
            # Generate forecast
            forecast = []
            last_timestamp = timestamps[-1]
            
            for i in range(1, days + 1):
                future_time = last_timestamp + timedelta(days=i)
                future_x = (future_time - start_time).total_seconds() / 86400
                predicted_value = slope * future_x + intercept
                forecast.append((future_time, predicted_value))
            
            return forecast
            
        except Exception as e:
            logger.error(f"Error generating forecast: {str(e)}")
            return []
    
    def get_performance_summary(self, model_name: str, days: int = 30) -> Dict[str, Any]:
        """Get comprehensive performance summary for a model"""
        try:
            summary = {
                "model_name": model_name,
                "analysis_period_days": days,
                "metrics": {},
                "overall_score": 0.0,
                "trends": {},
                "recommendations": []
            }
            
            # Analyze each metric
            for metric in PerformanceMetric:
                performance_data = self.get_model_performance(model_name, metric, days)
                
                if not performance_data:
                    continue
                
                values = [p.value for p in performance_data]
                
                metric_summary = {
                    "count": len(values),
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0,
                    "min": min(values),
                    "max": max(values),
                    "latest": values[-1] if values else 0
                }
                
                summary["metrics"][metric.value] = metric_summary
                
                # Analyze trends
                trend_analysis = self.analyze_trends(model_name, metric, days)
                if trend_analysis:
                    summary["trends"][metric.value] = {
                        "direction": trend_analysis.trend_direction,
                        "strength": trend_analysis.trend_strength,
                        "confidence": trend_analysis.confidence
                    }
            
            # Calculate overall score (weighted average of normalized metrics)
            if summary["metrics"]:
                weights = {
                    "quality_score": 0.3,
                    "response_time": 0.2,
                    "token_efficiency": 0.2,
                    "cost_efficiency": 0.15,
                    "accuracy": 0.15
                }
                
                weighted_score = 0.0
                total_weight = 0.0
                
                for metric_name, weight in weights.items():
                    if metric_name in summary["metrics"]:
                        # Normalize metric value (0-1 scale)
                        metric_data = summary["metrics"][metric_name]
                        normalized_value = min(1.0, max(0.0, metric_data["mean"]))
                        weighted_score += normalized_value * weight
                        total_weight += weight
                
                if total_weight > 0:
                    summary["overall_score"] = weighted_score / total_weight
            
            # Generate recommendations
            summary["recommendations"] = self._generate_recommendations(summary)
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating performance summary: {str(e)}")
            return {"error": str(e)}
    
    def _generate_recommendations(self, summary: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on performance analysis"""
        recommendations = []
        
        try:
            # Check overall score
            overall_score = summary.get("overall_score", 0)
            if overall_score < 0.6:
                recommendations.append("Overall performance is below optimal. Consider model fine-tuning or replacement.")
            
            # Check trends
            trends = summary.get("trends", {})
            for metric, trend_data in trends.items():
                if trend_data["direction"] == "declining" and trend_data["confidence"] > 0.7:
                    recommendations.append(f"{metric} is declining. Investigate potential causes.")
                elif trend_data["direction"] == "improving" and trend_data["confidence"] > 0.7:
                    recommendations.append(f"{metric} is improving. Consider scaling up usage.")
            
            # Check specific metrics
            metrics = summary.get("metrics", {})
            
            if "response_time" in metrics:
                avg_response_time = metrics["response_time"]["mean"]
                if avg_response_time > 5.0:  # 5 seconds
                    recommendations.append("Response time is high. Consider optimization or model change.")
            
            if "cost_efficiency" in metrics:
                cost_efficiency = metrics["cost_efficiency"]["mean"]
                if cost_efficiency < 0.5:
                    recommendations.append("Cost efficiency is low. Consider more efficient models.")
            
            if "quality_score" in metrics:
                quality_score = metrics["quality_score"]["mean"]
                if quality_score < 0.7:
                    recommendations.append("Quality score is below threshold. Review model parameters.")
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            recommendations.append("Error generating recommendations")
        
        return recommendations
    
    def get_model_rankings(self, metric: PerformanceMetric, days: int = 30) -> List[Dict[str, Any]]:
        """Get rankings of all models for a specific metric"""
        try:
            rankings = []
            
            for model_name in self.performance_stats["models_tracked"]:
                performance_data = self.get_model_performance(model_name, metric, days)
                
                if not performance_data:
                    continue
                
                values = [p.value for p in performance_data]
                mean_value = statistics.mean(values)
                
                rankings.append({
                    "model_name": model_name,
                    "metric": metric.value,
                    "mean_value": mean_value,
                    "sample_size": len(values),
                    "confidence": min(1.0, len(values) / 50)  # More samples = higher confidence
                })
            
            # Sort by mean value (descending)
            rankings.sort(key=lambda x: x["mean_value"], reverse=True)
            
            # Add rank
            for i, ranking in enumerate(rankings):
                ranking["rank"] = i + 1
            
            return rankings
            
        except Exception as e:
            logger.error(f"Error generating model rankings: {str(e)}")
            return []
    
    def get_comprehensive_report(self, days: int = 30) -> Dict[str, Any]:
        """Generate comprehensive analysis report"""
        try:
            report = {
                "report_date": datetime.now().isoformat(),
                "analysis_period_days": days,
                "summary": {
                    "total_models": len(self.performance_stats["models_tracked"]),
                    "total_measurements": self.performance_stats["total_measurements"],
                    "metrics_analyzed": list(self.performance_stats["metrics_tracked"])
                },
                "model_performances": {},
                "model_rankings": {},
                "trend_analyses": {},
                "comparisons": [],
                "insights": []
            }
            
            # Analyze each model
            for model_name in self.performance_stats["models_tracked"]:
                report["model_performances"][model_name] = self.get_performance_summary(model_name, days)
            
            # Generate rankings for each metric
            for metric in PerformanceMetric:
                rankings = self.get_model_rankings(metric, days)
                if rankings:
                    report["model_rankings"][metric.value] = rankings
            
            # Analyze trends
            for model_name in self.performance_stats["models_tracked"]:
                model_trends = {}
                for metric in PerformanceMetric:
                    trend = self.analyze_trends(model_name, metric, days)
                    if trend:
                        model_trends[metric.value] = {
                            "direction": trend.trend_direction,
                            "strength": trend.trend_strength,
                            "confidence": trend.confidence
                        }
                
                if model_trends:
                    report["trend_analyses"][model_name] = model_trends
            
            # Generate insights
            report["insights"] = self._generate_insights(report)
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating comprehensive report: {str(e)}")
            return {"error": str(e)}
    
    def _generate_insights(self, report: Dict[str, Any]) -> List[str]:
        """Generate insights from the comprehensive report"""
        insights = []
        
        try:
            # Find best performing model
            best_models = {}
            for metric, rankings in report.get("model_rankings", {}).items():
                if rankings:
                    best_model = rankings[0]
                    best_models[metric] = best_model["model_name"]
            
            if best_models:
                insights.append(f"Best performing models: {best_models}")
            
            # Find models with improving trends
            improving_models = []
            for model, trends in report.get("trend_analyses", {}).items():
                improving_metrics = [metric for metric, trend in trends.items() 
                                   if trend["direction"] == "improving" and trend["confidence"] > 0.7]
                if improving_metrics:
                    improving_models.append(f"{model}: {improving_metrics}")
            
            if improving_models:
                insights.append(f"Models with improving trends: {improving_models}")
            
            # Find models with declining trends
            declining_models = []
            for model, trends in report.get("trend_analyses", {}).items():
                declining_metrics = [metric for metric, trend in trends.items() 
                                   if trend["direction"] == "declining" and trend["confidence"] > 0.7]
                if declining_metrics:
                    declining_models.append(f"{model}: {declining_metrics}")
            
            if declining_models:
                insights.append(f"Models with declining trends: {declining_models}")
            
            # Performance distribution insights
            total_models = report["summary"]["total_models"]
            if total_models > 1:
                insights.append(f"Analyzing {total_models} models across {len(report['summary']['metrics_analyzed'])} metrics")
            
        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")
            insights.append("Error generating insights")
        
        return insights
    
    def _cleanup_old_data(self):
        """Clean up old performance data"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.max_history_days)
            
            for key in list(self.performance_history.keys()):
                self.performance_history[key] = [
                    p for p in self.performance_history[key]
                    if p.timestamp >= cutoff_date
                ]
                
                # Remove empty entries
                if not self.performance_history[key]:
                    del self.performance_history[key]
            
            # Clean up old comparisons (keep last 1000)
            if len(self.model_comparisons) > 1000:
                self.model_comparisons = self.model_comparisons[-1000:]
            
            # Clean up old trend analyses (keep last 100 per model)
            for key in list(self.trend_analyses.keys()):
                if len(self.trend_analyses[key]) > 100:
                    self.trend_analyses[key] = self.trend_analyses[key][-100:]
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {str(e)}")
    
    def _invalidate_cache(self):
        """Invalidate analysis cache"""
        self.analysis_cache.clear()
    
    def export_data(self, format: str = "json") -> Union[str, Dict[str, Any]]:
        """Export all analysis data"""
        try:
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "performance_history": {
                    key: [asdict(p) for p in data]
                    for key, data in self.performance_history.items()
                },
                "model_comparisons": [asdict(c) for c in self.model_comparisons],
                "trend_analyses": {
                    key: [asdict(t) for t in data]
                    for key, data in self.trend_analyses.items()
                },
                "performance_stats": {
                    "total_measurements": self.performance_stats["total_measurements"],
                    "models_tracked": list(self.performance_stats["models_tracked"]),
                    "metrics_tracked": list(self.performance_stats["metrics_tracked"]),
                    "last_analysis": self.performance_stats["last_analysis"]
                }
            }
            
            if format == "json":
                return json.dumps(export_data, indent=2, default=str)
            else:
                return export_data
                
        except Exception as e:
            logger.error(f"Error exporting data: {str(e)}")
            return {"error": str(e)}


# Global analyzer instance
_analyzer: Optional[AIHistoryAnalyzer] = None


def get_ai_history_analyzer(max_history_days: int = 365) -> AIHistoryAnalyzer:
    """Get or create global AI history analyzer"""
    global _analyzer
    if _analyzer is None:
        _analyzer = AIHistoryAnalyzer(max_history_days)
    return _analyzer


# Example usage and testing
async def main():
    """Example usage of the AI history analyzer"""
    analyzer = get_ai_history_analyzer()
    
    # Record some sample performance data
    models = ["gpt-4", "claude-3", "gemini-pro"]
    metrics = [PerformanceMetric.QUALITY_SCORE, PerformanceMetric.RESPONSE_TIME, PerformanceMetric.COST_EFFICIENCY]
    
    # Simulate performance data over time
    for i in range(100):
        for model in models:
            for metric in metrics:
                # Simulate varying performance
                if metric == PerformanceMetric.QUALITY_SCORE:
                    value = 0.7 + (i * 0.001) + np.random.normal(0, 0.05)
                elif metric == PerformanceMetric.RESPONSE_TIME:
                    value = 2.0 - (i * 0.01) + np.random.normal(0, 0.2)
                else:  # COST_EFFICIENCY
                    value = 0.8 + (i * 0.002) + np.random.normal(0, 0.03)
                
                analyzer.record_performance(
                    model_name=model,
                    model_type=ModelType.TEXT_GENERATION,
                    metric=metric,
                    value=max(0, min(1, value))  # Clamp to 0-1
                )
    
    # Generate comprehensive report
    report = analyzer.get_comprehensive_report(days=30)
    print("Comprehensive Report:")
    print(json.dumps(report, indent=2, default=str))
    
    # Compare models
    comparison = analyzer.compare_models("gpt-4", "claude-3", PerformanceMetric.QUALITY_SCORE)
    if comparison:
        print(f"\nModel Comparison: {comparison.model_a} vs {comparison.model_b}")
        print(f"Comparison Score: {comparison.comparison_score:.3f}")
        print(f"Confidence: {comparison.confidence:.3f}")


if __name__ == "__main__":
    asyncio.run(main())