"""
Content Performance Analytics Engine - Advanced Content Performance Tracking
======================================================================

This module provides comprehensive content performance analytics including:
- Real-time performance monitoring
- Predictive analytics and forecasting
- Content ROI analysis
- A/B testing analytics
- Performance benchmarking
- Trend analysis and insights
- Custom metrics and KPIs
- Performance optimization recommendations
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Metric type enumeration"""
    ENGAGEMENT = "engagement"
    TRAFFIC = "traffic"
    CONVERSION = "conversion"
    REVENUE = "revenue"
    SOCIAL = "social"
    SEO = "seo"
    QUALITY = "quality"
    CUSTOM = "custom"

class TimeGranularity(Enum):
    """Time granularity enumeration"""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"

class AnalysisType(Enum):
    """Analysis type enumeration"""
    TREND = "trend"
    CORRELATION = "correlation"
    PREDICTION = "prediction"
    COMPARISON = "comparison"
    SEGMENTATION = "segmentation"
    ANOMALY = "anomaly"

@dataclass
class PerformanceMetric:
    """Performance metric data structure"""
    metric_id: str
    name: str
    metric_type: MetricType
    value: float
    timestamp: datetime
    content_id: str
    dimensions: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceInsight:
    """Performance insight data structure"""
    insight_id: str
    content_id: str
    insight_type: AnalysisType
    title: str
    description: str
    confidence: float
    impact_score: float
    recommendations: List[str] = field(default_factory=list)
    supporting_data: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class PerformancePrediction:
    """Performance prediction data structure"""
    prediction_id: str
    content_id: str
    metric_type: MetricType
    predicted_value: float
    confidence_interval: Tuple[float, float]
    prediction_horizon: int  # days
    model_used: str
    accuracy_score: float
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class ABTestResult:
    """A/B test result data structure"""
    test_id: str
    content_id_a: str
    content_id_b: str
    metric_type: MetricType
    variant_a_performance: float
    variant_b_performance: float
    statistical_significance: float
    confidence_level: float
    sample_size: int
    winner: str
    improvement_percentage: float
    test_duration: int  # days
    created_at: datetime = field(default_factory=datetime.utcnow)

class ContentPerformanceAnalytics:
    """
    Advanced Content Performance Analytics Engine
    
    Provides comprehensive performance tracking, analysis, and optimization
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the Content Performance Analytics Engine"""
        self.config = config
        self.metrics_data = {}
        self.performance_models = {}
        self.insights_cache = {}
        self.predictions_cache = {}
        self.ab_tests = {}
        
        # Initialize models
        self._initialize_models()
        
        logger.info("Content Performance Analytics Engine initialized successfully")
    
    def _initialize_models(self):
        """Initialize machine learning models for analytics"""
        try:
            # Linear regression for trend analysis
            self.performance_models["trend_analysis"] = LinearRegression()
            
            # Random forest for predictions
            self.performance_models["performance_prediction"] = RandomForestRegressor(
                n_estimators=100,
                random_state=42
            )
            
            # Standard scaler for data preprocessing
            self.performance_models["scaler"] = StandardScaler()
            
            logger.info("Analytics models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise
    
    async def track_metric(self, metric: PerformanceMetric) -> bool:
        """Track a performance metric"""
        try:
            # Store metric
            if metric.content_id not in self.metrics_data:
                self.metrics_data[metric.content_id] = []
            
            self.metrics_data[metric.content_id].append(metric)
            
            # Update insights cache
            await self._update_insights_cache(metric)
            
            logger.info(f"Metric {metric.metric_id} tracked for content {metric.content_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error tracking metric: {e}")
            return False
    
    async def _update_insights_cache(self, metric: PerformanceMetric):
        """Update insights cache with new metric"""
        try:
            content_id = metric.content_id
            
            if content_id not in self.insights_cache:
                self.insights_cache[content_id] = {
                    "metrics": [],
                    "insights": [],
                    "last_updated": datetime.utcnow()
                }
            
            self.insights_cache[content_id]["metrics"].append(metric)
            self.insights_cache[content_id]["last_updated"] = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Error updating insights cache: {e}")
    
    async def get_performance_summary(self, content_id: str, 
                                    time_period: str = "30d") -> Dict[str, Any]:
        """Get comprehensive performance summary for content"""
        try:
            if content_id not in self.metrics_data:
                return {"error": "No metrics found for this content"}
            
            metrics = self.metrics_data[content_id]
            
            # Filter by time period
            end_date = datetime.utcnow()
            if time_period == "7d":
                start_date = end_date - timedelta(days=7)
            elif time_period == "30d":
                start_date = end_date - timedelta(days=30)
            elif time_period == "90d":
                start_date = end_date - timedelta(days=90)
            else:
                start_date = end_date - timedelta(days=30)
            
            filtered_metrics = [
                m for m in metrics 
                if start_date <= m.timestamp <= end_date
            ]
            
            if not filtered_metrics:
                return {"error": "No metrics found for the specified time period"}
            
            # Calculate summary statistics
            summary = {
                "content_id": content_id,
                "time_period": time_period,
                "total_metrics": len(filtered_metrics),
                "date_range": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat()
                },
                "metrics_by_type": {},
                "performance_trends": {},
                "top_performing_metrics": [],
                "insights": []
            }
            
            # Group metrics by type
            metrics_by_type = {}
            for metric in filtered_metrics:
                metric_type = metric.metric_type.value
                if metric_type not in metrics_by_type:
                    metrics_by_type[metric_type] = []
                metrics_by_type[metric_type].append(metric.value)
            
            # Calculate statistics for each metric type
            for metric_type, values in metrics_by_type.items():
                summary["metrics_by_type"][metric_type] = {
                    "count": len(values),
                    "mean": np.mean(values),
                    "median": np.median(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "trend": await self._calculate_trend(values)
                }
            
            # Get performance insights
            summary["insights"] = await self._generate_performance_insights(content_id, filtered_metrics)
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {"error": str(e)}
    
    async def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for a series of values"""
        try:
            if len(values) < 2:
                return "insufficient_data"
            
            # Simple trend calculation
            first_half = values[:len(values)//2]
            second_half = values[len(values)//2:]
            
            first_mean = np.mean(first_half)
            second_mean = np.mean(second_half)
            
            if second_mean > first_mean * 1.1:
                return "increasing"
            elif second_mean < first_mean * 0.9:
                return "decreasing"
            else:
                return "stable"
                
        except Exception as e:
            logger.error(f"Error calculating trend: {e}")
            return "unknown"
    
    async def _generate_performance_insights(self, content_id: str, 
                                           metrics: List[PerformanceMetric]) -> List[Dict[str, Any]]:
        """Generate performance insights from metrics"""
        try:
            insights = []
            
            # Group metrics by type
            metrics_by_type = {}
            for metric in metrics:
                metric_type = metric.metric_type.value
                if metric_type not in metrics_by_type:
                    metrics_by_type[metric_type] = []
                metrics_by_type[metric_type].append(metric)
            
            # Generate insights for each metric type
            for metric_type, type_metrics in metrics_by_type.items():
                values = [m.value for m in type_metrics]
                
                # High performance insight
                if np.mean(values) > 0.8:
                    insights.append({
                        "type": "high_performance",
                        "title": f"High {metric_type.title()} Performance",
                        "description": f"Content shows excellent {metric_type} performance with average score of {np.mean(values):.2f}",
                        "confidence": 0.9,
                        "impact_score": 0.8,
                        "recommendations": [
                            f"Maintain current {metric_type} strategy",
                            "Consider scaling similar content",
                            "Analyze what makes this content successful"
                        ]
                    })
                
                # Low performance insight
                elif np.mean(values) < 0.3:
                    insights.append({
                        "type": "low_performance",
                        "title": f"Low {metric_type.title()} Performance",
                        "description": f"Content shows poor {metric_type} performance with average score of {np.mean(values):.2f}",
                        "confidence": 0.9,
                        "impact_score": 0.9,
                        "recommendations": [
                            f"Improve {metric_type} optimization",
                            "Analyze competitor content",
                            "Consider content revision or removal"
                        ]
                    })
                
                # Trend insights
                trend = await self._calculate_trend(values)
                if trend == "increasing":
                    insights.append({
                        "type": "positive_trend",
                        "title": f"Improving {metric_type.title()} Trend",
                        "description": f"Content shows improving {metric_type} performance over time",
                        "confidence": 0.8,
                        "impact_score": 0.7,
                        "recommendations": [
                            "Continue current optimization efforts",
                            "Monitor for continued improvement",
                            "Consider increasing promotion"
                        ]
                    })
                elif trend == "decreasing":
                    insights.append({
                        "type": "negative_trend",
                        "title": f"Declining {metric_type.title()} Trend",
                        "description": f"Content shows declining {metric_type} performance over time",
                        "confidence": 0.8,
                        "impact_score": 0.8,
                        "recommendations": [
                            "Investigate cause of decline",
                            "Consider content refresh",
                            "Review optimization strategy"
                        ]
                    })
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating performance insights: {e}")
            return []
    
    async def predict_performance(self, content_id: str, 
                                metric_type: MetricType,
                                prediction_horizon: int = 30) -> PerformancePrediction:
        """Predict future performance for content"""
        try:
            if content_id not in self.metrics_data:
                raise ValueError(f"No metrics found for content {content_id}")
            
            metrics = self.metrics_data[content_id]
            
            # Filter metrics by type
            type_metrics = [
                m for m in metrics 
                if m.metric_type == metric_type
            ]
            
            if len(type_metrics) < 10:
                raise ValueError("Insufficient data for prediction")
            
            # Prepare data for prediction
            df = pd.DataFrame([
                {
                    "timestamp": m.timestamp,
                    "value": m.value,
                    "days_since_start": (m.timestamp - type_metrics[0].timestamp).days
                }
                for m in type_metrics
            ])
            
            # Sort by timestamp
            df = df.sort_values("timestamp")
            
            # Create features
            X = df[["days_since_start"]].values
            y = df["value"].values
            
            # Train model
            model = self.performance_models["performance_prediction"]
            model.fit(X, y)
            
            # Make prediction
            last_day = df["days_since_start"].max()
            future_day = last_day + prediction_horizon
            prediction = model.predict([[future_day]])[0]
            
            # Calculate confidence interval (simplified)
            y_pred = model.predict(X)
            mse = mean_squared_error(y, y_pred)
            std_error = np.sqrt(mse)
            confidence_interval = (
                prediction - 1.96 * std_error,
                prediction + 1.96 * std_error
            )
            
            # Calculate accuracy score
            accuracy_score = r2_score(y, y_pred)
            
            return PerformancePrediction(
                prediction_id=f"pred_{uuid.uuid4()}",
                content_id=content_id,
                metric_type=metric_type,
                predicted_value=prediction,
                confidence_interval=confidence_interval,
                prediction_horizon=prediction_horizon,
                model_used="RandomForestRegressor",
                accuracy_score=accuracy_score
            )
            
        except Exception as e:
            logger.error(f"Error predicting performance: {e}")
            raise
    
    async def run_ab_test(self, content_id_a: str, content_id_b: str,
                         metric_type: MetricType,
                         test_duration: int = 14) -> ABTestResult:
        """Run A/B test between two content pieces"""
        try:
            # Get metrics for both content pieces
            metrics_a = self.metrics_data.get(content_id_a, [])
            metrics_b = self.metrics_data.get(content_id_b, [])
            
            if not metrics_a or not metrics_b:
                raise ValueError("Insufficient data for A/B test")
            
            # Filter metrics by type
            type_metrics_a = [m for m in metrics_a if m.metric_type == metric_type]
            type_metrics_b = [m for m in metrics_b if m.metric_type == metric_type]
            
            if not type_metrics_a or not type_metrics_b:
                raise ValueError(f"No {metric_type.value} metrics found for A/B test")
            
            # Calculate performance metrics
            performance_a = np.mean([m.value for m in type_metrics_a])
            performance_b = np.mean([m.value for m in type_metrics_b])
            
            # Perform statistical test (simplified t-test)
            values_a = [m.value for m in type_metrics_a]
            values_b = [m.value for m in type_metrics_b]
            
            t_stat, p_value = stats.ttest_ind(values_a, values_b)
            
            # Determine winner
            if performance_b > performance_a:
                winner = "B"
                improvement = ((performance_b - performance_a) / performance_a) * 100
            else:
                winner = "A"
                improvement = ((performance_a - performance_b) / performance_b) * 100
            
            # Calculate confidence level
            confidence_level = (1 - p_value) * 100
            
            return ABTestResult(
                test_id=f"ab_test_{uuid.uuid4()}",
                content_id_a=content_id_a,
                content_id_b=content_id_b,
                metric_type=metric_type,
                variant_a_performance=performance_a,
                variant_b_performance=performance_b,
                statistical_significance=p_value,
                confidence_level=confidence_level,
                sample_size=len(values_a) + len(values_b),
                winner=winner,
                improvement_percentage=improvement,
                test_duration=test_duration
            )
            
        except Exception as e:
            logger.error(f"Error running A/B test: {e}")
            raise
    
    async def get_benchmark_analysis(self, content_id: str, 
                                   benchmark_group: str = "industry") -> Dict[str, Any]:
        """Get benchmark analysis for content"""
        try:
            if content_id not in self.metrics_data:
                return {"error": "No metrics found for this content"}
            
            metrics = self.metrics_data[content_id]
            
            # Calculate content performance
            content_performance = {}
            for metric in metrics:
                metric_type = metric.metric_type.value
                if metric_type not in content_performance:
                    content_performance[metric_type] = []
                content_performance[metric_type].append(metric.value)
            
            # Calculate averages
            content_averages = {
                metric_type: np.mean(values)
                for metric_type, values in content_performance.items()
            }
            
            # Mock benchmark data (in production, this would come from a database)
            benchmark_data = {
                "industry": {
                    "engagement": 0.65,
                    "traffic": 0.58,
                    "conversion": 0.12,
                    "seo": 0.72
                },
                "top_performers": {
                    "engagement": 0.85,
                    "traffic": 0.78,
                    "conversion": 0.25,
                    "seo": 0.90
                }
            }
            
            benchmark = benchmark_data.get(benchmark_group, benchmark_data["industry"])
            
            # Calculate benchmark comparison
            comparison = {}
            for metric_type, content_avg in content_averages.items():
                benchmark_avg = benchmark.get(metric_type, 0.5)
                comparison[metric_type] = {
                    "content_performance": content_avg,
                    "benchmark_performance": benchmark_avg,
                    "performance_ratio": content_avg / benchmark_avg if benchmark_avg > 0 else 0,
                    "performance_gap": content_avg - benchmark_avg,
                    "status": "above_benchmark" if content_avg > benchmark_avg else "below_benchmark"
                }
            
            return {
                "content_id": content_id,
                "benchmark_group": benchmark_group,
                "comparison": comparison,
                "overall_performance": np.mean(list(content_averages.values())),
                "benchmark_performance": np.mean(list(benchmark.values())),
                "performance_rank": "top_25%" if np.mean(list(content_averages.values())) > 0.8 else "average"
            }
            
        except Exception as e:
            logger.error(f"Error getting benchmark analysis: {e}")
            return {"error": str(e)}
    
    async def generate_performance_report(self, content_id: str, 
                                        time_period: str = "30d") -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        try:
            # Get performance summary
            summary = await self.get_performance_summary(content_id, time_period)
            
            # Get benchmark analysis
            benchmark = await self.get_benchmark_analysis(content_id)
            
            # Get predictions for key metrics
            predictions = {}
            if content_id in self.metrics_data:
                for metric_type in [MetricType.ENGAGEMENT, MetricType.TRAFFIC, MetricType.CONVERSION]:
                    try:
                        pred = await self.predict_performance(content_id, metric_type, 30)
                        predictions[metric_type.value] = {
                            "predicted_value": pred.predicted_value,
                            "confidence_interval": pred.confidence_interval,
                            "accuracy_score": pred.accuracy_score
                        }
                    except:
                        continue
            
            # Generate recommendations
            recommendations = []
            if "insights" in summary:
                for insight in summary["insights"]:
                    recommendations.extend(insight.get("recommendations", []))
            
            return {
                "content_id": content_id,
                "report_generated_at": datetime.utcnow().isoformat(),
                "time_period": time_period,
                "summary": summary,
                "benchmark_analysis": benchmark,
                "predictions": predictions,
                "recommendations": list(set(recommendations)),  # Remove duplicates
                "key_insights": [
                    insight["title"] for insight in summary.get("insights", [])
                ],
                "performance_score": summary.get("metrics_by_type", {}).get("engagement", {}).get("mean", 0.5)
            }
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return {"error": str(e)}
    
    async def export_analytics_data(self, content_id: str, 
                                  format: str = "json") -> Union[Dict[str, Any], str]:
        """Export analytics data in various formats"""
        try:
            if content_id not in self.metrics_data:
                return {"error": "No data found for this content"}
            
            metrics = self.metrics_data[content_id]
            
            # Prepare data
            data = {
                "content_id": content_id,
                "export_timestamp": datetime.utcnow().isoformat(),
                "total_metrics": len(metrics),
                "metrics": [
                    {
                        "metric_id": m.metric_id,
                        "name": m.name,
                        "type": m.metric_type.value,
                        "value": m.value,
                        "timestamp": m.timestamp.isoformat(),
                        "dimensions": m.dimensions,
                        "metadata": m.metadata
                    }
                    for m in metrics
                ]
            }
            
            if format == "json":
                return data
            elif format == "csv":
                # Convert to CSV format
                df = pd.DataFrame(data["metrics"])
                return df.to_csv(index=False)
            else:
                return data
                
        except Exception as e:
            logger.error(f"Error exporting analytics data: {e}")
            return {"error": str(e)}

# Example usage and testing
async def main():
    """Example usage of the Content Performance Analytics Engine"""
    try:
        # Initialize engine
        config = {
            "models": {
                "trend_analysis": "linear_regression",
                "performance_prediction": "random_forest"
            },
            "cache_size": 1000
        }
        
        engine = ContentPerformanceAnalytics(config)
        
        # Create sample metrics
        content_id = "sample_content_001"
        
        # Track some metrics
        for i in range(30):
            metric = PerformanceMetric(
                metric_id=f"metric_{i}",
                name="Engagement Score",
                metric_type=MetricType.ENGAGEMENT,
                value=0.5 + np.random.normal(0, 0.1),
                timestamp=datetime.utcnow() - timedelta(days=30-i),
                content_id=content_id
            )
            await engine.track_metric(metric)
        
        # Get performance summary
        print("Getting performance summary...")
        summary = await engine.get_performance_summary(content_id, "30d")
        print(f"Total metrics: {summary['total_metrics']}")
        print(f"Engagement trend: {summary['metrics_by_type']['engagement']['trend']}")
        
        # Get predictions
        print("\nGetting performance predictions...")
        prediction = await engine.predict_performance(content_id, MetricType.ENGAGEMENT, 30)
        print(f"Predicted engagement: {prediction.predicted_value:.3f}")
        print(f"Confidence interval: {prediction.confidence_interval}")
        
        # Get benchmark analysis
        print("\nGetting benchmark analysis...")
        benchmark = await engine.get_benchmark_analysis(content_id)
        print(f"Overall performance: {benchmark['overall_performance']:.3f}")
        print(f"Performance rank: {benchmark['performance_rank']}")
        
        # Generate performance report
        print("\nGenerating performance report...")
        report = await engine.generate_performance_report(content_id)
        print(f"Performance score: {report['performance_score']:.3f}")
        print(f"Key insights: {len(report['key_insights'])}")
        
        print("\nContent Performance Analytics Engine demo completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main demo: {e}")

if __name__ == "__main__":
    asyncio.run(main())
























