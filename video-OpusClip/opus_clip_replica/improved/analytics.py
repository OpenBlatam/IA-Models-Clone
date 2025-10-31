"""
Analytics Engine for OpusClip Improved
=====================================

Comprehensive analytics and insights generation for video content performance.
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from uuid import UUID, uuid4
import json
import numpy as np
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_

from .schemas import PlatformType, ClipType, QualityLevel
from .exceptions import DatabaseError, create_database_error
from .database import get_database_session

logger = logging.getLogger(__name__)


@dataclass
class MetricData:
    """Metric data structure"""
    metric_name: str
    value: float
    timestamp: datetime
    metadata: Dict[str, Any] = None


@dataclass
class TrendData:
    """Trend data structure"""
    metric_name: str
    time_series: List[Tuple[datetime, float]]
    trend_direction: str  # "up", "down", "stable"
    trend_strength: float  # 0-1
    metadata: Dict[str, Any] = None


@dataclass
class InsightData:
    """Insight data structure"""
    insight_type: str
    title: str
    description: str
    confidence: float
    impact: str  # "high", "medium", "low"
    recommendations: List[str]
    metadata: Dict[str, Any] = None


class AnalyticsEngine:
    """Advanced analytics engine for video content performance"""
    
    def __init__(self):
        self.metrics_cache = {}
        self.insights_cache = {}
        self.cache_ttl = 3600  # 1 hour
    
    async def get_metrics(
        self,
        project_id: Optional[UUID] = None,
        date_range: Optional[Dict[str, datetime]] = None,
        metrics: List[str] = None
    ) -> Dict[str, Any]:
        """Get comprehensive metrics for analysis"""
        try:
            if metrics is None:
                metrics = ["views", "engagement", "viral_score", "processing_time", "success_rate"]
            
            # Set default date range if not provided
            if date_range is None:
                end_date = datetime.utcnow()
                start_date = end_date - timedelta(days=30)
                date_range = {"start": start_date, "end": end_date}
            
            # Get metrics from database
            async with get_database_session() as session:
                metrics_data = {}
                
                for metric in metrics:
                    if metric == "views":
                        metrics_data[metric] = await self._get_views_metric(session, project_id, date_range)
                    elif metric == "engagement":
                        metrics_data[metric] = await self._get_engagement_metric(session, project_id, date_range)
                    elif metric == "viral_score":
                        metrics_data[metric] = await self._get_viral_score_metric(session, project_id, date_range)
                    elif metric == "processing_time":
                        metrics_data[metric] = await self._get_processing_time_metric(session, project_id, date_range)
                    elif metric == "success_rate":
                        metrics_data[metric] = await self._get_success_rate_metric(session, project_id, date_range)
                    elif metric == "content_quality":
                        metrics_data[metric] = await self._get_content_quality_metric(session, project_id, date_range)
                    elif metric == "platform_performance":
                        metrics_data[metric] = await self._get_platform_performance_metric(session, project_id, date_range)
                    elif metric == "user_activity":
                        metrics_data[metric] = await self._get_user_activity_metric(session, project_id, date_range)
                
                # Add summary statistics
                metrics_data["summary"] = await self._calculate_summary_stats(metrics_data)
                
                logger.info(f"Retrieved {len(metrics)} metrics for project {project_id}")
                return metrics_data
                
        except Exception as e:
            raise create_database_error("get_metrics", "analytics", e)
    
    async def get_trends(
        self,
        project_id: Optional[UUID] = None,
        date_range: Optional[Dict[str, datetime]] = None,
        group_by: Optional[str] = None
    ) -> Dict[str, List[TrendData]]:
        """Get trend analysis for metrics"""
        try:
            if date_range is None:
                end_date = datetime.utcnow()
                start_date = end_date - timedelta(days=30)
                date_range = {"start": start_date, "end": end_date}
            
            async with get_database_session() as session:
                trends = {}
                
                # Get trends for different metrics
                metrics = ["views", "engagement", "viral_score", "processing_time", "success_rate"]
                
                for metric in metrics:
                    trend_data = await self._calculate_trend(session, metric, project_id, date_range, group_by)
                    trends[metric] = trend_data
                
                logger.info(f"Calculated trends for {len(metrics)} metrics")
                return trends
                
        except Exception as e:
            raise create_database_error("get_trends", "analytics", e)
    
    async def generate_insights(
        self,
        metrics: Dict[str, Any],
        trends: Dict[str, List[TrendData]]
    ) -> List[InsightData]:
        """Generate AI-powered insights from metrics and trends"""
        try:
            insights = []
            
            # Performance insights
            performance_insights = await self._generate_performance_insights(metrics, trends)
            insights.extend(performance_insights)
            
            # Content insights
            content_insights = await self._generate_content_insights(metrics, trends)
            insights.extend(content_insights)
            
            # Platform insights
            platform_insights = await self._generate_platform_insights(metrics, trends)
            insights.extend(platform_insights)
            
            # User behavior insights
            user_insights = await self._generate_user_insights(metrics, trends)
            insights.extend(user_insights)
            
            # Optimization insights
            optimization_insights = await self._generate_optimization_insights(metrics, trends)
            insights.extend(optimization_insights)
            
            # Sort insights by impact and confidence
            insights.sort(key=lambda x: (x.impact == "high", x.confidence), reverse=True)
            
            logger.info(f"Generated {len(insights)} insights")
            return insights
            
        except Exception as e:
            logger.error(f"Failed to generate insights: {e}")
            return []
    
    async def generate_recommendations(
        self,
        metrics: Dict[str, Any],
        trends: Dict[str, List[TrendData]],
        insights: List[InsightData]
    ) -> List[str]:
        """Generate actionable recommendations"""
        try:
            recommendations = []
            
            # Performance recommendations
            perf_recs = await self._generate_performance_recommendations(metrics, trends)
            recommendations.extend(perf_recs)
            
            # Content recommendations
            content_recs = await self._generate_content_recommendations(metrics, trends)
            recommendations.extend(content_recs)
            
            # Platform recommendations
            platform_recs = await self._generate_platform_recommendations(metrics, trends)
            recommendations.extend(platform_recs)
            
            # Optimization recommendations
            opt_recs = await self._generate_optimization_recommendations(metrics, trends)
            recommendations.extend(opt_recs)
            
            # Remove duplicates and prioritize
            unique_recommendations = list(set(recommendations))
            prioritized_recommendations = await self._prioritize_recommendations(unique_recommendations, insights)
            
            logger.info(f"Generated {len(prioritized_recommendations)} recommendations")
            return prioritized_recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate recommendations: {e}")
            return []
    
    async def get_comparative_analysis(
        self,
        project_id: UUID,
        comparison_periods: List[Dict[str, datetime]]
    ) -> Dict[str, Any]:
        """Get comparative analysis across different time periods"""
        try:
            analysis = {}
            
            for i, period in enumerate(comparison_periods):
                period_name = f"period_{i+1}"
                
                # Get metrics for this period
                metrics = await self.get_metrics(project_id, period)
                trends = await self.get_trends(project_id, period)
                
                analysis[period_name] = {
                    "metrics": metrics,
                    "trends": trends,
                    "period": period
                }
            
            # Calculate comparisons
            comparisons = await self._calculate_period_comparisons(analysis)
            analysis["comparisons"] = comparisons
            
            logger.info(f"Generated comparative analysis for {len(comparison_periods)} periods")
            return analysis
            
        except Exception as e:
            raise create_database_error("get_comparative_analysis", "analytics", e)
    
    async def get_predictive_analytics(
        self,
        project_id: Optional[UUID] = None,
        forecast_days: int = 30
    ) -> Dict[str, Any]:
        """Get predictive analytics and forecasting"""
        try:
            # Get historical data
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=90)  # 3 months of data
            date_range = {"start": start_date, "end": end_date}
            
            metrics = await self.get_metrics(project_id, date_range)
            trends = await self.get_trends(project_id, date_range)
            
            # Generate forecasts
            forecasts = {}
            
            for metric_name, trend_data in trends.items():
                if trend_data:
                    forecast = await self._generate_forecast(trend_data[0], forecast_days)
                    forecasts[metric_name] = forecast
            
            # Generate predictions
            predictions = await self._generate_predictions(metrics, trends, forecast_days)
            
            result = {
                "forecasts": forecasts,
                "predictions": predictions,
                "forecast_period": forecast_days,
                "confidence_levels": await self._calculate_confidence_levels(forecasts)
            }
            
            logger.info(f"Generated predictive analytics for {forecast_days} days")
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate predictive analytics: {e}")
            return {}
    
    # Private methods for metric calculations
    async def _get_views_metric(self, session: AsyncSession, project_id: Optional[UUID], date_range: Dict[str, datetime]) -> Dict[str, Any]:
        """Get views metric data"""
        # Placeholder implementation - would query actual database
        return {
            "total_views": 10000,
            "unique_views": 8500,
            "average_views_per_video": 250,
            "top_performing_video": "video_123",
            "views_by_platform": {
                "youtube": 4000,
                "tiktok": 3000,
                "instagram": 2000,
                "linkedin": 1000
            },
            "views_trend": "increasing"
        }
    
    async def _get_engagement_metric(self, session: AsyncSession, project_id: Optional[UUID], date_range: Dict[str, datetime]) -> Dict[str, Any]:
        """Get engagement metric data"""
        return {
            "total_engagement": 2500,
            "engagement_rate": 0.25,
            "likes": 1500,
            "shares": 500,
            "comments": 500,
            "average_engagement_per_video": 62.5,
            "engagement_by_platform": {
                "youtube": 0.3,
                "tiktok": 0.4,
                "instagram": 0.2,
                "linkedin": 0.1
            }
        }
    
    async def _get_viral_score_metric(self, session: AsyncSession, project_id: Optional[UUID], date_range: Dict[str, datetime]) -> Dict[str, Any]:
        """Get viral score metric data"""
        return {
            "average_viral_score": 0.65,
            "high_viral_videos": 15,
            "viral_score_distribution": {
                "0-0.2": 5,
                "0.2-0.4": 10,
                "0.4-0.6": 20,
                "0.6-0.8": 15,
                "0.8-1.0": 5
            },
            "viral_factors": {
                "emotional_impact": 0.8,
                "shareability": 0.7,
                "trending_topics": 0.6,
                "visual_appeal": 0.9
            }
        }
    
    async def _get_processing_time_metric(self, session: AsyncSession, project_id: Optional[UUID], date_range: Dict[str, datetime]) -> Dict[str, Any]:
        """Get processing time metric data"""
        return {
            "average_processing_time": 45.5,
            "median_processing_time": 42.0,
            "processing_time_by_quality": {
                "low": 15.0,
                "medium": 30.0,
                "high": 60.0,
                "ultra": 120.0
            },
            "processing_efficiency": 0.85,
            "queue_wait_time": 5.2
        }
    
    async def _get_success_rate_metric(self, session: AsyncSession, project_id: Optional[UUID], date_range: Dict[str, datetime]) -> Dict[str, Any]:
        """Get success rate metric data"""
        return {
            "overall_success_rate": 0.95,
            "analysis_success_rate": 0.98,
            "generation_success_rate": 0.92,
            "export_success_rate": 0.96,
            "error_rate": 0.05,
            "common_errors": {
                "video_format_error": 0.02,
                "processing_timeout": 0.01,
                "ai_service_error": 0.01,
                "storage_error": 0.01
            }
        }
    
    async def _get_content_quality_metric(self, session: AsyncSession, project_id: Optional[UUID], date_range: Dict[str, datetime]) -> Dict[str, Any]:
        """Get content quality metric data"""
        return {
            "average_quality_score": 0.78,
            "quality_distribution": {
                "excellent": 0.2,
                "good": 0.4,
                "average": 0.3,
                "poor": 0.1
            },
            "quality_factors": {
                "video_resolution": 0.9,
                "audio_quality": 0.8,
                "content_relevance": 0.7,
                "engagement_potential": 0.8
            }
        }
    
    async def _get_platform_performance_metric(self, session: AsyncSession, project_id: Optional[UUID], date_range: Dict[str, datetime]) -> Dict[str, Any]:
        """Get platform performance metric data"""
        return {
            "platform_performance": {
                "youtube": {
                    "views": 4000,
                    "engagement_rate": 0.3,
                    "average_duration": 45.0,
                    "click_through_rate": 0.05
                },
                "tiktok": {
                    "views": 3000,
                    "engagement_rate": 0.4,
                    "average_duration": 30.0,
                    "completion_rate": 0.8
                },
                "instagram": {
                    "views": 2000,
                    "engagement_rate": 0.2,
                    "average_duration": 25.0,
                    "story_completion_rate": 0.7
                },
                "linkedin": {
                    "views": 1000,
                    "engagement_rate": 0.1,
                    "average_duration": 60.0,
                    "professional_engagement": 0.6
                }
            }
        }
    
    async def _get_user_activity_metric(self, session: AsyncSession, project_id: Optional[UUID], date_range: Dict[str, datetime]) -> Dict[str, Any]:
        """Get user activity metric data"""
        return {
            "active_users": 150,
            "new_users": 25,
            "user_retention_rate": 0.75,
            "average_session_duration": 12.5,
            "features_usage": {
                "video_analysis": 0.9,
                "clip_generation": 0.8,
                "batch_processing": 0.3,
                "analytics": 0.6
            },
            "user_satisfaction": 4.2
        }
    
    async def _calculate_summary_stats(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary statistics"""
        return {
            "total_videos_processed": 1000,
            "total_clips_generated": 5000,
            "total_exports": 3000,
            "average_processing_time": 45.5,
            "success_rate": 0.95,
            "user_satisfaction": 4.2,
            "system_uptime": 0.99
        }
    
    async def _calculate_trend(self, session: AsyncSession, metric_name: str, project_id: Optional[UUID], date_range: Dict[str, datetime], group_by: Optional[str]) -> List[TrendData]:
        """Calculate trend for a specific metric"""
        # Placeholder implementation - would calculate actual trends
        trend_data = TrendData(
            metric_name=metric_name,
            time_series=[
                (datetime.utcnow() - timedelta(days=i), 100 + i * 5 + np.random.normal(0, 10))
                for i in range(30, 0, -1)
            ],
            trend_direction="up",
            trend_strength=0.7,
            metadata={"project_id": str(project_id) if project_id else None}
        )
        return [trend_data]
    
    async def _generate_performance_insights(self, metrics: Dict[str, Any], trends: Dict[str, List[TrendData]]) -> List[InsightData]:
        """Generate performance-related insights"""
        insights = []
        
        # Processing time insight
        processing_time = metrics.get("processing_time", {})
        avg_time = processing_time.get("average_processing_time", 0)
        
        if avg_time > 60:
            insights.append(InsightData(
                insight_type="performance",
                title="High Processing Times",
                description=f"Average processing time is {avg_time:.1f} seconds, which is above optimal range",
                confidence=0.8,
                impact="high",
                recommendations=[
                    "Consider upgrading to GPU processing",
                    "Optimize video compression settings",
                    "Implement parallel processing for batch operations"
                ],
                metadata={"metric": "processing_time", "value": avg_time}
            ))
        
        # Success rate insight
        success_rate = metrics.get("success_rate", {})
        overall_rate = success_rate.get("overall_success_rate", 0)
        
        if overall_rate < 0.9:
            insights.append(InsightData(
                insight_type="performance",
                title="Low Success Rate",
                description=f"Overall success rate is {overall_rate:.1%}, below recommended threshold",
                confidence=0.9,
                impact="high",
                recommendations=[
                    "Investigate common error patterns",
                    "Improve error handling and retry logic",
                    "Monitor system resources and capacity"
                ],
                metadata={"metric": "success_rate", "value": overall_rate}
            ))
        
        return insights
    
    async def _generate_content_insights(self, metrics: Dict[str, Any], trends: Dict[str, List[TrendData]]) -> List[InsightData]:
        """Generate content-related insights"""
        insights = []
        
        # Viral score insight
        viral_score = metrics.get("viral_score", {})
        avg_viral = viral_score.get("average_viral_score", 0)
        
        if avg_viral > 0.7:
            insights.append(InsightData(
                insight_type="content",
                title="High Viral Potential",
                description=f"Average viral score is {avg_viral:.1%}, indicating strong content performance",
                confidence=0.8,
                impact="medium",
                recommendations=[
                    "Focus on similar content types",
                    "Increase production of high-performing content",
                    "Analyze viral factors for replication"
                ],
                metadata={"metric": "viral_score", "value": avg_viral}
            ))
        
        # Engagement insight
        engagement = metrics.get("engagement", {})
        engagement_rate = engagement.get("engagement_rate", 0)
        
        if engagement_rate < 0.2:
            insights.append(InsightData(
                insight_type="content",
                title="Low Engagement Rate",
                description=f"Engagement rate is {engagement_rate:.1%}, below industry average",
                confidence=0.7,
                impact="high",
                recommendations=[
                    "Improve content hooks and opening sequences",
                    "Add more interactive elements",
                    "Optimize for platform-specific engagement patterns"
                ],
                metadata={"metric": "engagement_rate", "value": engagement_rate}
            ))
        
        return insights
    
    async def _generate_platform_insights(self, metrics: Dict[str, Any], trends: Dict[str, List[TrendData]]) -> List[InsightData]:
        """Generate platform-related insights"""
        insights = []
        
        platform_performance = metrics.get("platform_performance", {})
        platform_data = platform_performance.get("platform_performance", {})
        
        # Find best performing platform
        best_platform = max(platform_data.items(), key=lambda x: x[1].get("engagement_rate", 0))
        worst_platform = min(platform_data.items(), key=lambda x: x[1].get("engagement_rate", 0))
        
        if best_platform[1].get("engagement_rate", 0) > worst_platform[1].get("engagement_rate", 0) * 2:
            insights.append(InsightData(
                insight_type="platform",
                title="Platform Performance Gap",
                description=f"{best_platform[0]} significantly outperforms {worst_platform[0]} in engagement",
                confidence=0.8,
                impact="medium",
                recommendations=[
                    f"Focus more resources on {best_platform[0]}",
                    f"Analyze {worst_platform[0]} optimization opportunities",
                    "Develop platform-specific content strategies"
                ],
                metadata={
                    "best_platform": best_platform[0],
                    "worst_platform": worst_platform[0],
                    "performance_gap": best_platform[1].get("engagement_rate", 0) / worst_platform[1].get("engagement_rate", 0.01)
                }
            ))
        
        return insights
    
    async def _generate_user_insights(self, metrics: Dict[str, Any], trends: Dict[str, List[TrendData]]) -> List[InsightData]:
        """Generate user behavior insights"""
        insights = []
        
        user_activity = metrics.get("user_activity", {})
        retention_rate = user_activity.get("user_retention_rate", 0)
        
        if retention_rate < 0.7:
            insights.append(InsightData(
                insight_type="user",
                title="Low User Retention",
                description=f"User retention rate is {retention_rate:.1%}, indicating potential user experience issues",
                confidence=0.7,
                impact="high",
                recommendations=[
                    "Improve onboarding experience",
                    "Add user tutorials and help documentation",
                    "Implement user feedback collection",
                    "Optimize feature discoverability"
                ],
                metadata={"metric": "retention_rate", "value": retention_rate}
            ))
        
        return insights
    
    async def _generate_optimization_insights(self, metrics: Dict[str, Any], trends: Dict[str, List[TrendData]]) -> List[InsightData]:
        """Generate optimization insights"""
        insights = []
        
        # Content quality insight
        content_quality = metrics.get("content_quality", {})
        avg_quality = content_quality.get("average_quality_score", 0)
        
        if avg_quality < 0.7:
            insights.append(InsightData(
                insight_type="optimization",
                title="Content Quality Improvement Needed",
                description=f"Average content quality score is {avg_quality:.1%}, below optimal range",
                confidence=0.8,
                impact="medium",
                recommendations=[
                    "Improve video resolution and audio quality",
                    "Enhance content relevance and engagement potential",
                    "Implement quality control processes",
                    "Provide content creation guidelines"
                ],
                metadata={"metric": "content_quality", "value": avg_quality}
            ))
        
        return insights
    
    async def _generate_performance_recommendations(self, metrics: Dict[str, Any], trends: Dict[str, List[TrendData]]) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        
        processing_time = metrics.get("processing_time", {})
        if processing_time.get("average_processing_time", 0) > 60:
            recommendations.append("Implement GPU acceleration for video processing")
            recommendations.append("Optimize video compression algorithms")
            recommendations.append("Add parallel processing capabilities")
        
        success_rate = metrics.get("success_rate", {})
        if success_rate.get("overall_success_rate", 0) < 0.9:
            recommendations.append("Improve error handling and retry mechanisms")
            recommendations.append("Monitor system resources and capacity")
            recommendations.append("Implement better input validation")
        
        return recommendations
    
    async def _generate_content_recommendations(self, metrics: Dict[str, Any], trends: Dict[str, List[TrendData]]) -> List[str]:
        """Generate content recommendations"""
        recommendations = []
        
        viral_score = metrics.get("viral_score", {})
        if viral_score.get("average_viral_score", 0) > 0.7:
            recommendations.append("Increase production of high viral potential content")
            recommendations.append("Analyze and replicate successful viral factors")
        
        engagement = metrics.get("engagement", {})
        if engagement.get("engagement_rate", 0) < 0.2:
            recommendations.append("Improve content hooks and opening sequences")
            recommendations.append("Add more interactive and engaging elements")
            recommendations.append("Optimize content for platform-specific algorithms")
        
        return recommendations
    
    async def _generate_platform_recommendations(self, metrics: Dict[str, Any], trends: Dict[str, List[TrendData]]) -> List[str]:
        """Generate platform recommendations"""
        recommendations = []
        
        platform_performance = metrics.get("platform_performance", {})
        platform_data = platform_performance.get("platform_performance", {})
        
        # Analyze platform performance
        for platform, data in platform_data.items():
            engagement_rate = data.get("engagement_rate", 0)
            if engagement_rate < 0.2:
                recommendations.append(f"Optimize content strategy for {platform}")
                recommendations.append(f"Improve {platform}-specific formatting and timing")
        
        return recommendations
    
    async def _generate_optimization_recommendations(self, metrics: Dict[str, Any], trends: Dict[str, List[TrendData]]) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        content_quality = metrics.get("content_quality", {})
        if content_quality.get("average_quality_score", 0) < 0.7:
            recommendations.append("Implement automated quality assessment")
            recommendations.append("Provide content creation best practices")
            recommendations.append("Add quality control checkpoints")
        
        return recommendations
    
    async def _prioritize_recommendations(self, recommendations: List[str], insights: List[InsightData]) -> List[str]:
        """Prioritize recommendations based on insights"""
        # Simple prioritization based on insight impact
        high_impact_insights = [i for i in insights if i.impact == "high"]
        medium_impact_insights = [i for i in insights if i.impact == "medium"]
        
        prioritized = []
        
        # Add recommendations from high impact insights first
        for insight in high_impact_insights:
            prioritized.extend(insight.recommendations)
        
        # Add recommendations from medium impact insights
        for insight in medium_impact_insights:
            prioritized.extend(insight.recommendations)
        
        # Add remaining recommendations
        for rec in recommendations:
            if rec not in prioritized:
                prioritized.append(rec)
        
        return prioritized[:10]  # Return top 10 recommendations
    
    async def _calculate_period_comparisons(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comparisons between different time periods"""
        comparisons = {}
        
        periods = list(analysis.keys())
        if len(periods) >= 2:
            period1 = periods[0]
            period2 = periods[1]
            
            metrics1 = analysis[period1]["metrics"]
            metrics2 = analysis[period2]["metrics"]
            
            # Compare key metrics
            comparisons["views_change"] = self._calculate_percentage_change(
                metrics1.get("views", {}).get("total_views", 0),
                metrics2.get("views", {}).get("total_views", 0)
            )
            
            comparisons["engagement_change"] = self._calculate_percentage_change(
                metrics1.get("engagement", {}).get("engagement_rate", 0),
                metrics2.get("engagement", {}).get("engagement_rate", 0)
            )
            
            comparisons["viral_score_change"] = self._calculate_percentage_change(
                metrics1.get("viral_score", {}).get("average_viral_score", 0),
                metrics2.get("viral_score", {}).get("average_viral_score", 0)
            )
        
        return comparisons
    
    def _calculate_percentage_change(self, old_value: float, new_value: float) -> float:
        """Calculate percentage change between two values"""
        if old_value == 0:
            return 0.0
        return ((new_value - old_value) / old_value) * 100
    
    async def _generate_forecast(self, trend_data: TrendData, forecast_days: int) -> Dict[str, Any]:
        """Generate forecast for a trend"""
        # Simple linear regression forecast
        time_series = trend_data.time_series
        
        if len(time_series) < 2:
            return {"error": "Insufficient data for forecasting"}
        
        # Extract values and calculate trend
        values = [point[1] for point in time_series]
        x = list(range(len(values)))
        
        # Simple linear regression
        n = len(values)
        sum_x = sum(x)
        sum_y = sum(values)
        sum_xy = sum(x[i] * values[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        intercept = (sum_y - slope * sum_x) / n
        
        # Generate forecast
        forecast_values = []
        last_date = time_series[-1][0]
        
        for i in range(1, forecast_days + 1):
            forecast_date = last_date + timedelta(days=i)
            forecast_value = intercept + slope * (len(values) + i)
            forecast_values.append((forecast_date, forecast_value))
        
        return {
            "forecast_values": forecast_values,
            "trend_slope": slope,
            "confidence": min(abs(slope) * 10, 1.0),  # Simple confidence calculation
            "forecast_period": forecast_days
        }
    
    async def _generate_predictions(self, metrics: Dict[str, Any], trends: Dict[str, List[TrendData]], forecast_days: int) -> Dict[str, Any]:
        """Generate predictions based on metrics and trends"""
        predictions = {}
        
        # Predict viral content
        viral_score = metrics.get("viral_score", {})
        avg_viral = viral_score.get("average_viral_score", 0)
        
        if avg_viral > 0.7:
            predictions["viral_content_prediction"] = {
                "prediction": "High viral potential content expected",
                "confidence": 0.8,
                "recommendation": "Increase production of similar content"
            }
        
        # Predict platform performance
        platform_performance = metrics.get("platform_performance", {})
        platform_data = platform_performance.get("platform_performance", {})
        
        best_platform = max(platform_data.items(), key=lambda x: x[1].get("engagement_rate", 0))
        predictions["platform_prediction"] = {
            "prediction": f"{best_platform[0]} will continue to be the best performing platform",
            "confidence": 0.7,
            "recommendation": f"Focus resources on {best_platform[0]} optimization"
        }
        
        return predictions
    
    async def _calculate_confidence_levels(self, forecasts: Dict[str, Any]) -> Dict[str, float]:
        """Calculate confidence levels for forecasts"""
        confidence_levels = {}
        
        for metric, forecast in forecasts.items():
            if "confidence" in forecast:
                confidence_levels[metric] = forecast["confidence"]
            else:
                confidence_levels[metric] = 0.5  # Default confidence
        
        return confidence_levels


# Global analytics engine instance
analytics_engine = AnalyticsEngine()





























