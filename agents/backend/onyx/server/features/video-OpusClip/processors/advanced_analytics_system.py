"""
Advanced Analytics System

Comprehensive analytics and reporting for video content performance,
audience insights, and optimization recommendations.
"""

from __future__ import annotations
from typing import List, Dict, Optional, Any, Tuple, Union
import asyncio
import numpy as np
import pandas as pd
import json
import time
from dataclasses import dataclass
from enum import Enum
import structlog
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import sqlite3
import pickle

from ..models.video_models import VideoClipRequest, VideoClipResponse
from ..error_handling import ErrorHandler, ProcessingError, ValidationError

logger = structlog.get_logger("advanced_analytics_system")
error_handler = ErrorHandler()

class MetricType(Enum):
    """Types of analytics metrics."""
    ENGAGEMENT = "engagement"
    REACH = "reach"
    RETENTION = "retention"
    CONVERSION = "conversion"
    QUALITY = "quality"
    VIRAL = "viral"

class TimeRange(Enum):
    """Time ranges for analytics."""
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"

@dataclass
class PerformanceMetric:
    """Individual performance metric."""
    metric_type: MetricType
    value: float
    timestamp: datetime
    platform: str
    content_id: str
    metadata: Dict[str, Any]

@dataclass
class AudienceInsight:
    """Audience demographic and behavioral insight."""
    demographic: str
    value: float
    percentage: float
    trend: str  # "increasing", "decreasing", "stable"
    confidence: float

@dataclass
class ContentAnalysis:
    """Comprehensive content analysis."""
    content_id: str
    title: str
    duration: float
    platform: str
    performance_score: float
    viral_potential: float
    audience_engagement: float
    quality_metrics: Dict[str, float]
    recommendations: List[str]
    created_at: datetime
    updated_at: datetime

@dataclass
class AnalyticsReport:
    """Comprehensive analytics report."""
    report_id: str
    content_analyses: List[ContentAnalysis]
    performance_summary: Dict[str, Any]
    audience_insights: List[AudienceInsight]
    trends: Dict[str, Any]
    recommendations: List[str]
    generated_at: datetime
    time_range: TimeRange

class PerformanceTracker:
    """Tracks and analyzes performance metrics."""
    
    def __init__(self):
        self.metrics_db = self._initialize_database()
        self.metric_cache = {}
    
    def _initialize_database(self) -> sqlite3.Connection:
        """Initialize SQLite database for metrics storage."""
        try:
            db_path = "/tmp/analytics.db"
            conn = sqlite3.connect(db_path)
            
            # Create metrics table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content_id TEXT NOT NULL,
                    platform TEXT NOT NULL,
                    metric_type TEXT NOT NULL,
                    value REAL NOT NULL,
                    timestamp DATETIME NOT NULL,
                    metadata TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create content table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS content_analyses (
                    content_id TEXT PRIMARY KEY,
                    title TEXT,
                    duration REAL,
                    platform TEXT,
                    performance_score REAL,
                    viral_potential REAL,
                    audience_engagement REAL,
                    quality_metrics TEXT,
                    recommendations TEXT,
                    created_at DATETIME,
                    updated_at DATETIME
                )
            ''')
            
            conn.commit()
            return conn
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            return None
    
    async def track_metric(self, metric: PerformanceMetric):
        """Track a performance metric."""
        try:
            if not self.metrics_db:
                return
            
            cursor = self.metrics_db.cursor()
            cursor.execute('''
                INSERT INTO performance_metrics 
                (content_id, platform, metric_type, value, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                metric.content_id,
                metric.platform,
                metric.metric_type.value,
                metric.value,
                metric.timestamp.isoformat(),
                json.dumps(metric.metadata)
            ))
            
            self.metrics_db.commit()
            
        except Exception as e:
            logger.error(f"Metric tracking failed: {e}")
    
    async def get_metrics(self, 
                         content_id: Optional[str] = None,
                         platform: Optional[str] = None,
                         metric_type: Optional[MetricType] = None,
                         time_range: TimeRange = TimeRange.WEEK) -> List[PerformanceMetric]:
        """Get performance metrics with filters."""
        try:
            if not self.metrics_db:
                return []
            
            # Build query
            query = "SELECT * FROM performance_metrics WHERE 1=1"
            params = []
            
            if content_id:
                query += " AND content_id = ?"
                params.append(content_id)
            
            if platform:
                query += " AND platform = ?"
                params.append(platform)
            
            if metric_type:
                query += " AND metric_type = ?"
                params.append(metric_type.value)
            
            # Add time range filter
            time_delta = self._get_time_delta(time_range)
            cutoff_time = datetime.now() - time_delta
            query += " AND timestamp >= ?"
            params.append(cutoff_time.isoformat())
            
            query += " ORDER BY timestamp DESC"
            
            cursor = self.metrics_db.cursor()
            cursor.execute(query, params)
            
            metrics = []
            for row in cursor.fetchall():
                metric = PerformanceMetric(
                    metric_type=MetricType(row[3]),
                    value=row[4],
                    timestamp=datetime.fromisoformat(row[5]),
                    platform=row[2],
                    content_id=row[1],
                    metadata=json.loads(row[6]) if row[6] else {}
                )
                metrics.append(metric)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Metrics retrieval failed: {e}")
            return []
    
    def _get_time_delta(self, time_range: TimeRange) -> timedelta:
        """Get time delta for time range."""
        deltas = {
            TimeRange.HOUR: timedelta(hours=1),
            TimeRange.DAY: timedelta(days=1),
            TimeRange.WEEK: timedelta(weeks=1),
            TimeRange.MONTH: timedelta(days=30),
            TimeRange.QUARTER: timedelta(days=90),
            TimeRange.YEAR: timedelta(days=365)
        }
        return deltas.get(time_range, timedelta(days=7))

class AudienceAnalyzer:
    """Analyzes audience demographics and behavior."""
    
    def __init__(self):
        self.audience_data = {}
        self.demographic_models = {}
        self._load_audience_models()
    
    def _load_audience_models(self):
        """Load audience demographic models."""
        try:
            # Placeholder - would load actual demographic models
            self.demographic_models = {
                "age_groups": {
                    "13-17": 0.15,
                    "18-24": 0.25,
                    "25-34": 0.30,
                    "35-44": 0.20,
                    "45+": 0.10
                },
                "genders": {
                    "male": 0.45,
                    "female": 0.50,
                    "other": 0.05
                },
                "interests": {
                    "technology": 0.30,
                    "entertainment": 0.25,
                    "lifestyle": 0.20,
                    "education": 0.15,
                    "sports": 0.10
                },
                "locations": {
                    "north_america": 0.40,
                    "europe": 0.25,
                    "asia": 0.20,
                    "other": 0.15
                }
            }
        except Exception as e:
            logger.error(f"Audience model loading failed: {e}")
    
    async def analyze_audience(self, 
                             content_id: str,
                             platform: str,
                             engagement_data: Dict[str, Any]) -> List[AudienceInsight]:
        """Analyze audience for specific content."""
        try:
            insights = []
            
            # Analyze age distribution
            age_insights = await self._analyze_age_distribution(content_id, platform, engagement_data)
            insights.extend(age_insights)
            
            # Analyze gender distribution
            gender_insights = await self._analyze_gender_distribution(content_id, platform, engagement_data)
            insights.extend(gender_insights)
            
            # Analyze interests
            interest_insights = await self._analyze_interests(content_id, platform, engagement_data)
            insights.extend(interest_insights)
            
            # Analyze geographic distribution
            location_insights = await self._analyze_locations(content_id, platform, engagement_data)
            insights.extend(location_insights)
            
            return insights
            
        except Exception as e:
            logger.error(f"Audience analysis failed: {e}")
            return []
    
    async def _analyze_age_distribution(self, content_id: str, platform: str, engagement_data: Dict[str, Any]) -> List[AudienceInsight]:
        """Analyze age distribution of audience."""
        try:
            insights = []
            
            # Get platform-specific age distribution
            platform_ages = self._get_platform_age_distribution(platform)
            
            for age_group, percentage in platform_ages.items():
                # Simulate content-specific adjustments
                content_multiplier = self._get_content_age_multiplier(content_id, age_group)
                adjusted_percentage = percentage * content_multiplier
                
                trend = self._calculate_trend(adjusted_percentage, platform_ages[age_group])
                confidence = min(0.9, 0.5 + abs(adjusted_percentage - percentage) * 2)
                
                insight = AudienceInsight(
                    demographic=f"age_{age_group}",
                    value=adjusted_percentage,
                    percentage=adjusted_percentage * 100,
                    trend=trend,
                    confidence=confidence
                )
                insights.append(insight)
            
            return insights
            
        except Exception as e:
            logger.error(f"Age distribution analysis failed: {e}")
            return []
    
    async def _analyze_gender_distribution(self, content_id: str, platform: str, engagement_data: Dict[str, Any]) -> List[AudienceInsight]:
        """Analyze gender distribution of audience."""
        try:
            insights = []
            
            platform_genders = self._get_platform_gender_distribution(platform)
            
            for gender, percentage in platform_genders.items():
                content_multiplier = self._get_content_gender_multiplier(content_id, gender)
                adjusted_percentage = percentage * content_multiplier
                
                trend = self._calculate_trend(adjusted_percentage, platform_genders[gender])
                confidence = min(0.9, 0.5 + abs(adjusted_percentage - percentage) * 2)
                
                insight = AudienceInsight(
                    demographic=f"gender_{gender}",
                    value=adjusted_percentage,
                    percentage=adjusted_percentage * 100,
                    trend=trend,
                    confidence=confidence
                )
                insights.append(insight)
            
            return insights
            
        except Exception as e:
            logger.error(f"Gender distribution analysis failed: {e}")
            return []
    
    async def _analyze_interests(self, content_id: str, platform: str, engagement_data: Dict[str, Any]) -> List[AudienceInsight]:
        """Analyze audience interests."""
        try:
            insights = []
            
            platform_interests = self._get_platform_interest_distribution(platform)
            
            for interest, percentage in platform_interests.items():
                content_multiplier = self._get_content_interest_multiplier(content_id, interest)
                adjusted_percentage = percentage * content_multiplier
                
                trend = self._calculate_trend(adjusted_percentage, platform_interests[interest])
                confidence = min(0.9, 0.5 + abs(adjusted_percentage - percentage) * 2)
                
                insight = AudienceInsight(
                    demographic=f"interest_{interest}",
                    value=adjusted_percentage,
                    percentage=adjusted_percentage * 100,
                    trend=trend,
                    confidence=confidence
                )
                insights.append(insight)
            
            return insights
            
        except Exception as e:
            logger.error(f"Interest analysis failed: {e}")
            return []
    
    async def _analyze_locations(self, content_id: str, platform: str, engagement_data: Dict[str, Any]) -> List[AudienceInsight]:
        """Analyze geographic distribution of audience."""
        try:
            insights = []
            
            platform_locations = self._get_platform_location_distribution(platform)
            
            for location, percentage in platform_locations.items():
                content_multiplier = self._get_content_location_multiplier(content_id, location)
                adjusted_percentage = percentage * content_multiplier
                
                trend = self._calculate_trend(adjusted_percentage, platform_locations[location])
                confidence = min(0.9, 0.5 + abs(adjusted_percentage - percentage) * 2)
                
                insight = AudienceInsight(
                    demographic=f"location_{location}",
                    value=adjusted_percentage,
                    percentage=adjusted_percentage * 100,
                    trend=trend,
                    confidence=confidence
                )
                insights.append(insight)
            
            return insights
            
        except Exception as e:
            logger.error(f"Location analysis failed: {e}")
            return []
    
    def _get_platform_age_distribution(self, platform: str) -> Dict[str, float]:
        """Get age distribution for platform."""
        distributions = {
            "tiktok": {"13-17": 0.25, "18-24": 0.35, "25-34": 0.25, "35-44": 0.10, "45+": 0.05},
            "youtube": {"13-17": 0.15, "18-24": 0.20, "25-34": 0.30, "35-44": 0.25, "45+": 0.10},
            "instagram": {"13-17": 0.20, "18-24": 0.30, "25-34": 0.35, "35-44": 0.10, "45+": 0.05},
            "twitter": {"13-17": 0.10, "18-24": 0.20, "25-34": 0.35, "35-44": 0.25, "45+": 0.10}
        }
        return distributions.get(platform, self.demographic_models["age_groups"])
    
    def _get_platform_gender_distribution(self, platform: str) -> Dict[str, float]:
        """Get gender distribution for platform."""
        distributions = {
            "tiktok": {"male": 0.40, "female": 0.55, "other": 0.05},
            "youtube": {"male": 0.55, "female": 0.40, "other": 0.05},
            "instagram": {"male": 0.35, "female": 0.60, "other": 0.05},
            "twitter": {"male": 0.60, "female": 0.35, "other": 0.05}
        }
        return distributions.get(platform, self.demographic_models["genders"])
    
    def _get_platform_interest_distribution(self, platform: str) -> Dict[str, float]:
        """Get interest distribution for platform."""
        distributions = {
            "tiktok": {"entertainment": 0.40, "lifestyle": 0.25, "dance": 0.20, "comedy": 0.15},
            "youtube": {"education": 0.30, "entertainment": 0.25, "technology": 0.20, "gaming": 0.15, "music": 0.10},
            "instagram": {"lifestyle": 0.35, "fashion": 0.25, "beauty": 0.20, "travel": 0.20},
            "twitter": {"news": 0.30, "technology": 0.25, "politics": 0.20, "sports": 0.15, "entertainment": 0.10}
        }
        return distributions.get(platform, self.demographic_models["interests"])
    
    def _get_platform_location_distribution(self, platform: str) -> Dict[str, float]:
        """Get location distribution for platform."""
        distributions = {
            "tiktok": {"asia": 0.40, "north_america": 0.25, "europe": 0.20, "other": 0.15},
            "youtube": {"north_america": 0.35, "europe": 0.25, "asia": 0.25, "other": 0.15},
            "instagram": {"north_america": 0.30, "europe": 0.30, "asia": 0.25, "other": 0.15},
            "twitter": {"north_america": 0.45, "europe": 0.30, "asia": 0.15, "other": 0.10}
        }
        return distributions.get(platform, self.demographic_models["locations"])
    
    def _get_content_age_multiplier(self, content_id: str, age_group: str) -> float:
        """Get content-specific age multiplier."""
        # Placeholder - would analyze content characteristics
        multipliers = {
            "13-17": 1.2,
            "18-24": 1.1,
            "25-34": 1.0,
            "35-44": 0.9,
            "45+": 0.8
        }
        return multipliers.get(age_group, 1.0)
    
    def _get_content_gender_multiplier(self, content_id: str, gender: str) -> float:
        """Get content-specific gender multiplier."""
        # Placeholder - would analyze content characteristics
        return 1.0
    
    def _get_content_interest_multiplier(self, content_id: str, interest: str) -> float:
        """Get content-specific interest multiplier."""
        # Placeholder - would analyze content characteristics
        return 1.0
    
    def _get_content_location_multiplier(self, content_id: str, location: str) -> float:
        """Get content-specific location multiplier."""
        # Placeholder - would analyze content characteristics
        return 1.0
    
    def _calculate_trend(self, current: float, baseline: float) -> str:
        """Calculate trend direction."""
        if current > baseline * 1.1:
            return "increasing"
        elif current < baseline * 0.9:
            return "decreasing"
        else:
            return "stable"

class ContentAnalyzer:
    """Analyzes content performance and quality."""
    
    def __init__(self):
        self.performance_tracker = PerformanceTracker()
        self.audience_analyzer = AudienceAnalyzer()
    
    async def analyze_content(self, 
                            content_id: str,
                            content_data: Dict[str, Any]) -> ContentAnalysis:
        """Perform comprehensive content analysis."""
        try:
            logger.info(f"Analyzing content: {content_id}")
            
            # Get performance metrics
            metrics = await self.performance_tracker.get_metrics(content_id=content_id)
            
            # Calculate performance score
            performance_score = await self._calculate_performance_score(metrics)
            
            # Calculate viral potential
            viral_potential = await self._calculate_viral_potential(content_data, metrics)
            
            # Calculate audience engagement
            audience_engagement = await self._calculate_audience_engagement(metrics)
            
            # Calculate quality metrics
            quality_metrics = await self._calculate_quality_metrics(content_data, metrics)
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(
                performance_score, viral_potential, audience_engagement, quality_metrics
            )
            
            # Analyze audience
            audience_insights = await self.audience_analyzer.analyze_audience(
                content_id, content_data.get("platform", "unknown"), content_data
            )
            
            return ContentAnalysis(
                content_id=content_id,
                title=content_data.get("title", "Untitled"),
                duration=content_data.get("duration", 0),
                platform=content_data.get("platform", "unknown"),
                performance_score=performance_score,
                viral_potential=viral_potential,
                audience_engagement=audience_engagement,
                quality_metrics=quality_metrics,
                recommendations=recommendations,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Content analysis failed: {e}")
            raise ProcessingError(f"Content analysis failed: {e}")
    
    async def _calculate_performance_score(self, metrics: List[PerformanceMetric]) -> float:
        """Calculate overall performance score."""
        try:
            if not metrics:
                return 0.5
            
            # Group metrics by type
            metric_groups = defaultdict(list)
            for metric in metrics:
                metric_groups[metric.metric_type].append(metric.value)
            
            # Calculate scores for each metric type
            scores = {}
            for metric_type, values in metric_groups.items():
                if values:
                    scores[metric_type] = np.mean(values)
            
            # Weighted average of all scores
            weights = {
                MetricType.ENGAGEMENT: 0.3,
                MetricType.REACH: 0.25,
                MetricType.RETENTION: 0.2,
                MetricType.CONVERSION: 0.15,
                MetricType.QUALITY: 0.1
            }
            
            total_score = 0.0
            total_weight = 0.0
            
            for metric_type, score in scores.items():
                weight = weights.get(metric_type, 0.1)
                total_score += score * weight
                total_weight += weight
            
            return total_score / total_weight if total_weight > 0 else 0.5
            
        except Exception as e:
            logger.error(f"Performance score calculation failed: {e}")
            return 0.5
    
    async def _calculate_viral_potential(self, content_data: Dict[str, Any], metrics: List[PerformanceMetric]) -> float:
        """Calculate viral potential score."""
        try:
            # Base viral potential from content characteristics
            base_potential = content_data.get("viral_score", 0.5)
            
            # Adjust based on metrics
            engagement_metrics = [m for m in metrics if m.metric_type == MetricType.ENGAGEMENT]
            if engagement_metrics:
                avg_engagement = np.mean([m.value for m in engagement_metrics])
                viral_potential = (base_potential + avg_engagement) / 2
            else:
                viral_potential = base_potential
            
            return min(viral_potential, 1.0)
            
        except Exception as e:
            logger.error(f"Viral potential calculation failed: {e}")
            return 0.5
    
    async def _calculate_audience_engagement(self, metrics: List[PerformanceMetric]) -> float:
        """Calculate audience engagement score."""
        try:
            engagement_metrics = [m for m in metrics if m.metric_type == MetricType.ENGAGEMENT]
            
            if not engagement_metrics:
                return 0.5
            
            # Calculate engagement score
            engagement_values = [m.value for m in engagement_metrics]
            
            # Weight recent metrics more heavily
            weights = np.linspace(1.0, 0.5, len(engagement_values))
            weighted_engagement = np.average(engagement_values, weights=weights)
            
            return min(weighted_engagement, 1.0)
            
        except Exception as e:
            logger.error(f"Audience engagement calculation failed: {e}")
            return 0.5
    
    async def _calculate_quality_metrics(self, content_data: Dict[str, Any], metrics: List[PerformanceMetric]) -> Dict[str, float]:
        """Calculate quality metrics."""
        try:
            quality_metrics = {}
            
            # Video quality
            quality_metrics["video_quality"] = content_data.get("video_quality", 0.8)
            
            # Audio quality
            quality_metrics["audio_quality"] = content_data.get("audio_quality", 0.8)
            
            # Content coherence
            quality_metrics["content_coherence"] = content_data.get("content_coherence", 0.7)
            
            # Technical quality
            quality_metrics["technical_quality"] = content_data.get("technical_quality", 0.8)
            
            # Overall quality
            quality_metrics["overall_quality"] = np.mean(list(quality_metrics.values()))
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Quality metrics calculation failed: {e}")
            return {"overall_quality": 0.5}
    
    async def _generate_recommendations(self, 
                                      performance_score: float,
                                      viral_potential: float,
                                      audience_engagement: float,
                                      quality_metrics: Dict[str, float]) -> List[str]:
        """Generate optimization recommendations."""
        try:
            recommendations = []
            
            # Performance recommendations
            if performance_score < 0.6:
                recommendations.append("Focus on improving overall performance metrics")
            
            # Viral potential recommendations
            if viral_potential < 0.5:
                recommendations.append("Increase viral potential by adding trending elements")
            
            # Engagement recommendations
            if audience_engagement < 0.6:
                recommendations.append("Improve audience engagement with interactive content")
            
            # Quality recommendations
            overall_quality = quality_metrics.get("overall_quality", 0.5)
            if overall_quality < 0.7:
                recommendations.append("Improve content quality for better viewer retention")
            
            # Specific quality recommendations
            if quality_metrics.get("video_quality", 0.8) < 0.7:
                recommendations.append("Upgrade video resolution and encoding quality")
            
            if quality_metrics.get("audio_quality", 0.8) < 0.7:
                recommendations.append("Improve audio quality and clarity")
            
            if quality_metrics.get("content_coherence", 0.7) < 0.6:
                recommendations.append("Improve content structure and flow")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            return ["Focus on creating high-quality, engaging content"]

class ReportGenerator:
    """Generates comprehensive analytics reports."""
    
    def __init__(self):
        self.content_analyzer = ContentAnalyzer()
        self.performance_tracker = PerformanceTracker()
    
    async def generate_report(self, 
                            content_ids: List[str],
                            time_range: TimeRange = TimeRange.WEEK) -> AnalyticsReport:
        """Generate comprehensive analytics report."""
        try:
            logger.info(f"Generating analytics report for {len(content_ids)} content items")
            
            # Analyze each content item
            content_analyses = []
            for content_id in content_ids:
                try:
                    # Get content data (placeholder)
                    content_data = await self._get_content_data(content_id)
                    analysis = await self.content_analyzer.analyze_content(content_id, content_data)
                    content_analyses.append(analysis)
                except Exception as e:
                    logger.error(f"Failed to analyze content {content_id}: {e}")
                    continue
            
            # Generate performance summary
            performance_summary = await self._generate_performance_summary(content_analyses)
            
            # Generate audience insights
            audience_insights = await self._generate_audience_insights(content_analyses)
            
            # Generate trends
            trends = await self._generate_trends(content_analyses, time_range)
            
            # Generate recommendations
            recommendations = await self._generate_report_recommendations(content_analyses)
            
            return AnalyticsReport(
                report_id=f"report_{int(time.time())}",
                content_analyses=content_analyses,
                performance_summary=performance_summary,
                audience_insights=audience_insights,
                trends=trends,
                recommendations=recommendations,
                generated_at=datetime.now(),
                time_range=time_range
            )
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            raise ProcessingError(f"Report generation failed: {e}")
    
    async def _get_content_data(self, content_id: str) -> Dict[str, Any]:
        """Get content data for analysis."""
        # Placeholder - would fetch from database
        return {
            "title": f"Content {content_id}",
            "duration": 30.0,
            "platform": "tiktok",
            "viral_score": 0.7,
            "video_quality": 0.8,
            "audio_quality": 0.8,
            "content_coherence": 0.7,
            "technical_quality": 0.8
        }
    
    async def _generate_performance_summary(self, analyses: List[ContentAnalysis]) -> Dict[str, Any]:
        """Generate performance summary."""
        try:
            if not analyses:
                return {"error": "No content analyses available"}
            
            performance_scores = [a.performance_score for a in analyses]
            viral_potentials = [a.viral_potential for a in analyses]
            audience_engagements = [a.audience_engagement for a in analyses]
            
            return {
                "total_content": len(analyses),
                "average_performance_score": np.mean(performance_scores),
                "average_viral_potential": np.mean(viral_potentials),
                "average_audience_engagement": np.mean(audience_engagements),
                "top_performing_content": max(analyses, key=lambda x: x.performance_score).content_id,
                "highest_viral_potential": max(analyses, key=lambda x: x.viral_potential).content_id
            }
            
        except Exception as e:
            logger.error(f"Performance summary generation failed: {e}")
            return {"error": str(e)}
    
    async def _generate_audience_insights(self, analyses: List[ContentAnalysis]) -> List[AudienceInsight]:
        """Generate audience insights from analyses."""
        try:
            # Placeholder - would aggregate audience data from analyses
            insights = []
            
            # Sample insights
            insights.append(AudienceInsight(
                demographic="age_18_24",
                value=0.35,
                percentage=35.0,
                trend="increasing",
                confidence=0.8
            ))
            
            insights.append(AudienceInsight(
                demographic="interest_technology",
                value=0.25,
                percentage=25.0,
                trend="stable",
                confidence=0.7
            ))
            
            return insights
            
        except Exception as e:
            logger.error(f"Audience insights generation failed: {e}")
            return []
    
    async def _generate_trends(self, analyses: List[ContentAnalysis], time_range: TimeRange) -> Dict[str, Any]:
        """Generate trend analysis."""
        try:
            # Placeholder - would analyze trends over time
            return {
                "performance_trend": "increasing",
                "engagement_trend": "stable",
                "viral_potential_trend": "increasing",
                "quality_trend": "stable"
            }
            
        except Exception as e:
            logger.error(f"Trend generation failed: {e}")
            return {"error": str(e)}
    
    async def _generate_report_recommendations(self, analyses: List[ContentAnalysis]) -> List[str]:
        """Generate report-level recommendations."""
        try:
            recommendations = []
            
            # Analyze overall performance
            avg_performance = np.mean([a.performance_score for a in analyses])
            if avg_performance < 0.6:
                recommendations.append("Overall content performance is below average - focus on engagement strategies")
            
            # Analyze viral potential
            avg_viral = np.mean([a.viral_potential for a in analyses])
            if avg_viral < 0.5:
                recommendations.append("Low viral potential across content - incorporate trending elements")
            
            # Analyze quality
            avg_quality = np.mean([a.quality_metrics.get("overall_quality", 0.5) for a in analyses])
            if avg_quality < 0.7:
                recommendations.append("Content quality needs improvement - invest in better production")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Report recommendations generation failed: {e}")
            return ["Focus on creating high-quality, engaging content"]

class AdvancedAnalyticsSystem:
    """Main advanced analytics system."""
    
    def __init__(self):
        self.performance_tracker = PerformanceTracker()
        self.audience_analyzer = AudienceAnalyzer()
        self.content_analyzer = ContentAnalyzer()
        self.report_generator = ReportGenerator()
    
    async def track_content_performance(self, 
                                      content_id: str,
                                      platform: str,
                                      metrics: Dict[str, float]) -> bool:
        """Track content performance metrics."""
        try:
            for metric_type, value in metrics.items():
                metric = PerformanceMetric(
                    metric_type=MetricType(metric_type),
                    value=value,
                    timestamp=datetime.now(),
                    platform=platform,
                    content_id=content_id,
                    metadata={}
                )
                await self.performance_tracker.track_metric(metric)
            
            return True
            
        except Exception as e:
            logger.error(f"Performance tracking failed: {e}")
            return False
    
    async def analyze_content_performance(self, content_id: str) -> ContentAnalysis:
        """Analyze performance of specific content."""
        try:
            content_data = await self._get_content_data(content_id)
            return await self.content_analyzer.analyze_content(content_id, content_data)
        except Exception as e:
            logger.error(f"Content performance analysis failed: {e}")
            raise ProcessingError(f"Content performance analysis failed: {e}")
    
    async def generate_analytics_report(self, 
                                      content_ids: List[str],
                                      time_range: TimeRange = TimeRange.WEEK) -> AnalyticsReport:
        """Generate comprehensive analytics report."""
        try:
            return await self.report_generator.generate_report(content_ids, time_range)
        except Exception as e:
            logger.error(f"Analytics report generation failed: {e}")
            raise ProcessingError(f"Analytics report generation failed: {e}")
    
    async def _get_content_data(self, content_id: str) -> Dict[str, Any]:
        """Get content data for analysis."""
        # Placeholder - would fetch from database
        return {
            "title": f"Content {content_id}",
            "duration": 30.0,
            "platform": "tiktok",
            "viral_score": 0.7,
            "video_quality": 0.8,
            "audio_quality": 0.8,
            "content_coherence": 0.7,
            "technical_quality": 0.8
        }

# Export the main class
__all__ = ["AdvancedAnalyticsSystem", "PerformanceTracker", "AudienceAnalyzer", "ContentAnalyzer", "ReportGenerator"]


