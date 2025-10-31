"""
Trend Analysis and Content Prediction System
===========================================

This module provides advanced trend analysis, content prediction, and market intelligence
for the Document Workflow Chain system.
"""

import asyncio
import logging
import json
import re
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import statistics
from collections import defaultdict, Counter
import hashlib
import math

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class TrendData:
    """Trend data structure"""
    topic: str
    category: str
    popularity_score: float
    growth_rate: float
    engagement_rate: float
    search_volume: int
    social_mentions: int
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ContentPrediction:
    """Content prediction result"""
    topic: str
    predicted_popularity: float
    predicted_engagement: float
    confidence_score: float
    recommended_content_type: str
    optimal_publishing_time: datetime
    target_audience: str
    content_suggestions: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)

@dataclass
class MarketIntelligence:
    """Market intelligence data"""
    market_size: int
    competition_level: float
    opportunity_score: float
    trending_topics: List[str]
    emerging_keywords: List[str]
    audience_insights: Dict[str, Any]
    content_gaps: List[str]
    recommendations: List[str]

class TrendAnalyzer:
    """Advanced trend analysis and prediction system"""
    
    def __init__(self):
        self.trend_data: List[TrendData] = []
        self.historical_data: Dict[str, List[float]] = defaultdict(list)
        self.prediction_models: Dict[str, Any] = {}
        self.market_indicators: Dict[str, float] = {}
        self.content_performance: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.audience_behavior: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Initialize trend categories
        self.trend_categories = {
            "technology": ["ai", "machine learning", "blockchain", "cloud", "automation"],
            "business": ["strategy", "leadership", "management", "entrepreneurship", "innovation"],
            "marketing": ["digital marketing", "seo", "social media", "content marketing", "analytics"],
            "lifestyle": ["health", "fitness", "travel", "food", "fashion"],
            "education": ["learning", "training", "skills", "development", "certification"]
        }
        
        # Initialize engagement factors
        self.engagement_factors = {
            "high_engagement": ["tutorial", "guide", "how-to", "tips", "secrets", "mistakes"],
            "medium_engagement": ["overview", "introduction", "basics", "fundamentals", "explanation"],
            "low_engagement": ["definition", "summary", "conclusion", "abstract", "overview"]
        }
    
    async def analyze_trends(
        self,
        time_period: int = 30,
        categories: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze trends across specified categories and time period
        
        Args:
            time_period: Number of days to analyze
            categories: List of categories to analyze
            
        Returns:
            Dict containing trend analysis results
        """
        try:
            categories = categories or list(self.trend_categories.keys())
            end_date = datetime.now()
            start_date = end_date - timedelta(days=time_period)
            
            # Filter trend data for the specified period
            period_data = [
                trend for trend in self.trend_data
                if start_date <= trend.timestamp <= end_date
            ]
            
            analysis_results = {
                "period": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat(),
                    "days": time_period
                },
                "categories": {},
                "overall_trends": {},
                "predictions": {},
                "recommendations": []
            }
            
            # Analyze trends by category
            for category in categories:
                category_data = [trend for trend in period_data if trend.category == category]
                if category_data:
                    analysis_results["categories"][category] = await self._analyze_category_trends(
                        category_data, category
                    )
            
            # Calculate overall trends
            analysis_results["overall_trends"] = await self._calculate_overall_trends(period_data)
            
            # Generate predictions
            analysis_results["predictions"] = await self._generate_trend_predictions(
                period_data, time_period
            )
            
            # Generate recommendations
            analysis_results["recommendations"] = await self._generate_trend_recommendations(
                analysis_results
            )
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error analyzing trends: {str(e)}")
            return {"error": str(e)}
    
    async def predict_content_success(
        self,
        topic: str,
        content_type: str,
        target_audience: str,
        publishing_time: Optional[datetime] = None
    ) -> ContentPrediction:
        """
        Predict content success based on trends and historical data
        
        Args:
            topic: Content topic
            content_type: Type of content (blog, article, guide, etc.)
            target_audience: Target audience
            publishing_time: Planned publishing time
            
        Returns:
            ContentPrediction: Prediction results
        """
        try:
            # Analyze topic trends
            topic_trends = await self._analyze_topic_trends(topic)
            
            # Predict popularity
            predicted_popularity = await self._predict_popularity(
                topic, content_type, topic_trends
            )
            
            # Predict engagement
            predicted_engagement = await self._predict_engagement(
                topic, content_type, target_audience, topic_trends
            )
            
            # Calculate confidence score
            confidence_score = await self._calculate_prediction_confidence(
                topic, content_type, topic_trends
            )
            
            # Recommend optimal content type
            recommended_content_type = await self._recommend_content_type(
                topic, target_audience, topic_trends
            )
            
            # Calculate optimal publishing time
            optimal_publishing_time = await self._calculate_optimal_publishing_time(
                topic, target_audience, publishing_time
            )
            
            # Generate content suggestions
            content_suggestions = await self._generate_content_suggestions(
                topic, content_type, topic_trends
            )
            
            # Identify risk factors
            risk_factors = await self._identify_risk_factors(
                topic, content_type, topic_trends
            )
            
            return ContentPrediction(
                topic=topic,
                predicted_popularity=predicted_popularity,
                predicted_engagement=predicted_engagement,
                confidence_score=confidence_score,
                recommended_content_type=recommended_content_type,
                optimal_publishing_time=optimal_publishing_time,
                target_audience=target_audience,
                content_suggestions=content_suggestions,
                risk_factors=risk_factors
            )
            
        except Exception as e:
            logger.error(f"Error predicting content success: {str(e)}")
            return ContentPrediction(
                topic=topic,
                predicted_popularity=0.5,
                predicted_engagement=0.5,
                confidence_score=0.0,
                recommended_content_type=content_type,
                optimal_publishing_time=datetime.now(),
                target_audience=target_audience,
                content_suggestions=[],
                risk_factors=["Prediction error"]
            )
    
    async def get_market_intelligence(
        self,
        market_segment: str,
        time_period: int = 90
    ) -> MarketIntelligence:
        """
        Get comprehensive market intelligence for a specific segment
        
        Args:
            market_segment: Market segment to analyze
            time_period: Time period for analysis
            
        Returns:
            MarketIntelligence: Market intelligence data
        """
        try:
            # Calculate market size
            market_size = await self._calculate_market_size(market_segment)
            
            # Assess competition level
            competition_level = await self._assess_competition_level(market_segment)
            
            # Calculate opportunity score
            opportunity_score = await self._calculate_opportunity_score(
                market_segment, market_size, competition_level
            )
            
            # Identify trending topics
            trending_topics = await self._identify_trending_topics(market_segment, time_period)
            
            # Find emerging keywords
            emerging_keywords = await self._find_emerging_keywords(market_segment, time_period)
            
            # Analyze audience insights
            audience_insights = await self._analyze_audience_insights(market_segment)
            
            # Identify content gaps
            content_gaps = await self._identify_content_gaps(market_segment)
            
            # Generate recommendations
            recommendations = await self._generate_market_recommendations(
                market_segment, opportunity_score, content_gaps
            )
            
            return MarketIntelligence(
                market_size=market_size,
                competition_level=competition_level,
                opportunity_score=opportunity_score,
                trending_topics=trending_topics,
                emerging_keywords=emerging_keywords,
                audience_insights=audience_insights,
                content_gaps=content_gaps,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Error getting market intelligence: {str(e)}")
            return MarketIntelligence(
                market_size=0,
                competition_level=0.0,
                opportunity_score=0.0,
                trending_topics=[],
                emerging_keywords=[],
                audience_insights={},
                content_gaps=[],
                recommendations=["Error in market analysis"]
            )
    
    async def add_trend_data(self, trend_data: TrendData):
        """Add new trend data to the system"""
        try:
            self.trend_data.append(trend_data)
            
            # Update historical data
            topic_key = f"{trend_data.category}_{trend_data.topic}"
            self.historical_data[topic_key].append({
                "timestamp": trend_data.timestamp,
                "popularity": trend_data.popularity_score,
                "engagement": trend_data.engagement_rate,
                "growth": trend_data.growth_rate
            })
            
            # Keep only recent data (last 365 days)
            cutoff_date = datetime.now() - timedelta(days=365)
            self.trend_data = [
                trend for trend in self.trend_data
                if trend.timestamp >= cutoff_date
            ]
            
            # Update historical data
            for topic_key in self.historical_data:
                self.historical_data[topic_key] = [
                    data for data in self.historical_data[topic_key]
                    if data["timestamp"] >= cutoff_date
                ]
            
        except Exception as e:
            logger.error(f"Error adding trend data: {str(e)}")
    
    async def _analyze_category_trends(
        self,
        category_data: List[TrendData],
        category: str
    ) -> Dict[str, Any]:
        """Analyze trends for a specific category"""
        try:
            if not category_data:
                return {"error": "No data available"}
            
            # Calculate average metrics
            avg_popularity = statistics.mean([trend.popularity_score for trend in category_data])
            avg_engagement = statistics.mean([trend.engagement_rate for trend in category_data])
            avg_growth = statistics.mean([trend.growth_rate for trend in category_data])
            
            # Find top performing topics
            top_topics = sorted(
                category_data,
                key=lambda x: x.popularity_score * x.engagement_rate,
                reverse=True
            )[:5]
            
            # Calculate trend direction
            trend_direction = await self._calculate_trend_direction(category_data)
            
            # Identify emerging topics
            emerging_topics = await self._identify_emerging_topics(category_data)
            
            return {
                "average_metrics": {
                    "popularity": avg_popularity,
                    "engagement": avg_engagement,
                    "growth_rate": avg_growth
                },
                "top_topics": [
                    {
                        "topic": trend.topic,
                        "popularity": trend.popularity_score,
                        "engagement": trend.engagement_rate,
                        "growth": trend.growth_rate
                    }
                    for trend in top_topics
                ],
                "trend_direction": trend_direction,
                "emerging_topics": emerging_topics,
                "total_topics": len(category_data)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing category trends: {str(e)}")
            return {"error": str(e)}
    
    async def _calculate_overall_trends(self, period_data: List[TrendData]) -> Dict[str, Any]:
        """Calculate overall trend metrics"""
        try:
            if not period_data:
                return {"error": "No data available"}
            
            # Calculate overall metrics
            total_popularity = sum(trend.popularity_score for trend in period_data)
            total_engagement = sum(trend.engagement_rate for trend in period_data)
            total_growth = sum(trend.growth_rate for trend in period_data)
            
            # Calculate averages
            avg_popularity = total_popularity / len(period_data)
            avg_engagement = total_engagement / len(period_data)
            avg_growth = total_growth / len(period_data)
            
            # Find most popular categories
            category_popularity = defaultdict(float)
            for trend in period_data:
                category_popularity[trend.category] += trend.popularity_score
            
            top_categories = sorted(
                category_popularity.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            
            # Calculate trend momentum
            momentum = await self._calculate_trend_momentum(period_data)
            
            return {
                "average_metrics": {
                    "popularity": avg_popularity,
                    "engagement": avg_engagement,
                    "growth_rate": avg_growth
                },
                "top_categories": [
                    {"category": cat, "popularity": pop}
                    for cat, pop in top_categories
                ],
                "trend_momentum": momentum,
                "total_topics": len(period_data)
            }
            
        except Exception as e:
            logger.error(f"Error calculating overall trends: {str(e)}")
            return {"error": str(e)}
    
    async def _generate_trend_predictions(
        self,
        period_data: List[TrendData],
        time_period: int
    ) -> Dict[str, Any]:
        """Generate trend predictions based on historical data"""
        try:
            predictions = {
                "short_term": {},  # Next 7 days
                "medium_term": {},  # Next 30 days
                "long_term": {}  # Next 90 days
            }
            
            # Group data by category
            category_data = defaultdict(list)
            for trend in period_data:
                category_data[trend.category].append(trend)
            
            # Generate predictions for each category
            for category, trends in category_data.items():
                if len(trends) >= 3:  # Need minimum data points
                    # Short-term prediction
                    predictions["short_term"][category] = await self._predict_category_trends(
                        trends, 7
                    )
                    
                    # Medium-term prediction
                    predictions["medium_term"][category] = await self._predict_category_trends(
                        trends, 30
                    )
                    
                    # Long-term prediction
                    predictions["long_term"][category] = await self._predict_category_trends(
                        trends, 90
                    )
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error generating trend predictions: {str(e)}")
            return {"error": str(e)}
    
    async def _generate_trend_recommendations(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on trend analysis"""
        try:
            recommendations = []
            
            # Analyze overall trends
            overall_trends = analysis_results.get("overall_trends", {})
            if overall_trends.get("trend_momentum", 0) > 0.7:
                recommendations.append("High trend momentum detected - consider increasing content production")
            elif overall_trends.get("trend_momentum", 0) < 0.3:
                recommendations.append("Low trend momentum - focus on evergreen content")
            
            # Analyze category performance
            categories = analysis_results.get("categories", {})
            for category, data in categories.items():
                if data.get("average_metrics", {}).get("engagement", 0) > 0.7:
                    recommendations.append(f"High engagement in {category} - consider expanding content in this area")
                elif data.get("average_metrics", {}).get("engagement", 0) < 0.3:
                    recommendations.append(f"Low engagement in {category} - review content strategy")
            
            # Analyze predictions
            predictions = analysis_results.get("predictions", {})
            short_term = predictions.get("short_term", {})
            for category, prediction in short_term.items():
                if prediction.get("predicted_growth", 0) > 0.5:
                    recommendations.append(f"Expected growth in {category} - prepare content calendar")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating trend recommendations: {str(e)}")
            return ["Error generating recommendations"]
    
    async def _analyze_topic_trends(self, topic: str) -> Dict[str, Any]:
        """Analyze trends for a specific topic"""
        try:
            # Find relevant trend data
            topic_data = [
                trend for trend in self.trend_data
                if topic.lower() in trend.topic.lower() or trend.topic.lower() in topic.lower()
            ]
            
            if not topic_data:
                return {"error": "No trend data available for topic"}
            
            # Calculate trend metrics
            avg_popularity = statistics.mean([trend.popularity_score for trend in topic_data])
            avg_engagement = statistics.mean([trend.engagement_rate for trend in topic_data])
            avg_growth = statistics.mean([trend.growth_rate for trend in topic_data])
            
            # Calculate trend direction
            trend_direction = await self._calculate_trend_direction(topic_data)
            
            # Find related topics
            related_topics = await self._find_related_topics(topic, topic_data)
            
            return {
                "topic": topic,
                "average_metrics": {
                    "popularity": avg_popularity,
                    "engagement": avg_engagement,
                    "growth_rate": avg_growth
                },
                "trend_direction": trend_direction,
                "related_topics": related_topics,
                "data_points": len(topic_data)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing topic trends: {str(e)}")
            return {"error": str(e)}
    
    async def _predict_popularity(
        self,
        topic: str,
        content_type: str,
        topic_trends: Dict[str, Any]
    ) -> float:
        """Predict content popularity"""
        try:
            base_popularity = topic_trends.get("average_metrics", {}).get("popularity", 0.5)
            
            # Adjust based on content type
            content_type_multipliers = {
                "blog": 1.0,
                "article": 0.9,
                "guide": 1.1,
                "tutorial": 1.2,
                "case_study": 0.8,
                "infographic": 1.3
            }
            
            multiplier = content_type_multipliers.get(content_type, 1.0)
            predicted_popularity = base_popularity * multiplier
            
            # Adjust based on trend direction
            trend_direction = topic_trends.get("trend_direction", "stable")
            if trend_direction == "rising":
                predicted_popularity *= 1.2
            elif trend_direction == "falling":
                predicted_popularity *= 0.8
            
            return min(1.0, max(0.0, predicted_popularity))
            
        except Exception as e:
            logger.error(f"Error predicting popularity: {str(e)}")
            return 0.5
    
    async def _predict_engagement(
        self,
        topic: str,
        content_type: str,
        target_audience: str,
        topic_trends: Dict[str, Any]
    ) -> float:
        """Predict content engagement"""
        try:
            base_engagement = topic_trends.get("average_metrics", {}).get("engagement", 0.5)
            
            # Adjust based on content type
            engagement_multipliers = {
                "blog": 1.0,
                "article": 0.8,
                "guide": 1.3,
                "tutorial": 1.4,
                "case_study": 1.1,
                "infographic": 1.2
            }
            
            multiplier = engagement_multipliers.get(content_type, 1.0)
            predicted_engagement = base_engagement * multiplier
            
            # Adjust based on audience
            audience_multipliers = {
                "general": 1.0,
                "professionals": 0.9,
                "beginners": 1.2,
                "experts": 0.8,
                "students": 1.1
            }
            
            audience_multiplier = audience_multipliers.get(target_audience, 1.0)
            predicted_engagement *= audience_multiplier
            
            return min(1.0, max(0.0, predicted_engagement))
            
        except Exception as e:
            logger.error(f"Error predicting engagement: {str(e)}")
            return 0.5
    
    async def _calculate_prediction_confidence(
        self,
        topic: str,
        content_type: str,
        topic_trends: Dict[str, Any]
    ) -> float:
        """Calculate confidence score for predictions"""
        try:
            confidence = 0.5  # Base confidence
            
            # Adjust based on data availability
            data_points = topic_trends.get("data_points", 0)
            if data_points >= 10:
                confidence += 0.3
            elif data_points >= 5:
                confidence += 0.2
            elif data_points >= 3:
                confidence += 0.1
            
            # Adjust based on trend consistency
            trend_direction = topic_trends.get("trend_direction", "stable")
            if trend_direction in ["rising", "falling"]:
                confidence += 0.1
            
            # Adjust based on content type familiarity
            if content_type in ["blog", "article"]:
                confidence += 0.1
            
            return min(1.0, max(0.0, confidence))
            
        except Exception as e:
            logger.error(f"Error calculating prediction confidence: {str(e)}")
            return 0.5
    
    async def _recommend_content_type(
        self,
        topic: str,
        target_audience: str,
        topic_trends: Dict[str, Any]
    ) -> str:
        """Recommend optimal content type"""
        try:
            engagement = topic_trends.get("average_metrics", {}).get("engagement", 0.5)
            popularity = topic_trends.get("average_metrics", {}).get("popularity", 0.5)
            
            # High engagement topics work well with interactive content
            if engagement > 0.7:
                if target_audience == "beginners":
                    return "tutorial"
                elif target_audience == "professionals":
                    return "case_study"
                else:
                    return "guide"
            
            # High popularity topics work well with accessible content
            elif popularity > 0.7:
                return "blog"
            
            # Medium metrics work well with comprehensive content
            else:
                return "article"
                
        except Exception as e:
            logger.error(f"Error recommending content type: {str(e)}")
            return "blog"
    
    async def _calculate_optimal_publishing_time(
        self,
        topic: str,
        target_audience: str,
        publishing_time: Optional[datetime]
    ) -> datetime:
        """Calculate optimal publishing time"""
        try:
            if publishing_time:
                return publishing_time
            
            # Default optimal times based on audience
            optimal_times = {
                "professionals": datetime.now().replace(hour=9, minute=0, second=0),  # 9 AM
                "students": datetime.now().replace(hour=14, minute=0, second=0),  # 2 PM
                "general": datetime.now().replace(hour=10, minute=0, second=0),  # 10 AM
                "beginners": datetime.now().replace(hour=19, minute=0, second=0),  # 7 PM
                "experts": datetime.now().replace(hour=8, minute=0, second=0)  # 8 AM
            }
            
            return optimal_times.get(target_audience, datetime.now())
            
        except Exception as e:
            logger.error(f"Error calculating optimal publishing time: {str(e)}")
            return datetime.now()
    
    async def _generate_content_suggestions(
        self,
        topic: str,
        content_type: str,
        topic_trends: Dict[str, Any]
    ) -> List[str]:
        """Generate content suggestions based on trends"""
        try:
            suggestions = []
            
            # Base suggestions
            suggestions.append(f"Create a comprehensive {content_type} about {topic}")
            
            # Add trend-based suggestions
            related_topics = topic_trends.get("related_topics", [])
            if related_topics:
                suggestions.append(f"Consider covering related topics: {', '.join(related_topics[:3])}")
            
            # Add engagement suggestions
            engagement = topic_trends.get("average_metrics", {}).get("engagement", 0.5)
            if engagement > 0.7:
                suggestions.append("Include interactive elements like quizzes or polls")
                suggestions.append("Add case studies and real-world examples")
            elif engagement < 0.3:
                suggestions.append("Focus on practical, actionable content")
                suggestions.append("Include step-by-step instructions")
            
            # Add content type specific suggestions
            if content_type == "guide":
                suggestions.append("Structure as a step-by-step walkthrough")
                suggestions.append("Include troubleshooting section")
            elif content_type == "tutorial":
                suggestions.append("Add screenshots or visual aids")
                suggestions.append("Include practice exercises")
            elif content_type == "case_study":
                suggestions.append("Include before/after comparisons")
                suggestions.append("Add lessons learned section")
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Error generating content suggestions: {str(e)}")
            return [f"Create content about {topic}"]
    
    async def _identify_risk_factors(
        self,
        topic: str,
        content_type: str,
        topic_trends: Dict[str, Any]
    ) -> List[str]:
        """Identify potential risk factors"""
        try:
            risk_factors = []
            
            # Check for declining trends
            trend_direction = topic_trends.get("trend_direction", "stable")
            if trend_direction == "falling":
                risk_factors.append("Topic shows declining interest")
            
            # Check for low engagement
            engagement = topic_trends.get("average_metrics", {}).get("engagement", 0.5)
            if engagement < 0.3:
                risk_factors.append("Low historical engagement for this topic")
            
            # Check for low popularity
            popularity = topic_trends.get("average_metrics", {}).get("popularity", 0.5)
            if popularity < 0.3:
                risk_factors.append("Low historical popularity for this topic")
            
            # Check for insufficient data
            data_points = topic_trends.get("data_points", 0)
            if data_points < 3:
                risk_factors.append("Limited historical data for accurate prediction")
            
            # Check for content type mismatch
            if content_type == "tutorial" and engagement < 0.4:
                risk_factors.append("Tutorial format may not be optimal for this topic")
            
            return risk_factors
            
        except Exception as e:
            logger.error(f"Error identifying risk factors: {str(e)}")
            return ["Unable to assess risks"]
    
    async def _calculate_trend_direction(self, trend_data: List[TrendData]) -> str:
        """Calculate trend direction (rising, falling, stable)"""
        try:
            if len(trend_data) < 2:
                return "stable"
            
            # Sort by timestamp
            sorted_data = sorted(trend_data, key=lambda x: x.timestamp)
            
            # Calculate growth rate over time
            early_popularity = sorted_data[0].popularity_score
            late_popularity = sorted_data[-1].popularity_score
            
            growth_rate = (late_popularity - early_popularity) / early_popularity if early_popularity > 0 else 0
            
            if growth_rate > 0.1:
                return "rising"
            elif growth_rate < -0.1:
                return "falling"
            else:
                return "stable"
                
        except Exception as e:
            logger.error(f"Error calculating trend direction: {str(e)}")
            return "stable"
    
    async def _identify_emerging_topics(self, category_data: List[TrendData]) -> List[str]:
        """Identify emerging topics in a category"""
        try:
            # Find topics with high growth rate
            emerging_topics = [
                trend.topic for trend in category_data
                if trend.growth_rate > 0.5 and trend.popularity_score > 0.3
            ]
            
            return emerging_topics[:5]  # Return top 5 emerging topics
            
        except Exception as e:
            logger.error(f"Error identifying emerging topics: {str(e)}")
            return []
    
    async def _calculate_trend_momentum(self, period_data: List[TrendData]) -> float:
        """Calculate overall trend momentum"""
        try:
            if len(period_data) < 2:
                return 0.5
            
            # Calculate average growth rate
            avg_growth = statistics.mean([trend.growth_rate for trend in period_data])
            
            # Calculate momentum score
            momentum = (avg_growth + 1) / 2  # Normalize to 0-1 range
            
            return min(1.0, max(0.0, momentum))
            
        except Exception as e:
            logger.error(f"Error calculating trend momentum: {str(e)}")
            return 0.5
    
    async def _predict_category_trends(
        self,
        trends: List[TrendData],
        days_ahead: int
    ) -> Dict[str, Any]:
        """Predict trends for a category"""
        try:
            # Simple linear regression for prediction
            if len(trends) < 3:
                return {"error": "Insufficient data for prediction"}
            
            # Sort by timestamp
            sorted_trends = sorted(trends, key=lambda x: x.timestamp)
            
            # Calculate average metrics
            avg_popularity = statistics.mean([trend.popularity_score for trend in sorted_trends])
            avg_engagement = statistics.mean([trend.engagement_rate for trend in sorted_trends])
            avg_growth = statistics.mean([trend.growth_rate for trend in sorted_trends])
            
            # Simple trend projection
            predicted_popularity = avg_popularity * (1 + avg_growth * (days_ahead / 30))
            predicted_engagement = avg_engagement * (1 + avg_growth * (days_ahead / 30))
            predicted_growth = avg_growth * (1 - (days_ahead / 365))  # Decay factor
            
            return {
                "predicted_popularity": min(1.0, max(0.0, predicted_popularity)),
                "predicted_engagement": min(1.0, max(0.0, predicted_engagement)),
                "predicted_growth": min(1.0, max(0.0, predicted_growth)),
                "confidence": 0.7 if len(trends) >= 5 else 0.5
            }
            
        except Exception as e:
            logger.error(f"Error predicting category trends: {str(e)}")
            return {"error": str(e)}
    
    async def _find_related_topics(self, topic: str, topic_data: List[TrendData]) -> List[str]:
        """Find topics related to the given topic"""
        try:
            # Simple keyword-based related topic finding
            topic_words = set(topic.lower().split())
            related_topics = []
            
            for trend in self.trend_data:
                if trend.topic.lower() != topic.lower():
                    trend_words = set(trend.topic.lower().split())
                    # Calculate word overlap
                    overlap = len(topic_words.intersection(trend_words))
                    if overlap > 0:
                        related_topics.append(trend.topic)
            
            # Return top related topics
            return related_topics[:5]
            
        except Exception as e:
            logger.error(f"Error finding related topics: {str(e)}")
            return []
    
    async def _calculate_market_size(self, market_segment: str) -> int:
        """Calculate market size for a segment"""
        try:
            # Simple market size calculation based on trend data
            segment_data = [
                trend for trend in self.trend_data
                if market_segment.lower() in trend.category.lower()
            ]
            
            if not segment_data:
                return 1000  # Default market size
            
            # Calculate based on search volume and social mentions
            total_search_volume = sum(trend.search_volume for trend in segment_data)
            total_social_mentions = sum(trend.social_mentions for trend in segment_data)
            
            # Estimate market size
            market_size = (total_search_volume * 10) + (total_social_mentions * 5)
            
            return max(1000, market_size)
            
        except Exception as e:
            logger.error(f"Error calculating market size: {str(e)}")
            return 1000
    
    async def _assess_competition_level(self, market_segment: str) -> float:
        """Assess competition level in a market segment"""
        try:
            # Simple competition assessment based on content performance
            segment_data = [
                trend for trend in self.trend_data
                if market_segment.lower() in trend.category.lower()
            ]
            
            if not segment_data:
                return 0.5  # Default competition level
            
            # Calculate competition based on popularity distribution
            popularities = [trend.popularity_score for trend in segment_data]
            avg_popularity = statistics.mean(popularities)
            std_popularity = statistics.stdev(popularities) if len(popularities) > 1 else 0
            
            # High standard deviation indicates high competition
            competition_level = min(1.0, std_popularity / avg_popularity if avg_popularity > 0 else 0.5)
            
            return competition_level
            
        except Exception as e:
            logger.error(f"Error assessing competition level: {str(e)}")
            return 0.5
    
    async def _calculate_opportunity_score(
        self,
        market_segment: str,
        market_size: int,
        competition_level: float
    ) -> float:
        """Calculate opportunity score for a market segment"""
        try:
            # Opportunity = Market Size * (1 - Competition Level)
            opportunity_score = (market_size / 10000) * (1 - competition_level)
            
            return min(1.0, max(0.0, opportunity_score))
            
        except Exception as e:
            logger.error(f"Error calculating opportunity score: {str(e)}")
            return 0.5
    
    async def _identify_trending_topics(self, market_segment: str, time_period: int) -> List[str]:
        """Identify trending topics in a market segment"""
        try:
            cutoff_date = datetime.now() - timedelta(days=time_period)
            segment_data = [
                trend for trend in self.trend_data
                if (market_segment.lower() in trend.category.lower() and
                    trend.timestamp >= cutoff_date)
            ]
            
            # Sort by growth rate and popularity
            trending_topics = sorted(
                segment_data,
                key=lambda x: x.growth_rate * x.popularity_score,
                reverse=True
            )
            
            return [trend.topic for trend in trending_topics[:10]]
            
        except Exception as e:
            logger.error(f"Error identifying trending topics: {str(e)}")
            return []
    
    async def _find_emerging_keywords(self, market_segment: str, time_period: int) -> List[str]:
        """Find emerging keywords in a market segment"""
        try:
            cutoff_date = datetime.now() - timedelta(days=time_period)
            segment_data = [
                trend for trend in self.trend_data
                if (market_segment.lower() in trend.category.lower() and
                    trend.timestamp >= cutoff_date)
            ]
            
            # Extract keywords from trending topics
            keywords = []
            for trend in segment_data:
                if trend.growth_rate > 0.5:  # High growth
                    keywords.extend(trend.topic.split())
            
            # Count keyword frequency
            keyword_counts = Counter(keywords)
            
            # Return top emerging keywords
            return [keyword for keyword, count in keyword_counts.most_common(10)]
            
        except Exception as e:
            logger.error(f"Error finding emerging keywords: {str(e)}")
            return []
    
    async def _analyze_audience_insights(self, market_segment: str) -> Dict[str, Any]:
        """Analyze audience insights for a market segment"""
        try:
            segment_data = [
                trend for trend in self.trend_data
                if market_segment.lower() in trend.category.lower()
            ]
            
            if not segment_data:
                return {"error": "No data available"}
            
            # Calculate audience preferences
            avg_engagement = statistics.mean([trend.engagement_rate for trend in segment_data])
            avg_popularity = statistics.mean([trend.popularity_score for trend in segment_data])
            
            # Determine audience type based on engagement patterns
            if avg_engagement > 0.7:
                audience_type = "highly_engaged"
            elif avg_engagement > 0.4:
                audience_type = "moderately_engaged"
            else:
                audience_type = "low_engagement"
            
            # Determine content preferences
            content_preferences = []
            if avg_popularity > 0.7:
                content_preferences.append("popular_topics")
            if avg_engagement > 0.6:
                content_preferences.append("interactive_content")
            
            return {
                "audience_type": audience_type,
                "average_engagement": avg_engagement,
                "average_popularity": avg_popularity,
                "content_preferences": content_preferences,
                "audience_size": len(segment_data)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing audience insights: {str(e)}")
            return {"error": str(e)}
    
    async def _identify_content_gaps(self, market_segment: str) -> List[str]:
        """Identify content gaps in a market segment"""
        try:
            # Simple content gap identification
            segment_data = [
                trend for trend in self.trend_data
                if market_segment.lower() in trend.category.lower()
            ]
            
            if not segment_data:
                return ["No data available for gap analysis"]
            
            # Find topics with low popularity but high engagement potential
            content_gaps = []
            for trend in segment_data:
                if trend.popularity_score < 0.4 and trend.engagement_rate > 0.6:
                    content_gaps.append(f"Under-explored topic: {trend.topic}")
            
            # Add general content gap suggestions
            content_gaps.extend([
                "Beginner-friendly content",
                "Advanced technical content",
                "Case studies and real-world examples",
                "Interactive tutorials and guides"
            ])
            
            return content_gaps[:5]  # Return top 5 content gaps
            
        except Exception as e:
            logger.error(f"Error identifying content gaps: {str(e)}")
            return ["Error in gap analysis"]
    
    async def _generate_market_recommendations(
        self,
        market_segment: str,
        opportunity_score: float,
        content_gaps: List[str]
    ) -> List[str]:
        """Generate market recommendations"""
        try:
            recommendations = []
            
            # Opportunity-based recommendations
            if opportunity_score > 0.7:
                recommendations.append("High opportunity market - consider aggressive content strategy")
            elif opportunity_score > 0.4:
                recommendations.append("Moderate opportunity - focus on quality content")
            else:
                recommendations.append("Low opportunity - consider niche targeting")
            
            # Content gap recommendations
            if content_gaps:
                recommendations.append("Focus on identified content gaps for competitive advantage")
            
            # General recommendations
            recommendations.extend([
                "Monitor trending topics regularly",
                "Engage with audience through interactive content",
                "Track performance metrics and adjust strategy",
                "Consider cross-platform content distribution"
            ])
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating market recommendations: {str(e)}")
            return ["Error generating recommendations"]

# Global instance
trend_analyzer = TrendAnalyzer()

# Example usage
if __name__ == "__main__":
    async def test_trend_analysis():
        print("📈 Testing Trend Analysis System")
        print("=" * 40)
        
        # Add sample trend data
        sample_trends = [
            TrendData(
                topic="Artificial Intelligence",
                category="technology",
                popularity_score=0.8,
                growth_rate=0.3,
                engagement_rate=0.7,
                search_volume=10000,
                social_mentions=5000,
                timestamp=datetime.now() - timedelta(days=1)
            ),
            TrendData(
                topic="Machine Learning",
                category="technology",
                popularity_score=0.7,
                growth_rate=0.4,
                engagement_rate=0.6,
                search_volume=8000,
                social_mentions=4000,
                timestamp=datetime.now() - timedelta(days=2)
            ),
            TrendData(
                topic="Digital Marketing",
                category="marketing",
                popularity_score=0.6,
                growth_rate=0.2,
                engagement_rate=0.5,
                search_volume=6000,
                social_mentions=3000,
                timestamp=datetime.now() - timedelta(days=3)
            )
        ]
        
        for trend in sample_trends:
            await trend_analyzer.add_trend_data(trend)
        
        # Test trend analysis
        analysis = await trend_analyzer.analyze_trends(time_period=30)
        print(f"Trend Analysis Results:")
        print(f"Categories analyzed: {list(analysis.get('categories', {}).keys())}")
        print(f"Overall trends: {analysis.get('overall_trends', {})}")
        print(f"Recommendations: {analysis.get('recommendations', [])}")
        
        # Test content prediction
        prediction = await trend_analyzer.predict_content_success(
            topic="Artificial Intelligence",
            content_type="blog",
            target_audience="professionals"
        )
        print(f"\nContent Prediction:")
        print(f"Topic: {prediction.topic}")
        print(f"Predicted Popularity: {prediction.predicted_popularity:.2f}")
        print(f"Predicted Engagement: {prediction.predicted_engagement:.2f}")
        print(f"Confidence Score: {prediction.confidence_score:.2f}")
        print(f"Recommended Content Type: {prediction.recommended_content_type}")
        print(f"Content Suggestions: {prediction.content_suggestions}")
        
        # Test market intelligence
        market_intel = await trend_analyzer.get_market_intelligence("technology")
        print(f"\nMarket Intelligence:")
        print(f"Market Size: {market_intel.market_size}")
        print(f"Competition Level: {market_intel.competition_level:.2f}")
        print(f"Opportunity Score: {market_intel.opportunity_score:.2f}")
        print(f"Trending Topics: {market_intel.trending_topics}")
        print(f"Recommendations: {market_intel.recommendations}")
    
    asyncio.run(test_trend_analysis())


