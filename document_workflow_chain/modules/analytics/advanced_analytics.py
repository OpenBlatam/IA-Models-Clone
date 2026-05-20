"""
Advanced Analytics and Predictive System
========================================

This module provides advanced analytics, trend analysis, and predictive
insights for document workflow chains and content performance.
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import statistics
import math

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics for content analysis"""
    timestamp: datetime
    chain_id: str
    document_id: str
    quality_score: float
    generation_time: float
    tokens_used: int
    word_count: int
    engagement_score: float
    seo_score: float
    readability_score: float

@dataclass
class TrendAnalysis:
    """Trend analysis results"""
    metric_name: str
    trend_direction: str  # "increasing", "decreasing", "stable"
    trend_strength: float  # 0-1
    change_percentage: float
    confidence_level: float
    data_points: int
    time_period: str

@dataclass
class PredictiveInsight:
    """Predictive insight"""
    insight_type: str
    prediction: str
    confidence: float
    timeframe: str
    supporting_data: Dict[str, Any]
    recommendations: List[str]

class AdvancedAnalytics:
    """Advanced analytics and predictive system"""
    
    def __init__(self):
        self.metrics_history: List[PerformanceMetrics] = []
        self.trend_cache: Dict[str, TrendAnalysis] = {}
        self.insights_cache: Dict[str, List[PredictiveInsight]] = {}
    
    async def record_metrics(self, metrics: PerformanceMetrics):
        """Record performance metrics"""
        try:
            self.metrics_history.append(metrics)
            
            # Keep only last 1000 records to manage memory
            if len(self.metrics_history) > 1000:
                self.metrics_history = self.metrics_history[-1000:]
            
            logger.info(f"Recorded metrics for document {metrics.document_id}")
            
        except Exception as e:
            logger.error(f"Error recording metrics: {str(e)}")
    
    async def analyze_trends(
        self,
        metric_name: str,
        time_period: str = "7d",
        chain_id: Optional[str] = None
    ) -> TrendAnalysis:
        """Analyze trends for a specific metric"""
        try:
            # Get relevant data
            data = await self._get_metric_data(metric_name, time_period, chain_id)
            
            if len(data) < 3:
                return TrendAnalysis(
                    metric_name=metric_name,
                    trend_direction="insufficient_data",
                    trend_strength=0.0,
                    change_percentage=0.0,
                    confidence_level=0.0,
                    data_points=len(data),
                    time_period=time_period
                )
            
            # Calculate trend
            trend_direction, trend_strength, change_percentage = self._calculate_trend(data)
            
            # Calculate confidence
            confidence = self._calculate_confidence(data, trend_strength)
            
            trend_analysis = TrendAnalysis(
                metric_name=metric_name,
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                change_percentage=change_percentage,
                confidence_level=confidence,
                data_points=len(data),
                time_period=time_period
            )
            
            # Cache result
            cache_key = f"{metric_name}_{time_period}_{chain_id or 'all'}"
            self.trend_cache[cache_key] = trend_analysis
            
            return trend_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing trends: {str(e)}")
            return TrendAnalysis(
                metric_name=metric_name,
                trend_direction="error",
                trend_strength=0.0,
                change_percentage=0.0,
                confidence_level=0.0,
                data_points=0,
                time_period=time_period
            )
    
    async def generate_predictive_insights(
        self,
        chain_id: Optional[str] = None,
        timeframe: str = "30d"
    ) -> List[PredictiveInsight]:
        """Generate predictive insights"""
        try:
            insights = []
            
            # Quality trend insight
            quality_trend = await self.analyze_trends("quality_score", "7d", chain_id)
            if quality_trend.trend_direction == "decreasing" and quality_trend.confidence_level > 0.7:
                insights.append(PredictiveInsight(
                    insight_type="quality_decline",
                    prediction=f"Content quality is declining by {abs(quality_trend.change_percentage):.1f}%",
                    confidence=quality_trend.confidence_level,
                    timeframe="next_7_days",
                    supporting_data={"trend_analysis": asdict(quality_trend)},
                    recommendations=[
                        "Review and improve prompt templates",
                        "Consider adjusting AI model parameters",
                        "Implement quality control checkpoints"
                    ]
                ))
            
            # Performance optimization insight
            generation_time_trend = await self.analyze_trends("generation_time", "7d", chain_id)
            if generation_time_trend.trend_direction == "increasing" and generation_time_trend.confidence_level > 0.6:
                insights.append(PredictiveInsight(
                    insight_type="performance_degradation",
                    prediction=f"Generation time is increasing by {generation_time_trend.change_percentage:.1f}%",
                    confidence=generation_time_trend.confidence_level,
                    timeframe="next_14_days",
                    supporting_data={"trend_analysis": asdict(generation_time_trend)},
                    recommendations=[
                        "Optimize AI client configuration",
                        "Consider implementing caching",
                        "Review system resource allocation"
                    ]
                ))
            
            # Token usage optimization insight
            token_trend = await self.analyze_trends("tokens_used", "7d", chain_id)
            if token_trend.trend_direction == "increasing" and token_trend.confidence_level > 0.6:
                insights.append(PredictiveInsight(
                    insight_type="cost_optimization",
                    prediction=f"Token usage is increasing by {token_trend.change_percentage:.1f}%",
                    confidence=token_trend.confidence_level,
                    timeframe="next_30_days",
                    supporting_data={"trend_analysis": asdict(token_trend)},
                    recommendations=[
                        "Optimize prompt length and complexity",
                        "Implement token usage monitoring",
                        "Consider using more efficient AI models"
                    ]
                ))
            
            # Engagement optimization insight
            engagement_trend = await self.analyze_trends("engagement_score", "7d", chain_id)
            if engagement_trend.trend_direction == "decreasing" and engagement_trend.confidence_level > 0.7:
                insights.append(PredictiveInsight(
                    insight_type="engagement_decline",
                    prediction=f"Content engagement is declining by {abs(engagement_trend.change_percentage):.1f}%",
                    confidence=engagement_trend.confidence_level,
                    timeframe="next_14_days",
                    supporting_data={"trend_analysis": asdict(engagement_trend)},
                    recommendations=[
                        "Enhance content templates with engagement elements",
                        "Add more interactive content features",
                        "Review audience targeting and messaging"
                    ]
                ))
            
            # Cache insights
            cache_key = f"insights_{chain_id or 'all'}_{timeframe}"
            self.insights_cache[cache_key] = insights
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating predictive insights: {str(e)}")
            return []
    
    async def get_performance_summary(
        self,
        chain_id: Optional[str] = None,
        time_period: str = "30d"
    ) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        try:
            # Get relevant metrics
            relevant_metrics = await self._get_metrics_for_period(chain_id, time_period)
            
            if not relevant_metrics:
                return {"error": "No data available for the specified period"}
            
            # Calculate summary statistics
            summary = {
                "period": time_period,
                "total_documents": len(relevant_metrics),
                "average_quality_score": statistics.mean([m.quality_score for m in relevant_metrics]),
                "average_generation_time": statistics.mean([m.generation_time for m in relevant_metrics]),
                "total_tokens_used": sum([m.tokens_used for m in relevant_metrics]),
                "average_word_count": statistics.mean([m.word_count for m in relevant_metrics]),
                "average_engagement_score": statistics.mean([m.engagement_score for m in relevant_metrics]),
                "average_seo_score": statistics.mean([m.seo_score for m in relevant_metrics]),
                "average_readability_score": statistics.mean([m.readability_score for m in relevant_metrics]),
                "quality_distribution": self._calculate_distribution([m.quality_score for m in relevant_metrics]),
                "performance_trends": {},
                "top_performing_documents": self._get_top_performers(relevant_metrics, "quality_score", 5)
            }
            
            # Add trend analysis for key metrics
            key_metrics = ["quality_score", "generation_time", "tokens_used", "engagement_score"]
            for metric in key_metrics:
                trend = await self.analyze_trends(metric, "7d", chain_id)
                summary["performance_trends"][metric] = asdict(trend)
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating performance summary: {str(e)}")
            return {"error": str(e)}
    
    async def get_optimization_recommendations(
        self,
        chain_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get optimization recommendations based on analytics"""
        try:
            recommendations = []
            
            # Get recent performance data
            recent_metrics = await self._get_metrics_for_period(chain_id, "7d")
            
            if not recent_metrics:
                return [{"type": "insufficient_data", "message": "Need more data to provide recommendations"}]
            
            # Quality optimization
            avg_quality = statistics.mean([m.quality_score for m in recent_metrics])
            if avg_quality < 0.7:
                recommendations.append({
                    "type": "quality_improvement",
                    "priority": "high",
                    "title": "Improve Content Quality",
                    "description": f"Average quality score is {avg_quality:.2f}, below optimal threshold",
                    "actions": [
                        "Review and enhance prompt templates",
                        "Implement quality scoring feedback loop",
                        "Consider using higher-quality AI models"
                    ],
                    "expected_impact": "15-25% quality improvement"
                })
            
            # Performance optimization
            avg_generation_time = statistics.mean([m.generation_time for m in recent_metrics])
            if avg_generation_time > 10.0:  # More than 10 seconds
                recommendations.append({
                    "type": "performance_optimization",
                    "priority": "medium",
                    "title": "Optimize Generation Speed",
                    "description": f"Average generation time is {avg_generation_time:.1f}s, above optimal threshold",
                    "actions": [
                        "Optimize AI client configuration",
                        "Implement request caching",
                        "Consider parallel processing"
                    ],
                    "expected_impact": "20-40% speed improvement"
                })
            
            # Cost optimization
            avg_tokens = statistics.mean([m.tokens_used for m in recent_metrics])
            if avg_tokens > 2000:  # High token usage
                recommendations.append({
                    "type": "cost_optimization",
                    "priority": "medium",
                    "title": "Reduce Token Usage",
                    "description": f"Average token usage is {avg_tokens:.0f}, above optimal threshold",
                    "actions": [
                        "Optimize prompt length and complexity",
                        "Use more efficient AI models",
                        "Implement prompt compression techniques"
                    ],
                    "expected_impact": "25-35% cost reduction"
                })
            
            # Engagement optimization
            avg_engagement = statistics.mean([m.engagement_score for m in recent_metrics])
            if avg_engagement < 0.6:
                recommendations.append({
                    "type": "engagement_optimization",
                    "priority": "high",
                    "title": "Improve Content Engagement",
                    "description": f"Average engagement score is {avg_engagement:.2f}, below optimal threshold",
                    "actions": [
                        "Add more interactive elements to content",
                        "Include questions and calls-to-action",
                        "Enhance content templates with engagement features"
                    ],
                    "expected_impact": "20-30% engagement improvement"
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating optimization recommendations: {str(e)}")
            return [{"type": "error", "message": str(e)}]
    
    def _get_metric_data(
        self,
        metric_name: str,
        time_period: str,
        chain_id: Optional[str] = None
    ) -> List[float]:
        """Get metric data for analysis"""
        try:
            # Calculate time threshold
            now = datetime.now()
            if time_period == "7d":
                threshold = now - timedelta(days=7)
            elif time_period == "30d":
                threshold = now - timedelta(days=30)
            else:
                threshold = now - timedelta(days=7)  # Default
            
            # Filter metrics
            relevant_metrics = [
                m for m in self.metrics_history
                if m.timestamp >= threshold and (chain_id is None or m.chain_id == chain_id)
            ]
            
            # Extract metric values
            if metric_name == "quality_score":
                return [m.quality_score for m in relevant_metrics]
            elif metric_name == "generation_time":
                return [m.generation_time for m in relevant_metrics]
            elif metric_name == "tokens_used":
                return [m.tokens_used for m in relevant_metrics]
            elif metric_name == "engagement_score":
                return [m.engagement_score for m in relevant_metrics]
            elif metric_name == "seo_score":
                return [m.seo_score for m in relevant_metrics]
            elif metric_name == "readability_score":
                return [m.readability_score for m in relevant_metrics]
            else:
                return []
                
        except Exception as e:
            logger.error(f"Error getting metric data: {str(e)}")
            return []
    
    def _calculate_trend(self, data: List[float]) -> Tuple[str, float, float]:
        """Calculate trend direction, strength, and percentage change"""
        try:
            if len(data) < 2:
                return "insufficient_data", 0.0, 0.0
            
            # Simple linear regression
            n = len(data)
            x = list(range(n))
            y = data
            
            # Calculate slope
            sum_x = sum(x)
            sum_y = sum(y)
            sum_xy = sum(x[i] * y[i] for i in range(n))
            sum_x2 = sum(x[i] ** 2 for i in range(n))
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
            
            # Calculate percentage change
            first_value = data[0]
            last_value = data[-1]
            change_percentage = ((last_value - first_value) / first_value) * 100 if first_value != 0 else 0
            
            # Determine trend direction and strength
            if abs(slope) < 0.01:  # Very small slope
                trend_direction = "stable"
                trend_strength = 0.1
            elif slope > 0:
                trend_direction = "increasing"
                trend_strength = min(1.0, abs(slope) * 10)  # Normalize
            else:
                trend_direction = "decreasing"
                trend_strength = min(1.0, abs(slope) * 10)  # Normalize
            
            return trend_direction, trend_strength, change_percentage
            
        except Exception as e:
            logger.error(f"Error calculating trend: {str(e)}")
            return "error", 0.0, 0.0
    
    def _calculate_confidence(self, data: List[float], trend_strength: float) -> float:
        """Calculate confidence level for trend analysis"""
        try:
            # Base confidence on data points and trend strength
            data_points_factor = min(1.0, len(data) / 10)  # More data = higher confidence
            trend_factor = trend_strength
            
            # Calculate variance (lower variance = higher confidence)
            if len(data) > 1:
                variance = statistics.variance(data)
                variance_factor = max(0.1, 1.0 - (variance * 10))  # Normalize variance
            else:
                variance_factor = 0.1
            
            confidence = (data_points_factor * 0.4 + trend_factor * 0.4 + variance_factor * 0.2)
            return min(1.0, max(0.0, confidence))
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}")
            return 0.0
    
    def _get_metrics_for_period(
        self,
        chain_id: Optional[str],
        time_period: str
    ) -> List[PerformanceMetrics]:
        """Get metrics for a specific time period"""
        try:
            now = datetime.now()
            if time_period == "7d":
                threshold = now - timedelta(days=7)
            elif time_period == "30d":
                threshold = now - timedelta(days=30)
            else:
                threshold = now - timedelta(days=7)
            
            return [
                m for m in self.metrics_history
                if m.timestamp >= threshold and (chain_id is None or m.chain_id == chain_id)
            ]
            
        except Exception as e:
            logger.error(f"Error getting metrics for period: {str(e)}")
            return []
    
    def _calculate_distribution(self, values: List[float]) -> Dict[str, int]:
        """Calculate distribution of values"""
        try:
            if not values:
                return {}
            
            # Define buckets
            buckets = {
                "excellent": 0,  # 0.8-1.0
                "good": 0,       # 0.6-0.8
                "average": 0,    # 0.4-0.6
                "poor": 0        # 0.0-0.4
            }
            
            for value in values:
                if value >= 0.8:
                    buckets["excellent"] += 1
                elif value >= 0.6:
                    buckets["good"] += 1
                elif value >= 0.4:
                    buckets["average"] += 1
                else:
                    buckets["poor"] += 1
            
            return buckets
            
        except Exception as e:
            logger.error(f"Error calculating distribution: {str(e)}")
            return {}
    
    def _get_top_performers(
        self,
        metrics: List[PerformanceMetrics],
        metric_name: str,
        count: int
    ) -> List[Dict[str, Any]]:
        """Get top performing documents"""
        try:
            # Sort by specified metric
            if metric_name == "quality_score":
                sorted_metrics = sorted(metrics, key=lambda m: m.quality_score, reverse=True)
            elif metric_name == "engagement_score":
                sorted_metrics = sorted(metrics, key=lambda m: m.engagement_score, reverse=True)
            else:
                sorted_metrics = sorted(metrics, key=lambda m: m.quality_score, reverse=True)
            
            # Return top performers
            top_performers = []
            for metric in sorted_metrics[:count]:
                top_performers.append({
                    "document_id": metric.document_id,
                    "chain_id": metric.chain_id,
                    "quality_score": metric.quality_score,
                    "engagement_score": metric.engagement_score,
                    "generation_time": metric.generation_time,
                    "timestamp": metric.timestamp.isoformat()
                })
            
            return top_performers
            
        except Exception as e:
            logger.error(f"Error getting top performers: {str(e)}")
            return []

# Global analytics instance
advanced_analytics = AdvancedAnalytics()

# Example usage
if __name__ == "__main__":
    async def test_analytics():
        print("🧪 Testing Advanced Analytics")
        print("=" * 40)
        
        # Create sample metrics
        import random
        from datetime import datetime, timedelta
        
        for i in range(20):
            metric = PerformanceMetrics(
                timestamp=datetime.now() - timedelta(days=i),
                chain_id="test_chain",
                document_id=f"doc_{i}",
                quality_score=0.7 + random.uniform(-0.2, 0.2),
                generation_time=5.0 + random.uniform(-2, 2),
                tokens_used=1500 + random.randint(-300, 300),
                word_count=800 + random.randint(-200, 200),
                engagement_score=0.6 + random.uniform(-0.2, 0.2),
                seo_score=0.8 + random.uniform(-0.1, 0.1),
                readability_score=0.7 + random.uniform(-0.1, 0.1)
            )
            await advanced_analytics.record_metrics(metric)
        
        # Test trend analysis
        quality_trend = await advanced_analytics.analyze_trends("quality_score", "7d")
        print(f"Quality trend: {quality_trend.trend_direction} ({quality_trend.change_percentage:.1f}%)")
        
        # Test performance summary
        summary = await advanced_analytics.get_performance_summary("test_chain", "7d")
        print(f"Average quality: {summary['average_quality_score']:.2f}")
        print(f"Total documents: {summary['total_documents']}")
        
        # Test insights
        insights = await advanced_analytics.generate_predictive_insights("test_chain")
        print(f"Generated {len(insights)} insights")
        
        # Test recommendations
        recommendations = await advanced_analytics.get_optimization_recommendations("test_chain")
        print(f"Generated {len(recommendations)} recommendations")
    
    asyncio.run(test_analytics())


