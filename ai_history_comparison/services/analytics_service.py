"""
Analytics Service

This service orchestrates analytics functionality including
trend analysis, performance analytics, and business intelligence.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from ..core.base import BaseService
from ..core.config import SystemConfig
from ..core.exceptions import AnalyticsError, AnalysisError

logger = logging.getLogger(__name__)


class AnalyticsService(BaseService[Dict[str, Any]]):
    """Service for managing analytics and business intelligence operations"""
    
    def __init__(self, config: Optional[SystemConfig] = None):
        super().__init__(config)
        self._analyzers = {}
        self._engines = {}
    
    async def _start(self) -> bool:
        """Start the analytics service"""
        try:
            # Initialize analyzers
            await self._initialize_analyzers()
            
            # Initialize engines
            await self._initialize_engines()
            
            logger.info("Analytics service started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start analytics service: {e}")
            return False
    
    async def _stop(self) -> bool:
        """Stop the analytics service"""
        try:
            # Cleanup engines
            await self._cleanup_engines()
            
            # Cleanup analyzers
            await self._cleanup_analyzers()
            
            logger.info("Analytics service stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop analytics service: {e}")
            return False
    
    async def _initialize_analyzers(self):
        """Initialize analytics analyzers"""
        try:
            # Import here to avoid circular imports
            from ..analyzers.trend_analyzer import TrendAnalyzer
            from ..analyzers.content_analyzer import ContentAnalyzer
            
            # Initialize trend analyzer
            if self.config.features.get("trend_analysis", False):
                self._analyzers["trend"] = TrendAnalyzer(self.config)
                await self._analyzers["trend"].initialize()
            
            # Initialize content analyzer for analytics
            if self.config.features.get("content_analysis", False):
                self._analyzers["content"] = ContentAnalyzer(self.config)
                await self._analyzers["content"].initialize()
            
            logger.info("Analytics analyzers initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize analytics analyzers: {e}")
            raise AnalyticsError(f"Failed to initialize analytics analyzers: {str(e)}")
    
    async def _initialize_engines(self):
        """Initialize analytics engines"""
        try:
            # Import here to avoid circular imports
            from ..engines.comparison_engine import ComparisonEngine
            
            # Initialize comparison engine
            if self.config.features.get("comparison_engine", False):
                self._engines["comparison"] = ComparisonEngine(self.config)
                await self._engines["comparison"].initialize()
            
            logger.info("Analytics engines initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize analytics engines: {e}")
            raise AnalyticsError(f"Failed to initialize analytics engines: {str(e)}")
    
    async def _cleanup_analyzers(self):
        """Cleanup analyzers"""
        try:
            for analyzer_name, analyzer in self._analyzers.items():
                await analyzer.shutdown()
            
            self._analyzers.clear()
            logger.info("Analytics analyzers cleaned up")
            
        except Exception as e:
            logger.error(f"Failed to cleanup analyzers: {e}")
    
    async def _cleanup_engines(self):
        """Cleanup engines"""
        try:
            for engine_name, engine in self._engines.items():
                await engine.shutdown()
            
            self._engines.clear()
            logger.info("Analytics engines cleaned up")
            
        except Exception as e:
            logger.error(f"Failed to cleanup engines: {e}")
    
    async def analyze_trends(self, data: List[Dict[str, Any]], metric: str, 
                           time_window: int = 30) -> Dict[str, Any]:
        """Analyze trends in data"""
        try:
            if not self._initialized:
                raise AnalyticsError("Analytics service not initialized")
            
            if "trend" not in self._analyzers:
                raise AnalyticsError("Trend analyzer not available")
            
            result = await self._analyzers["trend"].analyze_trends(data, metric)
            
            # Add time window information
            result["time_window"] = time_window
            result["analysis_timestamp"] = datetime.utcnow().isoformat()
            
            return result
            
        except Exception as e:
            logger.error(f"Trend analysis failed: {e}")
            raise AnalyticsError(f"Trend analysis failed: {str(e)}")
    
    async def predict_future(self, data: List[Dict[str, Any]], metric: str, 
                           prediction_days: int = 7) -> Dict[str, Any]:
        """Predict future values based on historical data"""
        try:
            if not self._initialized:
                raise AnalyticsError("Analytics service not initialized")
            
            if "trend" not in self._analyzers:
                raise AnalyticsError("Trend analyzer not available")
            
            result = await self._analyzers["trend"].predict_future(data, metric, prediction_days)
            
            # Add prediction metadata
            result["prediction_timestamp"] = datetime.utcnow().isoformat()
            result["confidence_level"] = self._calculate_prediction_confidence(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Future prediction failed: {e}")
            raise AnalyticsError(f"Future prediction failed: {str(e)}")
    
    async def detect_anomalies(self, data: List[Dict[str, Any]], metric: str, 
                             sensitivity: float = 2.0) -> List[Dict[str, Any]]:
        """Detect anomalies in data"""
        try:
            if not self._initialized:
                raise AnalyticsError("Analytics service not initialized")
            
            if "trend" not in self._analyzers:
                raise AnalyticsError("Trend analyzer not available")
            
            anomalies = await self._analyzers["trend"].detect_anomalies(data, metric)
            
            # Add anomaly metadata
            for anomaly in anomalies:
                anomaly["detection_timestamp"] = datetime.utcnow().isoformat()
                anomaly["sensitivity"] = sensitivity
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            raise AnalyticsError(f"Anomaly detection failed: {str(e)}")
    
    async def compare_content(self, content1: str, content2: str) -> Dict[str, Any]:
        """Compare two pieces of content"""
        try:
            if not self._initialized:
                raise AnalyticsError("Analytics service not initialized")
            
            if "comparison" not in self._engines:
                raise AnalyticsError("Comparison engine not available")
            
            result = await self._engines["comparison"].compare_content(content1, content2)
            
            # Add comparison metadata
            result["comparison_timestamp"] = datetime.utcnow().isoformat()
            result["comparison_type"] = "content_similarity"
            
            return result
            
        except Exception as e:
            logger.error(f"Content comparison failed: {e}")
            raise AnalyticsError(f"Content comparison failed: {str(e)}")
    
    async def compare_models(self, model1_results: Dict[str, Any], 
                           model2_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare two model results"""
        try:
            if not self._initialized:
                raise AnalyticsError("Analytics service not initialized")
            
            if "comparison" not in self._engines:
                raise AnalyticsError("Comparison engine not available")
            
            result = await self._engines["comparison"].compare_models(model1_results, model2_results)
            
            # Add comparison metadata
            result["comparison_timestamp"] = datetime.utcnow().isoformat()
            result["comparison_type"] = "model_performance"
            
            return result
            
        except Exception as e:
            logger.error(f"Model comparison failed: {e}")
            raise AnalyticsError(f"Model comparison failed: {str(e)}")
    
    async def find_similar_content(self, content: str, threshold: float = 0.8, 
                                 limit: int = 10) -> List[Dict[str, Any]]:
        """Find similar content pieces"""
        try:
            if not self._initialized:
                raise AnalyticsError("Analytics service not initialized")
            
            if "comparison" not in self._engines:
                raise AnalyticsError("Comparison engine not available")
            
            results = await self._engines["comparison"].find_similar_content(
                content, threshold
            )
            
            # Limit results
            limited_results = results[:limit]
            
            # Add search metadata
            for result in limited_results:
                result["search_timestamp"] = datetime.utcnow().isoformat()
                result["search_threshold"] = threshold
            
            return limited_results
            
        except Exception as e:
            logger.error(f"Similar content search failed: {e}")
            raise AnalyticsError(f"Similar content search failed: {str(e)}")
    
    async def generate_analytics_report(self, report_type: str = "comprehensive", 
                                      data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate comprehensive analytics report"""
        try:
            if not self._initialized:
                raise AnalyticsError("Analytics service not initialized")
            
            report = {
                "report_type": report_type,
                "generated_at": datetime.utcnow().isoformat(),
                "analytics_summary": {},
                "trends": {},
                "comparisons": {},
                "anomalies": {},
                "recommendations": []
            }
            
            # Generate trend analysis if data provided
            if data and "trend_data" in data:
                try:
                    trend_result = await self.analyze_trends(
                        data["trend_data"], 
                        data.get("metric", "value")
                    )
                    report["trends"] = trend_result
                except Exception as e:
                    logger.error(f"Failed to generate trend analysis: {e}")
                    report["trends"] = {"error": str(e)}
            
            # Generate content analysis if data provided
            if data and "content_data" in data:
                try:
                    if "content" in self._analyzers:
                        content_result = await self._analyzers["content"].analyze(
                            data["content_data"]
                        )
                        report["analytics_summary"]["content"] = content_result
                except Exception as e:
                    logger.error(f"Failed to generate content analysis: {e}")
                    report["analytics_summary"]["content"] = {"error": str(e)}
            
            # Generate recommendations
            report["recommendations"] = await self._generate_recommendations(report)
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate analytics report: {e}")
            raise AnalyticsError(f"Failed to generate analytics report: {str(e)}")
    
    def _calculate_prediction_confidence(self, prediction_result: Dict[str, Any]) -> float:
        """Calculate confidence level for predictions"""
        try:
            # Simple confidence calculation based on R-squared and data points
            r_squared = prediction_result.get("model_info", {}).get("r_squared", 0.5)
            data_points = prediction_result.get("model_info", {}).get("data_points", 10)
            
            # Base confidence from R-squared
            base_confidence = r_squared
            
            # Adjust for data points (more data = higher confidence)
            data_factor = min(1.0, data_points / 50.0)
            
            confidence = min(1.0, base_confidence * data_factor)
            return confidence
            
        except Exception as e:
            logger.error(f"Failed to calculate prediction confidence: {e}")
            return 0.5  # Default confidence
    
    async def _generate_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analytics results"""
        recommendations = []
        
        try:
            # Trend-based recommendations
            if "trends" in report and "trend_direction" in report["trends"]:
                trend_direction = report["trends"]["trend_direction"]
                if trend_direction == "decreasing":
                    recommendations.append("Consider investigating the declining trend and implementing corrective measures.")
                elif trend_direction == "increasing":
                    recommendations.append("The positive trend is encouraging. Consider scaling successful strategies.")
            
            # Anomaly-based recommendations
            if "anomalies" in report and report["anomalies"]:
                anomaly_count = len(report["anomalies"])
                if anomaly_count > 0:
                    recommendations.append(f"Detected {anomaly_count} anomalies. Review and investigate these data points.")
            
            # Content-based recommendations
            if "analytics_summary" in report and "content" in report["analytics_summary"]:
                content_analysis = report["analytics_summary"]["content"]
                if "quality_score" in content_analysis:
                    quality_score = content_analysis["quality_score"]
                    if quality_score < 0.6:
                        recommendations.append("Content quality is below optimal levels. Consider content improvement strategies.")
            
            if not recommendations:
                recommendations.append("No specific recommendations at this time. Continue monitoring.")
            
        except Exception as e:
            logger.error(f"Failed to generate recommendations: {e}")
            recommendations.append("Unable to generate recommendations due to analysis errors.")
        
        return recommendations
    
    def get_analytics_status(self) -> Dict[str, Any]:
        """Get analytics service status"""
        base_status = self.get_health_status()
        base_status.update({
            "analyzers": list(self._analyzers.keys()),
            "engines": list(self._engines.keys()),
            "features_enabled": {
                "trend_analysis": "trend" in self._analyzers,
                "content_analysis": "content" in self._analyzers,
                "comparison_engine": "comparison" in self._engines
            }
        })
        return base_status





















