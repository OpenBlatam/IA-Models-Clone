"""
AI History Analysis Service
==========================

Advanced AI history analysis service for tracking and comparing AI model performance over time.
"""

import asyncio
import logging
import json
import hashlib
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from uuid import UUID, uuid4
from dataclasses import dataclass, field
from enum import Enum
import redis
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, func, and_, or_

from ..schemas import (
    BusinessAgent, AgentRequest, AgentResponse, AgentAnalytics,
    AgentWorkflow, AgentCollaboration, AgentSettings,
    ErrorResponse, SuccessResponse
)
from ..exceptions import (
    AIHistoryNotFoundError, AIHistoryAnalysisError, AIHistoryValidationError,
    AIHistoryOptimizationError, AIHistorySystemError,
    create_agent_error, log_agent_error, handle_agent_error, get_error_response
)
from ..models import (
    db_manager, BusinessAgent as AgentModel, User
)
from ..config import get_settings

logger = logging.getLogger(__name__)


class AIHistoryType(str, Enum):
    """AI history type enumeration"""
    MODEL_PERFORMANCE = "model_performance"
    TRAINING_HISTORY = "training_history"
    PREDICTION_HISTORY = "prediction_history"
    OPTIMIZATION_HISTORY = "optimization_history"
    DEPLOYMENT_HISTORY = "deployment_history"
    ERROR_HISTORY = "error_history"
    USAGE_HISTORY = "usage_history"
    COST_HISTORY = "cost_history"
    CUSTOM = "custom"


class AnalysisType(str, Enum):
    """Analysis type enumeration"""
    TREND_ANALYSIS = "trend_analysis"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    PERFORMANCE_ANALYSIS = "performance_analysis"
    COST_ANALYSIS = "cost_analysis"
    USAGE_ANALYSIS = "usage_analysis"
    ERROR_ANALYSIS = "error_analysis"
    PREDICTIVE_ANALYSIS = "predictive_analysis"
    CORRELATION_ANALYSIS = "correlation_analysis"
    STATISTICAL_ANALYSIS = "statistical_analysis"
    CUSTOM = "custom"


class TimeGranularity(str, Enum):
    """Time granularity enumeration"""
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"
    CUSTOM = "custom"


@dataclass
class AIHistoryRecord:
    """AI history record"""
    record_id: str
    ai_model_id: str
    history_type: AIHistoryType
    timestamp: datetime
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0.0"
    tags: List[str] = field(default_factory=list)


@dataclass
class AnalysisResult:
    """Analysis result"""
    analysis_id: str
    analysis_type: AnalysisType
    time_range: Tuple[datetime, datetime]
    granularity: TimeGranularity
    results: Dict[str, Any]
    insights: List[str]
    recommendations: List[str]
    confidence_score: float
    analysis_timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ComparisonResult:
    """Comparison result"""
    comparison_id: str
    model_versions: List[str]
    comparison_metrics: List[str]
    results: Dict[str, Any]
    winner: str
    improvement_percentage: float
    statistical_significance: float
    comparison_timestamp: datetime = field(default_factory=datetime.utcnow)


class AIHistoryService:
    """Advanced AI history analysis service"""
    
    def __init__(self, db_session: AsyncSession, redis_client: redis.Redis):
        self.db = db_session
        self.redis = redis_client
        self.settings = get_settings()
        self._history_cache = {}
        self._analysis_cache = {}
        self._comparison_cache = {}
    
    async def record_ai_history(
        self,
        ai_model_id: str,
        history_type: AIHistoryType,
        data: Dict[str, Any],
        metadata: Dict[str, Any] = None,
        version: str = "1.0.0",
        tags: List[str] = None,
        user_id: str = None
    ) -> AIHistoryRecord:
        """Record AI history data"""
        try:
            # Validate history data
            await self._validate_history_data(ai_model_id, history_type, data)
            
            # Create history record
            record_id = str(uuid4())
            record_data = {
                "ai_model_id": ai_model_id,
                "history_type": history_type.value,
                "data": data,
                "metadata": metadata or {},
                "version": version,
                "tags": tags or [],
                "created_by": user_id or "system"
            }
            
            # Store in database
            history_record = await db_manager.create_ai_history(record_data)
            
            # Cache history data
            await self._cache_history_data(history_record)
            
            # Create AIHistoryRecord object
            record = AIHistoryRecord(
                record_id=str(history_record.id),
                ai_model_id=ai_model_id,
                history_type=history_type,
                timestamp=history_record.created_at,
                data=data,
                metadata=metadata or {},
                version=version,
                tags=tags or []
            )
            
            logger.info(f"AI history recorded successfully: {record_id}")
            
            return record
            
        except Exception as e:
            error = handle_agent_error(e, ai_model_id=ai_model_id, user_id=user_id)
            log_agent_error(error)
            raise error
    
    async def analyze_ai_history(
        self,
        ai_model_id: str,
        analysis_type: AnalysisType,
        time_range: Tuple[datetime, datetime],
        granularity: TimeGranularity = TimeGranularity.DAY,
        analysis_options: Dict[str, Any] = None,
        user_id: str = None
    ) -> AnalysisResult:
        """Analyze AI history data"""
        try:
            # Get history data
            history_data = await self.get_ai_history(
                ai_model_id, time_range[0], time_range[1]
            )
            
            if not history_data:
                raise AIHistoryNotFoundError(
                    "no_history_data",
                    f"No history data found for model {ai_model_id} in time range",
                    {"ai_model_id": ai_model_id, "time_range": time_range}
                )
            
            # Perform analysis based on type
            if analysis_type == AnalysisType.TREND_ANALYSIS:
                results = await self._perform_trend_analysis(history_data, granularity)
            elif analysis_type == AnalysisType.COMPARATIVE_ANALYSIS:
                results = await self._perform_comparative_analysis(history_data, granularity)
            elif analysis_type == AnalysisType.PERFORMANCE_ANALYSIS:
                results = await self._perform_performance_analysis(history_data, granularity)
            elif analysis_type == AnalysisType.COST_ANALYSIS:
                results = await self._perform_cost_analysis(history_data, granularity)
            elif analysis_type == AnalysisType.USAGE_ANALYSIS:
                results = await self._perform_usage_analysis(history_data, granularity)
            elif analysis_type == AnalysisType.ERROR_ANALYSIS:
                results = await self._perform_error_analysis(history_data, granularity)
            elif analysis_type == AnalysisType.PREDICTIVE_ANALYSIS:
                results = await self._perform_predictive_analysis(history_data, granularity)
            elif analysis_type == AnalysisType.CORRELATION_ANALYSIS:
                results = await self._perform_correlation_analysis(history_data, granularity)
            elif analysis_type == AnalysisType.STATISTICAL_ANALYSIS:
                results = await self._perform_statistical_analysis(history_data, granularity)
            else:
                results = await self._perform_custom_analysis(history_data, granularity, analysis_options)
            
            # Generate insights and recommendations
            insights = await self._generate_insights(results, analysis_type)
            recommendations = await self._generate_recommendations(results, analysis_type)
            
            # Calculate confidence score
            confidence_score = await self._calculate_confidence_score(results, history_data)
            
            # Create analysis result
            analysis_result = AnalysisResult(
                analysis_id=str(uuid4()),
                analysis_type=analysis_type,
                time_range=time_range,
                granularity=granularity,
                results=results,
                insights=insights,
                recommendations=recommendations,
                confidence_score=confidence_score
            )
            
            # Cache analysis result
            await self._cache_analysis_result(analysis_result)
            
            logger.info(f"AI history analysis completed: {ai_model_id}, analysis: {analysis_result.analysis_id}")
            
            return analysis_result
            
        except Exception as e:
            error = handle_agent_error(e, ai_model_id=ai_model_id, user_id=user_id)
            log_agent_error(error)
            raise error
    
    async def compare_ai_versions(
        self,
        model_versions: List[str],
        comparison_metrics: List[str],
        time_range: Tuple[datetime, datetime],
        comparison_options: Dict[str, Any] = None,
        user_id: str = None
    ) -> ComparisonResult:
        """Compare different AI model versions"""
        try:
            # Validate comparison data
            await self._validate_comparison_data(model_versions, comparison_metrics)
            
            # Get history data for each version
            version_data = {}
            for version in model_versions:
                version_data[version] = await self.get_ai_history(
                    version, time_range[0], time_range[1]
                )
            
            # Perform comparison analysis
            comparison_results = await self._perform_version_comparison(
                version_data, comparison_metrics, comparison_options
            )
            
            # Determine winner
            winner = await self._determine_winner(comparison_results, comparison_metrics)
            
            # Calculate improvement percentage
            improvement_percentage = await self._calculate_improvement_percentage(
                comparison_results, winner
            )
            
            # Calculate statistical significance
            statistical_significance = await self._calculate_statistical_significance(
                comparison_results, comparison_metrics
            )
            
            # Create comparison result
            comparison_result = ComparisonResult(
                comparison_id=str(uuid4()),
                model_versions=model_versions,
                comparison_metrics=comparison_metrics,
                results=comparison_results,
                winner=winner,
                improvement_percentage=improvement_percentage,
                statistical_significance=statistical_significance
            )
            
            # Cache comparison result
            await self._cache_comparison_result(comparison_result)
            
            logger.info(f"AI version comparison completed: {model_versions}, comparison: {comparison_result.comparison_id}")
            
            return comparison_result
            
        except Exception as e:
            error = handle_agent_error(e, model_versions=model_versions, user_id=user_id)
            log_agent_error(error)
            raise error
    
    async def get_ai_history(
        self,
        ai_model_id: str,
        start_time: datetime,
        end_time: datetime,
        history_type: AIHistoryType = None
    ) -> List[AIHistoryRecord]:
        """Get AI history data for a model"""
        try:
            # Try cache first
            cache_key = f"ai_history:{ai_model_id}:{start_time.isoformat()}:{end_time.isoformat()}"
            cached_data = await self._get_cached_history(cache_key)
            if cached_data:
                return cached_data
            
            # Get from database
            history_records = await db_manager.get_ai_history_by_model(
                ai_model_id, start_time, end_time, history_type.value if history_type else None
            )
            
            # Convert to AIHistoryRecord objects
            records = []
            for record in history_records:
                ai_record = AIHistoryRecord(
                    record_id=str(record.id),
                    ai_model_id=record.ai_model_id,
                    history_type=AIHistoryType(record.history_type),
                    timestamp=record.created_at,
                    data=record.data,
                    metadata=record.metadata,
                    version=record.version,
                    tags=record.tags
                )
                records.append(ai_record)
            
            # Cache history data
            await self._cache_history_data_list(cache_key, records)
            
            return records
            
        except Exception as e:
            error = handle_agent_error(e, ai_model_id=ai_model_id)
            log_agent_error(error)
            raise error
    
    async def get_analysis_result(self, analysis_id: str) -> Optional[AnalysisResult]:
        """Get analysis result by ID"""
        try:
            # Try cache first
            cached_result = await self._get_cached_analysis(analysis_id)
            if cached_result:
                return cached_result
            
            # Get from database
            analysis = await db_manager.get_analysis_by_id(analysis_id)
            if not analysis:
                return None
            
            return AnalysisResult(
                analysis_id=str(analysis.id),
                analysis_type=AnalysisType(analysis.analysis_type),
                time_range=(analysis.start_time, analysis.end_time),
                granularity=TimeGranularity(analysis.granularity),
                results=analysis.results,
                insights=analysis.insights,
                recommendations=analysis.recommendations,
                confidence_score=analysis.confidence_score,
                analysis_timestamp=analysis.created_at
            )
            
        except Exception as e:
            error = handle_agent_error(e, analysis_id=analysis_id)
            log_agent_error(error)
            raise error
    
    # Private helper methods
    async def _validate_history_data(
        self,
        ai_model_id: str,
        history_type: AIHistoryType,
        data: Dict[str, Any]
    ) -> None:
        """Validate history data"""
        if not ai_model_id or len(ai_model_id.strip()) == 0:
            raise AIHistoryValidationError(
                "invalid_ai_model_id",
                "AI model ID cannot be empty",
                {"ai_model_id": ai_model_id}
            )
        
        if not data:
            raise AIHistoryValidationError(
                "invalid_data",
                "History data cannot be empty",
                {"history_type": history_type}
            )
    
    async def _validate_comparison_data(
        self,
        model_versions: List[str],
        comparison_metrics: List[str]
    ) -> None:
        """Validate comparison data"""
        if len(model_versions) < 2:
            raise AIHistoryValidationError(
                "insufficient_versions",
                "At least 2 model versions required for comparison",
                {"model_versions": model_versions}
            )
        
        if not comparison_metrics:
            raise AIHistoryValidationError(
                "invalid_metrics",
                "Comparison metrics cannot be empty",
                {"comparison_metrics": comparison_metrics}
            )
    
    async def _perform_trend_analysis(
        self,
        history_data: List[AIHistoryRecord],
        granularity: TimeGranularity
    ) -> Dict[str, Any]:
        """Perform trend analysis on history data"""
        # Group data by time granularity
        grouped_data = await self._group_data_by_granularity(history_data, granularity)
        
        # Calculate trends for each metric
        trends = {}
        for metric in ["accuracy", "loss", "performance", "cost", "usage"]:
            if metric in grouped_data:
                trends[metric] = await self._calculate_trend(grouped_data[metric])
        
        # Calculate overall trend
        overall_trend = await self._calculate_overall_trend(trends)
        
        return {
            "trends": trends,
            "overall_trend": overall_trend,
            "data_points": len(history_data),
            "time_span": {
                "start": min(record.timestamp for record in history_data),
                "end": max(record.timestamp for record in history_data)
            }
        }
    
    async def _perform_comparative_analysis(
        self,
        history_data: List[AIHistoryRecord],
        granularity: TimeGranularity
    ) -> Dict[str, Any]:
        """Perform comparative analysis on history data"""
        # Group data by version
        version_data = {}
        for record in history_data:
            version = record.version
            if version not in version_data:
                version_data[version] = []
            version_data[version].append(record)
        
        # Compare versions
        comparisons = {}
        versions = list(version_data.keys())
        for i in range(len(versions)):
            for j in range(i + 1, len(versions)):
                v1, v2 = versions[i], versions[j]
                comparison = await self._compare_versions(
                    version_data[v1], version_data[v2]
                )
                comparisons[f"{v1}_vs_{v2}"] = comparison
        
        return {
            "comparisons": comparisons,
            "versions": versions,
            "best_version": await self._find_best_version(comparisons),
            "data_points": len(history_data)
        }
    
    async def _perform_performance_analysis(
        self,
        history_data: List[AIHistoryRecord],
        granularity: TimeGranularity
    ) -> Dict[str, Any]:
        """Perform performance analysis on history data"""
        # Extract performance metrics
        performance_metrics = []
        for record in history_data:
            if "performance" in record.data:
                performance_metrics.append(record.data["performance"])
        
        if not performance_metrics:
            return {"error": "No performance data found"}
        
        # Calculate performance statistics
        stats = {
            "mean": np.mean(performance_metrics),
            "median": np.median(performance_metrics),
            "std": np.std(performance_metrics),
            "min": np.min(performance_metrics),
            "max": np.max(performance_metrics),
            "percentiles": {
                "25th": np.percentile(performance_metrics, 25),
                "75th": np.percentile(performance_metrics, 75),
                "90th": np.percentile(performance_metrics, 90),
                "95th": np.percentile(performance_metrics, 95)
            }
        }
        
        # Calculate performance trends
        trends = await self._calculate_performance_trends(history_data, granularity)
        
        return {
            "statistics": stats,
            "trends": trends,
            "data_points": len(performance_metrics),
            "performance_score": await self._calculate_performance_score(stats)
        }
    
    async def _perform_cost_analysis(
        self,
        history_data: List[AIHistoryRecord],
        granularity: TimeGranularity
    ) -> Dict[str, Any]:
        """Perform cost analysis on history data"""
        # Extract cost data
        cost_data = []
        for record in history_data:
            if "cost" in record.data:
                cost_data.append(record.data["cost"])
        
        if not cost_data:
            return {"error": "No cost data found"}
        
        # Calculate cost statistics
        total_cost = sum(cost_data)
        avg_cost = total_cost / len(cost_data)
        
        # Calculate cost trends
        cost_trends = await self._calculate_cost_trends(history_data, granularity)
        
        # Calculate cost efficiency
        cost_efficiency = await self._calculate_cost_efficiency(history_data)
        
        return {
            "total_cost": total_cost,
            "average_cost": avg_cost,
            "cost_trends": cost_trends,
            "cost_efficiency": cost_efficiency,
            "data_points": len(cost_data)
        }
    
    async def _perform_usage_analysis(
        self,
        history_data: List[AIHistoryRecord],
        granularity: TimeGranularity
    ) -> Dict[str, Any]:
        """Perform usage analysis on history data"""
        # Extract usage data
        usage_data = []
        for record in history_data:
            if "usage" in record.data:
                usage_data.append(record.data["usage"])
        
        if not usage_data:
            return {"error": "No usage data found"}
        
        # Calculate usage statistics
        total_usage = sum(usage_data)
        avg_usage = total_usage / len(usage_data)
        
        # Calculate usage patterns
        usage_patterns = await self._calculate_usage_patterns(history_data, granularity)
        
        # Calculate usage efficiency
        usage_efficiency = await self._calculate_usage_efficiency(history_data)
        
        return {
            "total_usage": total_usage,
            "average_usage": avg_usage,
            "usage_patterns": usage_patterns,
            "usage_efficiency": usage_efficiency,
            "data_points": len(usage_data)
        }
    
    async def _perform_error_analysis(
        self,
        history_data: List[AIHistoryRecord],
        granularity: TimeGranularity
    ) -> Dict[str, Any]:
        """Perform error analysis on history data"""
        # Extract error data
        error_data = []
        for record in history_data:
            if "error" in record.data:
                error_data.append(record.data["error"])
        
        if not error_data:
            return {"error": "No error data found"}
        
        # Calculate error statistics
        error_rate = len(error_data) / len(history_data)
        error_types = {}
        for error in error_data:
            error_type = error.get("type", "unknown")
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        # Calculate error trends
        error_trends = await self._calculate_error_trends(history_data, granularity)
        
        return {
            "error_rate": error_rate,
            "error_types": error_types,
            "error_trends": error_trends,
            "total_errors": len(error_data),
            "data_points": len(history_data)
        }
    
    async def _perform_predictive_analysis(
        self,
        history_data: List[AIHistoryRecord],
        granularity: TimeGranularity
    ) -> Dict[str, Any]:
        """Perform predictive analysis on history data"""
        # Extract time series data
        time_series_data = []
        for record in history_data:
            if "performance" in record.data:
                time_series_data.append({
                    "timestamp": record.timestamp,
                    "value": record.data["performance"]
                })
        
        if len(time_series_data) < 10:
            return {"error": "Insufficient data for predictive analysis"}
        
        # Perform time series forecasting
        predictions = await self._perform_time_series_forecasting(time_series_data)
        
        # Calculate prediction confidence
        confidence = await self._calculate_prediction_confidence(predictions)
        
        return {
            "predictions": predictions,
            "confidence": confidence,
            "forecast_horizon": 30,  # days
            "data_points": len(time_series_data)
        }
    
    async def _perform_correlation_analysis(
        self,
        history_data: List[AIHistoryRecord],
        granularity: TimeGranularity
    ) -> Dict[str, Any]:
        """Perform correlation analysis on history data"""
        # Extract metrics for correlation
        metrics = {}
        for record in history_data:
            for key, value in record.data.items():
                if isinstance(value, (int, float)):
                    if key not in metrics:
                        metrics[key] = []
                    metrics[key].append(value)
        
        if len(metrics) < 2:
            return {"error": "Insufficient metrics for correlation analysis"}
        
        # Calculate correlations
        correlations = {}
        metric_names = list(metrics.keys())
        for i in range(len(metric_names)):
            for j in range(i + 1, len(metric_names)):
                metric1, metric2 = metric_names[i], metric_names[j]
                correlation = np.corrcoef(metrics[metric1], metrics[metric2])[0, 1]
                correlations[f"{metric1}_vs_{metric2}"] = correlation
        
        return {
            "correlations": correlations,
            "strong_correlations": {
                k: v for k, v in correlations.items() if abs(v) > 0.7
            },
            "metrics": metric_names,
            "data_points": len(history_data)
        }
    
    async def _perform_statistical_analysis(
        self,
        history_data: List[AIHistoryRecord],
        granularity: TimeGranularity
    ) -> Dict[str, Any]:
        """Perform statistical analysis on history data"""
        # Extract all numeric data
        numeric_data = {}
        for record in history_data:
            for key, value in record.data.items():
                if isinstance(value, (int, float)):
                    if key not in numeric_data:
                        numeric_data[key] = []
                    numeric_data[key].append(value)
        
        # Calculate statistics for each metric
        statistics = {}
        for metric, values in numeric_data.items():
            statistics[metric] = {
                "mean": np.mean(values),
                "median": np.median(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "count": len(values)
            }
        
        return {
            "statistics": statistics,
            "metrics": list(numeric_data.keys()),
            "data_points": len(history_data)
        }
    
    async def _perform_custom_analysis(
        self,
        history_data: List[AIHistoryRecord],
        granularity: TimeGranularity,
        analysis_options: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Perform custom analysis on history data"""
        # This would implement custom analysis logic based on options
        return {
            "custom_analysis": "Custom analysis completed",
            "data_points": len(history_data),
            "options": analysis_options or {}
        }
    
    async def _group_data_by_granularity(
        self,
        history_data: List[AIHistoryRecord],
        granularity: TimeGranularity
    ) -> Dict[str, List[float]]:
        """Group data by time granularity"""
        grouped_data = {}
        
        for record in history_data:
            # Determine time bucket based on granularity
            if granularity == TimeGranularity.HOUR:
                time_key = record.timestamp.replace(minute=0, second=0, microsecond=0)
            elif granularity == TimeGranularity.DAY:
                time_key = record.timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
            elif granularity == TimeGranularity.WEEK:
                # Get start of week
                days_since_monday = record.timestamp.weekday()
                time_key = (record.timestamp - timedelta(days=days_since_monday)).replace(hour=0, minute=0, second=0, microsecond=0)
            elif granularity == TimeGranularity.MONTH:
                time_key = record.timestamp.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            else:
                time_key = record.timestamp
            
            time_key_str = time_key.isoformat()
            
            # Extract numeric values
            for key, value in record.data.items():
                if isinstance(value, (int, float)):
                    if key not in grouped_data:
                        grouped_data[key] = {}
                    if time_key_str not in grouped_data[key]:
                        grouped_data[key][time_key_str] = []
                    grouped_data[key][time_key_str].append(value)
        
        # Convert to lists of values
        result = {}
        for metric, time_data in grouped_data.items():
            result[metric] = []
            for time_key in sorted(time_data.keys()):
                values = time_data[time_key]
                result[metric].append(np.mean(values))  # Average values in each time bucket
        
        return result
    
    async def _calculate_trend(self, values: List[float]) -> Dict[str, Any]:
        """Calculate trend for a series of values"""
        if len(values) < 2:
            return {"trend": "insufficient_data", "slope": 0, "r_squared": 0}
        
        # Simple linear regression
        x = np.arange(len(values))
        y = np.array(values)
        
        # Calculate slope and intercept
        slope = np.polyfit(x, y, 1)[0]
        
        # Calculate R-squared
        y_pred = slope * x + np.mean(y)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Determine trend direction
        if slope > 0.01:
            trend = "increasing"
        elif slope < -0.01:
            trend = "decreasing"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "slope": slope,
            "r_squared": r_squared,
            "strength": abs(slope) * r_squared
        }
    
    async def _calculate_overall_trend(self, trends: Dict[str, Any]) -> str:
        """Calculate overall trend from individual trends"""
        if not trends:
            return "unknown"
        
        trend_scores = {"increasing": 1, "stable": 0, "decreasing": -1}
        total_score = 0
        total_weight = 0
        
        for metric, trend_data in trends.items():
            weight = trend_data.get("strength", 0)
            trend = trend_data.get("trend", "stable")
            score = trend_scores.get(trend, 0)
            
            total_score += score * weight
            total_weight += weight
        
        if total_weight == 0:
            return "stable"
        
        avg_score = total_score / total_weight
        
        if avg_score > 0.1:
            return "increasing"
        elif avg_score < -0.1:
            return "decreasing"
        else:
            return "stable"
    
    async def _compare_versions(
        self,
        version1_data: List[AIHistoryRecord],
        version2_data: List[AIHistoryRecord]
    ) -> Dict[str, Any]:
        """Compare two model versions"""
        # Extract performance metrics
        v1_performance = [r.data.get("performance", 0) for r in version1_data if "performance" in r.data]
        v2_performance = [r.data.get("performance", 0) for r in version2_data if "performance" in r.data]
        
        if not v1_performance or not v2_performance:
            return {"error": "Insufficient performance data"}
        
        # Calculate statistics
        v1_mean = np.mean(v1_performance)
        v2_mean = np.mean(v2_performance)
        
        # Calculate improvement
        improvement = ((v2_mean - v1_mean) / v1_mean) * 100 if v1_mean != 0 else 0
        
        return {
            "version1_mean": v1_mean,
            "version2_mean": v2_mean,
            "improvement_percentage": improvement,
            "better_version": "version2" if improvement > 0 else "version1",
            "statistical_significance": 0.95  # Placeholder
        }
    
    async def _find_best_version(self, comparisons: Dict[str, Any]) -> str:
        """Find the best version from comparisons"""
        version_scores = {}
        
        for comparison_key, comparison_data in comparisons.items():
            if "error" in comparison_data:
                continue
            
            versions = comparison_key.split("_vs_")
            v1, v2 = versions[0], versions[1]
            
            improvement = comparison_data.get("improvement_percentage", 0)
            
            if improvement > 0:
                version_scores[v2] = version_scores.get(v2, 0) + 1
            else:
                version_scores[v1] = version_scores.get(v1, 0) + 1
        
        if not version_scores:
            return "unknown"
        
        return max(version_scores, key=version_scores.get)
    
    async def _calculate_performance_trends(
        self,
        history_data: List[AIHistoryRecord],
        granularity: TimeGranularity
    ) -> Dict[str, Any]:
        """Calculate performance trends"""
        # Group data by granularity
        grouped_data = await self._group_data_by_granularity(history_data, granularity)
        
        if "performance" not in grouped_data:
            return {"error": "No performance data found"}
        
        # Calculate trend
        trend = await self._calculate_trend(grouped_data["performance"])
        
        return {
            "trend": trend,
            "data_points": len(grouped_data["performance"])
        }
    
    async def _calculate_performance_score(self, stats: Dict[str, Any]) -> float:
        """Calculate overall performance score"""
        # Weighted combination of statistics
        weights = {
            "mean": 0.4,
            "std": -0.2,  # Lower std is better
            "min": 0.2,
            "max": 0.2
        }
        
        score = 0
        total_weight = 0
        
        for metric, weight in weights.items():
            if metric in stats:
                score += stats[metric] * weight
                total_weight += abs(weight)
        
        return score / total_weight if total_weight > 0 else 0
    
    async def _calculate_cost_trends(
        self,
        history_data: List[AIHistoryRecord],
        granularity: TimeGranularity
    ) -> Dict[str, Any]:
        """Calculate cost trends"""
        # Group data by granularity
        grouped_data = await self._group_data_by_granularity(history_data, granularity)
        
        if "cost" not in grouped_data:
            return {"error": "No cost data found"}
        
        # Calculate trend
        trend = await self._calculate_trend(grouped_data["cost"])
        
        return {
            "trend": trend,
            "data_points": len(grouped_data["cost"])
        }
    
    async def _calculate_cost_efficiency(self, history_data: List[AIHistoryRecord]) -> float:
        """Calculate cost efficiency"""
        total_cost = 0
        total_performance = 0
        
        for record in history_data:
            if "cost" in record.data and "performance" in record.data:
                total_cost += record.data["cost"]
                total_performance += record.data["performance"]
        
        return total_performance / total_cost if total_cost > 0 else 0
    
    async def _calculate_usage_patterns(
        self,
        history_data: List[AIHistoryRecord],
        granularity: TimeGranularity
    ) -> Dict[str, Any]:
        """Calculate usage patterns"""
        # Group data by granularity
        grouped_data = await self._group_data_by_granularity(history_data, granularity)
        
        if "usage" not in grouped_data:
            return {"error": "No usage data found"}
        
        # Calculate trend
        trend = await self._calculate_trend(grouped_data["usage"])
        
        return {
            "trend": trend,
            "data_points": len(grouped_data["usage"])
        }
    
    async def _calculate_usage_efficiency(self, history_data: List[AIHistoryRecord]) -> float:
        """Calculate usage efficiency"""
        total_usage = 0
        total_performance = 0
        
        for record in history_data:
            if "usage" in record.data and "performance" in record.data:
                total_usage += record.data["usage"]
                total_performance += record.data["performance"]
        
        return total_performance / total_usage if total_usage > 0 else 0
    
    async def _calculate_error_trends(
        self,
        history_data: List[AIHistoryRecord],
        granularity: TimeGranularity
    ) -> Dict[str, Any]:
        """Calculate error trends"""
        # Count errors by time bucket
        error_counts = {}
        
        for record in history_data:
            if "error" in record.data:
                # Determine time bucket
                if granularity == TimeGranularity.HOUR:
                    time_key = record.timestamp.replace(minute=0, second=0, microsecond=0)
                elif granularity == TimeGranularity.DAY:
                    time_key = record.timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
                else:
                    time_key = record.timestamp
                
                time_key_str = time_key.isoformat()
                error_counts[time_key_str] = error_counts.get(time_key_str, 0) + 1
        
        # Convert to list of values
        error_values = list(error_counts.values())
        
        if not error_values:
            return {"error": "No error data found"}
        
        # Calculate trend
        trend = await self._calculate_trend(error_values)
        
        return {
            "trend": trend,
            "data_points": len(error_values)
        }
    
    async def _perform_time_series_forecasting(
        self,
        time_series_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Perform time series forecasting"""
        # Simple linear trend forecasting
        values = [d["value"] for d in time_series_data]
        x = np.arange(len(values))
        
        # Fit linear trend
        slope, intercept = np.polyfit(x, values, 1)
        
        # Generate future predictions
        future_x = np.arange(len(values), len(values) + 30)  # 30 days ahead
        future_values = slope * future_x + intercept
        
        return {
            "future_values": future_values.tolist(),
            "trend_slope": slope,
            "trend_intercept": intercept,
            "forecast_dates": [(datetime.now() + timedelta(days=i)).isoformat() for i in range(1, 31)]
        }
    
    async def _calculate_prediction_confidence(self, predictions: Dict[str, Any]) -> float:
        """Calculate prediction confidence"""
        # Simple confidence calculation based on trend strength
        slope = abs(predictions.get("trend_slope", 0))
        
        # Higher slope = lower confidence (more volatile)
        confidence = max(0.5, 1.0 - slope * 10)
        
        return min(0.95, confidence)
    
    async def _perform_version_comparison(
        self,
        version_data: Dict[str, List[AIHistoryRecord]],
        comparison_metrics: List[str],
        comparison_options: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Perform version comparison analysis"""
        results = {}
        
        for metric in comparison_metrics:
            metric_results = {}
            
            for version, data in version_data.items():
                # Extract metric values
                values = []
                for record in data:
                    if metric in record.data:
                        values.append(record.data[metric])
                
                if values:
                    metric_results[version] = {
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "count": len(values)
                    }
            
            results[metric] = metric_results
        
        return results
    
    async def _determine_winner(
        self,
        comparison_results: Dict[str, Any],
        comparison_metrics: List[str]
    ) -> str:
        """Determine the winning version"""
        version_scores = {}
        
        for metric in comparison_metrics:
            if metric not in comparison_results:
                continue
            
            metric_results = comparison_results[metric]
            versions = list(metric_results.keys())
            
            if len(versions) < 2:
                continue
            
            # Find best version for this metric
            best_version = max(versions, key=lambda v: metric_results[v]["mean"])
            
            # Award points
            for version in versions:
                if version not in version_scores:
                    version_scores[version] = 0
                
                if version == best_version:
                    version_scores[version] += 1
        
        if not version_scores:
            return "unknown"
        
        return max(version_scores, key=version_scores.get)
    
    async def _calculate_improvement_percentage(
        self,
        comparison_results: Dict[str, Any],
        winner: str
    ) -> float:
        """Calculate improvement percentage"""
        if winner == "unknown":
            return 0.0
        
        # Calculate average improvement across all metrics
        improvements = []
        
        for metric, metric_results in comparison_results.items():
            if winner not in metric_results:
                continue
            
            winner_value = metric_results[winner]["mean"]
            
            # Compare with other versions
            for version, results in metric_results.items():
                if version != winner:
                    other_value = results["mean"]
                    if other_value != 0:
                        improvement = ((winner_value - other_value) / other_value) * 100
                        improvements.append(improvement)
        
        return np.mean(improvements) if improvements else 0.0
    
    async def _calculate_statistical_significance(
        self,
        comparison_results: Dict[str, Any],
        comparison_metrics: List[str]
    ) -> float:
        """Calculate statistical significance"""
        # Simplified statistical significance calculation
        # In practice, you'd use proper statistical tests
        
        total_comparisons = 0
        significant_comparisons = 0
        
        for metric in comparison_metrics:
            if metric not in comparison_results:
                continue
            
            metric_results = comparison_results[metric]
            versions = list(metric_results.keys())
            
            if len(versions) < 2:
                continue
            
            # Simple significance test based on standard deviation
            for i in range(len(versions)):
                for j in range(i + 1, len(versions)):
                    v1, v2 = versions[i], versions[1]
                    
                    mean1 = metric_results[v1]["mean"]
                    mean2 = metric_results[v2]["mean"]
                    std1 = metric_results[v1]["std"]
                    std2 = metric_results[v2]["std"]
                    
                    # Simple significance test
                    if abs(mean1 - mean2) > (std1 + std2) / 2:
                        significant_comparisons += 1
                    
                    total_comparisons += 1
        
        return significant_comparisons / total_comparisons if total_comparisons > 0 else 0.0
    
    async def _generate_insights(
        self,
        results: Dict[str, Any],
        analysis_type: AnalysisType
    ) -> List[str]:
        """Generate insights from analysis results"""
        insights = []
        
        if analysis_type == AnalysisType.TREND_ANALYSIS:
            overall_trend = results.get("overall_trend", "unknown")
            insights.append(f"Overall trend is {overall_trend}")
            
            if "trends" in results:
                for metric, trend_data in results["trends"].items():
                    trend = trend_data.get("trend", "unknown")
                    strength = trend_data.get("strength", 0)
                    insights.append(f"{metric} shows {trend} trend with strength {strength:.2f}")
        
        elif analysis_type == AnalysisType.PERFORMANCE_ANALYSIS:
            if "performance_score" in results:
                score = results["performance_score"]
                insights.append(f"Overall performance score: {score:.2f}")
            
            if "statistics" in results:
                stats = results["statistics"]
                if "performance" in stats:
                    mean_perf = stats["performance"]["mean"]
                    insights.append(f"Average performance: {mean_perf:.2f}")
        
        elif analysis_type == AnalysisType.COST_ANALYSIS:
            if "total_cost" in results:
                total_cost = results["total_cost"]
                insights.append(f"Total cost: ${total_cost:.2f}")
            
            if "cost_efficiency" in results:
                efficiency = results["cost_efficiency"]
                insights.append(f"Cost efficiency: {efficiency:.2f}")
        
        return insights
    
    async def _generate_recommendations(
        self,
        results: Dict[str, Any],
        analysis_type: AnalysisType
    ) -> List[str]:
        """Generate recommendations from analysis results"""
        recommendations = []
        
        if analysis_type == AnalysisType.TREND_ANALYSIS:
            overall_trend = results.get("overall_trend", "unknown")
            
            if overall_trend == "decreasing":
                recommendations.append("Consider model retraining or hyperparameter optimization")
            elif overall_trend == "increasing":
                recommendations.append("Continue current approach, performance is improving")
            else:
                recommendations.append("Monitor performance closely for any changes")
        
        elif analysis_type == AnalysisType.PERFORMANCE_ANALYSIS:
            if "performance_score" in results:
                score = results["performance_score"]
                if score < 0.7:
                    recommendations.append("Performance is below optimal, consider optimization")
                elif score > 0.9:
                    recommendations.append("Performance is excellent, maintain current approach")
        
        elif analysis_type == AnalysisType.COST_ANALYSIS:
            if "cost_efficiency" in results:
                efficiency = results["cost_efficiency"]
                if efficiency < 1.0:
                    recommendations.append("Cost efficiency is low, consider cost optimization")
        
        return recommendations
    
    async def _calculate_confidence_score(
        self,
        results: Dict[str, Any],
        history_data: List[AIHistoryRecord]
    ) -> float:
        """Calculate confidence score for analysis"""
        # Base confidence on data quality and quantity
        data_points = len(history_data)
        
        # More data points = higher confidence
        if data_points < 10:
            base_confidence = 0.5
        elif data_points < 50:
            base_confidence = 0.7
        elif data_points < 100:
            base_confidence = 0.8
        else:
            base_confidence = 0.9
        
        # Adjust based on data quality
        quality_factors = []
        
        # Check for missing data
        missing_data_ratio = 0
        for record in history_data:
            if not record.data:
                missing_data_ratio += 1
        
        missing_data_ratio /= len(history_data)
        quality_factors.append(1 - missing_data_ratio)
        
        # Check for data consistency
        if "statistics" in results:
            stats = results["statistics"]
            for metric, stat_data in stats.items():
                if "std" in stat_data:
                    # Lower std relative to mean = higher quality
                    if stat_data["mean"] != 0:
                        cv = stat_data["std"] / abs(stat_data["mean"])
                        quality_factors.append(max(0, 1 - cv))
        
        # Calculate final confidence
        if quality_factors:
            quality_score = np.mean(quality_factors)
        else:
            quality_score = 1.0
        
        final_confidence = base_confidence * quality_score
        
        return min(0.95, max(0.1, final_confidence))
    
    # Caching methods
    async def _cache_history_data(self, history_record: Any) -> None:
        """Cache history data"""
        cache_key = f"ai_history_record:{history_record.id}"
        history_data = {
            "id": str(history_record.id),
            "ai_model_id": history_record.ai_model_id,
            "history_type": history_record.history_type,
            "data": history_record.data,
            "timestamp": history_record.created_at.isoformat()
        }
        
        await self.redis.setex(
            cache_key,
            3600,  # 1 hour
            json.dumps(history_data)
        )
    
    async def _cache_history_data_list(self, cache_key: str, records: List[AIHistoryRecord]) -> None:
        """Cache list of history data"""
        history_data = []
        for record in records:
            history_data.append({
                "record_id": record.record_id,
                "ai_model_id": record.ai_model_id,
                "history_type": record.history_type.value,
                "timestamp": record.timestamp.isoformat(),
                "data": record.data,
                "metadata": record.metadata,
                "version": record.version,
                "tags": record.tags
            })
        
        await self.redis.setex(
            cache_key,
            1800,  # 30 minutes
            json.dumps(history_data)
        )
    
    async def _get_cached_history(self, cache_key: str) -> Optional[List[AIHistoryRecord]]:
        """Get cached history data"""
        cached_data = await self.redis.get(cache_key)
        
        if cached_data:
            history_data = json.loads(cached_data)
            records = []
            for data in history_data:
                record = AIHistoryRecord(
                    record_id=data["record_id"],
                    ai_model_id=data["ai_model_id"],
                    history_type=AIHistoryType(data["history_type"]),
                    timestamp=datetime.fromisoformat(data["timestamp"]),
                    data=data["data"],
                    metadata=data["metadata"],
                    version=data["version"],
                    tags=data["tags"]
                )
                records.append(record)
            return records
        
        return None
    
    async def _cache_analysis_result(self, analysis_result: AnalysisResult) -> None:
        """Cache analysis result"""
        cache_key = f"ai_analysis:{analysis_result.analysis_id}"
        analysis_data = {
            "analysis_id": analysis_result.analysis_id,
            "analysis_type": analysis_result.analysis_type.value,
            "time_range": [t.isoformat() for t in analysis_result.time_range],
            "granularity": analysis_result.granularity.value,
            "results": analysis_result.results,
            "insights": analysis_result.insights,
            "recommendations": analysis_result.recommendations,
            "confidence_score": analysis_result.confidence_score,
            "analysis_timestamp": analysis_result.analysis_timestamp.isoformat()
        }
        
        await self.redis.setex(
            cache_key,
            1800,  # 30 minutes
            json.dumps(analysis_data)
        )
    
    async def _get_cached_analysis(self, analysis_id: str) -> Optional[AnalysisResult]:
        """Get cached analysis result"""
        cache_key = f"ai_analysis:{analysis_id}"
        cached_data = await self.redis.get(cache_key)
        
        if cached_data:
            analysis_data = json.loads(cached_data)
            return AnalysisResult(
                analysis_id=analysis_data["analysis_id"],
                analysis_type=AnalysisType(analysis_data["analysis_type"]),
                time_range=(datetime.fromisoformat(analysis_data["time_range"][0]), 
                           datetime.fromisoformat(analysis_data["time_range"][1])),
                granularity=TimeGranularity(analysis_data["granularity"]),
                results=analysis_data["results"],
                insights=analysis_data["insights"],
                recommendations=analysis_data["recommendations"],
                confidence_score=analysis_data["confidence_score"],
                analysis_timestamp=datetime.fromisoformat(analysis_data["analysis_timestamp"])
            )
        
        return None
    
    async def _cache_comparison_result(self, comparison_result: ComparisonResult) -> None:
        """Cache comparison result"""
        cache_key = f"ai_comparison:{comparison_result.comparison_id}"
        comparison_data = {
            "comparison_id": comparison_result.comparison_id,
            "model_versions": comparison_result.model_versions,
            "comparison_metrics": comparison_result.comparison_metrics,
            "results": comparison_result.results,
            "winner": comparison_result.winner,
            "improvement_percentage": comparison_result.improvement_percentage,
            "statistical_significance": comparison_result.statistical_significance,
            "comparison_timestamp": comparison_result.comparison_timestamp.isoformat()
        }
        
        await self.redis.setex(
            cache_key,
            1800,  # 30 minutes
            json.dumps(comparison_data)
        )



























