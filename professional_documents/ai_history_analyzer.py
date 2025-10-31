"""
AI History Analyzer
==================

Advanced AI history analysis and evolution tracking for document generation.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass, field
from uuid import uuid4
import json
import numpy as np
from collections import defaultdict, Counter
import hashlib
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


class AnalysisType(str, Enum):
    """Analysis type."""
    QUALITY_EVOLUTION = "quality_evolution"
    PATTERN_DETECTION = "pattern_detection"
    PERFORMANCE_TRACKING = "performance_tracking"
    CONTENT_ANALYSIS = "content_analysis"
    USER_BEHAVIOR = "user_behavior"
    MODEL_EVOLUTION = "model_evolution"


class EvolutionStage(str, Enum):
    """Evolution stage."""
    INITIAL = "initial"
    IMPROVING = "improving"
    STABLE = "stable"
    DECLINING = "declining"
    BREAKTHROUGH = "breakthrough"


@dataclass
class AIHistoryEntry:
    """AI history entry."""
    entry_id: str
    timestamp: datetime
    model_version: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    performance_metrics: Dict[str, float]
    user_feedback: Optional[Dict[str, Any]] = None
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvolutionPattern:
    """Evolution pattern."""
    pattern_id: str
    pattern_type: str
    description: str
    confidence: float
    time_range: Tuple[datetime, datetime]
    affected_metrics: List[str]
    trend_direction: str
    significance_score: float
    recommendations: List[str] = field(default_factory=list)


@dataclass
class QualityEvolution:
    """Quality evolution tracking."""
    evolution_id: str
    metric_name: str
    time_series: List[Tuple[datetime, float]]
    trend: str
    improvement_rate: float
    volatility: float
    current_stage: EvolutionStage
    predicted_future: List[Tuple[datetime, float]]
    confidence_interval: Tuple[float, float]


@dataclass
class ModelEvolution:
    """Model evolution tracking."""
    model_id: str
    version: str
    evolution_stage: EvolutionStage
    performance_metrics: Dict[str, float]
    training_data_size: int
    model_architecture: Dict[str, Any]
    hyperparameters: Dict[str, Any]
    created_at: datetime
    deployed_at: Optional[datetime] = None
    retired_at: Optional[datetime] = None
    successor_model: Optional[str] = None


class AIHistoryAnalyzer:
    """AI history analyzer for tracking evolution and patterns."""
    
    def __init__(self, history_file: str = "ai_history.json"):
        self.history_file = Path(history_file)
        self.history_entries: List[AIHistoryEntry] = []
        self.evolution_patterns: List[EvolutionPattern] = []
        self.quality_evolutions: Dict[str, QualityEvolution] = {}
        self.model_evolutions: Dict[str, ModelEvolution] = {}
        self.analysis_cache: Dict[str, Any] = {}
        
        self._load_history()
    
    def _load_history(self):
        """Load AI history from file."""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                    self.history_entries = [
                        AIHistoryEntry(
                            entry_id=entry["entry_id"],
                            timestamp=datetime.fromisoformat(entry["timestamp"]),
                            model_version=entry["model_version"],
                            input_data=entry["input_data"],
                            output_data=entry["output_data"],
                            performance_metrics=entry["performance_metrics"],
                            user_feedback=entry.get("user_feedback"),
                            context=entry.get("context", {}),
                            metadata=entry.get("metadata", {})
                        )
                        for entry in data.get("entries", [])
                    ]
                logger.info(f"Loaded {len(self.history_entries)} AI history entries")
            except Exception as e:
                logger.error(f"Failed to load AI history: {str(e)}")
    
    def _save_history(self):
        """Save AI history to file."""
        try:
            data = {
                "entries": [
                    {
                        "entry_id": entry.entry_id,
                        "timestamp": entry.timestamp.isoformat(),
                        "model_version": entry.model_version,
                        "input_data": entry.input_data,
                        "output_data": entry.output_data,
                        "performance_metrics": entry.performance_metrics,
                        "user_feedback": entry.user_feedback,
                        "context": entry.context,
                        "metadata": entry.metadata
                    }
                    for entry in self.history_entries
                ]
            }
            with open(self.history_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save AI history: {str(e)}")
    
    async def add_history_entry(
        self,
        model_version: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        performance_metrics: Dict[str, float],
        user_feedback: Optional[Dict[str, Any]] = None,
        context: Dict[str, Any] = None
    ) -> AIHistoryEntry:
        """Add new AI history entry."""
        
        entry = AIHistoryEntry(
            entry_id=str(uuid4()),
            timestamp=datetime.now(),
            model_version=model_version,
            input_data=input_data,
            output_data=output_data,
            performance_metrics=performance_metrics,
            user_feedback=user_feedback,
            context=context or {}
        )
        
        self.history_entries.append(entry)
        self._save_history()
        
        # Clear analysis cache to force recalculation
        self.analysis_cache.clear()
        
        logger.info(f"Added AI history entry: {entry.entry_id}")
        
        return entry
    
    async def analyze_quality_evolution(
        self,
        metric_name: str,
        time_window: Optional[timedelta] = None
    ) -> QualityEvolution:
        """Analyze quality evolution for a specific metric."""
        
        cache_key = f"quality_evolution_{metric_name}_{time_window}"
        if cache_key in self.analysis_cache:
            return self.analysis_cache[cache_key]
        
        # Filter entries by time window
        if time_window:
            cutoff_time = datetime.now() - time_window
            entries = [e for e in self.history_entries if e.timestamp >= cutoff_time]
        else:
            entries = self.history_entries
        
        # Extract metric values
        time_series = []
        for entry in entries:
            if metric_name in entry.performance_metrics:
                time_series.append((entry.timestamp, entry.performance_metrics[metric_name]))
        
        if len(time_series) < 2:
            raise ValueError(f"Insufficient data for metric {metric_name}")
        
        # Sort by timestamp
        time_series.sort(key=lambda x: x[0])
        
        # Calculate trend
        values = [v for _, v in time_series]
        trend = self._calculate_trend(values)
        
        # Calculate improvement rate
        improvement_rate = self._calculate_improvement_rate(values)
        
        # Calculate volatility
        volatility = np.std(values) if len(values) > 1 else 0
        
        # Determine evolution stage
        current_stage = self._determine_evolution_stage(values, trend, improvement_rate)
        
        # Predict future values
        predicted_future = self._predict_future_values(time_series, 10)
        
        # Calculate confidence interval
        confidence_interval = self._calculate_confidence_interval(values)
        
        evolution = QualityEvolution(
            evolution_id=str(uuid4()),
            metric_name=metric_name,
            time_series=time_series,
            trend=trend,
            improvement_rate=improvement_rate,
            volatility=volatility,
            current_stage=current_stage,
            predicted_future=predicted_future,
            confidence_interval=confidence_interval
        )
        
        self.quality_evolutions[metric_name] = evolution
        self.analysis_cache[cache_key] = evolution
        
        return evolution
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction."""
        if len(values) < 2:
            return "stable"
        
        # Simple linear regression
        x = np.arange(len(values))
        y = np.array(values)
        
        # Calculate slope
        slope = np.polyfit(x, y, 1)[0]
        
        if slope > 0.01:
            return "improving"
        elif slope < -0.01:
            return "declining"
        else:
            return "stable"
    
    def _calculate_improvement_rate(self, values: List[float]) -> float:
        """Calculate improvement rate."""
        if len(values) < 2:
            return 0.0
        
        # Calculate percentage change from first to last value
        first_value = values[0]
        last_value = values[-1]
        
        if first_value == 0:
            return 0.0
        
        return ((last_value - first_value) / first_value) * 100
    
    def _determine_evolution_stage(
        self,
        values: List[float],
        trend: str,
        improvement_rate: float
    ) -> EvolutionStage:
        """Determine evolution stage."""
        
        if improvement_rate > 20:
            return EvolutionStage.BREAKTHROUGH
        elif improvement_rate > 5:
            return EvolutionStage.IMPROVING
        elif improvement_rate < -5:
            return EvolutionStage.DECLINING
        elif trend == "stable" and abs(improvement_rate) < 2:
            return EvolutionStage.STABLE
        else:
            return EvolutionStage.INITIAL
    
    def _predict_future_values(
        self,
        time_series: List[Tuple[datetime, float]],
        num_predictions: int
    ) -> List[Tuple[datetime, float]]:
        """Predict future values using simple linear regression."""
        
        if len(time_series) < 2:
            return []
        
        # Extract timestamps and values
        timestamps = [t for t, _ in time_series]
        values = [v for _, v in time_series]
        
        # Convert timestamps to numeric values
        x = np.array([(t - timestamps[0]).total_seconds() for t in timestamps])
        y = np.array(values)
        
        # Fit linear regression
        coeffs = np.polyfit(x, y, 1)
        
        # Generate future predictions
        last_timestamp = timestamps[-1]
        predictions = []
        
        for i in range(1, num_predictions + 1):
            future_time = last_timestamp + timedelta(days=i)
            future_x = (future_time - timestamps[0]).total_seconds()
            predicted_value = coeffs[0] * future_x + coeffs[1]
            predictions.append((future_time, predicted_value))
        
        return predictions
    
    def _calculate_confidence_interval(self, values: List[float]) -> Tuple[float, float]:
        """Calculate confidence interval for values."""
        if len(values) < 2:
            return (0.0, 0.0)
        
        mean = np.mean(values)
        std = np.std(values)
        
        # 95% confidence interval
        margin = 1.96 * std / np.sqrt(len(values))
        
        return (mean - margin, mean + margin)
    
    async def detect_evolution_patterns(
        self,
        analysis_type: AnalysisType,
        time_window: Optional[timedelta] = None
    ) -> List[EvolutionPattern]:
        """Detect evolution patterns in AI history."""
        
        cache_key = f"patterns_{analysis_type}_{time_window}"
        if cache_key in self.analysis_cache:
            return self.analysis_cache[cache_key]
        
        patterns = []
        
        if analysis_type == AnalysisType.QUALITY_EVOLUTION:
            patterns = await self._detect_quality_patterns(time_window)
        elif analysis_type == AnalysisType.PATTERN_DETECTION:
            patterns = await self._detect_general_patterns(time_window)
        elif analysis_type == AnalysisType.PERFORMANCE_TRACKING:
            patterns = await self._detect_performance_patterns(time_window)
        elif analysis_type == AnalysisType.CONTENT_ANALYSIS:
            patterns = await self._detect_content_patterns(time_window)
        elif analysis_type == AnalysisType.USER_BEHAVIOR:
            patterns = await self._detect_user_behavior_patterns(time_window)
        elif analysis_type == AnalysisType.MODEL_EVOLUTION:
            patterns = await self._detect_model_evolution_patterns(time_window)
        
        self.analysis_cache[cache_key] = patterns
        
        return patterns
    
    async def _detect_quality_patterns(
        self,
        time_window: Optional[timedelta]
    ) -> List[EvolutionPattern]:
        """Detect quality evolution patterns."""
        
        patterns = []
        
        # Get all available metrics
        all_metrics = set()
        for entry in self.history_entries:
            all_metrics.update(entry.performance_metrics.keys())
        
        for metric in all_metrics:
            try:
                evolution = await self.analyze_quality_evolution(metric, time_window)
                
                # Create pattern based on evolution
                if evolution.current_stage == EvolutionStage.BREAKTHROUGH:
                    pattern = EvolutionPattern(
                        pattern_id=str(uuid4()),
                        pattern_type="quality_breakthrough",
                        description=f"Breakthrough improvement in {metric}",
                        confidence=0.9,
                        time_range=(evolution.time_series[0][0], evolution.time_series[-1][0]),
                        affected_metrics=[metric],
                        trend_direction=evolution.trend,
                        significance_score=evolution.improvement_rate / 100,
                        recommendations=[
                            f"Investigate factors contributing to {metric} breakthrough",
                            "Consider applying similar approaches to other metrics",
                            "Document best practices for {metric} improvement"
                        ]
                    )
                    patterns.append(pattern)
                
                elif evolution.current_stage == EvolutionStage.DECLINING:
                    pattern = EvolutionPattern(
                        pattern_id=str(uuid4()),
                        pattern_type="quality_decline",
                        description=f"Declining performance in {metric}",
                        confidence=0.8,
                        time_range=(evolution.time_series[0][0], evolution.time_series[-1][0]),
                        affected_metrics=[metric],
                        trend_direction=evolution.trend,
                        significance_score=abs(evolution.improvement_rate) / 100,
                        recommendations=[
                            f"Investigate causes of {metric} decline",
                            "Review recent model changes",
                            "Consider rollback or intervention"
                        ]
                    )
                    patterns.append(pattern)
                
            except Exception as e:
                logger.warning(f"Failed to analyze quality pattern for {metric}: {str(e)}")
        
        return patterns
    
    async def _detect_general_patterns(
        self,
        time_window: Optional[timedelta]
    ) -> List[EvolutionPattern]:
        """Detect general evolution patterns."""
        
        patterns = []
        
        # Filter entries by time window
        if time_window:
            cutoff_time = datetime.now() - time_window
            entries = [e for e in self.history_entries if e.timestamp >= cutoff_time]
        else:
            entries = self.history_entries
        
        if len(entries) < 10:
            return patterns
        
        # Detect usage patterns
        usage_by_hour = defaultdict(int)
        for entry in entries:
            usage_by_hour[entry.timestamp.hour] += 1
        
        peak_hour = max(usage_by_hour.items(), key=lambda x: x[1])
        
        if peak_hour[1] > len(entries) * 0.2:  # More than 20% of usage in one hour
            pattern = EvolutionPattern(
                pattern_id=str(uuid4()),
                pattern_type="usage_peak",
                description=f"Peak usage at hour {peak_hour[0]}:00",
                confidence=0.7,
                time_range=(entries[0].timestamp, entries[-1].timestamp),
                affected_metrics=["usage_frequency"],
                trend_direction="stable",
                significance_score=peak_hour[1] / len(entries),
                recommendations=[
                    "Consider scaling resources during peak hours",
                    "Optimize performance for peak usage patterns",
                    "Implement load balancing strategies"
                ]
            )
            patterns.append(pattern)
        
        # Detect model version patterns
        version_usage = Counter(entry.model_version for entry in entries)
        most_used_version = version_usage.most_common(1)[0]
        
        if most_used_version[1] > len(entries) * 0.8:  # More than 80% usage of one version
            pattern = EvolutionPattern(
                pattern_id=str(uuid4()),
                pattern_type="version_dominance",
                description=f"Version {most_used_version[0]} dominates usage",
                confidence=0.8,
                time_range=(entries[0].timestamp, entries[-1].timestamp),
                affected_metrics=["model_version"],
                trend_direction="stable",
                significance_score=most_used_version[1] / len(entries),
                recommendations=[
                    f"Consider deprecating older versions",
                    "Focus optimization efforts on version {most_used_version[0]}",
                    "Plan migration strategy for other versions"
                ]
            )
            patterns.append(pattern)
        
        return patterns
    
    async def _detect_performance_patterns(
        self,
        time_window: Optional[timedelta]
    ) -> List[EvolutionPattern]:
        """Detect performance-related patterns."""
        
        patterns = []
        
        # Filter entries by time window
        if time_window:
            cutoff_time = datetime.now() - time_window
            entries = [e for e in self.history_entries if e.timestamp >= cutoff_time]
        else:
            entries = self.history_entries
        
        if len(entries) < 5:
            return patterns
        
        # Analyze response time patterns
        response_times = []
        for entry in entries:
            if "response_time" in entry.performance_metrics:
                response_times.append(entry.performance_metrics["response_time"])
        
        if response_times:
            avg_response_time = np.mean(response_times)
            response_time_std = np.std(response_times)
            
            if response_time_std > avg_response_time * 0.5:  # High variability
                pattern = EvolutionPattern(
                    pattern_id=str(uuid4()),
                    pattern_type="performance_volatility",
                    description="High variability in response times",
                    confidence=0.7,
                    time_range=(entries[0].timestamp, entries[-1].timestamp),
                    affected_metrics=["response_time"],
                    trend_direction="volatile",
                    significance_score=response_time_std / avg_response_time,
                    recommendations=[
                        "Investigate causes of response time variability",
                        "Implement performance monitoring",
                        "Consider caching strategies"
                    ]
                )
                patterns.append(pattern)
        
        return patterns
    
    async def _detect_content_patterns(
        self,
        time_window: Optional[timedelta]
    ) -> List[EvolutionPattern]:
        """Detect content-related patterns."""
        
        patterns = []
        
        # Filter entries by time window
        if time_window:
            cutoff_time = datetime.now() - time_window
            entries = [e for e in self.history_entries if e.timestamp >= cutoff_time]
        else:
            entries = self.history_entries
        
        if len(entries) < 5:
            return patterns
        
        # Analyze content length patterns
        content_lengths = []
        for entry in entries:
            if "content" in entry.input_data:
                content_lengths.append(len(str(entry.input_data["content"])))
        
        if content_lengths:
            avg_length = np.mean(content_lengths)
            length_std = np.std(content_lengths)
            
            if length_std > avg_length * 0.8:  # High variability in content length
                pattern = EvolutionPattern(
                    pattern_id=str(uuid4()),
                    pattern_type="content_length_variability",
                    description="High variability in input content length",
                    confidence=0.6,
                    time_range=(entries[0].timestamp, entries[-1].timestamp),
                    affected_metrics=["content_length"],
                    trend_direction="volatile",
                    significance_score=length_std / avg_length,
                    recommendations=[
                        "Consider content length normalization",
                        "Implement adaptive processing based on content size",
                        "Optimize for different content length ranges"
                    ]
                )
                patterns.append(pattern)
        
        return patterns
    
    async def _detect_user_behavior_patterns(
        self,
        time_window: Optional[timedelta]
    ) -> List[EvolutionPattern]:
        """Detect user behavior patterns."""
        
        patterns = []
        
        # Filter entries by time window
        if time_window:
            cutoff_time = datetime.now() - time_window
            entries = [e for e in self.history_entries if e.timestamp >= cutoff_time]
        else:
            entries = self.history_entries
        
        if len(entries) < 10:
            return patterns
        
        # Analyze user feedback patterns
        feedback_scores = []
        for entry in entries:
            if entry.user_feedback and "rating" in entry.user_feedback:
                feedback_scores.append(entry.user_feedback["rating"])
        
        if feedback_scores:
            avg_rating = np.mean(feedback_scores)
            
            if avg_rating < 3.0:  # Low average rating
                pattern = EvolutionPattern(
                    pattern_id=str(uuid4()),
                    pattern_type="low_user_satisfaction",
                    description=f"Low average user rating: {avg_rating:.2f}",
                    confidence=0.8,
                    time_range=(entries[0].timestamp, entries[-1].timestamp),
                    affected_metrics=["user_satisfaction"],
                    trend_direction="declining",
                    significance_score=(5.0 - avg_rating) / 5.0,
                    recommendations=[
                        "Investigate causes of low user satisfaction",
                        "Review recent model changes",
                        "Implement user feedback collection improvements",
                        "Consider user experience enhancements"
                    ]
                )
                patterns.append(pattern)
        
        return patterns
    
    async def _detect_model_evolution_patterns(
        self,
        time_window: Optional[timedelta]
    ) -> List[EvolutionPattern]:
        """Detect model evolution patterns."""
        
        patterns = []
        
        # Filter entries by time window
        if time_window:
            cutoff_time = datetime.now() - time_window
            entries = [e for e in self.history_entries if e.timestamp >= cutoff_time]
        else:
            entries = self.history_entries
        
        if len(entries) < 5:
            return patterns
        
        # Analyze model version transitions
        version_transitions = []
        for i in range(1, len(entries)):
            if entries[i].model_version != entries[i-1].model_version:
                version_transitions.append(entries[i].timestamp)
        
        if len(version_transitions) > 0:
            # Calculate average time between version changes
            if len(version_transitions) > 1:
                time_diffs = []
                for i in range(1, len(version_transitions)):
                    time_diffs.append((version_transitions[i] - version_transitions[i-1]).total_seconds())
                
                avg_time_between_changes = np.mean(time_diffs)
                
                if avg_time_between_changes < 86400:  # Less than 1 day
                    pattern = EvolutionPattern(
                        pattern_id=str(uuid4()),
                        pattern_type="rapid_model_changes",
                        description="Rapid model version changes",
                        confidence=0.7,
                        time_range=(entries[0].timestamp, entries[-1].timestamp),
                        affected_metrics=["model_version"],
                        trend_direction="volatile",
                        significance_score=1.0 / (avg_time_between_changes / 86400),
                        recommendations=[
                            "Consider stabilizing model versions",
                            "Implement proper testing before deployment",
                            "Document model change rationale"
                        ]
                    )
                    patterns.append(pattern)
        
        return patterns
    
    async def get_evolution_summary(
        self,
        time_window: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """Get comprehensive evolution summary."""
        
        # Filter entries by time window
        if time_window:
            cutoff_time = datetime.now() - time_window
            entries = [e for e in self.history_entries if e.timestamp >= cutoff_time]
        else:
            entries = self.history_entries
        
        if not entries:
            return {"error": "No data available for the specified time window"}
        
        # Basic statistics
        total_entries = len(entries)
        time_span = (entries[-1].timestamp - entries[0].timestamp).total_seconds()
        
        # Model version distribution
        version_distribution = Counter(entry.model_version for entry in entries)
        
        # Performance metrics summary
        all_metrics = set()
        for entry in entries:
            all_metrics.update(entry.performance_metrics.keys())
        
        metrics_summary = {}
        for metric in all_metrics:
            values = [entry.performance_metrics[metric] for entry in entries if metric in entry.performance_metrics]
            if values:
                metrics_summary[metric] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "count": len(values)
                }
        
        # User feedback summary
        feedback_entries = [e for e in entries if e.user_feedback]
        feedback_summary = {}
        if feedback_entries:
            ratings = [f["rating"] for f in feedback_entries if "rating" in f]
            if ratings:
                feedback_summary = {
                    "average_rating": np.mean(ratings),
                    "rating_count": len(ratings),
                    "rating_std": np.std(ratings)
                }
        
        # Recent trends
        recent_entries = entries[-10:] if len(entries) >= 10 else entries
        recent_trends = {}
        for metric in all_metrics:
            recent_values = [entry.performance_metrics[metric] for entry in recent_entries if metric in entry.performance_metrics]
            if len(recent_values) >= 2:
                recent_trends[metric] = self._calculate_trend(recent_values)
        
        return {
            "summary": {
                "total_entries": total_entries,
                "time_span_days": time_span / 86400,
                "date_range": {
                    "start": entries[0].timestamp.isoformat(),
                    "end": entries[-1].timestamp.isoformat()
                }
            },
            "model_versions": dict(version_distribution),
            "performance_metrics": metrics_summary,
            "user_feedback": feedback_summary,
            "recent_trends": recent_trends,
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    async def export_analysis(
        self,
        analysis_type: str,
        format: str = "json"
    ) -> Union[str, bytes]:
        """Export analysis results."""
        
        if analysis_type == "quality_evolution":
            data = {name: {
                "evolution_id": evolution.evolution_id,
                "metric_name": evolution.metric_name,
                "trend": evolution.trend,
                "improvement_rate": evolution.improvement_rate,
                "volatility": evolution.volatility,
                "current_stage": evolution.current_stage.value,
                "time_series": [(t.isoformat(), v) for t, v in evolution.time_series],
                "predicted_future": [(t.isoformat(), v) for t, v in evolution.predicted_future],
                "confidence_interval": evolution.confidence_interval
            } for name, evolution in self.quality_evolutions.items()}
        
        elif analysis_type == "evolution_patterns":
            data = [{
                "pattern_id": pattern.pattern_id,
                "pattern_type": pattern.pattern_type,
                "description": pattern.description,
                "confidence": pattern.confidence,
                "time_range": [pattern.time_range[0].isoformat(), pattern.time_range[1].isoformat()],
                "affected_metrics": pattern.affected_metrics,
                "trend_direction": pattern.trend_direction,
                "significance_score": pattern.significance_score,
                "recommendations": pattern.recommendations
            } for pattern in self.evolution_patterns]
        
        elif analysis_type == "summary":
            data = await self.get_evolution_summary()
        
        else:
            raise ValueError(f"Unknown analysis type: {analysis_type}")
        
        if format == "json":
            return json.dumps(data, indent=2)
        elif format == "pickle":
            return pickle.dumps(data)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    async def get_analytics(self) -> Dict[str, Any]:
        """Get AI history analytics."""
        
        total_entries = len(self.history_entries)
        unique_models = len(set(entry.model_version for entry in self.history_entries))
        total_patterns = len(self.evolution_patterns)
        total_evolutions = len(self.quality_evolutions)
        
        # Time range
        if self.history_entries:
            time_range = (self.history_entries[-1].timestamp - self.history_entries[0].timestamp).total_seconds()
        else:
            time_range = 0
        
        # Recent activity
        recent_cutoff = datetime.now() - timedelta(days=7)
        recent_entries = len([e for e in self.history_entries if e.timestamp >= recent_cutoff])
        
        return {
            "total_entries": total_entries,
            "unique_models": unique_models,
            "total_patterns": total_patterns,
            "total_evolutions": total_evolutions,
            "time_range_days": time_range / 86400,
            "recent_entries_7d": recent_entries,
            "analysis_cache_size": len(self.analysis_cache),
            "last_analysis": datetime.now().isoformat()
        }



























