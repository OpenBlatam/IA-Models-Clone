"""
AI History Comparison System - AI Evolution Tracker

This module provides advanced AI model evolution tracking, version comparison,
and performance regression detection capabilities.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json
import asyncio
from collections import defaultdict
import statistics

from .ai_history_analyzer import AIHistoryAnalyzer, MetricType, HistoryEntry
from .advanced_ml_engine import ml_engine

logger = logging.getLogger(__name__)

class EvolutionMetric(Enum):
    """Types of evolution metrics"""
    QUALITY_IMPROVEMENT = "quality_improvement"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    CONSISTENCY_CHANGE = "consistency_change"
    BIAS_DRIFT = "bias_drift"
    VARIANCE_CHANGE = "variance_change"
    STYLE_DRIFT = "style_drift"
    TOPIC_DRIFT = "topic_drift"

class RegressionType(Enum):
    """Types of performance regressions"""
    QUALITY_REGRESSION = "quality_regression"
    PERFORMANCE_REGRESSION = "performance_regression"
    CONSISTENCY_REGRESSION = "consistency_regression"
    BIAS_REGRESSION = "bias_regression"
    VARIANCE_REGRESSION = "variance_regression"

@dataclass
class ModelVersion:
    """Model version information"""
    version: str
    release_date: datetime
    description: str
    performance_metrics: Dict[str, float]
    total_entries: int
    quality_score: float
    consistency_score: float

@dataclass
class EvolutionAnalysis:
    """AI model evolution analysis result"""
    model_versions: List[ModelVersion]
    evolution_trends: Dict[str, str]
    regression_detected: bool
    regression_details: List[Dict[str, Any]]
    improvement_areas: List[str]
    degradation_areas: List[str]
    recommendations: List[str]
    confidence_score: float
    analysis_timestamp: datetime

@dataclass
class PerformanceRegression:
    """Performance regression detection result"""
    regression_type: RegressionType
    severity: str  # "low", "medium", "high", "critical"
    affected_metrics: List[str]
    regression_magnitude: float
    confidence: float
    affected_versions: List[str]
    recommendations: List[str]
    detection_timestamp: datetime

class AIEvolutionTracker:
    """
    Advanced AI model evolution tracking and analysis system
    """
    
    def __init__(self, analyzer: AIHistoryAnalyzer):
        """Initialize AI evolution tracker"""
        self.analyzer = analyzer
        self.model_versions: Dict[str, ModelVersion] = {}
        self.evolution_history: List[EvolutionAnalysis] = []
        self.regression_thresholds = {
            "quality_regression": 0.1,  # 10% decrease
            "performance_regression": 0.15,  # 15% decrease
            "consistency_regression": 0.2,  # 20% decrease
            "bias_regression": 0.25,  # 25% increase in bias
            "variance_regression": 0.3  # 30% increase in variance
        }
        
        logger.info("AI Evolution Tracker initialized")

    def track_model_version(self, version: str, release_date: datetime, 
                           description: str = "") -> ModelVersion:
        """Track a new model version"""
        try:
            # Get entries for this model version
            version_entries = [e for e in self.analyzer.history_entries 
                             if e.model_version == version]
            
            if not version_entries:
                logger.warning(f"No entries found for model version: {version}")
                return None
            
            # Calculate performance metrics
            performance_metrics = self._calculate_version_metrics(version_entries)
            
            # Calculate quality and consistency scores
            quality_score = self._calculate_quality_score(version_entries)
            consistency_score = self._calculate_consistency_score(version_entries)
            
            # Create model version
            model_version = ModelVersion(
                version=version,
                release_date=release_date,
                description=description,
                performance_metrics=performance_metrics,
                total_entries=len(version_entries),
                quality_score=quality_score,
                consistency_score=consistency_score
            )
            
            self.model_versions[version] = model_version
            logger.info(f"Model version tracked: {version}")
            return model_version
            
        except Exception as e:
            logger.error(f"Error tracking model version {version}: {e}")
            return None

    def analyze_evolution(self, time_window: Optional[timedelta] = None) -> EvolutionAnalysis:
        """Analyze AI model evolution over time"""
        try:
            # Get model versions to analyze
            versions_to_analyze = list(self.model_versions.values())
            
            if len(versions_to_analyze) < 2:
                return EvolutionAnalysis(
                    model_versions=versions_to_analyze,
                    evolution_trends={},
                    regression_detected=False,
                    regression_details=[],
                    improvement_areas=[],
                    degradation_areas=[],
                    recommendations=["Insufficient data for evolution analysis"],
                    confidence_score=0.0,
                    analysis_timestamp=datetime.now()
                )
            
            # Sort versions by release date
            versions_to_analyze.sort(key=lambda v: v.release_date)
            
            # Analyze evolution trends
            evolution_trends = self._analyze_evolution_trends(versions_to_analyze)
            
            # Detect regressions
            regression_detected, regression_details = self._detect_regressions(versions_to_analyze)
            
            # Identify improvement and degradation areas
            improvement_areas, degradation_areas = self._identify_change_areas(versions_to_analyze)
            
            # Generate recommendations
            recommendations = self._generate_evolution_recommendations(
                evolution_trends, regression_details, improvement_areas, degradation_areas
            )
            
            # Calculate confidence score
            confidence_score = self._calculate_evolution_confidence(versions_to_analyze)
            
            # Create evolution analysis
            analysis = EvolutionAnalysis(
                model_versions=versions_to_analyze,
                evolution_trends=evolution_trends,
                regression_detected=regression_detected,
                regression_details=regression_details,
                improvement_areas=improvement_areas,
                degradation_areas=degradation_areas,
                recommendations=recommendations,
                confidence_score=confidence_score,
                analysis_timestamp=datetime.now()
            )
            
            self.evolution_history.append(analysis)
            logger.info("Evolution analysis completed")
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing evolution: {e}")
            return EvolutionAnalysis(
                model_versions=[],
                evolution_trends={},
                regression_detected=False,
                regression_details=[],
                improvement_areas=[],
                degradation_areas=[],
                recommendations=[f"Analysis failed: {str(e)}"],
                confidence_score=0.0,
                analysis_timestamp=datetime.now()
            )

    def detect_performance_regression(self, version_1: str, version_2: str) -> List[PerformanceRegression]:
        """Detect performance regressions between two model versions"""
        try:
            if version_1 not in self.model_versions or version_2 not in self.model_versions:
                raise ValueError("One or both model versions not found")
            
            v1 = self.model_versions[version_1]
            v2 = self.model_versions[version_2]
            
            regressions = []
            
            # Check quality regression
            quality_change = (v2.quality_score - v1.quality_score) / v1.quality_score
            if quality_change < -self.regression_thresholds["quality_regression"]:
                severity = self._determine_regression_severity(abs(quality_change))
                regressions.append(PerformanceRegression(
                    regression_type=RegressionType.QUALITY_REGRESSION,
                    severity=severity,
                    affected_metrics=["quality_score"],
                    regression_magnitude=abs(quality_change),
                    confidence=min(abs(quality_change) * 2, 1.0),
                    affected_versions=[version_1, version_2],
                    recommendations=self._get_quality_regression_recommendations(quality_change),
                    detection_timestamp=datetime.now()
                ))
            
            # Check consistency regression
            consistency_change = (v2.consistency_score - v1.consistency_score) / v1.consistency_score
            if consistency_change < -self.regression_thresholds["consistency_regression"]:
                severity = self._determine_regression_severity(abs(consistency_change))
                regressions.append(PerformanceRegression(
                    regression_type=RegressionType.CONSISTENCY_REGRESSION,
                    severity=severity,
                    affected_metrics=["consistency_score"],
                    regression_magnitude=abs(consistency_change),
                    confidence=min(abs(consistency_change) * 2, 1.0),
                    affected_versions=[version_1, version_2],
                    recommendations=self._get_consistency_regression_recommendations(consistency_change),
                    detection_timestamp=datetime.now()
                ))
            
            # Check performance metrics regression
            performance_regressions = self._check_performance_metrics_regression(v1, v2)
            regressions.extend(performance_regressions)
            
            logger.info(f"Performance regression detection completed: {len(regressions)} regressions found")
            return regressions
            
        except Exception as e:
            logger.error(f"Error detecting performance regression: {e}")
            return []

    def compare_model_versions(self, version_1: str, version_2: str) -> Dict[str, Any]:
        """Comprehensive comparison between two model versions"""
        try:
            if version_1 not in self.model_versions or version_2 not in self.model_versions:
                raise ValueError("One or both model versions not found")
            
            v1 = self.model_versions[version_1]
            v2 = self.model_versions[version_2]
            
            # Calculate differences
            quality_diff = v2.quality_score - v1.quality_score
            consistency_diff = v2.consistency_score - v1.consistency_score
            entries_diff = v2.total_entries - v1.total_entries
            
            # Performance metrics comparison
            performance_comparison = {}
            for metric in v1.performance_metrics:
                if metric in v2.performance_metrics:
                    v1_val = v1.performance_metrics[metric]
                    v2_val = v2.performance_metrics[metric]
                    if v1_val != 0:
                        change_percent = ((v2_val - v1_val) / abs(v1_val)) * 100
                    else:
                        change_percent = 0
                    performance_comparison[metric] = {
                        "v1_value": v1_val,
                        "v2_value": v2_val,
                        "change": v2_val - v1_val,
                        "change_percent": change_percent
                    }
            
            # Determine overall trend
            overall_trend = "improving" if quality_diff > 0 and consistency_diff > 0 else \
                          "declining" if quality_diff < 0 and consistency_diff < 0 else "mixed"
            
            # Generate insights
            insights = self._generate_version_comparison_insights(v1, v2, performance_comparison)
            
            return {
                "version_1": {
                    "version": v1.version,
                    "release_date": v1.release_date.isoformat(),
                    "description": v1.description,
                    "quality_score": v1.quality_score,
                    "consistency_score": v1.consistency_score,
                    "total_entries": v1.total_entries
                },
                "version_2": {
                    "version": v2.version,
                    "release_date": v2.release_date.isoformat(),
                    "description": v2.description,
                    "quality_score": v2.quality_score,
                    "consistency_score": v2.consistency_score,
                    "total_entries": v2.total_entries
                },
                "comparison": {
                    "quality_difference": quality_diff,
                    "consistency_difference": consistency_diff,
                    "entries_difference": entries_diff,
                    "overall_trend": overall_trend,
                    "performance_metrics": performance_comparison
                },
                "insights": insights,
                "comparison_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error comparing model versions: {e}")
            return {"error": str(e)}

    def get_evolution_timeline(self) -> List[Dict[str, Any]]:
        """Get evolution timeline with key milestones"""
        try:
            timeline = []
            
            # Sort versions by release date
            sorted_versions = sorted(self.model_versions.values(), key=lambda v: v.release_date)
            
            for i, version in enumerate(sorted_versions):
                timeline_entry = {
                    "version": version.version,
                    "release_date": version.release_date.isoformat(),
                    "description": version.description,
                    "quality_score": version.quality_score,
                    "consistency_score": version.consistency_score,
                    "total_entries": version.total_entries,
                    "milestone_type": "initial" if i == 0 else "update"
                }
                
                # Add milestone information
                if i > 0:
                    prev_version = sorted_versions[i-1]
                    quality_change = version.quality_score - prev_version.quality_score
                    consistency_change = version.consistency_score - prev_version.consistency_score
                    
                    if quality_change > 0.1:
                        timeline_entry["milestone_type"] = "quality_improvement"
                    elif quality_change < -0.1:
                        timeline_entry["milestone_type"] = "quality_regression"
                    elif consistency_change > 0.1:
                        timeline_entry["milestone_type"] = "consistency_improvement"
                    elif consistency_change < -0.1:
                        timeline_entry["milestone_type"] = "consistency_regression"
                
                timeline.append(timeline_entry)
            
            return timeline
            
        except Exception as e:
            logger.error(f"Error getting evolution timeline: {e}")
            return []

    def predict_future_performance(self, target_version: str = None) -> Dict[str, Any]:
        """Predict future model performance based on evolution trends"""
        try:
            if len(self.model_versions) < 3:
                return {"error": "Insufficient data for prediction (need at least 3 versions)"}
            
            # Get historical data
            sorted_versions = sorted(self.model_versions.values(), key=lambda v: v.release_date)
            
            # Prepare data for prediction
            quality_scores = [v.quality_score for v in sorted_versions]
            consistency_scores = [v.consistency_score for v in sorted_versions]
            dates = [v.release_date for v in sorted_versions]
            
            # Simple linear regression for prediction
            x = np.arange(len(quality_scores))
            
            # Predict quality score
            quality_slope, quality_intercept = np.polyfit(x, quality_scores, 1)
            next_quality = quality_slope * len(quality_scores) + quality_intercept
            
            # Predict consistency score
            consistency_slope, consistency_intercept = np.polyfit(x, consistency_scores, 1)
            next_consistency = consistency_slope * len(consistency_scores) + consistency_intercept
            
            # Calculate confidence based on R-squared
            quality_r2 = self._calculate_r_squared(quality_scores, x, quality_slope, quality_intercept)
            consistency_r2 = self._calculate_r_squared(consistency_scores, x, consistency_slope, consistency_intercept)
            
            # Generate predictions for next few versions
            predictions = []
            for i in range(1, 4):  # Predict next 3 versions
                future_quality = quality_slope * (len(quality_scores) + i) + quality_intercept
                future_consistency = consistency_slope * (len(consistency_scores) + i) + consistency_intercept
                
                predictions.append({
                    "version_number": len(quality_scores) + i,
                    "predicted_quality_score": max(0, min(1, future_quality)),  # Clamp to [0,1]
                    "predicted_consistency_score": max(0, min(1, future_consistency)),  # Clamp to [0,1]
                    "confidence": (quality_r2 + consistency_r2) / 2
                })
            
            return {
                "current_performance": {
                    "quality_score": quality_scores[-1],
                    "consistency_score": consistency_scores[-1]
                },
                "trend_analysis": {
                    "quality_trend": "improving" if quality_slope > 0 else "declining",
                    "consistency_trend": "improving" if consistency_slope > 0 else "declining",
                    "quality_slope": quality_slope,
                    "consistency_slope": consistency_slope
                },
                "predictions": predictions,
                "confidence_metrics": {
                    "quality_r_squared": quality_r2,
                    "consistency_r_squared": consistency_r2,
                    "overall_confidence": (quality_r2 + consistency_r2) / 2
                },
                "prediction_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error predicting future performance: {e}")
            return {"error": str(e)}

    def _calculate_version_metrics(self, entries: List[HistoryEntry]) -> Dict[str, float]:
        """Calculate performance metrics for a model version"""
        if not entries:
            return {}
        
        metrics = {}
        
        # Readability metrics
        readability_scores = [e.metrics.readability_score for e in entries]
        metrics["avg_readability"] = statistics.mean(readability_scores)
        metrics["readability_std"] = statistics.stdev(readability_scores) if len(readability_scores) > 1 else 0
        
        # Sentiment metrics
        sentiment_scores = [e.metrics.sentiment_score for e in entries]
        metrics["avg_sentiment"] = statistics.mean(sentiment_scores)
        metrics["sentiment_std"] = statistics.stdev(sentiment_scores) if len(sentiment_scores) > 1 else 0
        
        # Length metrics
        word_counts = [e.metrics.word_count for e in entries]
        metrics["avg_word_count"] = statistics.mean(word_counts)
        metrics["word_count_std"] = statistics.stdev(word_counts) if len(word_counts) > 1 else 0
        
        # Complexity metrics
        complexity_scores = [e.metrics.complexity_score for e in entries]
        metrics["avg_complexity"] = statistics.mean(complexity_scores)
        metrics["complexity_std"] = statistics.stdev(complexity_scores) if len(complexity_scores) > 1 else 0
        
        # Topic diversity
        topic_diversity_scores = [e.metrics.topic_diversity for e in entries]
        metrics["avg_topic_diversity"] = statistics.mean(topic_diversity_scores)
        
        return metrics

    def _calculate_quality_score(self, entries: List[HistoryEntry]) -> float:
        """Calculate overall quality score for a model version"""
        if not entries:
            return 0.0
        
        # Weighted combination of quality metrics
        readability_scores = [e.metrics.readability_score for e in entries]
        sentiment_scores = [e.metrics.sentiment_score for e in entries]
        complexity_scores = [e.metrics.complexity_score for e in entries]
        
        # Normalize and weight metrics
        avg_readability = statistics.mean(readability_scores) / 100.0  # Normalize to [0,1]
        avg_sentiment = (statistics.mean(sentiment_scores) + 1) / 2.0  # Normalize to [0,1]
        avg_complexity = statistics.mean(complexity_scores)  # Already in [0,1]
        
        # Weighted quality score
        quality_score = (0.4 * avg_readability + 0.3 * avg_sentiment + 0.3 * avg_complexity)
        return max(0, min(1, quality_score))

    def _calculate_consistency_score(self, entries: List[HistoryEntry]) -> float:
        """Calculate consistency score for a model version"""
        if len(entries) < 2:
            return 1.0  # Perfect consistency for single entry
        
        # Calculate variance in key metrics
        readability_scores = [e.metrics.readability_score for e in entries]
        sentiment_scores = [e.metrics.sentiment_score for e in entries]
        complexity_scores = [e.metrics.complexity_score for e in entries]
        
        # Calculate coefficient of variation (CV = std/mean)
        readability_cv = statistics.stdev(readability_scores) / statistics.mean(readability_scores) if statistics.mean(readability_scores) != 0 else 0
        sentiment_cv = statistics.stdev(sentiment_scores) / abs(statistics.mean(sentiment_scores)) if statistics.mean(sentiment_scores) != 0 else 0
        complexity_cv = statistics.stdev(complexity_scores) / statistics.mean(complexity_scores) if statistics.mean(complexity_scores) != 0 else 0
        
        # Consistency score (lower CV = higher consistency)
        avg_cv = (readability_cv + sentiment_cv + complexity_cv) / 3
        consistency_score = max(0, 1 - avg_cv)  # Convert to [0,1] scale
        
        return consistency_score

    def _analyze_evolution_trends(self, versions: List[ModelVersion]) -> Dict[str, str]:
        """Analyze evolution trends across model versions"""
        trends = {}
        
        if len(versions) < 2:
            return trends
        
        # Quality trend
        quality_scores = [v.quality_score for v in versions]
        if len(quality_scores) > 1:
            quality_slope = (quality_scores[-1] - quality_scores[0]) / (len(quality_scores) - 1)
            trends["quality"] = "improving" if quality_slope > 0.01 else "declining" if quality_slope < -0.01 else "stable"
        
        # Consistency trend
        consistency_scores = [v.consistency_score for v in versions]
        if len(consistency_scores) > 1:
            consistency_slope = (consistency_scores[-1] - consistency_scores[0]) / (len(consistency_scores) - 1)
            trends["consistency"] = "improving" if consistency_slope > 0.01 else "declining" if consistency_slope < -0.01 else "stable"
        
        # Volume trend
        entry_counts = [v.total_entries for v in versions]
        if len(entry_counts) > 1:
            volume_slope = (entry_counts[-1] - entry_counts[0]) / (len(entry_counts) - 1)
            trends["volume"] = "increasing" if volume_slope > 0 else "decreasing" if volume_slope < 0 else "stable"
        
        return trends

    def _detect_regressions(self, versions: List[ModelVersion]) -> Tuple[bool, List[Dict[str, Any]]]:
        """Detect regressions in model evolution"""
        regressions = []
        regression_detected = False
        
        if len(versions) < 2:
            return False, regressions
        
        # Check for quality regressions
        for i in range(1, len(versions)):
            prev_version = versions[i-1]
            curr_version = versions[i]
            
            quality_change = (curr_version.quality_score - prev_version.quality_score) / prev_version.quality_score
            if quality_change < -self.regression_thresholds["quality_regression"]:
                regressions.append({
                    "type": "quality_regression",
                    "severity": self._determine_regression_severity(abs(quality_change)),
                    "from_version": prev_version.version,
                    "to_version": curr_version.version,
                    "magnitude": abs(quality_change),
                    "affected_metric": "quality_score"
                })
                regression_detected = True
            
            # Check for consistency regressions
            consistency_change = (curr_version.consistency_score - prev_version.consistency_score) / prev_version.consistency_score
            if consistency_change < -self.regression_thresholds["consistency_regression"]:
                regressions.append({
                    "type": "consistency_regression",
                    "severity": self._determine_regression_severity(abs(consistency_change)),
                    "from_version": prev_version.version,
                    "to_version": curr_version.version,
                    "magnitude": abs(consistency_change),
                    "affected_metric": "consistency_score"
                })
                regression_detected = True
        
        return regression_detected, regressions

    def _identify_change_areas(self, versions: List[ModelVersion]) -> Tuple[List[str], List[str]]:
        """Identify areas of improvement and degradation"""
        improvement_areas = []
        degradation_areas = []
        
        if len(versions) < 2:
            return improvement_areas, degradation_areas
        
        first_version = versions[0]
        last_version = versions[-1]
        
        # Check quality changes
        quality_change = last_version.quality_score - first_version.quality_score
        if quality_change > 0.05:
            improvement_areas.append("Overall Quality")
        elif quality_change < -0.05:
            degradation_areas.append("Overall Quality")
        
        # Check consistency changes
        consistency_change = last_version.consistency_score - first_version.consistency_score
        if consistency_change > 0.05:
            improvement_areas.append("Consistency")
        elif consistency_change < -0.05:
            degradation_areas.append("Consistency")
        
        # Check performance metrics changes
        for metric in first_version.performance_metrics:
            if metric in last_version.performance_metrics:
                v1_val = first_version.performance_metrics[metric]
                v2_val = last_version.performance_metrics[metric]
                if v1_val != 0:
                    change_percent = ((v2_val - v1_val) / abs(v1_val)) * 100
                    if change_percent > 10:
                        improvement_areas.append(metric.replace("_", " ").title())
                    elif change_percent < -10:
                        degradation_areas.append(metric.replace("_", " ").title())
        
        return improvement_areas, degradation_areas

    def _generate_evolution_recommendations(self, trends: Dict[str, str], 
                                          regressions: List[Dict[str, Any]], 
                                          improvements: List[str], 
                                          degradations: List[str]) -> List[str]:
        """Generate recommendations based on evolution analysis"""
        recommendations = []
        
        # Trend-based recommendations
        if trends.get("quality") == "declining":
            recommendations.append("Focus on improving content quality through better training data or model tuning")
        
        if trends.get("consistency") == "declining":
            recommendations.append("Implement consistency checks and validation in the model pipeline")
        
        # Regression-based recommendations
        if regressions:
            recommendations.append("Investigate recent regressions and consider rolling back problematic versions")
        
        # Improvement-based recommendations
        if improvements:
            recommendations.append(f"Continue building on improvements in: {', '.join(improvements)}")
        
        # Degradation-based recommendations
        if degradations:
            recommendations.append(f"Address degradations in: {', '.join(degradations)}")
        
        # General recommendations
        if not recommendations:
            recommendations.append("Model evolution is stable. Continue monitoring for any changes")
        
        return recommendations

    def _calculate_evolution_confidence(self, versions: List[ModelVersion]) -> float:
        """Calculate confidence score for evolution analysis"""
        if len(versions) < 2:
            return 0.0
        
        # Base confidence on number of versions and data volume
        version_confidence = min(len(versions) / 5.0, 1.0)  # Max confidence at 5+ versions
        
        total_entries = sum(v.total_entries for v in versions)
        data_confidence = min(total_entries / 1000.0, 1.0)  # Max confidence at 1000+ entries
        
        # Combine confidences
        overall_confidence = (version_confidence + data_confidence) / 2
        return overall_confidence

    def _determine_regression_severity(self, magnitude: float) -> str:
        """Determine regression severity based on magnitude"""
        if magnitude > 0.5:
            return "critical"
        elif magnitude > 0.3:
            return "high"
        elif magnitude > 0.15:
            return "medium"
        else:
            return "low"

    def _get_quality_regression_recommendations(self, quality_change: float) -> List[str]:
        """Get recommendations for quality regression"""
        recommendations = [
            "Review recent training data for quality issues",
            "Check for data drift or distribution changes",
            "Consider retraining with higher quality data",
            "Implement quality validation in the pipeline"
        ]
        
        if abs(quality_change) > 0.3:
            recommendations.append("Consider rolling back to previous model version")
        
        return recommendations

    def _get_consistency_regression_recommendations(self, consistency_change: float) -> List[str]:
        """Get recommendations for consistency regression"""
        recommendations = [
            "Review model parameters for consistency",
            "Check for input data variations",
            "Implement consistency validation",
            "Consider ensemble methods for stability"
        ]
        
        if abs(consistency_change) > 0.3:
            recommendations.append("Investigate model architecture changes")
        
        return recommendations

    def _check_performance_metrics_regression(self, v1: ModelVersion, v2: ModelVersion) -> List[PerformanceRegression]:
        """Check for performance metrics regressions"""
        regressions = []
        
        for metric in v1.performance_metrics:
            if metric in v2.performance_metrics:
                v1_val = v1.performance_metrics[metric]
                v2_val = v2.performance_metrics[metric]
                
                if v1_val != 0:
                    change_percent = (v2_val - v1_val) / abs(v1_val)
                    
                    # Check for significant regression
                    if change_percent < -self.regression_thresholds["performance_regression"]:
                        severity = self._determine_regression_severity(abs(change_percent))
                        regressions.append(PerformanceRegression(
                            regression_type=RegressionType.PERFORMANCE_REGRESSION,
                            severity=severity,
                            affected_metrics=[metric],
                            regression_magnitude=abs(change_percent),
                            confidence=min(abs(change_percent) * 2, 1.0),
                            affected_versions=[v1.version, v2.version],
                            recommendations=[f"Investigate regression in {metric}"],
                            detection_timestamp=datetime.now()
                        ))
        
        return regressions

    def _generate_version_comparison_insights(self, v1: ModelVersion, v2: ModelVersion, 
                                           performance_comparison: Dict[str, Any]) -> List[str]:
        """Generate insights for version comparison"""
        insights = []
        
        # Quality insights
        quality_diff = v2.quality_score - v1.quality_score
        if abs(quality_diff) > 0.05:
            if quality_diff > 0:
                insights.append(f"Quality improved by {quality_diff:.3f} points")
            else:
                insights.append(f"Quality declined by {abs(quality_diff):.3f} points")
        
        # Consistency insights
        consistency_diff = v2.consistency_score - v1.consistency_score
        if abs(consistency_diff) > 0.05:
            if consistency_diff > 0:
                insights.append(f"Consistency improved by {consistency_diff:.3f} points")
            else:
                insights.append(f"Consistency declined by {abs(consistency_diff):.3f} points")
        
        # Performance metrics insights
        significant_changes = [metric for metric, data in performance_comparison.items() 
                             if abs(data["change_percent"]) > 10]
        
        if significant_changes:
            insights.append(f"Significant changes in: {', '.join(significant_changes)}")
        
        return insights

    def _calculate_r_squared(self, y_values: List[float], x_values: np.ndarray, 
                           slope: float, intercept: float) -> float:
        """Calculate R-squared for regression quality"""
        try:
            y_pred = slope * x_values + intercept
            y_mean = np.mean(y_values)
            
            ss_res = np.sum((y_values - y_pred) ** 2)
            ss_tot = np.sum((y_values - y_mean) ** 2)
            
            if ss_tot == 0:
                return 0.0
            
            r_squared = 1 - (ss_res / ss_tot)
            return max(0, min(1, r_squared))  # Clamp to [0,1]
            
        except:
            return 0.0

    def get_model_version_summary(self) -> Dict[str, Any]:
        """Get summary of all tracked model versions"""
        if not self.model_versions:
            return {"message": "No model versions tracked"}
        
        versions = list(self.model_versions.values())
        versions.sort(key=lambda v: v.release_date)
        
        return {
            "total_versions": len(versions),
            "latest_version": versions[-1].version if versions else None,
            "earliest_version": versions[0].version if versions else None,
            "version_range": {
                "start_date": versions[0].release_date.isoformat() if versions else None,
                "end_date": versions[-1].release_date.isoformat() if versions else None
            },
            "quality_range": {
                "min": min(v.quality_score for v in versions),
                "max": max(v.quality_score for v in versions),
                "avg": statistics.mean(v.quality_score for v in versions)
            },
            "consistency_range": {
                "min": min(v.consistency_score for v in versions),
                "max": max(v.consistency_score for v in versions),
                "avg": statistics.mean(v.consistency_score for v in versions)
            },
            "total_entries": sum(v.total_entries for v in versions)
        }


# Global evolution tracker instance
evolution_tracker = None

def initialize_evolution_tracker(analyzer: AIHistoryAnalyzer):
    """Initialize global evolution tracker"""
    global evolution_tracker
    evolution_tracker = AIEvolutionTracker(analyzer)
    logger.info("Evolution tracker initialized")

def track_model_version(version: str, release_date: datetime, description: str = "") -> ModelVersion:
    """Track a new model version"""
    if evolution_tracker:
        return evolution_tracker.track_model_version(version, release_date, description)
    else:
        raise RuntimeError("Evolution tracker not initialized")

def analyze_evolution(time_window: Optional[timedelta] = None) -> EvolutionAnalysis:
    """Analyze AI model evolution"""
    if evolution_tracker:
        return evolution_tracker.analyze_evolution(time_window)
    else:
        raise RuntimeError("Evolution tracker not initialized")

def detect_performance_regression(version_1: str, version_2: str) -> List[PerformanceRegression]:
    """Detect performance regressions between versions"""
    if evolution_tracker:
        return evolution_tracker.detect_performance_regression(version_1, version_2)
    else:
        raise RuntimeError("Evolution tracker not initialized")



























