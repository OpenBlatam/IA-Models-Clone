"""
Advanced Pydantic Schemas for Email Sequence API

This module contains advanced schemas for AI enhancements, analytics,
and sophisticated email sequence features.
"""

from typing import Any, List, Dict, Optional, Union, Tuple
from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, Field, validator
from enum import Enum


# AI Enhancement Schemas
class ContentOptimizationRequest(BaseModel):
    """Request schema for content optimization"""
    subject: str = Field(..., min_length=1, max_length=255)
    content: str = Field(..., min_length=1)
    target_audience: str = Field(..., min_length=1, max_length=500)
    goal: str = Field(default="engage", regex="^(engage|convert|inform|nurture)$")
    optimization_focus: Optional[List[str]] = Field(
        default=["subject_line", "content_clarity", "cta_effectiveness"],
        description="Areas to focus optimization on"
    )


class ContentOptimizationResponse(BaseModel):
    """Response schema for content optimization"""
    original_content: str
    optimized_content: str
    improvements: List[str]
    confidence_score: float = Field(ge=0.0, le=1.0)
    estimated_improvement: float = Field(ge=0.0, le=100.0)
    optimization_details: Dict[str, Any] = Field(default_factory=dict)


class SentimentAnalysisRequest(BaseModel):
    """Request schema for sentiment analysis"""
    subject: str = Field(..., min_length=1, max_length=255)
    content: str = Field(..., min_length=1)
    analysis_depth: str = Field(default="standard", regex="^(basic|standard|detailed)$")


class SentimentAnalysisResponse(BaseModel):
    """Response schema for sentiment analysis"""
    sentiment: str = Field(regex="^(positive|negative|neutral|mixed)$")
    confidence: float = Field(ge=0.0, le=1.0)
    emotions: Dict[str, float] = Field(description="Detected emotions with confidence scores")
    tone_suggestions: List[str] = Field(description="Suggestions for tone improvement")
    sentiment_breakdown: Dict[str, Any] = Field(default_factory=dict)


class PersonalizationRequest(BaseModel):
    """Request schema for content personalization"""
    template_content: str = Field(..., min_length=1)
    subscriber_data: Dict[str, Any] = Field(..., description="Subscriber information")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)
    personalization_level: str = Field(default="moderate", regex="^(light|moderate|deep)$")


class PersonalizationResponse(BaseModel):
    """Response schema for content personalization"""
    personalized_content: str
    original_content: str
    personalization_applied: bool
    personalization_techniques: List[str]
    confidence_score: float = Field(ge=0.0, le=1.0)


class SendTimePredictionRequest(BaseModel):
    """Request schema for send time prediction"""
    subscriber_data: Dict[str, Any] = Field(..., description="Subscriber information")
    sequence_data: Dict[str, Any] = Field(..., description="Sequence information")
    prediction_horizon: int = Field(default=7, ge=1, le=30, description="Days to predict ahead")


class SendTimePredictionResponse(BaseModel):
    """Response schema for send time prediction"""
    optimal_day: str
    optimal_hour: int = Field(ge=0, le=23)
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    alternative_times: List[Dict[str, Any]] = Field(default_factory=list)


# Advanced Analytics Schemas
class CohortAnalysisRequest(BaseModel):
    """Request schema for cohort analysis"""
    sequence_id: UUID
    cohort_type: str = Field(regex="^(acquisition|behavioral|retention|revenue)$")
    period_days: int = Field(default=30, ge=7, le=365)
    granularity: str = Field(default="weekly", regex="^(daily|weekly|monthly)$")


class CohortAnalysisResponse(BaseModel):
    """Response schema for cohort analysis"""
    cohort_period: str
    cohort_size: int
    retention_rates: Dict[str, float]
    revenue_per_cohort: float
    lifetime_value: float
    cohort_insights: List[str] = Field(default_factory=list)


class AdvancedSegmentationRequest(BaseModel):
    """Request schema for advanced segmentation"""
    sequence_id: UUID
    segment_type: str = Field(regex="^(rfm|behavioral|predictive|lifecycle)$")
    min_segment_size: int = Field(default=10, ge=1)
    max_segments: int = Field(default=10, ge=1, le=50)


class AdvancedSegmentationResponse(BaseModel):
    """Response schema for advanced segmentation"""
    segment_name: str
    segment_type: str
    size: int
    characteristics: Dict[str, Any]
    performance_metrics: Dict[str, float]
    recommendations: List[str]
    segment_health: str = Field(regex="^(healthy|at_risk|inactive)$")


class PredictiveModelRequest(BaseModel):
    """Request schema for predictive modeling"""
    sequence_id: UUID
    target_metric: str = Field(regex="^(open_rate|click_rate|conversion_rate|churn_probability)$")
    model_type: str = Field(default="auto", regex="^(auto|random_forest|neural_network|logistic_regression)$")
    training_period_days: int = Field(default=90, ge=30, le=365)


class PredictiveModelResponse(BaseModel):
    """Response schema for predictive modeling"""
    model_type: str
    accuracy: float = Field(ge=0.0, le=1.0)
    features: List[str]
    predictions: Dict[str, Any]
    confidence_intervals: Dict[str, Tuple[float, float]]
    model_insights: List[str] = Field(default_factory=list)


class LifetimeValueRequest(BaseModel):
    """Request schema for lifetime value calculation"""
    sequence_id: UUID
    subscriber_id: Optional[UUID] = None
    calculation_method: str = Field(default="standard", regex="^(standard|advanced|predictive)$")
    time_horizon_days: int = Field(default=365, ge=30, le=1095)


class LifetimeValueResponse(BaseModel):
    """Response schema for lifetime value calculation"""
    average_ltv: float
    median_ltv: float
    ltv_by_segment: Dict[str, float]
    ltv_trend: str = Field(regex="^(increasing|decreasing|stable)$")
    predicted_ltv: float
    ltv_insights: List[str] = Field(default_factory=list)


class AttributionAnalysisRequest(BaseModel):
    """Request schema for attribution analysis"""
    sequence_id: UUID
    conversion_event: str = Field(default="purchase")
    attribution_model: str = Field(default="multi_touch", regex="^(first_touch|last_touch|multi_touch|time_decay)$")
    lookback_days: int = Field(default=30, ge=1, le=90)


class AttributionAnalysisResponse(BaseModel):
    """Response schema for attribution analysis"""
    attribution_model: str
    touchpoint_attribution: Dict[str, float]
    sequence_attribution: Dict[str, float]
    conversion_paths: List[Dict[str, Any]]
    attribution_insights: List[str] = Field(default_factory=list)


class InsightsReportRequest(BaseModel):
    """Request schema for insights report generation"""
    sequence_id: UUID
    report_type: str = Field(default="comprehensive", regex="^(comprehensive|performance|engagement|conversion)$")
    include_recommendations: bool = Field(default=True)
    include_benchmarks: bool = Field(default=True)


class InsightsReportResponse(BaseModel):
    """Response schema for insights report generation"""
    executive_summary: Dict[str, Any]
    key_insights: List[str]
    recommendations: List[str]
    performance_metrics: Dict[str, float]
    benchmark_comparison: Dict[str, Any] = Field(default_factory=dict)
    trend_analysis: Dict[str, Any] = Field(default_factory=dict)
    generated_at: datetime = Field(default_factory=datetime.utcnow)


# Advanced Sequence Features Schemas
class AIOptimizationRequest(BaseModel):
    """Request schema for AI sequence optimization"""
    sequence_id: UUID
    optimization_goals: List[str] = Field(
        default=["engagement", "conversion"],
        description="Goals for optimization"
    )
    optimization_level: str = Field(default="moderate", regex="^(light|moderate|aggressive)$")
    preserve_brand_voice: bool = Field(default=True)


class AIOptimizationResponse(BaseModel):
    """Response schema for AI sequence optimization"""
    sequence_id: UUID
    optimization_completed: bool
    optimized_steps: List[Dict[str, Any]]
    total_improvements: int
    average_confidence: float = Field(ge=0.0, le=1.0)
    optimization_summary: Dict[str, Any] = Field(default_factory=dict)


class SmartSchedulingRequest(BaseModel):
    """Request schema for smart scheduling"""
    sequence_id: UUID
    subscriber_segments: List[str] = Field(default_factory=list)
    scheduling_strategy: str = Field(default="optimal", regex="^(optimal|balanced|aggressive)$")
    timezone_consideration: bool = Field(default=True)


class SmartSchedulingResponse(BaseModel):
    """Response schema for smart scheduling"""
    sequence_id: UUID
    scheduling_analysis_completed: bool
    recommendations: List[Dict[str, Any]]
    implementation_notes: List[str]
    expected_improvement: float = Field(ge=0.0, le=100.0)


# Competitor Analysis Schemas
class CompetitorAnalysisRequest(BaseModel):
    """Request schema for competitor analysis"""
    competitor_emails: List[Dict[str, str]] = Field(..., description="Competitor email data")
    analysis_focus: List[str] = Field(
        default=["subject_lines", "content_themes", "cta_patterns"],
        description="Areas to focus analysis on"
    )


class CompetitorAnalysisResponse(BaseModel):
    """Response schema for competitor analysis"""
    common_patterns: List[str]
    content_themes: List[str]
    subject_line_strategies: List[str]
    cta_patterns: List[str]
    opportunities: List[str]
    recommendations: List[str]
    competitive_insights: Dict[str, Any] = Field(default_factory=dict)


# A/B Testing Schemas
class ABTestRequest(BaseModel):
    """Request schema for A/B testing"""
    sequence_id: UUID
    test_type: str = Field(regex="^(subject_line|content|send_time|frequency)$")
    variants: List[Dict[str, Any]] = Field(..., min_items=2, max_items=5)
    test_duration_days: int = Field(default=14, ge=7, le=30)
    success_metric: str = Field(default="open_rate", regex="^(open_rate|click_rate|conversion_rate)$")


class ABTestResponse(BaseModel):
    """Response schema for A/B testing"""
    test_id: UUID
    test_status: str = Field(regex="^(running|completed|paused)$")
    variants: List[Dict[str, Any]]
    current_winner: Optional[str] = None
    confidence_level: float = Field(ge=0.0, le=1.0)
    test_results: Dict[str, Any] = Field(default_factory=dict)


# Performance Optimization Schemas
class PerformanceOptimizationRequest(BaseModel):
    """Request schema for performance optimization"""
    sequence_id: UUID
    optimization_areas: List[str] = Field(
        default=["delivery", "engagement", "conversion"],
        description="Areas to optimize"
    )
    target_improvement: float = Field(default=20.0, ge=5.0, le=100.0)


class PerformanceOptimizationResponse(BaseModel):
    """Response schema for performance optimization"""
    sequence_id: UUID
    optimization_completed: bool
    improvements_applied: List[str]
    expected_improvement: float
    optimization_details: Dict[str, Any]
    monitoring_recommendations: List[str] = Field(default_factory=list)






























