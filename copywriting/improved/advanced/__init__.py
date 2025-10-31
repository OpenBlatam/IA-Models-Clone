"""
Advanced Features Module
=======================

Advanced AI engine, analytics, and content optimization capabilities.
"""

from .ai_engine import (
    AIEngine,
    AIProvider,
    AIProviderConfig,
    OpenAIEngine,
    AnthropicEngine,
    GoogleEngine,
    AIEngineManager,
    ai_engine_manager
)

from .analytics import (
    AnalyticsEngine,
    MetricType,
    TimeRange,
    MetricData,
    PerformanceMetrics,
    QualityMetrics,
    UsageMetrics,
    EngagementMetrics,
    TrendData,
    analytics_engine
)

from .content_optimizer import (
    ContentOptimizer,
    OptimizationStrategy,
    OptimizationGoal,
    OptimizationResult,
    ABTestResult,
    ContentInsight,
    content_optimizer
)

__all__ = [
    # AI Engine
    "AIEngine",
    "AIProvider", 
    "AIProviderConfig",
    "OpenAIEngine",
    "AnthropicEngine",
    "GoogleEngine",
    "AIEngineManager",
    "ai_engine_manager",
    
    # Analytics
    "AnalyticsEngine",
    "MetricType",
    "TimeRange", 
    "MetricData",
    "PerformanceMetrics",
    "QualityMetrics",
    "UsageMetrics",
    "EngagementMetrics",
    "TrendData",
    "analytics_engine",
    
    # Content Optimizer
    "ContentOptimizer",
    "OptimizationStrategy",
    "OptimizationGoal", 
    "OptimizationResult",
    "ABTestResult",
    "ContentInsight",
    "content_optimizer"
]






























