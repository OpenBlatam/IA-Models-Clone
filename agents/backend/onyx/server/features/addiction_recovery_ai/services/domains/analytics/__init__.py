"""
Analytics domain services
"""

from services.domains import register_service

try:
    from services.analytics_service import AnalyticsService
    from services.advanced_data_analysis_service import AdvancedDataAnalysisService
    from services.advanced_metrics_service import AdvancedMetricsService
    from services.behavioral_analysis_service import BehavioralAnalysisService
    from services.temporal_pattern_analysis_service import TemporalPatternAnalysisService
    
    def register_services():
        register_service("analytics", "analytics", AnalyticsService)
        register_service("analytics", "advanced_data", AdvancedDataAnalysisService)
        register_service("analytics", "advanced_metrics", AdvancedMetricsService)
        register_service("analytics", "behavioral", BehavioralAnalysisService)
        register_service("analytics", "temporal_patterns", TemporalPatternAnalysisService)
except ImportError:
    pass



