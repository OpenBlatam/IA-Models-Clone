"""
Assessment domain services
"""

from services.domains import register_service

try:
    from services.analytics_service import AnalyticsService
    from services.assessment_service import AssessmentService
    from services.sentiment_service import SentimentService
    
    def register_services():
        register_service("assessment", "analytics", AnalyticsService)
        register_service("assessment", "assessment", AssessmentService)
        register_service("assessment", "sentiment", SentimentService)
except ImportError:
    pass



