"""
Services Module
===============

Módulo de servicios del sistema.
"""

from .manual.manual_service import ManualService
from .analytics.analytics_service import AnalyticsService
from .cache.cache_service import CacheService
from .notification.notification_service import NotificationService
from .rating.rating_service import RatingService
from .recommendation.recommendation_service import RecommendationService
from .semantic_search_service import SemanticSearchService
from .share.share_service import ShareService
from .template.template_service import TemplateService

from .manual_service import ManualService as ManualServiceLegacy
from .analytics_service import AnalyticsService as AnalyticsServiceLegacy
from .cache_service import CacheService as CacheServiceLegacy
from .notification_service import NotificationService as NotificationServiceLegacy
from .rating_service import RatingService as RatingServiceLegacy
from .recommendation_service import RecommendationService as RecommendationServiceLegacy
from .share_service import ShareService as ShareServiceLegacy
from .template_service import TemplateService as TemplateServiceLegacy

__all__ = [
    "ManualService",
    "AnalyticsService",
    "CacheService",
    "NotificationService",
    "RatingService",
    "RecommendationService",
    "SemanticSearchService",
    "ShareService",
    "TemplateService",
    "ManualServiceLegacy",
    "AnalyticsServiceLegacy",
    "CacheServiceLegacy",
    "NotificationServiceLegacy",
    "RatingServiceLegacy",
    "RecommendationServiceLegacy",
    "ShareServiceLegacy",
    "TemplateServiceLegacy",
]
