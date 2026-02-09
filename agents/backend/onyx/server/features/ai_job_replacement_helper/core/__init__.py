"""
Core modules for AI Job Replacement Helper
"""

from .gamification import GamificationService
from .steps_guide import StepsGuideService
from .linkedin_integration import LinkedInIntegrationService
from .recommendations import RecommendationService
from .notifications import NotificationsService
from .mentoring import MentoringService
from .cv_analyzer import CVAnalyzerService
from .interview_simulator import InterviewSimulatorService
from .challenges import ChallengesService
from .analytics import AnalyticsService
from .community import CommunityService
from .job_platforms import JobPlatformsService
from .application_tracker import ApplicationTrackerService
from .auth import AuthService
from .messaging import MessagingService
from .events import EventsService
from .resources import ResourcesService
from .reports import ReportsService
from .cache import CacheService, cache_service
from .templates import TemplatesService

__all__ = [
    "GamificationService",
    "StepsGuideService",
    "LinkedInIntegrationService",
    "RecommendationService",
    "NotificationsService",
    "MentoringService",
    "CVAnalyzerService",
    "InterviewSimulatorService",
    "ChallengesService",
    "AnalyticsService",
    "CommunityService",
    "JobPlatformsService",
    "ApplicationTrackerService",
    "AuthService",
    "MessagingService",
    "EventsService",
    "ResourcesService",
    "ReportsService",
    "CacheService",
    "cache_service",
    "TemplatesService",
]

