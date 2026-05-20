"""
Dependency injection for services
Using FastAPI's dependency injection system
"""

from functools import lru_cache
from typing import Annotated

try:
    from core.addiction_analyzer import AddictionAnalyzer
    from core.recovery_planner import RecoveryPlanner
    from core.progress_tracker import ProgressTracker
    from core.relapse_prevention import RelapsePrevention
    from services.counseling_service import CounselingService
    from services.motivation_service import MotivationService
    from services.notification_service import NotificationService
    from services.analytics_service import AnalyticsService
    from services.gamification_service import GamificationService
    from services.emergency_service import EmergencyService
except ImportError:
    from .core.addiction_analyzer import AddictionAnalyzer
    from .core.recovery_planner import RecoveryPlanner
    from .core.progress_tracker import ProgressTracker
    from .core.relapse_prevention import RelapsePrevention
    from .services.counseling_service import CounselingService
    from .services.motivation_service import MotivationService
    from .services.notification_service import NotificationService
    from .services.analytics_service import AnalyticsService
    from .services.gamification_service import GamificationService
    from .services.emergency_service import EmergencyService


# Core services
@lru_cache()
def get_addiction_analyzer() -> AddictionAnalyzer:
    """Get AddictionAnalyzer instance"""
    return AddictionAnalyzer()


@lru_cache()
def get_recovery_planner() -> RecoveryPlanner:
    """Get RecoveryPlanner instance"""
    return RecoveryPlanner()


@lru_cache()
def get_progress_tracker() -> ProgressTracker:
    """Get ProgressTracker instance"""
    return ProgressTracker()


@lru_cache()
def get_relapse_prevention() -> RelapsePrevention:
    """Get RelapsePrevention instance"""
    return RelapsePrevention()


# Support services
@lru_cache()
def get_counseling_service() -> CounselingService:
    """Get CounselingService instance"""
    return CounselingService()


@lru_cache()
def get_motivation_service() -> MotivationService:
    """Get MotivationService instance"""
    return MotivationService()


@lru_cache()
def get_notification_service() -> NotificationService:
    """Get NotificationService instance"""
    return NotificationService()


@lru_cache()
def get_analytics_service() -> AnalyticsService:
    """Get AnalyticsService instance"""
    return AnalyticsService()


@lru_cache()
def get_gamification_service() -> GamificationService:
    """Get GamificationService instance"""
    return GamificationService()


@lru_cache()
def get_emergency_service() -> EmergencyService:
    """Get EmergencyService instance"""
    return EmergencyService()


# Type aliases for dependency injection
AddictionAnalyzerDep = Annotated[AddictionAnalyzer, get_addiction_analyzer]
RecoveryPlannerDep = Annotated[RecoveryPlanner, get_recovery_planner]
ProgressTrackerDep = Annotated[ProgressTracker, get_progress_tracker]
RelapsePreventionDep = Annotated[RelapsePrevention, get_relapse_prevention]
CounselingServiceDep = Annotated[CounselingService, get_counseling_service]
MotivationServiceDep = Annotated[MotivationService, get_motivation_service]
NotificationServiceDep = Annotated[NotificationService, get_notification_service]
AnalyticsServiceDep = Annotated[AnalyticsService, get_analytics_service]
GamificationServiceDep = Annotated[GamificationService, get_gamification_service]
EmergencyServiceDep = Annotated[EmergencyService, get_emergency_service]

