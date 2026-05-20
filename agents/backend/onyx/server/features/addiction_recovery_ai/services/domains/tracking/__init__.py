"""
Tracking domain services
"""

from services.domains import register_service

try:
    from services.health_tracking_service import HealthTrackingService
    from services.habit_tracking_service import HabitTrackingService
    from services.progress_tracker_service import ProgressTrackerService
    from services.withdrawal_tracking_service import WithdrawalTrackingService
    from services.symptom_tracking_service import SymptomTrackingService
    from services.location_tracking_service import LocationTrackingService
    
    def register_services():
        register_service("tracking", "health", HealthTrackingService)
        register_service("tracking", "habit", HabitTrackingService)
        register_service("tracking", "progress", ProgressTrackerService)
        register_service("tracking", "withdrawal", WithdrawalTrackingService)
        register_service("tracking", "symptom", SymptomTrackingService)
        register_service("tracking", "location", LocationTrackingService)
except ImportError:
    pass



