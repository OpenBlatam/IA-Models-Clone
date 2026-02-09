"""
Recovery domain services
"""

from services.domains import register_service

try:
    from services.recovery_planner_service import RecoveryPlannerService
    from services.progress_tracker_service import ProgressTrackerService
    from services.relapse_prevention_service import RelapsePreventionService
    
    def register_services():
        register_service("recovery", "planner", RecoveryPlannerService)
        register_service("recovery", "progress", ProgressTrackerService)
        register_service("recovery", "relapse_prevention", RelapsePreventionService)
except ImportError:
    pass



