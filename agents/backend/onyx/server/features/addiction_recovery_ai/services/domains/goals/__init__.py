"""
Goals domain services
"""

from services.domains import register_service

try:
    from services.goals_service import GoalsService
    from services.advanced_goal_tracking_service import AdvancedGoalTrackingService
    from services.long_term_goals_service import LongTermGoalsService
    
    def register_services():
        register_service("goals", "goals", GoalsService)
        register_service("goals", "advanced_tracking", AdvancedGoalTrackingService)
        register_service("goals", "long_term", LongTermGoalsService)
except ImportError:
    pass



