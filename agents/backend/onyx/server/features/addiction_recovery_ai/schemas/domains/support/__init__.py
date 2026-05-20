"""
Support domain schemas
"""

from schemas.domains import register_schema

try:
    from schemas.support import (
        CoachingSessionRequest,
        CoachingSessionResponse,
        MotivationResponse,
        MilestoneRequest,
        MilestoneResponse,
        AchievementsResponse
    )
    
    def register_schemas():
        register_schema("support", "CoachingSessionRequest", CoachingSessionRequest)
        register_schema("support", "CoachingSessionResponse", CoachingSessionResponse)
        register_schema("support", "MotivationResponse", MotivationResponse)
        register_schema("support", "MilestoneRequest", MilestoneRequest)
        register_schema("support", "MilestoneResponse", MilestoneResponse)
        register_schema("support", "AchievementsResponse", AchievementsResponse)
except ImportError:
    pass



