"""
Support domain services
"""

from services.domains import register_service

try:
    from services.counseling_service import CounselingService
    from services.motivation_service import MotivationService
    from services.community_service import CommunityService
    from services.mentorship_service import MentorshipService
    
    def register_services():
        register_service("support", "counseling", CounselingService)
        register_service("support", "motivation", MotivationService)
        register_service("support", "community", CommunityService)
        register_service("support", "mentorship", MentorshipService)
except ImportError:
    pass



