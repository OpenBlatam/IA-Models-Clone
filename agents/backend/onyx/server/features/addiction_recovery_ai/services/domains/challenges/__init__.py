"""
Challenges domain services
"""

from services.domains import register_service

try:
    from services.challenge_service import ChallengeService
    
    def register_services():
        register_service("challenges", "challenge", ChallengeService)
except ImportError:
    pass



