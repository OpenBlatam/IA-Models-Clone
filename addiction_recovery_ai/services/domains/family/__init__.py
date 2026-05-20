"""
Family domain services
"""

from services.domains import register_service

try:
    from services.family_tracking_service import FamilyTrackingService
    
    def register_services():
        register_service("family", "tracking", FamilyTrackingService)
except ImportError:
    pass



