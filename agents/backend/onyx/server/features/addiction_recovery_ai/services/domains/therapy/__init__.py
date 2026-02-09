"""
Therapy domain services
"""

from services.domains import register_service

try:
    from services.virtual_therapy_service import VirtualTherapyService
    from services.vr_ar_therapy_service import VRARTherapyService
    from services.group_therapy_integration_service import GroupTherapyIntegrationService
    from services.alternative_therapy_integration_service import AlternativeTherapyIntegrationService
    from services.realtime_coaching_service import RealtimeCoachingService
    
    def register_services():
        register_service("therapy", "virtual", VirtualTherapyService)
        register_service("therapy", "vr_ar", VRARTherapyService)
        register_service("therapy", "group", GroupTherapyIntegrationService)
        register_service("therapy", "alternative", AlternativeTherapyIntegrationService)
        register_service("therapy", "realtime_coaching", RealtimeCoachingService)
except ImportError:
    pass



