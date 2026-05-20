"""
Emergency domain services
"""

from services.domains import register_service

try:
    from services.emergency_service import EmergencyService
    from services.emergency_integration_service import EmergencyIntegrationService
    from services.emergency_services_integration_service import EmergencyServicesIntegrationService
    
    def register_services():
        register_service("emergency", "emergency", EmergencyService)
        register_service("emergency", "integration", EmergencyIntegrationService)
        register_service("emergency", "services_integration", EmergencyServicesIntegrationService)
except ImportError:
    pass



