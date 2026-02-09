"""
Integration domain services
"""

from services.domains import register_service

try:
    from services.health_integration_service import HealthIntegrationService
    from services.iot_integration_service import IoTIntegrationService
    from services.ehr_integration_service import EHRIntegrationService
    from services.telemedicine_integration_service import TelemedicineIntegrationService
    from services.third_party_integration_service import ThirdPartyIntegrationService
    from services.blockchain_integration_service import BlockchainIntegrationService
    
    def register_services():
        register_service("integrations", "health", HealthIntegrationService)
        register_service("integrations", "iot", IoTIntegrationService)
        register_service("integrations", "ehr", EHRIntegrationService)
        register_service("integrations", "telemedicine", TelemedicineIntegrationService)
        register_service("integrations", "third_party", ThirdPartyIntegrationService)
        register_service("integrations", "blockchain", BlockchainIntegrationService)
except ImportError:
    pass



