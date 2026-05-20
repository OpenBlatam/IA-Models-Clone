"""
Wearables domain services
"""

from services.domains import register_service

try:
    from services.wearable_service import WearableService
    from services.health_monitoring_device_service import HealthMonitoringDeviceService
    from services.advanced_health_device_integration_service import AdvancedHealthDeviceIntegrationService
    from services.medical_device_integration_service import MedicalDeviceIntegrationService
    
    def register_services():
        register_service("wearables", "wearable", WearableService)
        register_service("wearables", "health_monitoring", HealthMonitoringDeviceService)
        register_service("wearables", "advanced_health", AdvancedHealthDeviceIntegrationService)
        register_service("wearables", "medical", MedicalDeviceIntegrationService)
except ImportError:
    pass



