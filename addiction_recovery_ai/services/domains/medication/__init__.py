"""
Medication domain services
"""

from services.domains import register_service

try:
    from services.medication_service import MedicationService
    from services.advanced_medication_service import AdvancedMedicationService
    from services.advanced_medication_tracking_service import AdvancedMedicationTrackingService
    
    def register_services():
        register_service("medication", "medication", MedicationService)
        register_service("medication", "advanced", AdvancedMedicationService)
        register_service("medication", "advanced_tracking", AdvancedMedicationTrackingService)
except ImportError:
    pass



