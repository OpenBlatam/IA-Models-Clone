"""
Emergency domain schemas
"""

from schemas.domains import register_schema

try:
    from schemas.emergency import (
        CreateEmergencyContactRequest,
        EmergencyContactResponse,
        EmergencyContactsListResponse,
        TriggerEmergencyRequest,
        EmergencyProtocolResponse,
        CrisisResourceResponse,
        CrisisResourcesResponse
    )
    
    def register_schemas():
        register_schema("emergency", "CreateEmergencyContactRequest", CreateEmergencyContactRequest)
        register_schema("emergency", "EmergencyContactResponse", EmergencyContactResponse)
        register_schema("emergency", "EmergencyContactsListResponse", EmergencyContactsListResponse)
        register_schema("emergency", "TriggerEmergencyRequest", TriggerEmergencyRequest)
        register_schema("emergency", "EmergencyProtocolResponse", EmergencyProtocolResponse)
        register_schema("emergency", "CrisisResourceResponse", CrisisResourceResponse)
        register_schema("emergency", "CrisisResourcesResponse", CrisisResourcesResponse)
except ImportError:
    pass



