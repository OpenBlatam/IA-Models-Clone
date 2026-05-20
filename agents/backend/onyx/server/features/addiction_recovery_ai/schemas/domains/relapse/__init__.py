"""
Relapse prevention domain schemas
"""

from schemas.domains import register_schema

try:
    from schemas.relapse import (
        RelapseRiskCheckRequest,
        RelapseRiskResponse,
        CopingStrategiesRequest,
        CopingStrategiesResponse,
        EmergencyPlanRequest,
        EmergencyPlanResponse
    )
    
    def register_schemas():
        register_schema("relapse", "RelapseRiskCheckRequest", RelapseRiskCheckRequest)
        register_schema("relapse", "RelapseRiskResponse", RelapseRiskResponse)
        register_schema("relapse", "CopingStrategiesRequest", CopingStrategiesRequest)
        register_schema("relapse", "CopingStrategiesResponse", CopingStrategiesResponse)
        register_schema("relapse", "EmergencyPlanRequest", EmergencyPlanRequest)
        register_schema("relapse", "EmergencyPlanResponse", EmergencyPlanResponse)
except ImportError:
    pass



