"""
Recovery domain schemas
"""

from schemas.domains import register_schema

try:
    from schemas.recovery_plan import (
        CreateRecoveryPlanRequest,
        RecoveryPlanResponse,
        UpdateRecoveryPlanRequest
    )
    
    def register_schemas():
        register_schema("recovery", "CreateRecoveryPlanRequest", CreateRecoveryPlanRequest)
        register_schema("recovery", "RecoveryPlanResponse", RecoveryPlanResponse)
        register_schema("recovery", "UpdateRecoveryPlanRequest", UpdateRecoveryPlanRequest)
except ImportError:
    pass



