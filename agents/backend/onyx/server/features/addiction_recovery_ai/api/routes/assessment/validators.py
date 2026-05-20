"""
Validation functions for assessment endpoints
"""

try:
    from schemas.assessment import AssessmentRequest, UpdateProfileRequest
    from utils.validators import AddictionTypeValidator
    from utils.errors import ValidationError
except ImportError:
    from ...schemas.assessment import AssessmentRequest, UpdateProfileRequest
    from ...utils.validators import AddictionTypeValidator
    from ...utils.errors import ValidationError


async def validate_assessment_request(request: AssessmentRequest) -> None:
    """Validate assessment request"""
    if not AddictionTypeValidator.validate_type(request.addiction_type):
        raise ValidationError(
            f"Tipo de adicción no válido. Tipos válidos: {AddictionTypeValidator.get_valid_types()}"
        )


async def validate_profile_update(request: UpdateProfileRequest) -> None:
    """Validate profile update request"""
    has_updates = any([
        request.email, request.name, request.addiction_type, request.additional_info
    ])
    
    if not has_updates:
        raise ValidationError("Al menos un campo debe ser proporcionado para actualizar")

