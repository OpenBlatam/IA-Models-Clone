"""
Validation functions for support endpoints
"""

try:
    from schemas.support import CoachingRequest, MotivationRequest
    from utils.validators import validate_user_id
    from utils.guards import guard_not_empty
    from utils.errors import ValidationError
except ImportError:
    from ...schemas.support import CoachingRequest, MotivationRequest
    from ...utils.validators import validate_user_id
    from ...utils.guards import guard_not_empty
    from ...utils.errors import ValidationError


async def validate_coaching_request(request: CoachingRequest) -> None:
    """Validate coaching request"""
    validate_user_id(request.user_id)
    guard_not_empty(request.topic, "topic")
    guard_not_empty(request.situation, "situation")


async def validate_motivation_request(request: MotivationRequest) -> None:
    """Validate motivation request"""
    validate_user_id(request.user_id)
    if request.days_sober < 0:
        raise ValidationError("days_sober cannot be negative")

