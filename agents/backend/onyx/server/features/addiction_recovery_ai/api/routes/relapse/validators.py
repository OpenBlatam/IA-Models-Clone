"""
Validation functions for relapse endpoints
"""

try:
    from schemas.relapse import RelapseRiskRequest
    from utils.validators import validate_user_id
    from utils.guards import guard_in_range, guard_non_negative
    from utils.errors import ValidationError
except ImportError:
    from ...schemas.relapse import RelapseRiskRequest
    from ...utils.validators import validate_user_id
    from ...utils.guards import guard_in_range, guard_non_negative
    from ...utils.errors import ValidationError


async def validate_relapse_risk_request(request: RelapseRiskRequest) -> None:
    """Validate relapse risk request"""
    validate_user_id(request.user_id)
    guard_non_negative(request.days_sober, "days_sober")
    guard_in_range(request.stress_level, 0, 10, "stress_level")
    guard_in_range(request.support_level, 0, 10, "support_level")

