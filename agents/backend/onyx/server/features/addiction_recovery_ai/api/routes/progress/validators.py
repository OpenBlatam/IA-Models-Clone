"""
Validation functions for progress endpoints
"""

try:
    from schemas.progress import LogEntryRequest
    from utils.validators import (
        validate_user_id,
        validate_date_string,
        validate_date_not_future
    )
    from utils.errors import ValidationError
except ImportError:
    from ...schemas.progress import LogEntryRequest
    from ...utils.validators import (
        validate_user_id,
        validate_date_string,
        validate_date_not_future
    )
    from ...utils.errors import ValidationError


async def validate_log_entry_request(request: LogEntryRequest) -> None:
    """Validate log entry request"""
    validate_user_id(request.user_id)
    
    entry_date = validate_date_string(request.date, "date")
    validate_date_not_future(entry_date, "date")

