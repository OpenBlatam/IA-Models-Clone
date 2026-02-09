"""
Data transformers for progress endpoints
Pure functions for transforming request/response data
"""

from typing import Dict, Any, List
from schemas.progress import LogEntryRequest, LogEntryResponse
from utils.transformers import transform_dict, pick_fields


def transform_log_entry_request_to_dict(
    request: LogEntryRequest
) -> Dict[str, Any]:
    """
    Transform LogEntryRequest to dictionary (RORO pattern)
    
    Args:
        request: LogEntryRequest object
    
    Returns:
        Dictionary with log entry data
    """
    # Guard clause
    if not request:
        raise ValueError("request cannot be None")
    
    # Return object (RORO pattern)
    return request.model_dump()


def transform_entry_to_response(
    entry: Dict[str, Any]
) -> LogEntryResponse:
    """
    Transform entry dictionary to LogEntryResponse (RORO pattern)
    
    Args:
        entry: Entry dictionary
    
    Returns:
        LogEntryResponse object
    """
    # Guard clause
    if not entry:
        raise ValueError("entry cannot be empty")
    
    # Return object (RORO pattern)
    return LogEntryResponse(
        entry_id=entry.get("entry_id", ""),
        user_id=entry.get("user_id", ""),
        date=entry.get("date", ""),
        mood=entry.get("mood", ""),
        cravings_level=entry.get("cravings_level", 0),
        triggers_encountered=entry.get("triggers_encountered", []),
        consumed=entry.get("consumed", False),
        notes=entry.get("notes")
    )


def extract_progress_summary(
    progress: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Extract summary from progress (RORO pattern)
    
    Args:
        progress: Full progress dictionary
    
    Returns:
        Summary dictionary
    """
    # Guard clause
    if not progress:
        raise ValueError("progress cannot be empty")
    
    # Pick only summary fields
    summary_fields = [
        "user_id",
        "days_sober",
        "streak_days",
        "progress_percentage"
    ]
    
    # Return object (RORO pattern)
    return pick_fields(progress, summary_fields)

