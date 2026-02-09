"""
Audit API for tracking system activities and changes.
Refactored to use BaseRouter for reduced duplication.
"""

from fastapi import Depends, Query
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta

from .base_router import BaseRouter
from ...utils.file_helpers import get_iso_timestamp, parse_iso_date, get_timestamp_id
from ...utils.data_helpers import filter_by_fields, filter_by_date_range, sort_by_field, paginate_items, count_by_key, get_most_common_key, increment_dict_value
import logging

logger = logging.getLogger(__name__)

# Create base router instance
base = BaseRouter(
    prefix="/api/audit",
    tags=["Audit"],
    require_authentication=True,
    require_rate_limit=False
)

router = base.router


class AuditLogEntry(BaseModel):
    """Audit log entry model."""
    action: str = Field(..., description="Action performed")
    resource: str = Field(..., description="Resource affected")
    resource_id: Optional[str] = Field(None, description="Resource ID")
    user_id: Optional[str] = Field(None, description="User who performed the action")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional details")
    ip_address: Optional[str] = Field(None, description="IP address")
    timestamp: str = Field(default_factory=get_iso_timestamp)


# In-memory audit log (would be database in production)
audit_logs: List[Dict[str, Any]] = []


def log_audit_event(
    action: str,
    resource: str,
    resource_id: Optional[str] = None,
    user_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    ip_address: Optional[str] = None
):
    """
    Internal function to log audit events.
    """
    entry = {
        "id": get_timestamp_id("audit_"),
        "action": action,
        "resource": resource,
        "resource_id": resource_id,
        "user_id": user_id,
        "details": details or {},
        "ip_address": ip_address,
        "timestamp": get_iso_timestamp()
    }
    audit_logs.append(entry)
    # Keep only last 10000 entries
    if len(audit_logs) > 10000:
        audit_logs.pop(0)
    logger.info(f"Audit: {action} on {resource} by {user_id}")


@router.post("/log")
@base.timed_endpoint("create_audit_log")
async def create_audit_log(
    entry: AuditLogEntry,
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """
    Create an audit log entry.
    """
    base.log_request("create_audit_log", action=entry.action, resource=entry.resource)
    
    log_entry = {
        "id": get_timestamp_id("audit_"),
        **entry.model_dump()
    }
    audit_logs.append(log_entry)
    
    return base.success(log_entry, message="Audit log entry created")


@router.get("/logs")
@base.timed_endpoint("get_audit_logs")
async def get_audit_logs(
    action: Optional[str] = Query(None, description="Filter by action"),
    resource: Optional[str] = Query(None, description="Filter by resource"),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    start_date: Optional[str] = Query(None, description="Start date (ISO format)"),
    end_date: Optional[str] = Query(None, description="End date (ISO format)"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """
    Get audit logs with optional filters.
    """
    base.log_request("get_audit_logs", action=action, resource=resource, user_id=user_id)
    
    # Apply field filters using helper
    filtered = filter_by_fields(
        audit_logs,
        {
            "action": action,
            "resource": resource,
            "user_id": user_id
        }
    )
    
    # Date filter using helper
    if start_date and end_date:
        start = parse_iso_date(start_date)
        end = parse_iso_date(end_date)
        if start and end:
            filtered = filter_by_date_range(filtered, start, end, "timestamp")
    
    # Sort by timestamp (newest first) using helper
    filtered = sort_by_field(filtered, "timestamp", reverse=True, default_value="")
    
    # Pagination using helper
    paginated, total, page = paginate_items(filtered, offset, limit)
    
    return base.paginated(
        items=paginated,
        total=total,
        page=page,
        page_size=limit
    )


@router.get("/stats")
@base.timed_endpoint("get_audit_stats")
async def get_audit_stats(
    days: int = Query(30, ge=1, le=365, description="Number of days"),
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """
    Get audit statistics.
    """
    base.log_request("get_audit_stats", days=days)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Filter by date
    recent_logs = [
        log for log in audit_logs
        if log.get("timestamp") and
        parse_iso_date(log["timestamp"]) and
        start_date <= parse_iso_date(log["timestamp"]) <= end_date
    ]
    
    # Calculate statistics using helpers
    actions = count_by_key(recent_logs, "action", "unknown")
    resources = count_by_key(recent_logs, "resource", "unknown")
    users = count_by_key(recent_logs, "user_id", "unknown")
    
    return base.success({
        "period_days": days,
        "total_events": len(recent_logs),
        "by_action": actions,
        "by_resource": resources,
        "by_user": users,
        "most_common_action": get_most_common_key(actions),
        "most_common_resource": get_most_common_key(resources),
        "most_active_user": get_most_common_key(users)
    })


@router.get("/activity/timeline")
@base.timed_endpoint("get_activity_timeline")
async def get_activity_timeline(
    hours: int = Query(24, ge=1, le=168, description="Number of hours"),
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """
    Get activity timeline for visualization.
    """
    base.log_request("get_activity_timeline", hours=hours)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(hours=hours)
    
    # Filter by date
    recent_logs = [
        log for log in audit_logs
        if log.get("timestamp") and
        parse_iso_date(log["timestamp"]) and
        start_date <= parse_iso_date(log["timestamp"]) <= end_date
    ]
    
    # Group by hour
    hourly_activity = {}
    for log in recent_logs:
        timestamp = parse_iso_date(log["timestamp"])
        if timestamp:
            hour_key = timestamp.strftime("%Y-%m-%d %H:00")
            increment_dict_value(hourly_activity, hour_key)
    
    # Fill missing hours
    timeline = []
    for i in range(hours):
        hour = (end_date - timedelta(hours=i)).strftime("%Y-%m-%d %H:00")
        timeline.append({
            "hour": hour,
            "count": hourly_activity.get(hour, 0)
        })
    
    timeline.reverse()
    
    return base.success({
        "timeline": timeline,
        "total_events": len(recent_logs),
        "peak_hour": max(timeline, key=lambda x: x["count"]) if timeline else None
    })




