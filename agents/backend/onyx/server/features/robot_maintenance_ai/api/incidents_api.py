"""
Incidents API for managing maintenance incidents and tickets.
Refactored to use BaseRouter for reduced duplication.
"""

from fastapi import Depends, Query
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime

from .base_router import BaseRouter
from ..utils.file_helpers import get_iso_timestamp, get_timestamp_id, create_resource, update_resource
from ..utils.data_helpers import filter_by_fields, paginate_items

# Create base router instance
base = BaseRouter(
    prefix="/api/incidents",
    tags=["Incidents"],
    require_authentication=True,
    require_rate_limit=False
)

router = base.router


class CreateIncidentRequest(BaseModel):
    """Request to create an incident."""
    title: str = Field(..., min_length=5, max_length=200, description="Incident title")
    description: str = Field(..., min_length=10, description="Detailed description")
    robot_type: str = Field(..., description="Type of robot affected")
    severity: str = Field("medium", description="Severity: low, medium, high, critical")
    maintenance_type: Optional[str] = Field(None, description="Related maintenance type")
    sensor_data: Optional[Dict[str, Any]] = Field(None, description="Relevant sensor data")


class UpdateIncidentRequest(BaseModel):
    """Request to update an incident."""
    status: Optional[str] = Field(None, description="New status")
    priority: Optional[str] = Field(None, description="New priority")
    assigned_to: Optional[str] = Field(None, description="Assignee")
    notes: Optional[str] = Field(None, description="Additional notes")
    resolution: Optional[str] = Field(None, description="Resolution description")


@router.post("/create")
@base.timed_endpoint("create_incident")
async def create_incident(
    request: CreateIncidentRequest,
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """
    Create a new maintenance incident.
    """
    base.log_request("create_incident", robot_type=request.robot_type, severity=request.severity)
    
    incident = create_resource(
        {
            "title": request.title,
            "description": request.description,
            "robot_type": request.robot_type,
            "maintenance_type": request.maintenance_type,
            "severity": request.severity,
            "status": "open",
            "priority": "medium",
            "sensor_data": request.sensor_data,
            "resolved_at": None,
            "assigned_to": None,
            "notes": [],
            "resolution": None
        },
        id_prefix="incident_"
    )
    
    return base.success(incident, message="Incident created successfully")


@router.get("/list")
@base.timed_endpoint("list_incidents")
async def list_incidents(
    robot_type: Optional[str] = Query(None, description="Filter by robot type"),
    status: Optional[str] = Query(None, description="Filter by status"),
    severity: Optional[str] = Query(None, description="Filter by severity"),
    limit: int = Query(50, ge=1, le=500, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """
    List incidents with optional filters.
    """
    base.log_request("list_incidents", robot_type=robot_type, status=status, severity=severity)
    
    # Placeholder - would query database
    incidents = []
    
    # Apply filters using helper
    filtered = filter_by_fields(
        incidents,
        {
            "robot_type": robot_type,
            "status": status,
            "severity": severity
        }
    )
    
    # Pagination using helper
    paginated, total, page = paginate_items(filtered, offset, limit)
    
    return base.paginated(
        items=paginated,
        total=total,
        page=page,
        page_size=limit,
        message=f"Found {total} incidents"
    )


@router.get("/{incident_id}")
@base.timed_endpoint("get_incident")
async def get_incident(
    incident_id: str,
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """
    Get details of a specific incident.
    """
    base.log_request("get_incident", incident_id=incident_id)
    
    # Placeholder - would query database
    incident = {
        "id": incident_id,
        "title": "Sample Incident",
        "description": "This is a placeholder",
        "status": "open"
    }
    
    return base.success(incident)


@router.put("/{incident_id}")
@base.timed_endpoint("update_incident")
async def update_incident(
    incident_id: str,
    request: UpdateIncidentRequest,
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """
    Update an incident.
    """
    base.log_request("update_incident", incident_id=incident_id)
    
    # Placeholder - would update database
    updates = {}
    if request.status:
        updates["status"] = request.status
    if request.priority:
        updates["priority"] = request.priority
    if request.assigned_to:
        updates["assigned_to"] = request.assigned_to
    if request.resolution:
        updates["resolution"] = request.resolution
        updates["resolved_at"] = get_iso_timestamp()
        updates["status"] = "resolved"
    
    # Create updated resource with updated_at automatically set
    updated = update_resource(
        {"id": incident_id},
        updates
    )
    
    return base.success(updated, message="Incident updated successfully")


@router.post("/{incident_id}/add-note")
@base.timed_endpoint("add_incident_note")
async def add_incident_note(
    incident_id: str,
    note: str = Field(..., min_length=1, description="Note content"),
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """
    Add a note to an incident.
    """
    base.log_request("add_incident_note", incident_id=incident_id)
    
    note_entry = create_resource(
        {
            "incident_id": incident_id,
            "content": note
        },
        id_prefix="note_"
    )
    
    return base.success(note_entry, message="Note added successfully")


@router.post("/{incident_id}/resolve")
@base.timed_endpoint("resolve_incident")
async def resolve_incident(
    incident_id: str,
    resolution: str = Field(..., min_length=10, description="Resolution description"),
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """
    Resolve an incident.
    """
    base.log_request("resolve_incident", incident_id=incident_id)
    
    resolved = {
        "id": incident_id,
        "status": "resolved",
        "resolution": resolution,
        "resolved_at": get_iso_timestamp(),
        "updated_at": get_iso_timestamp()
    }
    
    return base.success(resolved, message="Incident resolved successfully")


@router.get("/stats/summary")
@base.timed_endpoint("get_incidents_summary")
async def get_incidents_summary(
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """
    Get summary statistics of incidents.
    """
    base.log_request("get_incidents_summary")
    
    # Placeholder - would calculate from database
    summary = {
        "total": 0,
        "open": 0,
        "resolved": 0,
        "by_severity": {
            "low": 0,
            "medium": 0,
            "high": 0,
            "critical": 0
        },
        "by_robot_type": {},
        "average_resolution_time": "0 hours"
    }
    
    return base.success(summary)




