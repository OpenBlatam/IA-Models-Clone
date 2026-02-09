"""Alert routes"""

from fastapi import APIRouter
from typing import List, Optional

from models.schemas import (
    AlertRequest,
    AlertResponse,
)
from utils.dependencies import AlertServiceDep
from utils.exceptions import NotFoundError

router = APIRouter(prefix="/alerts", tags=["Alerts"])


@router.post("", response_model=AlertResponse, status_code=201)
async def create_alert(
    request: AlertRequest,
    alert_service: AlertServiceDep
) -> AlertResponse:
    """Create a new alert"""
    alert = await alert_service.create_alert(request)
    return alert


@router.get("", response_model=List[AlertResponse])
async def get_alerts(
    shipment_id: Optional[str] = None,
    container_id: Optional[str] = None,
    is_read: Optional[bool] = None,
    limit: int = 100,
    offset: int = 0,
    alert_service: AlertServiceDep = None
) -> List[AlertResponse]:
    """Get alerts with optional filtering"""
    if limit > 1000:
        limit = 1000  # Enforce max limit
    
    alerts = await alert_service.get_alerts(
        shipment_id=shipment_id,
        container_id=container_id,
        is_read=is_read,
        limit=limit,
        offset=offset
    )
    return alerts


@router.get("/{alert_id}", response_model=AlertResponse)
async def get_alert(
    alert_id: str,
    alert_service: AlertServiceDep
) -> AlertResponse:
    """Get alert by ID"""
    alert = await alert_service.get_alert(alert_id)
    if not alert:
        raise NotFoundError("Alert", alert_id)
    return alert


@router.patch("/{alert_id}/read", response_model=AlertResponse)
async def mark_alert_read(
    alert_id: str,
    alert_service: AlertServiceDep
) -> AlertResponse:
    """Mark an alert as read"""
    alert = await alert_service.mark_alert_read(alert_id)
    if not alert:
        raise NotFoundError("Alert", alert_id)
    return alert


@router.delete("/{alert_id}")
async def delete_alert(
    alert_id: str,
    alert_service: AlertServiceDep
):
    """Delete an alert"""
    success = await alert_service.delete_alert(alert_id)
    if not success:
        raise NotFoundError("Alert", alert_id)
    return {"message": "Alert deleted successfully"}

