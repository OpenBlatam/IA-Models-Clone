"""Alert service for managing alerts and notifications"""

from typing import List, Optional
from datetime import datetime
import uuid
import logging

from models.schemas import (
    AlertRequest,
    AlertResponse,
)

logger = logging.getLogger(__name__)


class AlertService:
    """Service for managing alerts"""
    
    def __init__(self):
        """Initialize alert service"""
        self.alerts = {}  # In-memory storage
    
    async def create_alert(self, request: AlertRequest) -> AlertResponse:
        """Create a new alert"""
        try:
            alert_id = f"ALT{str(uuid.uuid4())[:8].upper()}"
            
            alert = AlertResponse(
                alert_id=alert_id,
                shipment_id=request.shipment_id,
                container_id=request.container_id,
                alert_type=request.alert_type,
                message=request.message,
                priority=request.priority,
                is_read=False,
                created_at=datetime.now()
            )
            
            self.alerts[alert_id] = alert
            
            logger.info(f"Alert created: {alert_id}")
            return alert
            
        except Exception as e:
            logger.error(f"Error creating alert: {str(e)}")
            raise
    
    async def get_alert(self, alert_id: str) -> Optional[AlertResponse]:
        """Get alert by ID"""
        return self.alerts.get(alert_id)
    
    async def get_alerts(
        self,
        shipment_id: Optional[str] = None,
        container_id: Optional[str] = None,
        is_read: Optional[bool] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[AlertResponse]:
        """Get alerts with optional filtering"""
        alerts = list(self.alerts.values())
        
        if shipment_id:
            alerts = [a for a in alerts if a.shipment_id == shipment_id]
        
        if container_id:
            alerts = [a for a in alerts if a.container_id == container_id]
        
        if is_read is not None:
            alerts = [a for a in alerts if a.is_read == is_read]
        
        # Sort by created_at descending
        alerts.sort(key=lambda x: x.created_at, reverse=True)
        
        return alerts[offset:offset + limit]
    
    async def mark_alert_read(self, alert_id: str) -> Optional[AlertResponse]:
        """Mark an alert as read"""
        alert = await self.get_alert(alert_id)
        if not alert:
            return None
        
        alert.is_read = True
        alert.read_at = datetime.now()
        
        self.alerts[alert_id] = alert
        return alert
    
    async def delete_alert(self, alert_id: str) -> bool:
        """Delete an alert"""
        if alert_id not in self.alerts:
            return False
        
        del self.alerts[alert_id]
        logger.info(f"Alert deleted: {alert_id}")
        return True













