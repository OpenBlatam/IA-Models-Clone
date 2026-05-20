"""Report routes"""

from fastapi import APIRouter
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

from utils.dependencies import (
    ShipmentServiceDep,
    TrackingServiceDep,
    InvoiceServiceDep,
)
from utils.cache import cache_service

router = APIRouter(prefix="/reports", tags=["Reports"])


@router.get("/dashboard")
async def get_dashboard_stats(
    shipment_service: ShipmentServiceDep = None,
    tracking_service: TrackingServiceDep = None,
    invoice_service: InvoiceServiceDep = None
) -> Dict[str, Any]:
    """Get dashboard statistics"""
    # Try cache first
    cached = await cache_service.get("reports:dashboard")
    if cached:
        return cached
    
    all_shipments = await shipment_service.get_shipments()
    
    # Count by status
    status_counts = {}
    for shipment in all_shipments:
        status = shipment.status.value
        status_counts[status] = status_counts.get(status, 0) + 1
    
    # Get summary data
    departing = await tracking_service.get_departing_this_week()
    arriving = await tracking_service.get_arriving_this_week()
    in_transit = await tracking_service.get_in_transit()
    
    # Get invoices
    invoices = await invoice_service.get_all_invoices(limit=1000)
    total_revenue = sum(inv.total for inv in invoices)
    
    stats = {
        "shipments": {
            "total": len(all_shipments),
            "by_status": status_counts,
            "departing_this_week": len(departing),
            "arriving_this_week": len(arriving),
            "in_transit": len(in_transit)
        },
        "revenue": {
            "total_usd": total_revenue,
            "invoice_count": len(invoices)
        },
        "timestamp": datetime.now()
    }
    
    # Cache for 5 minutes
    await cache_service.set("reports:dashboard", stats, ttl=300)
    
    return stats


@router.get("/shipments")
async def get_shipment_report(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    shipment_service: ShipmentServiceDep = None
) -> Dict[str, Any]:
    """Get shipment report"""
    if not start_date:
        start_date = datetime.now() - timedelta(days=30)
    if not end_date:
        end_date = datetime.now()
    
    all_shipments = await shipment_service.get_shipments()
    
    # Filter by date range
    filtered = [
        s for s in all_shipments
        if start_date <= s.created_at <= end_date
    ]
    
    return {
        "period": {
            "start": start_date,
            "end": end_date
        },
        "total_shipments": len(filtered),
        "shipments": filtered
    }

