"""API module for Logistics AI Platform"""

from .forwarding_routes import router as forwarding_router
from .invoice_routes import router as invoice_router
from .document_routes import router as document_router
from .alert_routes import router as alert_router
from .tracking_routes import router as tracking_router
from .insurance_routes import router as insurance_router
from .report_routes import router as report_router
from .metrics_routes import router as metrics_router

__all__ = [
    "forwarding_router",
    "invoice_router",
    "document_router",
    "alert_router",
    "tracking_router",
    "insurance_router",
    "report_router",
    "metrics_router",
]








