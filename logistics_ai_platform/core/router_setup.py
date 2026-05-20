"""
Router configuration

This module provides functions to register all application routers.
"""

from fastapi import FastAPI

from api import (
    forwarding_router,
    invoice_router,
    document_router,
    alert_router,
    tracking_router,
    insurance_router,
    report_router,
    metrics_router,
)


def setup_routers(app: FastAPI) -> None:
    """Register all application routers"""
    app.include_router(forwarding_router)
    app.include_router(invoice_router)
    app.include_router(document_router)
    app.include_router(alert_router)
    app.include_router(tracking_router)
    app.include_router(insurance_router)
    app.include_router(report_router)
    app.include_router(metrics_router)







