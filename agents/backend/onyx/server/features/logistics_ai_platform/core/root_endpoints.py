"""
Root endpoints

This module provides root-level endpoints for the application.
"""

import time
from fastapi import FastAPI, status
from fastapi.responses import JSONResponse

from utils.health import get_health_status
from utils.metrics import get_metrics_collector
from utils.cache import cache_service
from utils.scalability.background_tasks import background_task_queue
from utils.logger import logger


API_VERSION = "1.0.0"
SERVICE_NAME = "Logistics AI Platform"


def setup_root_endpoints(app: FastAPI) -> None:
    """Register root-level endpoints"""
    
    @app.get("/", tags=["Root"])
    async def root():
        """
        Root endpoint with API information
        
        Returns comprehensive API information including available endpoints,
        version, and service status.
        """
        return {
            "service": SERVICE_NAME,
            "version": API_VERSION,
            "status": "running",
            "description": "Comprehensive freight forwarding and logistics management system",
            "api_version": API_VERSION,
            "endpoints": {
                "forwarding": {
                    "quotes": "/forwarding/quotes",
                    "bookings": "/forwarding/bookings",
                    "shipments": "/forwarding/shipments",
                    "containers": "/forwarding/containers"
                },
                "tracking": {
                    "track_shipment": "/tracking/shipment/{shipment_id}",
                    "track_container": "/tracking/container/{container_id}",
                    "tracking_history": "/tracking/shipment/{shipment_id}/history",
                    "summary": "/tracking/summary"
                },
                "invoices": {
                    "create": "/invoices",
                    "list": "/invoices",
                    "get": "/invoices/{invoice_id}",
                    "by_shipment": "/invoices/shipment/{shipment_id}"
                },
                "documents": {
                    "upload": "/documents",
                    "get": "/documents/{document_id}",
                    "by_shipment": "/documents/shipment/{shipment_id}",
                    "delete": "/documents/{document_id}"
                },
                "alerts": {
                    "create": "/alerts",
                    "list": "/alerts",
                    "get": "/alerts/{alert_id}",
                    "mark_read": "/alerts/{alert_id}/read",
                    "delete": "/alerts/{alert_id}"
                },
                "insurance": {
                    "create": "/insurance",
                    "get": "/insurance/{insurance_id}",
                    "by_shipment": "/insurance/shipment/{shipment_id}"
                },
                "reports": {
                    "dashboard": "/reports/dashboard",
                    "shipments": "/reports/shipments"
                },
                "metrics": {
                    "prometheus": "/metrics",
                    "info": "/metrics/info"
                },
                "docs": "/docs",
                "redoc": "/redoc",
                "health": "/health",
                "ready": "/ready"
            }
        }
    
    @app.get("/health", tags=["Health"])
    async def health_check():
        """
        Health check endpoint
        
        Returns comprehensive health status including:
        - Overall service status
        - Individual service health (cache, database, etc.)
        - Timestamp
        - Version information
        """
        try:
            health_data = await get_health_status()
            
            # Determine HTTP status code
            http_status = status.HTTP_200_OK
            if health_data["status"] == "unhealthy":
                http_status = status.HTTP_503_SERVICE_UNAVAILABLE
            elif health_data["status"] == "degraded":
                http_status = status.HTTP_200_OK
            
            return JSONResponse(
                content=health_data,
                status_code=http_status
            )
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return JSONResponse(
                content={
                    "status": "unhealthy",
                    "error": str(e),
                    "timestamp": time.time()
                },
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE
            )
    
    @app.get("/ready", tags=["Health"])
    async def readiness_check():
        """
        Readiness check endpoint
        
        Returns whether the service is ready to accept traffic.
        More strict than health check - only returns healthy if all
        critical services are operational.
        """
        try:
            health_data = await get_health_status()
            
            # Check if all critical services are healthy
            all_healthy = all(
                service["status"] == "healthy"
                for service in health_data.get("services", {}).values()
            )
            
            if all_healthy:
                return JSONResponse(
                    content={
                        "status": "ready",
                        "timestamp": health_data["timestamp"],
                        "version": API_VERSION
                    },
                    status_code=status.HTTP_200_OK
                )
            else:
                return JSONResponse(
                    content={
                        "status": "not_ready",
                        "timestamp": health_data["timestamp"],
                        "services": health_data.get("services", {}),
                        "version": API_VERSION
                    },
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE
                )
        except Exception as e:
            logger.error(f"Readiness check failed: {e}")
            return JSONResponse(
                content={
                    "status": "not_ready",
                    "error": str(e),
                    "timestamp": time.time()
                },
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE
            )







