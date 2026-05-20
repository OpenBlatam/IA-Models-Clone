"""API routes for Artist Manager AI."""

from fastapi import APIRouter
from . import calendar, routines, protocols, wardrobe, dashboard, advanced, monitoring, websocket, graphql

router = APIRouter(prefix="/artist-manager", tags=["artist-manager"])

router.include_router(calendar.router)
router.include_router(routines.router)
router.include_router(protocols.router)
router.include_router(wardrobe.router)
router.include_router(dashboard.router)
router.include_router(advanced.router)
router.include_router(monitoring.router)
router.include_router(websocket.router)
router.include_router(graphql.router)

__all__ = ["router", "calendar", "routines", "protocols", "wardrobe", "dashboard", "advanced", "monitoring", "websocket", "graphql"]

