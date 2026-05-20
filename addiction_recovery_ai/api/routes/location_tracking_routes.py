"""
Location tracking routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse

try:
    from services.location_tracking_service import LocationTrackingService
except ImportError:
    from ...services.location_tracking_service import LocationTrackingService

router = APIRouter()

location_tracking = LocationTrackingService()


@router.post("/location/add")
async def add_location(
    user_id: str = Body(...),
    location_type: str = Body(...),
    name: str = Body(...),
    latitude: float = Body(...),
    longitude: float = Body(...)
):
    """Agrega una ubicación"""
    try:
        location = location_tracking.add_location(
            user_id, location_type, name, latitude, longitude
        )
        return JSONResponse(content=location)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error agregando ubicación: {str(e)}")


@router.post("/location/check-proximity")
async def check_location_proximity(
    user_id: str = Body(...),
    current_latitude: float = Body(...),
    current_longitude: float = Body(...)
):
    """Verifica proximidad a ubicaciones registradas"""
    try:
        proximity = location_tracking.check_location_proximity(
            user_id, current_latitude, current_longitude
        )
        return JSONResponse(content=proximity)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error verificando proximidad: {str(e)}")



