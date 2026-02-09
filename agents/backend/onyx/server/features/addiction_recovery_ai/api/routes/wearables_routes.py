"""
Wearables routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse

try:
    from services.wearable_service import WearableService
except ImportError:
    from ...services.wearable_service import WearableService

router = APIRouter()

wearable = WearableService()


@router.post("/wearable/register")
async def register_wearable(
    user_id: str = Body(...),
    device_type: str = Body(...),
    device_name: str = Body(...),
    device_id: str = Body(...)
):
    """Registra un dispositivo wearable"""
    try:
        device = wearable.register_device(user_id, device_type, device_name, device_id)
        return JSONResponse(content=device)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error registrando dispositivo: {str(e)}")



