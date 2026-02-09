"""
Integrations Router - Handles IoT, wearable, pharmacy, and other integration endpoints
"""

from fastapi import APIRouter, HTTPException, Form, Query
from fastapi.responses import JSONResponse
from typing import Optional
import json

from ...api.services_locator import get_service
from ...utils.logger import logger

router = APIRouter(prefix="/dermatology", tags=["integrations"])


@router.post("/iot/register")
async def register_iot_device(
    user_id: str = Form(...),
    device_type: str = Form(...),
    device_id: str = Form(...),
    metadata: Optional[str] = Form(None)
):
    """Registra dispositivo IoT"""
    try:
        iot_integration = get_service("iot_integration")
        metadata_dict = json.loads(metadata) if metadata else {}
        device = iot_integration.register_device(user_id, device_type, device_id, metadata_dict)
        return JSONResponse(content={"success": True, "device": device.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/iot/devices/{user_id}")
async def get_iot_devices(user_id: str):
    """Obtiene dispositivos IoT del usuario"""
    try:
        iot_integration = get_service("iot_integration")
        devices = iot_integration.get_user_devices(user_id)
        return JSONResponse(content={
            "success": True,
            "devices": [d.to_dict() for d in devices]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/iot/data")
async def sync_iot_data(
    user_id: str = Form(...),
    device_id: str = Form(...),
    data: str = Form(...)
):
    """Sincroniza datos de dispositivo IoT"""
    try:
        iot_integration = get_service("iot_integration")
        data_dict = json.loads(data)
        result = iot_integration.sync_device_data(user_id, device_id, data_dict)
        return JSONResponse(content={"success": True, "result": result.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/wearable/register")
async def register_wearable(
    user_id: str = Form(...),
    device_type: str = Form(...),
    device_id: str = Form(...),
    metadata: Optional[str] = Form(None)
):
    """Registra dispositivo wearable"""
    try:
        wearable_integration = get_service("wearable_integration")
        metadata_dict = json.loads(metadata) if metadata else {}
        device = wearable_integration.register_device(user_id, device_type, device_id, metadata_dict)
        return JSONResponse(content={"success": True, "device": device.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/wearable/sync")
async def sync_wearable_data(
    user_id: str = Form(...),
    device_id: str = Form(...),
    data: str = Form(...)
):
    """Sincroniza datos de wearable"""
    try:
        wearable_integration = get_service("wearable_integration")
        data_dict = json.loads(data)
        result = wearable_integration.sync_data(user_id, device_id, data_dict)
        return JSONResponse(content={"success": True, "result": result.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/wearable/insights/{user_id}")
async def get_wearable_insights(user_id: str):
    """Obtiene insights de datos wearable"""
    try:
        wearable_integration = get_service("wearable_integration")
        insights = wearable_integration.get_insights(user_id)
        return JSONResponse(content={
            "success": True,
            "insights": insights.to_dict() if hasattr(insights, 'to_dict') else insights
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/pharmacy/register")
async def register_pharmacy(
    pharmacy_id: str = Form(...),
    name: str = Form(...),
    location: str = Form(...),
    metadata: Optional[str] = Form(None)
):
    """Registra farmacia"""
    try:
        pharmacy_integration = get_service("pharmacy_integration")
        metadata_dict = json.loads(metadata) if metadata else {}
        pharmacy = pharmacy_integration.register_pharmacy(
            pharmacy_id, name, location, metadata_dict
        )
        return JSONResponse(content={"success": True, "pharmacy": pharmacy.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/pharmacy/nearby")
async def get_nearby_pharmacies(
    latitude: float = Query(...),
    longitude: float = Query(...),
    radius: float = Query(5.0)
):
    """Obtiene farmacias cercanas"""
    try:
        pharmacy_integration = get_service("pharmacy_integration")
        pharmacies = pharmacy_integration.get_nearby_pharmacies(latitude, longitude, radius)
        return JSONResponse(content={
            "success": True,
            "pharmacies": [p.to_dict() for p in pharmacies]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/pharmacy/product-availability")
async def check_product_availability(
    pharmacy_id: str = Form(...),
    product_id: str = Form(...)
):
    """Verifica disponibilidad de producto en farmacia"""
    try:
        pharmacy_integration = get_service("pharmacy_integration")
        availability = pharmacy_integration.check_product_availability(pharmacy_id, product_id)
        return JSONResponse(content={
            "success": True,
            "availability": availability.to_dict() if hasattr(availability, 'to_dict') else availability
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")




