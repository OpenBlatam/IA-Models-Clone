"""
IoT/ERP Routes - Endpoints para IoT, ERP, AR/VR y más
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Optional, List
from datetime import datetime
import logging

from ..services.ar_vr_service import ARVRService
from ..services.iot_service import IoTService, SensorType
from ..services.intelligent_inventory_service import IntelligentInventoryService
from ..services.realtime_analytics_service import RealtimeAnalyticsService
from ..services.erp_integration_service import ERPIntegrationService, ERPProvider
from ..services.compliance_service import ComplianceService, ComplianceType, ComplianceStatus
from ..services.storage_service import StorageService
from ..services.auth_service import AuthService

logger = logging.getLogger(__name__)

router = APIRouter()

# Inicializar servicios
ar_vr_service = ARVRService()
iot_service = IoTService()
inventory_service = IntelligentInventoryService()
realtime_analytics_service = RealtimeAnalyticsService()
erp_service = ERPIntegrationService()
compliance_service = ComplianceService()
storage_service = StorageService()
auth_service = AuthService()


def verify_token(authorization: Optional[str] = None):
    """Verificar token"""
    if authorization:
        token = authorization.replace("Bearer ", "")
        payload = auth_service.verify_token(token)
        if payload:
            return payload
    return None


@router.post("/ar-vr/experience/{store_id}")
async def create_ar_experience(
    store_id: str,
    experience_name: str,
    description: str
):
    """Crear experiencia AR"""
    experience = ar_vr_service.create_ar_experience(store_id, experience_name, description)
    return experience


@router.post("/ar-vr/vr-tour/{store_id}")
async def create_vr_tour(
    store_id: str,
    tour_name: str,
    scenes: List[dict]
):
    """Crear tour VR"""
    tour = ar_vr_service.create_vr_tour(store_id, tour_name, scenes)
    return tour


@router.get("/ar-vr/preview/{store_id}")
async def get_ar_preview(store_id: str):
    """Obtener preview AR"""
    design = storage_service.load_design(store_id)
    if not design:
        raise HTTPException(status_code=404, detail="Diseño no encontrado")
    
    preview = ar_vr_service.generate_ar_preview(store_id, design.dict())
    return preview


@router.post("/iot/devices/{store_id}")
async def register_iot_device(
    store_id: str,
    device_name: str,
    device_type: str,
    location: str,
    capabilities: List[str]
):
    """Registrar dispositivo IoT"""
    device = iot_service.register_device(store_id, device_name, device_type, location, capabilities)
    return device


@router.post("/iot/sensors/{device_id}")
async def add_sensor(
    device_id: str,
    sensor_type: str,
    unit: str = "default"
):
    """Agregar sensor"""
    try:
        sensor_type_enum = SensorType(sensor_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Tipo de sensor inválido: {sensor_type}")
    
    sensor = iot_service.add_sensor(device_id, sensor_type_enum, unit)
    return sensor


@router.post("/iot/readings/{sensor_id}")
async def record_reading(
    sensor_id: str,
    value: float,
    timestamp: Optional[str] = None
):
    """Registrar lectura de sensor"""
    reading = iot_service.record_reading(sensor_id, value, timestamp)
    return reading


@router.get("/iot/analytics/{store_id}")
async def get_iot_analytics(
    store_id: str,
    hours: int = 24
):
    """Obtener analytics de IoT"""
    analytics = iot_service.get_store_analytics(store_id, hours)
    return analytics


@router.post("/inventory/products/{store_id}")
async def add_product(
    store_id: str,
    product_name: str,
    sku: str,
    initial_stock: int,
    unit_price: float,
    category: Optional[str] = None,
    reorder_point: Optional[int] = None
):
    """Agregar producto al inventario"""
    product = inventory_service.add_product(
        store_id, product_name, sku, initial_stock, unit_price, category, reorder_point
    )
    return product


@router.post("/inventory/transactions/{product_id}")
async def record_transaction(
    product_id: str,
    transaction_type: str,
    quantity: int,
    notes: Optional[str] = None
):
    """Registrar transacción de inventario"""
    try:
        transaction = inventory_service.record_transaction(product_id, transaction_type, quantity, notes)
        return transaction
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/inventory/predict/{product_id}")
async def predict_demand(
    product_id: str,
    days: int = 30
):
    """Predecir demanda"""
    prediction = await inventory_service.predict_demand(product_id, days)
    return prediction


@router.get("/inventory/alerts/{store_id}")
async def get_low_stock_alerts(store_id: str):
    """Obtener alertas de stock bajo"""
    alerts = inventory_service.get_low_stock_alerts(store_id)
    return {"store_id": store_id, "alerts": alerts}


@router.post("/analytics/metrics/{store_id}")
async def record_metric(
    store_id: str,
    metric_name: str,
    value: float,
    tags: Optional[dict] = None
):
    """Registrar métrica en tiempo real"""
    metric = realtime_analytics_service.record_metric(store_id, metric_name, value, tags)
    return metric


@router.get("/analytics/dashboard/{store_id}")
async def get_realtime_dashboard(
    store_id: str,
    time_window_minutes: int = 60
):
    """Obtener dashboard en tiempo real"""
    dashboard = realtime_analytics_service.get_realtime_dashboard(store_id, time_window_minutes)
    return dashboard


@router.get("/analytics/history/{store_id}/{metric_name}")
async def get_metric_history(
    store_id: str,
    metric_name: str,
    hours: int = 24
):
    """Obtener historial de métrica"""
    history = realtime_analytics_service.get_metric_history(store_id, metric_name, hours)
    return history


@router.post("/erp/register/{store_id}")
async def register_erp(
    store_id: str,
    provider: str,
    connection_config: dict,
    sync_frequency: str = "daily"
):
    """Registrar integración ERP"""
    try:
        provider_enum = ERPProvider(provider)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Proveedor ERP inválido: {provider}")
    
    integration = erp_service.register_erp(store_id, provider_enum, connection_config, sync_frequency)
    return integration


@router.post("/erp/sync/inventory/{integration_id}")
async def sync_inventory(integration_id: str):
    """Sincronizar inventario con ERP"""
    try:
        sync = await erp_service.sync_inventory(integration_id)
        return sync
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/erp/test/{integration_id}")
async def test_erp_connection(integration_id: str):
    """Probar conexión ERP"""
    result = erp_service.test_connection(integration_id)
    return result


@router.post("/compliance/assess/{store_id}")
async def assess_compliance(
    store_id: str,
    location: Optional[str] = None
):
    """Evaluar compliance del diseño"""
    design = storage_service.load_design(store_id)
    if not design:
        raise HTTPException(status_code=404, detail="Diseño no encontrado")
    
    assessment = compliance_service.assess_compliance(store_id, design.dict(), location)
    return assessment


@router.get("/compliance/certificate/{assessment_id}")
async def get_compliance_certificate(assessment_id: str):
    """Obtener certificado de compliance"""
    certificate = compliance_service.get_compliance_certificate(assessment_id)
    
    if not certificate:
        raise HTTPException(status_code=404, detail="Certificado no disponible o evaluación no compliant")
    
    return certificate




