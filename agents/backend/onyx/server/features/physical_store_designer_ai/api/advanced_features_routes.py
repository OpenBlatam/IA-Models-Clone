"""
Advanced Features Routes - Endpoints para funcionalidades avanzadas
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Optional, List
from datetime import datetime
import logging

from ..services.generative_ai_service import GenerativeAIService
from ..services.workflow_automation_service import WorkflowAutomationService, WorkflowStatus
from ..services.booking_service import BookingService, BookingStatus
from ..services.roi_analysis_service import ROIAnalysisService
from ..services.auto_documentation_service import AutoDocumentationService
from ..services.payment_integration_service import PaymentIntegrationService, PaymentProvider, PaymentStatus
from ..services.storage_service import StorageService
from ..services.auth_service import AuthService

logger = logging.getLogger(__name__)

router = APIRouter()

# Inicializar servicios
generative_ai_service = GenerativeAIService()
workflow_service = WorkflowAutomationService()
booking_service = BookingService()
roi_service = ROIAnalysisService()
doc_service = AutoDocumentationService()
payment_service = PaymentIntegrationService()
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


@router.post("/generative-ai/image/{store_id}")
async def generate_store_image(
    store_id: str,
    description: str,
    style: str = "realistic",
    resolution: str = "1024x1024"
):
    """Generar imagen del local"""
    image = await generative_ai_service.generate_store_image(store_id, description, style, resolution)
    return image


@router.post("/generative-ai/video/{store_id}")
async def generate_store_video(
    store_id: str,
    script: str,
    duration: int = 30
):
    """Generar video promocional"""
    video = await generative_ai_service.generate_store_video(store_id, script, duration)
    return video


@router.post("/generative-ai/marketing-copy/{store_id}")
async def generate_marketing_copy(
    store_id: str,
    copy_type: str = "social_media"
):
    """Generar copy de marketing"""
    design = storage_service.load_design(store_id)
    if not design:
        raise HTTPException(status_code=404, detail="Diseño no encontrado")
    
    copy = await generative_ai_service.generate_marketing_copy(design.dict(), copy_type)
    return copy


@router.post("/workflows/create")
async def create_workflow(
    name: str,
    description: str,
    steps: List[dict],
    trigger_event: Optional[str] = None
):
    """Crear workflow"""
    workflow = workflow_service.create_workflow(name, description, steps, trigger_event)
    return workflow


@router.post("/workflows/{workflow_id}/activate")
async def activate_workflow(workflow_id: str):
    """Activar workflow"""
    activated = workflow_service.activate_workflow(workflow_id)
    
    if not activated:
        raise HTTPException(status_code=404, detail="Workflow no encontrado")
    
    return {"message": "Workflow activado", "workflow_id": workflow_id}


@router.post("/workflows/{workflow_id}/execute")
async def execute_workflow(
    workflow_id: str,
    context: Optional[dict] = None
):
    """Ejecutar workflow"""
    try:
        execution = await workflow_service.execute_workflow(workflow_id, context)
        return execution
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/workflows")
async def get_workflows():
    """Obtener workflows"""
    workflows = workflow_service.get_workflows()
    return {"workflows": workflows}


@router.post("/bookings/services/{store_id}")
async def create_service(
    store_id: str,
    name: str,
    duration_minutes: int,
    price: float,
    description: Optional[str] = None
):
    """Crear servicio reservable"""
    service = booking_service.create_service(store_id, name, duration_minutes, price, description)
    return service


@router.post("/bookings/availability/{store_id}")
async def set_availability(
    store_id: str,
    day_of_week: int,
    start_time: str,
    end_time: str,
    is_available: bool = True
):
    """Configurar disponibilidad"""
    availability = booking_service.set_availability(store_id, day_of_week, start_time, end_time, is_available)
    return availability


@router.post("/bookings/create")
async def create_booking(
    store_id: str,
    service_id: str,
    customer_name: str,
    customer_email: str,
    customer_phone: Optional[str],
    booking_date: str,
    booking_time: str,
    notes: Optional[str] = None
):
    """Crear reserva"""
    try:
        booking = booking_service.create_booking(
            store_id, service_id, customer_name, customer_email,
            customer_phone, booking_date, booking_time, notes
        )
        return booking
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/bookings/{store_id}")
async def get_bookings(
    store_id: str,
    date: Optional[str] = None,
    status: Optional[str] = None
):
    """Obtener reservas"""
    status_enum = None
    if status:
        try:
            status_enum = BookingStatus(status)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Status inválido: {status}")
    
    bookings = booking_service.get_bookings(store_id, date, status_enum)
    return {"store_id": store_id, "bookings": bookings}


@router.get("/bookings/availability/{store_id}/{date}")
async def get_available_slots(
    store_id: str,
    date: str,
    service_id: str
):
    """Obtener slots disponibles"""
    slots = booking_service.get_available_slots(store_id, date, service_id)
    return {"store_id": store_id, "date": date, "available_slots": slots}


@router.post("/roi/calculate")
async def calculate_roi(
    initial_investment: float,
    monthly_revenue: float,
    monthly_costs: float,
    months: int = 12
):
    """Calcular ROI"""
    roi = roi_service.calculate_roi(initial_investment, monthly_revenue, monthly_costs, months)
    return roi


@router.post("/roi/report/{store_id}")
async def generate_roi_report(store_id: str):
    """Generar reporte completo de ROI"""
    design = storage_service.load_design(store_id)
    if not design:
        raise HTTPException(status_code=404, detail="Diseño no encontrado")
    
    financial_data = design.financial_analysis.dict() if design.financial_analysis else {}
    report = await roi_service.generate_roi_report(store_id, financial_data)
    return report


@router.post("/roi/compare")
async def compare_roi_scenarios(scenarios: List[dict]):
    """Comparar escenarios de ROI"""
    comparison = roi_service.compare_scenarios(scenarios)
    return comparison


@router.post("/documentation/generate/{store_id}")
async def generate_documentation(store_id: str):
    """Generar documentación completa"""
    design = storage_service.load_design(store_id)
    if not design:
        raise HTTPException(status_code=404, detail="Diseño no encontrado")
    
    documentation = await doc_service.generate_project_documentation(store_id, design.dict())
    return documentation


@router.get("/documentation/{document_id}")
async def get_documentation(document_id: str):
    """Obtener documentación"""
    document = doc_service.get_document(document_id)
    
    if not document:
        raise HTTPException(status_code=404, detail="Documento no encontrado")
    
    return document


@router.get("/documentation/{document_id}/markdown")
async def export_documentation_markdown(document_id: str):
    """Exportar documentación a Markdown"""
    markdown = doc_service.export_to_markdown(document_id)
    
    if not markdown:
        raise HTTPException(status_code=404, detail="Documento no encontrado")
    
    from fastapi.responses import Response
    return Response(content=markdown, media_type="text/markdown")


@router.post("/payments/providers/register")
async def register_payment_provider(
    provider: str,
    api_key: str,
    secret_key: Optional[str] = None,
    is_active: bool = True
):
    """Registrar proveedor de pago"""
    try:
        provider_enum = PaymentProvider(provider)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Proveedor inválido: {provider}")
    
    config = payment_service.register_provider(provider_enum, api_key, secret_key, is_active)
    return config


@router.post("/payments/create")
async def create_payment_intent(
    amount: float,
    currency: str = "USD",
    provider: str = "stripe",
    description: Optional[str] = None,
    metadata: Optional[dict] = None
):
    """Crear intención de pago"""
    try:
        provider_enum = PaymentProvider(provider)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Proveedor inválido: {provider}")
    
    payment = await payment_service.create_payment_intent(
        amount, currency, provider_enum, description, metadata
    )
    return payment


@router.post("/payments/{payment_id}/process")
async def process_payment(
    payment_id: str,
    payment_method: str,
    provider: Optional[str] = None
):
    """Procesar pago"""
    provider_enum = None
    if provider:
        try:
            provider_enum = PaymentProvider(provider)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Proveedor inválido: {provider}")
    
    try:
        payment = await payment_service.process_payment(payment_id, payment_method, provider_enum)
        return payment
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/payments/{payment_id}/refund")
async def refund_payment(
    payment_id: str,
    amount: Optional[float] = None,
    reason: Optional[str] = None
):
    """Reembolsar pago"""
    try:
        refund = await payment_service.refund_payment(payment_id, amount, reason)
        return refund
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/payments/statistics")
async def get_payment_statistics(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """Obtener estadísticas de pagos"""
    start = datetime.fromisoformat(start_date) if start_date else None
    end = datetime.fromisoformat(end_date) if end_date else None
    
    stats = payment_service.get_payment_statistics(start, end)
    return stats




