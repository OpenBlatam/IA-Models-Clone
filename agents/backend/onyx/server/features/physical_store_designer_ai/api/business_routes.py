"""
Business Routes - Endpoints para funcionalidades de negocio
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional
import logging

from ..services.vendor_service import VendorService
from ..services.billing_service import BillingService, SubscriptionPlan, PaymentStatus
from ..services.sentiment_analysis_service import SentimentAnalysisService
from ..services.ml_recommendations_service import MLRecommendationsService
from ..services.social_media_service import SocialMediaService
from ..services.ab_testing_service import ABTestingService, TestStatus
from ..services.storage_service import StorageService
from ..services.auth_service import AuthService

logger = logging.getLogger(__name__)

router = APIRouter()

# Inicializar servicios
vendor_service = VendorService()
billing_service = BillingService()
sentiment_service = SentimentAnalysisService()
ml_recommendations_service = MLRecommendationsService()
social_media_service = SocialMediaService()
ab_testing_service = ABTestingService()
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


@router.post("/vendors/register")
async def register_vendor(
    name: str,
    category: str,
    contact_info: dict,
    specialties: List[str],
    rating: Optional[float] = None
):
    """Registrar nuevo proveedor"""
    vendor = vendor_service.register_vendor(name, category, contact_info, specialties, rating)
    return vendor


@router.get("/vendors")
async def get_vendors(
    category: Optional[str] = None,
    min_rating: Optional[float] = None
):
    """Obtener proveedores"""
    vendors = vendor_service.get_vendors(category, min_rating)
    return {"vendors": vendors}


@router.post("/vendors/quotations/{store_id}")
async def request_quotation(
    store_id: str,
    vendor_id: str,
    items: List[dict],
    requirements: Optional[str] = None
):
    """Solicitar cotización"""
    try:
        quotation = vendor_service.request_quotation(store_id, vendor_id, items, requirements)
        return quotation
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/vendors/quotations/{store_id}")
async def get_quotations(store_id: str):
    """Obtener cotizaciones"""
    quotations = vendor_service.get_quotations(store_id)
    return {"store_id": store_id, "quotations": quotations}


@router.get("/vendors/recommend/{store_type}")
async def recommend_vendors(store_type: str, needs: List[str]):
    """Recomendar proveedores"""
    recommendations = vendor_service.recommend_vendors(store_type, needs)
    return {"recommendations": recommendations}


@router.post("/billing/subscribe")
async def create_subscription(
    user_id: str,
    plan: str,
    payment_method: Optional[str] = None
):
    """Crear suscripción"""
    try:
        plan_enum = SubscriptionPlan(plan)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Plan inválido: {plan}")
    
    subscription = billing_service.create_subscription(user_id, plan_enum, payment_method)
    return subscription


@router.get("/billing/subscription/{user_id}")
async def get_subscription(user_id: str):
    """Obtener suscripción"""
    subscription = billing_service.get_subscription(user_id)
    if not subscription:
        raise HTTPException(status_code=404, detail="Suscripción no encontrada")
    return subscription


@router.get("/billing/invoices/{user_id}")
async def get_invoices(user_id: str):
    """Obtener facturas"""
    invoices = billing_service.get_invoices(user_id)
    return {"user_id": user_id, "invoices": invoices}


@router.post("/billing/pay/{invoice_id}")
async def process_payment(
    invoice_id: str,
    payment_method: str,
    amount: Optional[float] = None
):
    """Procesar pago"""
    try:
        payment = billing_service.process_payment(invoice_id, payment_method, amount)
        return payment
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/billing/check-feature/{user_id}")
async def check_feature_access(user_id: str, feature: str):
    """Verificar acceso a funcionalidad"""
    has_access = billing_service.check_feature_access(user_id, feature)
    return {"user_id": user_id, "feature": feature, "has_access": has_access}


@router.post("/sentiment/analyze")
async def analyze_sentiment(text: str):
    """Analizar sentimiento de texto"""
    sentiment = await sentiment_service.analyze_sentiment(text)
    return sentiment


@router.post("/sentiment/feedback/{store_id}")
async def analyze_feedback_sentiment(store_id: str):
    """Analizar sentimiento de feedbacks"""
    from ..services.feedback_service import FeedbackService
    
    feedback_service = FeedbackService()
    feedback_list = feedback_service.get_feedback(store_id)
    
    analysis = await sentiment_service.analyze_feedback_sentiment(feedback_list)
    return analysis


@router.get("/ml/recommendations/{user_id}")
async def get_ml_recommendations(
    user_id: str,
    store_id: Optional[str] = None
):
    """Obtener recomendaciones ML personalizadas"""
    # Obtener historial del usuario (simplificado)
    all_designs = storage_service.list_designs()
    user_designs = [d for d in all_designs]  # En producción, filtrar por user_id
    
    current_design = None
    if store_id:
        design = storage_service.load_design(store_id)
        if design:
            current_design = design.dict()
    
    recommendations = await ml_recommendations_service.generate_personalized_recommendations(
        user_id, [d for d in user_designs], current_design
    )
    
    return recommendations


@router.get("/ml/success-recommendations")
async def get_success_recommendations(store_type: str, location: Optional[str] = None):
    """Obtener recomendaciones basadas en éxito"""
    from ..core.models import StoreType
    
    try:
        store_type_enum = StoreType(store_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Tipo de tienda inválido: {store_type}")
    
    recommendations = await ml_recommendations_service.recommend_based_on_success(
        store_type_enum, location
    )
    
    return recommendations


@router.post("/social/generate-content/{store_id}")
async def generate_social_content(
    store_id: str,
    platform: str = "instagram"
):
    """Generar contenido para redes sociales"""
    design = storage_service.load_design(store_id)
    if not design:
        raise HTTPException(status_code=404, detail="Diseño no encontrado")
    
    content = social_media_service.generate_social_media_content(design.dict(), platform)
    return content


@router.get("/social/calendar/{store_id}")
async def get_content_calendar(
    store_id: str,
    days: int = 30
):
    """Obtener calendario de contenido"""
    design = storage_service.load_design(store_id)
    if not design:
        raise HTTPException(status_code=404, detail="Diseño no encontrado")
    
    calendar = social_media_service.create_content_calendar(design.dict(), days)
    return {"store_id": store_id, "calendar": calendar}


@router.post("/social/opening-announcement/{store_id}")
async def generate_opening_announcement(
    store_id: str,
    opening_date: str
):
    """Generar anuncio de apertura"""
    design = storage_service.load_design(store_id)
    if not design:
        raise HTTPException(status_code=404, detail="Diseño no encontrado")
    
    announcement = social_media_service.generate_opening_announcement(design.dict(), opening_date)
    return announcement


@router.post("/ab-testing/create")
async def create_ab_test(
    test_name: str,
    variants: List[dict],
    traffic_split: Optional[dict] = None
):
    """Crear test A/B"""
    try:
        test = ab_testing_service.create_test(test_name, variants, traffic_split)
        return test
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/ab-testing/{test_id}/start")
async def start_ab_test(test_id: str):
    """Iniciar test A/B"""
    started = ab_testing_service.start_test(test_id)
    
    if not started:
        raise HTTPException(status_code=404, detail="Test no encontrado")
    
    return {"message": "Test iniciado", "test_id": test_id}


@router.get("/ab-testing/{test_id}/assign")
async def assign_variant(test_id: str, user_id: str):
    """Asignar variante a usuario"""
    variant = ab_testing_service.assign_variant(test_id, user_id)
    
    if not variant:
        raise HTTPException(status_code=404, detail="Test no encontrado o no está corriendo")
    
    return {"test_id": test_id, "user_id": user_id, "variant": variant}


@router.post("/ab-testing/{test_id}/conversion")
async def record_conversion(
    test_id: str,
    user_id: str,
    conversion_type: str = "click"
):
    """Registrar conversión"""
    recorded = ab_testing_service.record_conversion(test_id, user_id, conversion_type)
    
    if not recorded:
        raise HTTPException(status_code=400, detail="No se pudo registrar conversión")
    
    return {"message": "Conversión registrada"}


@router.get("/ab-testing/{test_id}/results")
async def get_ab_test_results(test_id: str):
    """Obtener resultados del test"""
    results = ab_testing_service.get_test_results(test_id)
    
    if "error" in results:
        raise HTTPException(status_code=404, detail=results["error"])
    
    return results


@router.post("/ab-testing/{test_id}/complete")
async def complete_ab_test(test_id: str):
    """Completar test A/B"""
    completed = ab_testing_service.complete_test(test_id)
    
    if not completed:
        raise HTTPException(status_code=404, detail="Test no encontrado")
    
    return {"message": "Test completado", "test_id": test_id}




