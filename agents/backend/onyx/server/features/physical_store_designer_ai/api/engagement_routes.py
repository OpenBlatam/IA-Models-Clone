"""
Engagement Routes - Endpoints para engagement y funcionalidades avanzadas
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Optional
from datetime import datetime
import logging

from ..services.analytics_service import AnalyticsService
from ..services.gamification_service import GamificationService
from ..services.marketplace_service import MarketplaceService, ListingStatus
from ..services.crm_service import CRMService, ContactStatus
from ..services.loyalty_service import LoyaltyService, RewardType
from ..services.realtime_competitor_service import RealtimeCompetitorService
from ..services.storage_service import StorageService
from ..services.auth_service import AuthService

logger = logging.getLogger(__name__)

router = APIRouter()

# Inicializar servicios
analytics_service = AnalyticsService()
gamification_service = GamificationService()
marketplace_service = MarketplaceService()
crm_service = CRMService()
loyalty_service = LoyaltyService()
realtime_competitor_service = RealtimeCompetitorService()
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


@router.post("/analytics/track")
async def track_event(
    event_type: str,
    user_id: Optional[str] = None,
    store_id: Optional[str] = None,
    properties: Optional[dict] = None
):
    """Rastrear evento"""
    event = analytics_service.track_event(event_type, user_id, store_id, properties)
    return event


@router.get("/analytics")
async def get_analytics(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    event_type: Optional[str] = None
):
    """Obtener analytics"""
    start = datetime.fromisoformat(start_date) if start_date else None
    end = datetime.fromisoformat(end_date) if end_date else None
    
    analytics = analytics_service.get_analytics(start, end, event_type)
    return analytics


@router.get("/analytics/user/{user_id}/journey")
async def get_user_journey(user_id: str):
    """Obtener journey de usuario"""
    journey = analytics_service.get_user_journey(user_id)
    return journey


@router.get("/gamification/profile/{user_id}")
async def get_gamification_profile(user_id: str):
    """Obtener perfil de gamificación"""
    profile = gamification_service.get_user_profile(user_id)
    return profile


@router.post("/gamification/award-points/{user_id}")
async def award_points(
    user_id: str,
    points: int,
    reason: str
):
    """Otorgar puntos"""
    result = gamification_service.award_points(user_id, points, reason)
    return result


@router.get("/gamification/achievements/{user_id}")
async def get_achievements(user_id: str):
    """Obtener logros"""
    achievements = gamification_service.get_achievements(user_id)
    return {"user_id": user_id, "achievements": achievements}


@router.get("/gamification/leaderboard")
async def get_leaderboard(limit: int = 10):
    """Obtener leaderboard"""
    leaderboard = gamification_service.get_leaderboard(limit)
    return {"leaderboard": leaderboard}


@router.get("/gamification/badges/{user_id}")
async def get_badges(user_id: str):
    """Obtener badges"""
    badges = gamification_service.get_badges(user_id)
    return {"user_id": user_id, "badges": badges}


@router.post("/marketplace/listings")
async def create_listing(
    store_id: str,
    seller_id: str,
    price: float,
    title: str,
    description: str,
    category: Optional[str] = None,
    tags: Optional[list] = None
):
    """Crear listing"""
    listing = marketplace_service.create_listing(
        store_id, seller_id, price, title, description, category, tags
    )
    return listing


@router.post("/marketplace/listings/{listing_id}/publish")
async def publish_listing(listing_id: str):
    """Publicar listing"""
    published = marketplace_service.publish_listing(listing_id)
    
    if not published:
        raise HTTPException(status_code=404, detail="Listing no encontrado")
    
    return {"message": "Listing publicado", "listing_id": listing_id}


@router.get("/marketplace/listings")
async def get_listings(
    category: Optional[str] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    tags: Optional[list] = None,
    sort_by: str = "recent"
):
    """Obtener listings"""
    listings = marketplace_service.get_listings(category, min_price, max_price, tags, sort_by)
    return {"listings": listings}


@router.get("/marketplace/listings/{listing_id}")
async def get_listing_details(listing_id: str):
    """Obtener detalles de listing"""
    listing = marketplace_service.get_listing_details(listing_id)
    
    if not listing:
        raise HTTPException(status_code=404, detail="Listing no encontrado")
    
    return listing


@router.post("/marketplace/listings/{listing_id}/purchase")
async def purchase_listing(
    listing_id: str,
    buyer_id: str
):
    """Comprar listing"""
    try:
        purchase = marketplace_service.purchase_listing(listing_id, buyer_id)
        return purchase
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/marketplace/listings/{listing_id}/review")
async def add_review(
    listing_id: str,
    reviewer_id: str,
    rating: int,
    comment: str
):
    """Agregar review"""
    try:
        review = marketplace_service.add_review(listing_id, reviewer_id, rating, comment)
        return review
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/crm/contacts")
async def create_contact(
    name: str,
    email: str,
    phone: Optional[str] = None,
    company: Optional[str] = None,
    source: Optional[str] = None
):
    """Crear contacto"""
    contact = crm_service.create_contact(name, email, phone, company, source)
    return contact


@router.get("/crm/contacts")
async def get_contacts(status: Optional[str] = None):
    """Obtener contactos"""
    status_enum = None
    if status:
        try:
            status_enum = ContactStatus(status)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Status inválido: {status}")
    
    contacts = crm_service.get_contacts(status_enum)
    return {"contacts": contacts}


@router.post("/crm/deals")
async def create_deal(
    contact_id: str,
    title: str,
    value: float,
    expected_close_date: Optional[str] = None
):
    """Crear deal"""
    deal = crm_service.create_deal(contact_id, title, value, expected_close_date)
    return deal


@router.get("/crm/deals")
async def get_deals(stage: Optional[str] = None):
    """Obtener deals"""
    deals = crm_service.get_deals(stage)
    return {"deals": deals}


@router.get("/crm/pipeline")
async def get_pipeline():
    """Obtener pipeline"""
    pipeline = crm_service.get_pipeline()
    return pipeline


@router.post("/crm/sync/{store_id}")
async def sync_with_crm(
    store_id: str,
    contact_id: str
):
    """Sincronizar diseño con CRM"""
    try:
        sync = crm_service.sync_with_design(store_id, contact_id)
        return sync
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/loyalty/enroll/{user_id}")
async def enroll_loyalty(user_id: str, tier: str = "bronze"):
    """Inscribir en programa de lealtad"""
    member = loyalty_service.enroll_member(user_id, tier)
    return member


@router.get("/loyalty/{user_id}")
async def get_loyalty_stats(user_id: str):
    """Obtener estadísticas de lealtad"""
    stats = loyalty_service.get_loyalty_stats(user_id)
    return stats


@router.post("/loyalty/points/{user_id}")
async def add_loyalty_points(
    user_id: str,
    points: int,
    reason: str
):
    """Agregar puntos de lealtad"""
    result = loyalty_service.add_points(user_id, points, reason)
    return result


@router.get("/loyalty/rewards/{user_id}")
async def get_available_rewards(user_id: str):
    """Obtener recompensas disponibles"""
    rewards = loyalty_service.get_available_rewards(user_id)
    return {"user_id": user_id, "rewards": rewards}


@router.post("/loyalty/claim/{user_id}")
async def claim_reward(
    user_id: str,
    reward_id: str
):
    """Reclamar recompensa"""
    try:
        reward = loyalty_service.claim_reward(user_id, reward_id)
        return reward
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/loyalty/referral/{user_id}")
async def get_referral_code(user_id: str):
    """Obtener código de referido"""
    code = loyalty_service.get_referral_code(user_id)
    return {"user_id": user_id, "referral_code": code}


@router.post("/loyalty/referral")
async def process_referral(
    referrer_id: str,
    referred_id: str
):
    """Procesar referido"""
    result = loyalty_service.process_referral(referrer_id, referred_id)
    return result


@router.post("/competitor/monitoring/start")
async def start_competitor_monitoring(
    store_id: str,
    location: str,
    store_type: str,
    monitoring_frequency: str = "daily"
):
    """Iniciar monitoreo de competencia en tiempo real"""
    job = await realtime_competitor_service.start_monitoring(
        store_id, location, store_type, monitoring_frequency
    )
    return job


@router.get("/competitor/monitoring/{store_id}")
async def get_realtime_competitor_analysis(store_id: str):
    """Obtener análisis de competencia en tiempo real"""
    analysis = await realtime_competitor_service.get_realtime_analysis(store_id)
    return analysis


@router.get("/competitor/monitoring/{store_id}/history")
async def get_monitoring_history(store_id: str):
    """Obtener historial de monitoreo"""
    history = realtime_competitor_service.get_monitoring_history(store_id)
    return {"store_id": store_id, "history": history}


@router.get("/competitor/compare/{store_id}")
async def compare_with_competitors(store_id: str):
    """Comparar diseño con competidores"""
    design = storage_service.load_design(store_id)
    if not design:
        raise HTTPException(status_code=404, detail="Diseño no encontrado")
    
    comparison = await realtime_competitor_service.compare_with_competitors(
        store_id, design.dict()
    )
    return comparison

