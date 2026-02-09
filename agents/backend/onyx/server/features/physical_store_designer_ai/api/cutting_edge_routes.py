"""
Cutting Edge Routes - Endpoints para tecnologías de vanguardia
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Optional, List
from datetime import datetime
import logging

from ..services.advanced_ml_service import AdvancedMLService
from ..services.voice_assistant_service import VoiceAssistantService, VoicePlatform
from ..services.mixed_reality_service import MixedRealityService
from ..services.realtime_market_analysis_service import RealtimeMarketAnalysisService
from ..services.collaborative_filtering_service import CollaborativeFilteringService
from ..services.smart_energy_service import SmartEnergyService
from ..services.storage_service import StorageService
from ..services.auth_service import AuthService

logger = logging.getLogger(__name__)

router = APIRouter()

# Inicializar servicios
ml_service = AdvancedMLService()
voice_service = VoiceAssistantService()
mr_service = MixedRealityService()
market_service = RealtimeMarketAnalysisService()
collab_service = CollaborativeFilteringService()
energy_service = SmartEnergyService()
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


@router.post("/ml/train")
async def train_model(
    model_name: str,
    model_type: str,
    training_data: List[dict],
    parameters: Optional[dict] = None
):
    """Entrenar modelo ML personalizado"""
    model = await ml_service.train_custom_model(model_name, model_type, training_data, parameters)
    return model


@router.post("/ml/predict/{model_id}")
async def predict_with_model(
    model_id: str,
    input_data: dict
):
    """Hacer predicción con modelo"""
    try:
        prediction = await ml_service.predict_with_model(model_id, input_data)
        return prediction
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/ml/insights/{store_id}")
async def generate_ml_insights(
    store_id: str,
    data_type: str = "sales"
):
    """Generar insights usando ML"""
    insights = await ml_service.generate_insights_with_ml(store_id, data_type)
    return insights


@router.post("/voice/skills/{store_id}")
async def create_voice_skill(
    store_id: str,
    skill_name: str,
    platform: str,
    intents: List[dict]
):
    """Crear skill de voz"""
    try:
        platform_enum = VoicePlatform(platform)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Plataforma inválida: {platform}")
    
    skill = voice_service.create_voice_skill(store_id, skill_name, platform_enum, intents)
    return skill


@router.post("/voice/command/{skill_id}")
async def process_voice_command(
    skill_id: str,
    command: str,
    user_id: Optional[str] = None
):
    """Procesar comando de voz"""
    try:
        interaction = voice_service.process_voice_command(skill_id, command, user_id)
        return interaction
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/voice/analytics/{skill_id}")
async def get_voice_analytics(
    skill_id: str,
    days: int = 30
):
    """Obtener analytics de voz"""
    analytics = voice_service.get_voice_analytics(skill_id, days)
    return analytics


@router.post("/mr/experience/{store_id}")
async def create_mr_experience(
    store_id: str,
    experience_name: str,
    description: str,
    mr_type: str = "hololens"
):
    """Crear experiencia de realidad mixta"""
    experience = mr_service.create_mr_experience(store_id, experience_name, description, mr_type)
    return experience


@router.get("/mr/showroom/{store_id}")
async def create_virtual_showroom(store_id: str):
    """Crear showroom virtual en MR"""
    design = storage_service.load_design(store_id)
    if not design:
        raise HTTPException(status_code=404, detail="Diseño no encontrado")
    
    showroom = mr_service.create_virtual_showroom(store_id, design.dict())
    return showroom


@router.post("/mr/session/{experience_id}")
async def start_mr_session(
    experience_id: str,
    user_id: str
):
    """Iniciar sesión MR"""
    session = mr_service.start_mr_session(experience_id, user_id)
    return session


@router.post("/market/data/{store_id}")
async def record_market_data(
    store_id: str,
    data_type: str,
    value: dict,
    metadata: Optional[dict] = None
):
    """Registrar dato de mercado"""
    data = market_service.record_market_data(store_id, data_type, value, metadata)
    return data


@router.get("/market/trends/{store_id}")
async def analyze_market_trends(
    store_id: str,
    category: str
):
    """Analizar tendencias de mercado"""
    trends = await market_service.analyze_market_trends(store_id, category)
    return trends


@router.get("/market/intelligence/{store_id}")
async def get_market_intelligence(store_id: str):
    """Obtener inteligencia de mercado"""
    intelligence = await market_service.get_market_intelligence(store_id)
    return intelligence


@router.post("/collaborative/preference")
async def record_preference(
    user_id: str,
    item_id: str,
    rating: float,
    item_type: str = "design"
):
    """Registrar preferencia de usuario"""
    preference = collab_service.record_preference(user_id, item_id, rating, item_type)
    return preference


@router.get("/collaborative/recommend/{user_id}")
async def get_recommendations(
    user_id: str,
    limit: int = 10
):
    """Obtener recomendaciones colaborativas"""
    recommendations = collab_service.recommend_items(user_id, limit)
    return {"user_id": user_id, "recommendations": recommendations}


@router.get("/collaborative/similar/{user_id}")
async def find_similar_users(
    user_id: str,
    limit: int = 10
):
    """Encontrar usuarios similares"""
    similar = collab_service.find_similar_users(user_id, limit)
    return {"user_id": user_id, "similar_users": similar}


@router.post("/energy/devices/{store_id}")
async def register_energy_device(
    store_id: str,
    device_name: str,
    device_type: str,
    power_rating_watts: float,
    location: str
):
    """Registrar dispositivo de energía"""
    device = energy_service.register_energy_device(
        store_id, device_name, device_type, power_rating_watts, location
    )
    return device


@router.post("/energy/consumption/{device_id}")
async def record_consumption(
    device_id: str,
    energy_kwh: float,
    timestamp: Optional[str] = None
):
    """Registrar consumo de energía"""
    consumption = energy_service.record_consumption(device_id, energy_kwh, timestamp)
    return consumption


@router.get("/energy/usage/{store_id}")
async def calculate_energy_usage(
    store_id: str,
    days: int = 30
):
    """Calcular uso de energía"""
    usage = energy_service.calculate_energy_usage(store_id, days)
    return usage


@router.get("/energy/optimize/{store_id}")
async def generate_energy_optimization(store_id: str):
    """Generar optimización de energía"""
    optimization = energy_service.generate_energy_optimization(store_id)
    return optimization




