"""
Final Tech Routes - Endpoints para tecnologías finales avanzadas
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Optional, List
from datetime import datetime
import logging

from ..services.multimodal_ai_service import MultimodalAIService
from ..services.predictive_behavior_service import PredictiveBehaviorService
from ..services.smart_waste_service import SmartWasteService, WasteType
from ..services.traffic_flow_service import TrafficFlowService
from ..services.renewable_energy_service import RenewableEnergyService, RenewableSource
from ..services.hybrid_recommendations_service import HybridRecommendationsService
from ..services.storage_service import StorageService
from ..services.auth_service import AuthService

logger = logging.getLogger(__name__)

router = APIRouter()

# Inicializar servicios
multimodal_service = MultimodalAIService()
behavior_service = PredictiveBehaviorService()
waste_service = SmartWasteService()
traffic_service = TrafficFlowService()
renewable_service = RenewableEnergyService()
hybrid_recs_service = HybridRecommendationsService()
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


@router.post("/multimodal/generate/{store_id}")
async def generate_multimodal_content(
    store_id: str,
    prompt: str,
    input_type: str = "text",
    output_types: List[str] = ["text", "image"]
):
    """Generar contenido multimodal"""
    content = await multimodal_service.generate_multimodal_content(store_id, prompt, input_type, output_types)
    return content


@router.post("/multimodal/analyze")
async def analyze_multimodal_input(inputs: dict):
    """Analizar entrada multimodal"""
    analysis = await multimodal_service.analyze_multimodal_input(inputs)
    return analysis


@router.post("/behavior/record/{store_id}")
async def record_behavior(
    store_id: str,
    customer_id: str,
    behavior_type: str,
    data: dict
):
    """Registrar comportamiento"""
    behavior = behavior_service.record_behavior(store_id, customer_id, behavior_type, data)
    return behavior


@router.get("/behavior/predict/{customer_id}")
async def predict_customer_behavior(
    customer_id: str,
    time_horizon: str = "next_visit"
):
    """Predecir comportamiento del cliente"""
    prediction = await behavior_service.predict_customer_behavior(customer_id, time_horizon)
    return prediction


@router.get("/behavior/traffic/{store_id}")
async def predict_store_traffic(
    store_id: str,
    date: str
):
    """Predecir tráfico de la tienda"""
    prediction = await behavior_service.predict_store_traffic(store_id, date)
    return prediction


@router.post("/waste/bins/{store_id}")
async def register_waste_bin(
    store_id: str,
    bin_name: str,
    waste_type: str,
    location: str,
    capacity_liters: float,
    sensor_enabled: bool = True
):
    """Registrar contenedor de residuos"""
    try:
        waste_type_enum = WasteType(waste_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Tipo de residuo inválido: {waste_type}")
    
    bin_data = waste_service.register_waste_bin(
        store_id, bin_name, waste_type_enum, location, capacity_liters, sensor_enabled
    )
    return bin_data


@router.post("/waste/update-level/{bin_id}")
async def update_bin_level(
    bin_id: str,
    level_liters: float
):
    """Actualizar nivel del contenedor"""
    try:
        bin_data = waste_service.update_bin_level(bin_id, level_liters)
        return bin_data
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/waste/analytics/{store_id}")
async def get_waste_analytics(
    store_id: str,
    days: int = 30
):
    """Obtener analytics de residuos"""
    analytics = waste_service.get_waste_analytics(store_id, days)
    return analytics


@router.post("/traffic/record/{store_id}")
async def record_traffic_point(
    store_id: str,
    location: str,
    timestamp: Optional[str] = None,
    metadata: Optional[dict] = None
):
    """Registrar punto de tráfico"""
    point = traffic_service.record_traffic_point(store_id, location, timestamp, metadata)
    return point


@router.get("/traffic/analyze/{store_id}")
async def analyze_traffic_flow(
    store_id: str,
    hours: int = 24
):
    """Analizar flujo de tráfico"""
    analysis = traffic_service.analyze_traffic_flow(store_id, hours)
    return analysis


@router.get("/traffic/heatmap/{store_id}")
async def generate_flow_heatmap(
    store_id: str,
    hours: int = 24
):
    """Generar heatmap de flujo"""
    heatmap = traffic_service.generate_flow_heatmap(store_id, hours)
    return heatmap


@router.get("/traffic/optimize/{store_id}")
async def optimize_traffic_flow(store_id: str):
    """Optimizar flujo de tráfico"""
    optimization = traffic_service.optimize_traffic_flow(store_id)
    return optimization


@router.post("/renewable/install/{store_id}")
async def install_renewable_system(
    store_id: str,
    system_name: str,
    source: str,
    capacity_kw: float,
    installation_date: str
):
    """Instalar sistema de energía renovable"""
    try:
        source_enum = RenewableSource(source)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Fuente renovable inválida: {source}")
    
    system = renewable_service.install_renewable_system(
        store_id, system_name, source_enum, capacity_kw, installation_date
    )
    return system


@router.post("/renewable/generation/{system_id}")
async def record_generation(
    system_id: str,
    energy_kwh: float,
    timestamp: Optional[str] = None
):
    """Registrar generación de energía"""
    try:
        generation = renewable_service.record_generation(system_id, energy_kwh, timestamp)
        return generation
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/renewable/savings/{store_id}")
async def calculate_energy_savings(
    store_id: str,
    days: int = 30
):
    """Calcular ahorros de energía"""
    savings = renewable_service.calculate_energy_savings(store_id, days)
    return savings


@router.get("/renewable/credits/{store_id}")
async def generate_energy_credits(store_id: str):
    """Generar créditos de energía renovable"""
    credits = renewable_service.generate_energy_credits(store_id)
    return credits


@router.get("/recommendations/hybrid/{user_id}")
async def get_hybrid_recommendations(
    user_id: str,
    context: Optional[dict] = None
):
    """Obtener recomendaciones híbridas"""
    recommendations = await hybrid_recs_service.generate_hybrid_recommendations(user_id, context)
    return recommendations


@router.get("/recommendations/explain/{user_id}")
async def explain_recommendation(
    user_id: str,
    item_id: str
):
    """Explicar recomendación"""
    explanation = await hybrid_recs_service.explain_recommendation(user_id, item_id)
    return explanation




