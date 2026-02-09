"""
Advanced Routes - Endpoints para funcionalidades avanzadas
"""

from fastapi import APIRouter, HTTPException
from typing import List, Optional, Dict, Any

from ..services.design_comparison_service import DesignComparisonService
from ..services.technical_plans_service import TechnicalPlansService
from ..services.feedback_service import FeedbackService
from ..services.recommendation_service import RecommendationService
from ..services.location_analysis_service import LocationAnalysisService
from ..services.versioning_service import VersioningService
from ..services.storage_service import StorageService
from ..core.route_helpers import handle_route_errors, track_route_metrics

router = APIRouter()

# Inicializar servicios
comparison_service = DesignComparisonService()
technical_service = TechnicalPlansService()
feedback_service = FeedbackService()
recommendation_service = RecommendationService()
location_service = LocationAnalysisService()
versioning_service = VersioningService()
storage_service = StorageService()


@router.post("/compare/designs")
@handle_route_errors
@track_route_metrics("advanced.compare_designs")
async def compare_designs(store_ids: List[str], criteria: Optional[List[str]] = None):
    """Comparar múltiples diseños"""
    designs = []
    for store_id in store_ids:
        design = storage_service.load_design(store_id)
        if not design:
            raise HTTPException(status_code=404, detail=f"Diseño {store_id} no encontrado")
        designs.append(design)
    
    return comparison_service.compare_designs(designs, criteria)


@router.get("/technical-plans/{store_id}")
@handle_route_errors
@track_route_metrics("advanced.get_technical_plans")
async def get_technical_plans(store_id: str):
    """Obtener planos técnicos detallados"""
    design = storage_service.load_design(store_id)
    if not design:
        raise HTTPException(status_code=404, detail="Diseño no encontrado")
    
    plans = technical_service.generate_technical_plan(
        layout=design.layout,
        store_type=design.store_type,
        style=design.style
    )
    
    return {
        "store_id": store_id,
        "store_name": design.store_name,
        "technical_plans": plans
    }


@router.post("/feedback/{store_id}")
@handle_route_errors
@track_route_metrics("advanced.add_feedback")
async def add_feedback(
    store_id: str,
    feedback_type: str,
    content: str,
    rating: Optional[int] = None,
    user_id: Optional[str] = None
):
    """Agregar feedback a un diseño"""
    design = storage_service.load_design(store_id)
    if not design:
        raise HTTPException(status_code=404, detail="Diseño no encontrado")
    
    return feedback_service.add_feedback(
        store_id=store_id,
        feedback_type=feedback_type,
        content=content,
        rating=rating,
        user_id=user_id
    )


@router.get("/feedback/{store_id}")
@handle_route_errors
@track_route_metrics("advanced.get_feedback")
async def get_feedback(store_id: str):
    """Obtener feedback de un diseño"""
    feedback_list = feedback_service.get_feedback(store_id)
    
    design = storage_service.load_design(store_id)
    if not design:
        raise HTTPException(status_code=404, detail="Diseño no encontrado")
    
    suggestions = feedback_service.generate_improvement_suggestions(design, feedback_list)
    
    return {
        "store_id": store_id,
        "feedback": feedback_list,
        "improvement_suggestions": suggestions
    }


@router.get("/recommendations/{store_id}")
@handle_route_errors
@track_route_metrics("advanced.get_recommendations")
async def get_recommendations(store_id: str):
    """Obtener recomendaciones inteligentes"""
    design = storage_service.load_design(store_id)
    if not design:
        raise HTTPException(status_code=404, detail="Diseño no encontrado")
    
    recommendations = recommendation_service.generate_recommendations(design)
    
    return {
        "store_id": store_id,
        "store_name": design.store_name,
        "recommendations": recommendations
    }


@router.get("/location/analyze")
@handle_route_errors
@track_route_metrics("advanced.analyze_location")
async def analyze_location(location: str, store_type: str):
    """Analizar ubicación"""
    from ..core.models import StoreType
    
    try:
        store_type_enum = StoreType(store_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Tipo de tienda inválido: {store_type}")
    
    analysis = await location_service.analyze_location(location, store_type_enum)
    
    return {
        "location": location,
        "store_type": store_type,
        "analysis": analysis
    }


@router.post("/versions/{store_id}")
@handle_route_errors
@track_route_metrics("advanced.create_version")
async def create_version(
    store_id: str,
    changes: dict,
    version_notes: Optional[str] = None
):
    """Crear nueva versión de un diseño"""
    design = storage_service.load_design(store_id)
    if not design:
        raise HTTPException(status_code=404, detail="Diseño no encontrado")
    
    return versioning_service.create_version(
        original_design=design,
        changes=changes,
        version_notes=version_notes
    )


@router.get("/versions/{store_id}")
@handle_route_errors
@track_route_metrics("advanced.get_versions")
async def get_versions(store_id: str):
    """Obtener todas las versiones de un diseño"""
    return versioning_service.get_version_history(store_id)


@router.get("/versions/{store_id}/compare")
@handle_route_errors
@track_route_metrics("advanced.compare_versions")
async def compare_versions(store_id: str, version1: int, version2: int):
    """Comparar dos versiones"""
    return versioning_service.compare_versions(store_id, version1, version2)


@router.post("/versions/{store_id}/approve")
@handle_route_errors
@track_route_metrics("advanced.approve_version")
async def approve_version(store_id: str, version_number: int):
    """Aprobar una versión"""
    approved = versioning_service.approve_version(store_id, version_number)
    if not approved:
        raise HTTPException(status_code=404, detail="Versión no encontrada")
    
    return {
        "message": "Versión aprobada",
        "store_id": store_id,
        "version_number": version_number
    }




