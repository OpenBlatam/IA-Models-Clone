"""
Analysis Routes - Endpoints para análisis adicionales
"""

from fastapi import APIRouter, HTTPException
from typing import Optional
import logging

from ..core.models import StoreType
from ..services.competitor_analysis_service import CompetitorAnalysisService
from ..services.financial_analysis_service import FinancialAnalysisService
from ..services.inventory_service import InventoryService
from ..services.metrics_service import MetricsService
from ..services.storage_service import StorageService

logger = logging.getLogger(__name__)

router = APIRouter()

# Inicializar servicios
competitor_service = CompetitorAnalysisService()
financial_service = FinancialAnalysisService()
inventory_service = InventoryService()
metrics_service = MetricsService()
storage_service = StorageService()


@router.get("/analysis/competitor/{store_id}")
async def get_competitor_analysis(store_id: str):
    """Obtener análisis de competencia para un diseño"""
    design = storage_service.load_design(store_id)
    if not design:
        raise HTTPException(status_code=404, detail="Diseño no encontrado")
    
    analysis = await competitor_service.analyze_competitors(
        store_type=design.store_type,
        location=None,  # Podría extraerse del diseño
        store_name=design.store_name
    )
    
    return {
        "store_id": store_id,
        "store_name": design.store_name,
        "analysis": analysis
    }


@router.get("/analysis/financial/{store_id}")
async def get_financial_analysis(store_id: str):
    """Obtener análisis financiero para un diseño"""
    design = storage_service.load_design(store_id)
    if not design:
        raise HTTPException(status_code=404, detail="Diseño no encontrado")
    
    analysis = financial_service.generate_financial_analysis(
        store_type=design.store_type,
        decoration_plan=design.decoration_plan,
        location=None
    )
    
    return {
        "store_id": store_id,
        "store_name": design.store_name,
        "analysis": analysis
    }


@router.get("/analysis/inventory/{store_id}")
async def get_inventory_recommendations(store_id: str):
    """Obtener recomendaciones de inventario para un diseño"""
    design = storage_service.load_design(store_id)
    if not design:
        raise HTTPException(status_code=404, detail="Diseño no encontrado")
    
    recommendations = inventory_service.generate_inventory_recommendations(
        store_type=design.store_type,
        store_size=design.layout.dimensions
    )
    
    return {
        "store_id": store_id,
        "store_name": design.store_name,
        "recommendations": recommendations
    }


@router.get("/analysis/kpis/{store_id}")
async def get_kpis(store_id: str):
    """Obtener KPIs y métricas para un diseño"""
    design = storage_service.load_design(store_id)
    if not design:
        raise HTTPException(status_code=404, detail="Diseño no encontrado")
    
    kpis = metrics_service.generate_kpis(
        store_type=design.store_type,
        financial_analysis=design.financial_analysis
    )
    
    dashboard = metrics_service.generate_dashboard_metrics(
        store_type=design.store_type
    )
    
    return {
        "store_id": store_id,
        "store_name": design.store_name,
        "kpis": kpis,
        "dashboard": dashboard
    }


@router.get("/analysis/full/{store_id}")
async def get_full_analysis(store_id: str):
    """Obtener análisis completo (competencia, financiero, inventario, KPIs)"""
    design = storage_service.load_design(store_id)
    if not design:
        raise HTTPException(status_code=404, detail="Diseño no encontrado")
    
    # Si el diseño ya tiene análisis, devolverlos
    if all([
        design.competitor_analysis,
        design.financial_analysis,
        design.inventory_recommendations,
        design.kpis
    ]):
        return {
            "store_id": store_id,
            "store_name": design.store_name,
            "competitor_analysis": design.competitor_analysis,
            "financial_analysis": design.financial_analysis,
            "inventory_recommendations": design.inventory_recommendations,
            "kpis": design.kpis
        }
    
    # Generar análisis si no existen
    competitor_analysis = await competitor_service.analyze_competitors(
        store_type=design.store_type,
        location=None,
        store_name=design.store_name
    )
    
    financial_analysis = financial_service.generate_financial_analysis(
        store_type=design.store_type,
        decoration_plan=design.decoration_plan,
        location=None
    )
    
    inventory_recommendations = inventory_service.generate_inventory_recommendations(
        store_type=design.store_type,
        store_size=design.layout.dimensions
    )
    
    kpis = metrics_service.generate_kpis(
        store_type=design.store_type,
        financial_analysis=financial_analysis
    )
    
    return {
        "store_id": store_id,
        "store_name": design.store_name,
        "competitor_analysis": competitor_analysis,
        "financial_analysis": financial_analysis,
        "inventory_recommendations": inventory_recommendations,
        "kpis": kpis
    }

