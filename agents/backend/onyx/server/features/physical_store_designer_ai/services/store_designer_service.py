"""
Store Designer Service - Servicio principal que orquesta todo el diseño
"""

import uuid
from typing import Optional
from datetime import datetime

from ..core.models import (
    StoreDesignRequest,
    StoreDesign,
    StoreLayout,
    StoreType,
    DesignStyle
)
from ..core.logging_config import get_logger
from .visualization_service import VisualizationService
from .marketing_service import MarketingService
from .decoration_service import DecorationService
from .competitor_analysis_service import CompetitorAnalysisService
from .financial_analysis_service import FinancialAnalysisService
from .inventory_service import InventoryService
from .metrics_service import MetricsService

logger = get_logger(__name__)


class StoreDesignerService:
    """Servicio principal para generar diseños completos de locales"""
    
    def __init__(
        self,
        visualization_service: Optional[VisualizationService] = None,
        marketing_service: Optional[MarketingService] = None,
        decoration_service: Optional[DecorationService] = None,
        competitor_analysis_service: Optional[CompetitorAnalysisService] = None,
        financial_analysis_service: Optional[FinancialAnalysisService] = None,
        inventory_service: Optional[InventoryService] = None,
        metrics_service: Optional[MetricsService] = None
    ):
        self.visualization_service = visualization_service or VisualizationService()
        self.marketing_service = marketing_service or MarketingService()
        self.decoration_service = decoration_service or DecorationService()
        self.competitor_analysis_service = competitor_analysis_service or CompetitorAnalysisService()
        self.financial_analysis_service = financial_analysis_service or FinancialAnalysisService()
        self.inventory_service = inventory_service or InventoryService()
        self.metrics_service = metrics_service or MetricsService()
    
    async def generate_store_design(
        self,
        request: StoreDesignRequest
    ) -> StoreDesign:
        """Generar diseño completo del local"""
        
        logger.info(
            f"Generando diseño para tienda: {request.store_name}",
            extra={
                "store_name": request.store_name,
                "store_type": request.store_type.value,
                "style": request.style_preference.value if request.style_preference else None
            }
        )
        
        # Determinar estilo
        style = request.style_preference or DesignStyle.MODERN
        
        # Generar layout
        layout = self._generate_layout(request)
        
        # Generar visualizaciones
        visualizations = self.visualization_service.generate_store_visualization(
            store_type=request.store_type,
            style=style,
            store_name=request.store_name,
            additional_info=request.additional_info
        )
        
        # Generar plan de marketing
        marketing_plan = await self.marketing_service.generate_marketing_plan(
            store_type=request.store_type,
            store_name=request.store_name,
            target_audience=request.target_audience,
            location=request.location
        )
        
        # Generar plan de decoración
        decoration_plan = self.decoration_service.generate_decoration_plan(
            store_type=request.store_type,
            style=style,
            dimensions=request.dimensions,
            budget_range=request.budget_range
        )
        
        # Generar análisis adicionales
        competitor_analysis = await self.competitor_analysis_service.analyze_competitors(
            store_type=request.store_type,
            location=request.location,
            store_name=request.store_name
        )
        
        financial_analysis = self.financial_analysis_service.generate_financial_analysis(
            store_type=request.store_type,
            decoration_plan=decoration_plan,
            location=request.location
        )
        
        inventory_recommendations = self.inventory_service.generate_inventory_recommendations(
            store_type=request.store_type,
            store_size=request.dimensions
        )
        
        kpis = self.metrics_service.generate_kpis(
            store_type=request.store_type,
            financial_analysis=financial_analysis
        )
        
        # Crear diseño completo
        store_id = str(uuid.uuid4())
        description = self._generate_description(request, style)
        
        design = StoreDesign(
            store_id=store_id,
            store_name=request.store_name,
            store_type=request.store_type,
            style=style,
            layout=layout,
            visualizations=visualizations,
            marketing_plan=marketing_plan,
            decoration_plan=decoration_plan,
            description=description,
            competitor_analysis=competitor_analysis,
            financial_analysis=financial_analysis,
            inventory_recommendations=inventory_recommendations,
            kpis=kpis
        )
        
        logger.info(
            f"Diseño generado exitosamente: {store_id}",
            extra={
                "store_id": store_id,
                "store_name": request.store_name,
                "store_type": request.store_type.value,
                "has_competitor_analysis": competitor_analysis is not None,
                "has_financial_analysis": financial_analysis is not None
            }
        )
        return design
    
    def _generate_layout(self, request: StoreDesignRequest) -> StoreLayout:
        """Generar layout del local"""
        
        # Dimensiones por defecto si no se proporcionan
        dimensions = request.dimensions or {
            "width": 10.0,
            "length": 15.0,
            "height": 3.0
        }
        
        # Zonas básicas según tipo de tienda
        zones = []
        if request.store_type == StoreType.RESTAURANT:
            zones = [
                {"name": "Área de comedor", "size": "60%", "purpose": "Mesas y sillas"},
                {"name": "Cocina", "size": "25%", "purpose": "Preparación de alimentos"},
                {"name": "Recepción", "size": "10%", "purpose": "Entrada y caja"},
                {"name": "Almacén", "size": "5%", "purpose": "Almacenamiento"}
            ]
        elif request.store_type == StoreType.CAFE:
            zones = [
                {"name": "Área de clientes", "size": "70%", "purpose": "Mesas y asientos"},
                {"name": "Barra de café", "size": "20%", "purpose": "Preparación y servicio"},
                {"name": "Almacén", "size": "10%", "purpose": "Almacenamiento"}
            ]
        elif request.store_type == StoreType.BOUTIQUE:
            zones = [
                {"name": "Área de exhibición", "size": "60%", "purpose": "Productos"},
                {"name": "Probadores", "size": "20%", "purpose": "Vestidores"},
                {"name": "Caja", "size": "10%", "purpose": "Pago"},
                {"name": "Almacén", "size": "10%", "purpose": "Inventario"}
            ]
        else:
            zones = [
                {"name": "Área de ventas", "size": "70%", "purpose": "Exhibición de productos"},
                {"name": "Caja", "size": "15%", "purpose": "Punto de pago"},
                {"name": "Almacén", "size": "15%", "purpose": "Inventario"}
            ]
        
        return StoreLayout(
            dimensions=dimensions,
            zones=zones,
            furniture_placement=[],
            traffic_flow={
                "entrance": "Entrada principal",
                "exit": "Salida cerca de caja",
                "flow": "Flujo circular para maximizar exposición"
            },
            accessibility={
                "ramp": "Rampa de acceso si es necesario",
                "width": "Pasillos de mínimo 1.2m",
                "restroom": "Baños accesibles"
            }
        )
    
    def _generate_description(
        self,
        request: StoreDesignRequest,
        style: DesignStyle
    ) -> str:
        """Generar descripción del diseño"""
        
        style_names = {
            DesignStyle.MODERN: "moderno",
            DesignStyle.CLASSIC: "clásico",
            DesignStyle.MINIMALIST: "minimalista",
            DesignStyle.INDUSTRIAL: "industrial",
            DesignStyle.RUSTIC: "rústico",
            DesignStyle.LUXURY: "de lujo",
            DesignStyle.ECO_FRIENDLY: "ecológico",
            DesignStyle.VINTAGE: "vintage"
        }
        
        return (
            f"Diseño {style_names.get(style, 'moderno')} para {request.store_name}, "
            f"una {request.store_type.value} que combina funcionalidad y estética. "
            f"El diseño incluye un layout optimizado, plan de marketing completo, "
            f"y decoración detallada para crear una experiencia única para los clientes."
        )

