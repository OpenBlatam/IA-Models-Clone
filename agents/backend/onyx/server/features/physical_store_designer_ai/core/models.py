"""
Data models for Physical Store Designer AI
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum


class StoreType(str, Enum):
    """Tipos de tienda"""
    RETAIL = "retail"
    RESTAURANT = "restaurant"
    CAFE = "cafe"
    BOUTIQUE = "boutique"
    SUPERMARKET = "supermarket"
    PHARMACY = "pharmacy"
    ELECTRONICS = "electronics"
    CLOTHING = "clothing"
    FURNITURE = "furniture"
    OTHER = "other"


class DesignStyle(str, Enum):
    """Estilos de diseño"""
    MODERN = "modern"
    CLASSIC = "classic"
    MINIMALIST = "minimalist"
    INDUSTRIAL = "industrial"
    RUSTIC = "rustic"
    LUXURY = "luxury"
    ECO_FRIENDLY = "eco_friendly"
    VINTAGE = "vintage"


class ChatMessage(BaseModel):
    """Mensaje del chat"""
    role: str = Field(..., description="Rol: 'user' o 'assistant'")
    content: str = Field(..., min_length=1, max_length=10000, description="Contenido del mensaje")
    timestamp: datetime = Field(default_factory=datetime.now)
    
    @validator("role")
    def validate_role(cls, v):
        """Validar rol del mensaje"""
        if v not in ["user", "assistant"]:
            raise ValueError("role debe ser 'user' o 'assistant'")
        return v


class ChatSession(BaseModel):
    """Sesión de chat"""
    session_id: str
    messages: List[ChatMessage] = Field(default_factory=list)
    store_info: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class StoreLayout(BaseModel):
    """Layout del local"""
    dimensions: Dict[str, float] = Field(..., description="Dimensiones en metros: {width, length, height}")
    zones: List[Dict[str, Any]] = Field(default_factory=list, description="Zonas del local")
    furniture_placement: List[Dict[str, Any]] = Field(default_factory=list, description="Ubicación de muebles")
    traffic_flow: Dict[str, Any] = Field(default_factory=dict, description="Flujo de tráfico")
    accessibility: Dict[str, Any] = Field(default_factory=dict, description="Accesibilidad")
    
    @validator("dimensions")
    def validate_dimensions(cls, v):
        """Validar dimensiones"""
        required_keys = {"width", "length", "height"}
        if not all(key in v for key in required_keys):
            raise ValueError(f"dimensions debe incluir: {required_keys}")
        for key, value in v.items():
            if value <= 0:
                raise ValueError(f"dimension {key} debe ser mayor a 0")
            if value > 1000:  # Límite razonable de 1000 metros
                raise ValueError(f"dimension {key} no puede ser mayor a 1000 metros")
        return v


class StoreVisualization(BaseModel):
    """Visualización del local"""
    image_url: Optional[str] = Field(None, description="URL de imagen generada")
    image_prompt: str = Field(..., min_length=10, max_length=2000, description="Prompt usado para generar la imagen")
    view_type: str = Field(..., description="Tipo de vista: 'exterior', 'interior', 'layout'")
    style: DesignStyle = Field(..., description="Estilo de diseño aplicado")
    
    @validator("view_type")
    def validate_view_type(cls, v):
        """Validar tipo de vista"""
        valid_types = ["exterior", "interior", "layout"]
        if v not in valid_types:
            raise ValueError(f"view_type debe ser uno de: {valid_types}")
        return v


class MarketingPlan(BaseModel):
    """Plan de marketing y ventas"""
    target_audience: str = Field(..., description="Audiencia objetivo")
    marketing_strategy: List[str] = Field(default_factory=list, description="Estrategias de marketing")
    sales_tactics: List[str] = Field(default_factory=list, description="Tácticas de ventas")
    pricing_strategy: str = Field(..., description="Estrategia de precios")
    promotion_ideas: List[str] = Field(default_factory=list, description="Ideas de promoción")
    social_media_plan: Dict[str, Any] = Field(default_factory=dict, description="Plan de redes sociales")
    opening_strategy: str = Field(..., description="Estrategia de apertura")


class DecorationPlan(BaseModel):
    """Plan de decoración"""
    color_scheme: Dict[str, str] = Field(..., description="Esquema de colores")
    lighting_plan: Dict[str, Any] = Field(default_factory=dict, description="Plan de iluminación")
    furniture_recommendations: List[Dict[str, Any]] = Field(default_factory=list, description="Recomendaciones de muebles")
    decoration_elements: List[Dict[str, Any]] = Field(default_factory=list, description="Elementos decorativos")
    materials: List[str] = Field(default_factory=list, description="Materiales recomendados")
    budget_estimate: Dict[str, float] = Field(default_factory=dict, description="Estimación de presupuesto")


class StoreDesign(BaseModel):
    """Diseño completo del local"""
    store_id: str
    store_name: str
    store_type: StoreType
    style: DesignStyle
    
    layout: StoreLayout
    visualizations: List[StoreVisualization] = Field(default_factory=list)
    marketing_plan: MarketingPlan
    decoration_plan: DecorationPlan
    
    description: str = Field(..., description="Descripción del diseño")
    
    # Análisis adicionales (opcionales)
    competitor_analysis: Optional[Dict[str, Any]] = Field(None, description="Análisis de competencia")
    financial_analysis: Optional[Dict[str, Any]] = Field(None, description="Análisis financiero")
    inventory_recommendations: Optional[Dict[str, Any]] = Field(None, description="Recomendaciones de inventario")
    kpis: Optional[Dict[str, Any]] = Field(None, description="KPIs y métricas")
    
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class StoreDesignRequest(BaseModel):
    """Request para generar diseño de local"""
    store_name: str = Field(..., min_length=1, max_length=200, description="Nombre de la tienda")
    store_type: StoreType = Field(..., description="Tipo de tienda")
    style_preference: Optional[DesignStyle] = Field(None, description="Preferencia de estilo")
    budget_range: Optional[str] = Field(None, max_length=50, description="Rango de presupuesto")
    location: Optional[str] = Field(None, max_length=500, description="Ubicación del local")
    target_audience: Optional[str] = Field(None, max_length=500, description="Audiencia objetivo")
    special_requirements: Optional[str] = Field(None, max_length=2000, description="Requisitos especiales")
    dimensions: Optional[Dict[str, float]] = Field(None, description="Dimensiones del local")
    additional_info: Optional[str] = Field(None, max_length=5000, description="Información adicional")
    
    @validator("dimensions")
    def validate_dimensions(cls, v):
        """Validar dimensiones si se proporcionan"""
        if v is None:
            return v
        required_keys = {"width", "length", "height"}
        if not all(key in v for key in required_keys):
            raise ValueError(f"dimensions debe incluir: {required_keys}")
        for key, value in v.items():
            if value <= 0:
                raise ValueError(f"dimension {key} debe ser mayor a 0")
            if value > 1000:
                raise ValueError(f"dimension {key} no puede ser mayor a 1000 metros")
        return v
    
    @validator("budget_range")
    def validate_budget_range(cls, v):
        """Validar rango de presupuesto"""
        if v is None:
            return v
        valid_ranges = ["bajo", "medio", "alto", "premium", "low", "medium", "high"]
        if v.lower() not in valid_ranges:
            # Permitir otros valores pero advertir
            pass
        return v

