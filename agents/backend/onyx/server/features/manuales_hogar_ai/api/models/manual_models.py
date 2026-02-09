"""
Manual API Models
================

Modelos Pydantic para endpoints de manuales.
"""

from typing import Optional
from pydantic import BaseModel, Field


class ManualTextRequest(BaseModel):
    """Request para generar manual desde texto."""
    problem_description: str = Field(..., description="Descripción del problema")
    category: str = Field(
        default="general",
        description="Categoría: plomeria, techos, carpinteria, electricidad, albanileria, pintura, herreria, jardineria, general"
    )
    model: Optional[str] = Field(None, description="Modelo de IA a usar (opcional)")
    include_safety: bool = Field(True, description="Incluir advertencias de seguridad")
    include_tools: bool = Field(True, description="Incluir lista de herramientas")
    include_materials: bool = Field(True, description="Incluir lista de materiales")


class ManualResponse(BaseModel):
    """Response del manual generado."""
    success: bool
    manual: Optional[str] = None
    category: Optional[str] = None
    model_used: Optional[str] = None
    tokens_used: Optional[int] = None
    format: str = "lego"
    image_analysis: Optional[str] = None
    detected_category: Optional[str] = None
    images_count: Optional[int] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Response de health check."""
    status: str
    healthy: bool
    timestamp: Optional[str] = None
    error: Optional[str] = None

