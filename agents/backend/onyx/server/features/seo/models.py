from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

from typing import Any, List, Dict, Optional
import logging
import asyncio
class SEOScrapeRequest(BaseModel):
    url: str = Field(..., description="URL a analizar o scrapear", example="https://ejemplo.com")
    options: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Opciones adicionales para el scraping")

class SEOAnalysis(BaseModel):
    """Análisis SEO detallado de una página web"""
    title: Optional[str] = Field(None, description="Título de la página")
    meta_description: Optional[str] = Field(None, description="Meta descripción")
    h1_tags: List[str] = Field(default_factory=list, description="Tags H1 encontrados")
    h2_tags: List[str] = Field(default_factory=list, description="Tags H2 encontrados")
    h3_tags: List[str] = Field(default_factory=list, description="Tags H3 encontrados")
    images: List[Dict[str, str]] = Field(default_factory=list, description="Imágenes con alt text")
    links: List[Dict[str, str]] = Field(default_factory=list, description="Enlaces internos y externos")
    keywords: List[str] = Field(default_factory=list, description="Palabras clave identificadas")
    content_length: int = Field(0, description="Longitud del contenido en caracteres")
    load_time: Optional[float] = Field(None, description="Tiempo de carga en segundos")
    seo_score: Optional[int] = Field(None, description="Puntuación SEO (0-100)")
    recommendations: List[str] = Field(default_factory=list, description="Recomendaciones de mejora SEO")
    technical_issues: List[str] = Field(default_factory=list, description="Problemas técnicos identificados")
    mobile_friendly: Optional[bool] = Field(None, description="¿Es compatible con móviles?")
    page_speed: Optional[str] = Field(None, description="Velocidad de la página")
    social_media_tags: Dict[str, str] = Field(default_factory=dict, description="Tags de redes sociales")

class SEOScrapeResponse(BaseModel):
    success: bool = Field(..., example=True)
    data: Optional[SEOAnalysis] = Field(None, description="Datos extraídos o analizados")
    error: Optional[str] = Field(None, description="Mensaje de error si falla el scraping")
    raw_html: Optional[str] = Field(None, description="HTML raw de la página")
    analysis_summary: Optional[str] = Field(None, description="Resumen del análisis en lenguaje natural") 