"""
Schemas de requests para la API
"""

from pydantic import BaseModel, HttpUrl, Field, field_validator
from typing import Optional, List

from .constants import (
    DEFAULT_MODEL,
    DEFAULT_MAX_TOKENS,
    MIN_MAX_TOKENS,
    MAX_MAX_TOKENS,
    EXTRACT_STRATEGIES,
    DEFAULT_EXTRACT_STRATEGY,
    DEFAULT_MAX_CONCURRENT,
    MIN_MAX_CONCURRENT,
    MAX_MAX_CONCURRENT,
    MAX_BATCH_URLS,
    MIN_BATCH_URLS,
)


class BaseExtractionRequest(BaseModel):
    """
    Clase base para requests de extracción.
    Contiene campos comunes compartidos entre ExtractContentRequest y BatchExtractRequest.
    """
    use_javascript: Optional[bool] = Field(
        default=False,
        description="Si True, usa Playwright para renderizar JavaScript (más lento pero más completo)"
    )
    extract_strategy: Optional[str] = Field(
        default=DEFAULT_EXTRACT_STRATEGY,
        description=f"Estrategia de extracción. Opciones: {', '.join(EXTRACT_STRATEGIES)}"
    )

    @field_validator('extract_strategy')
    @classmethod
    def validate_extract_strategy(cls, v: Optional[str]) -> Optional[str]:
        """Valida que la estrategia de extracción sea válida."""
        if v is not None and v not in EXTRACT_STRATEGIES:
            raise ValueError(
                f"extract_strategy debe ser uno de: {', '.join(EXTRACT_STRATEGIES)}"
            )
        return v


class ExtractContentRequest(BaseExtractionRequest):
    """Request para extraer contenido de una página web"""
    url: HttpUrl = Field(..., description="URL de la página web a extraer")
    model: Optional[str] = Field(
        default=DEFAULT_MODEL,
        description="Modelo de OpenRouter a usar"
    )
    max_tokens: Optional[int] = Field(
        default=DEFAULT_MAX_TOKENS,
        ge=MIN_MAX_TOKENS,
        le=MAX_MAX_TOKENS,
        description=f"Máximo de tokens en la respuesta (entre {MIN_MAX_TOKENS} y {MAX_MAX_TOKENS})"
    )


class BatchExtractRequest(BaseExtractionRequest):
    """Request para extraer contenido de múltiples páginas web"""
    urls: List[HttpUrl] = Field(
        ...,
        description="Lista de URLs a extraer",
        min_length=MIN_BATCH_URLS,
        max_length=MAX_BATCH_URLS
    )
    max_concurrent: Optional[int] = Field(
        default=DEFAULT_MAX_CONCURRENT,
        ge=MIN_MAX_CONCURRENT,
        le=MAX_MAX_CONCURRENT,
        description=f"Máximo de requests concurrentes (entre {MIN_MAX_CONCURRENT} y {MAX_MAX_CONCURRENT})"
    )
