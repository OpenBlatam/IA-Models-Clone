"""
Controller para extracción de contenido web
"""

import logging
from fastapi import APIRouter, Depends, Query
from ..schemas.requests import ExtractContentRequest, BatchExtractRequest
from ..schemas.responses import ExtractContentResponse, ErrorResponse, BatchExtractResponse
from ...application.use_cases.extract_content_use_case import ExtractContentUseCase
from ...infrastructure.web_scraper.scraper import WebScraper
from ...infrastructure.openrouter.client import OpenRouterClient
from ...infrastructure.cache.content_cache import ContentCache
from ...infrastructure.utils.error_handler import handle_extraction_error
from ...core.dependencies import (
    get_web_scraper,
    get_openrouter_client,
    get_content_cache
)
from .response_builders import build_extract_content_response, build_batch_extract_response
from .request_helpers import convert_urls_to_strings, get_first_url_or_default

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/extract", tags=["Extract"])


def get_extract_use_case(
    web_scraper: WebScraper = Depends(get_web_scraper),
    openrouter_client: OpenRouterClient = Depends(get_openrouter_client),
    cache: ContentCache = Depends(get_content_cache)
) -> ExtractContentUseCase:
    """Dependency para obtener ExtractContentUseCase"""
    return ExtractContentUseCase(web_scraper, openrouter_client, cache)


@router.post(
    "",
    response_model=ExtractContentResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}}
)
async def extract_content(
    request: ExtractContentRequest,
    use_cache: bool = Query(True, description="Usar cache si está disponible"),
    use_case: ExtractContentUseCase = Depends(get_extract_use_case)
):
    """
    Extrae información completa de una página web usando OpenRouter
    
    - **url**: URL de la página web a extraer
    - **model**: Modelo de OpenRouter a usar (opcional)
    - **max_tokens**: Máximo de tokens en la respuesta (opcional)
    - **use_cache**: Usar cache si está disponible (opcional)
    - **use_javascript**: Renderizar JavaScript con Playwright (opcional)
    - **extract_strategy**: Estrategia de extracción (opcional)
    """
    try:
        result = await use_case.execute(
            url=request.url,
            model=request.model,
            max_tokens=request.max_tokens,
            use_cache=use_cache,
            use_javascript=request.use_javascript,
            extract_strategy=request.extract_strategy
        )
        
        return build_extract_content_response(result)
        
    except Exception as e:
        raise handle_extraction_error(e, str(request.url))


@router.delete("/cache")
async def clear_cache(cache: ContentCache = Depends(get_content_cache)):
    """Limpiar cache de contenido"""
    cache.clear()
    return {"success": True, "message": "Cache limpiado exitosamente"}


@router.get("/cache/stats")
async def cache_stats(cache: ContentCache = Depends(get_content_cache)):
    """Obtener estadísticas del cache"""
    return cache.stats()


@router.post(
    "/batch",
    response_model=BatchExtractResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}}
)
async def extract_batch(
    request: BatchExtractRequest,
    web_scraper: WebScraper = Depends(get_web_scraper)
):
    """
    Extrae contenido de múltiples URLs en paralelo
    
    - **urls**: Lista de URLs a extraer (máximo 50)
    - **use_javascript**: Si True, usa Playwright (opcional)
    - **extract_strategy**: Estrategia de extracción (opcional)
    - **max_concurrent**: Máximo de requests concurrentes (1-20, default: 5)
    """
    try:
        urls = convert_urls_to_strings(request.urls)
        results = await web_scraper.scrape_batch(
            urls=urls,
            use_javascript=request.use_javascript,
            extract_strategy=request.extract_strategy,
            max_concurrent=request.max_concurrent
        )
        
        return build_batch_extract_response(urls, results)
        
    except Exception as e:
        raise handle_extraction_error(e, get_first_url_or_default(request.urls))

