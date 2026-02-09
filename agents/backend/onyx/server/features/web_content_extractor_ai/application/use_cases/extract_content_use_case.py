"""
Caso de uso para extraer información de páginas web
"""

import logging
from typing import Dict, Any, Optional, Union
from ...infrastructure.web_scraper.scraper import WebScraper
from ...infrastructure.openrouter.client import OpenRouterClient
from ...infrastructure.cache.content_cache import ContentCache
from .content_formatters import ContentFormatter, ResultBuilder

# Importar constantes desde schemas
try:
    from ...api.v1.schemas.constants import (
        DEFAULT_MODEL,
        DEFAULT_MAX_TOKENS,
        DEFAULT_EXTRACT_STRATEGY
    )
except ImportError:
    # Fallback si no se puede importar
    DEFAULT_MODEL = "anthropic/claude-3.5-sonnet"
    DEFAULT_MAX_TOKENS = 4000
    DEFAULT_EXTRACT_STRATEGY = "auto"

logger = logging.getLogger(__name__)


class ExtractContentUseCase:
    """Caso de uso para extraer información de páginas web"""
    
    def __init__(
        self,
        web_scraper: WebScraper,
        openrouter_client: OpenRouterClient,
        cache: Optional[ContentCache] = None
    ):
        self.web_scraper = web_scraper
        self.openrouter_client = openrouter_client
        self.cache = cache
    
    async def execute(
        self,
        url: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        use_javascript: bool = False,
        extract_strategy: Optional[str] = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Ejecuta la extracción de contenido de una página web
        
        Args:
            url: URL de la página a extraer
            model: Modelo de OpenRouter a usar (None para usar default)
            max_tokens: Máximo de tokens en la respuesta (None para usar default)
            use_javascript: Si True, usa Playwright para renderizar JavaScript
            extract_strategy: Estrategia de extracción (None para usar default)
            use_cache: Usar cache si está disponible
            
        Returns:
            Dict con información extraída y procesada
        """
        # Usar valores por defecto si no se proporcionan
        model = model or DEFAULT_MODEL
        max_tokens = max_tokens or DEFAULT_MAX_TOKENS
        extract_strategy = extract_strategy or DEFAULT_EXTRACT_STRATEGY
        
        # Verificar cache
        if use_cache and self.cache:
            cached = self.cache.get(url)
            if cached:
                logger.info(f"Contenido obtenido del cache para {url}")
                return cached
        
        try:
            # 1. Scrapear contenido web con scraper avanzado
            logger.info(f"Scrapeando contenido de {url} (estrategia: {extract_strategy})")
            scraped_data = await self.web_scraper.scrape(
                url=url,
                use_javascript=use_javascript,
                extract_strategy=extract_strategy
            )
            
            # 2. Preparar contenido para OpenRouter con información enriquecida
            content_summary = ContentFormatter.build_content_summary(scraped_data)
            
            # 3. Procesar con OpenRouter
            logger.info(f"Procesando contenido con OpenRouter (modelo: {model})")
            extracted_info = await self.openrouter_client.extract_content(
                web_content=content_summary,
                url=url,
                model=model,
                max_tokens=max_tokens
            )
            
            # 4. Combinar resultados con metadatos enriquecidos
            result = ResultBuilder.build_result(url, scraped_data, extracted_info)
            
            # Guardar en cache
            if use_cache and self.cache:
                self.cache.set(url, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error en ExtractContentUseCase: {e}")
            raise

