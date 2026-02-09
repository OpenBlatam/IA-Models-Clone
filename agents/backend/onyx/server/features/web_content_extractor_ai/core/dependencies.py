"""
Dependencias compartidas para inyección de dependencias
"""

from functools import lru_cache
from infrastructure.web_scraper.scraper import WebScraper
from infrastructure.openrouter.client import OpenRouterClient
from infrastructure.cache.content_cache import ContentCache
from config import settings


@lru_cache()
def get_web_scraper() -> WebScraper:
    """Obtener instancia de WebScraper"""
    return WebScraper()


@lru_cache()
def get_openrouter_client() -> OpenRouterClient:
    """Obtener instancia de OpenRouterClient"""
    return OpenRouterClient()


@lru_cache()
def get_content_cache() -> ContentCache:
    """Obtener instancia de ContentCache"""
    return ContentCache(
        maxsize=settings.cache_max_size,
        ttl=settings.cache_ttl
    )








