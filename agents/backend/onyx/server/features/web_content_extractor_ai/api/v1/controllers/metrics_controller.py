"""
Controller para métricas y estadísticas
"""

import logging
from fastapi import APIRouter, Depends
from ...infrastructure.cache.content_cache import ContentCache
from ...core.dependencies import get_content_cache

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/metrics", tags=["Metrics"])


@router.get("/stats")
async def get_stats(cache: ContentCache = Depends(get_content_cache)):
    """Obtener estadísticas generales del servicio"""
    return {
        "cache": cache.stats(),
        "service": {
            "name": "web-content-extractor-ai",
            "version": "1.0.0"
        }
    }

