"""
Manejo centralizado de errores
"""

import logging
from typing import Dict, Any, Optional
from fastapi import HTTPException, status

logger = logging.getLogger(__name__)


class ExtractionError(Exception):
    """Error personalizado para extracción"""
    pass


class ScrapingError(ExtractionError):
    """Error durante scraping"""
    pass


class ProcessingError(ExtractionError):
    """Error durante procesamiento con IA"""
    pass


def handle_extraction_error(error: Exception, url: str) -> HTTPException:
    """Convertir excepciones a HTTPException apropiadas"""
    
    if isinstance(error, ScrapingError):
        logger.error(f"Error de scraping para {url}: {error}")
        return HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Error al acceder a la URL: {str(error)}"
        )
    
    if isinstance(error, ProcessingError):
        logger.error(f"Error de procesamiento para {url}: {error}")
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Error al procesar contenido: {str(error)}"
        )
    
    if isinstance(error, ValueError):
        logger.warning(f"Error de validación para {url}: {error}")
        return HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error)
        )
    
    logger.error(f"Error inesperado para {url}: {error}", exc_info=True)
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Error interno del servidor"
    )








