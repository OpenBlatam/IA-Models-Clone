"""
Performance Optimizations - Optimizaciones de performance
=========================================================

Optimizaciones para mejorar throughput, latencia y eficiencia.
"""

import logging
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import ORJSONResponse
import orjson

logger = logging.getLogger(__name__)


def optimize_app(app: FastAPI) -> FastAPI:
    """
    Aplica optimizaciones de performance a la aplicación FastAPI.
    
    Args:
        app: Aplicación FastAPI
    
    Returns:
        Aplicación optimizada
    """
    # Response compression
    enable_response_compression(app)
    
    # Connection pooling
    enable_connection_pooling()
    
    # Query optimization
    enable_query_optimization()
    
    # JSON serialization optimization
    enable_fast_json_serialization(app)
    
    logger.info("Performance optimizations applied")
    return app


def enable_response_compression(app: FastAPI, minimum_size: int = 500):
    """
    Habilita compresión de respuestas.
    
    Args:
        app: Aplicación FastAPI
        minimum_size: Tamaño mínimo para comprimir (bytes)
    """
    app.add_middleware(GZipMiddleware, minimum_size=minimum_size)


def enable_fast_json_serialization(app: FastAPI):
    """
    Habilita serialización JSON rápida usando orjson.
    
    Args:
        app: Aplicación FastAPI
    """
    # Configurar default response class para usar orjson
    app.default_response_class = ORJSONResponse


def enable_connection_pooling():
    """Habilita connection pooling para servicios externos"""
    # Connection pooling se configura en los clientes específicos
    # (Redis, HTTP, etc.)
    pass


def enable_query_optimization():
    """Habilita optimizaciones de queries"""
    # Las optimizaciones de queries se aplican en los repositorios
    pass


class ResponseOptimizer:
    """Optimizador de respuestas HTTP"""
    
    @staticmethod
    def optimize_response(data: dict, compress: bool = True) -> dict:
        """
        Optimiza respuesta antes de enviarla.
        
        Args:
            data: Datos de respuesta
            compress: Si comprimir (ya manejado por middleware)
        
        Returns:
            Datos optimizados
        """
        # Remover campos None para reducir tamaño
        return {k: v for k, v in data.items() if v is not None}
    
    @staticmethod
    def paginate_response(
        items: list,
        total: int,
        page: int = 1,
        page_size: int = 100
    ) -> dict:
        """
        Crea respuesta paginada optimizada.
        
        Args:
            items: Items de la página
            total: Total de items
            page: Página actual
            page_size: Tamaño de página
        
        Returns:
            Respuesta paginada
        """
        return {
            "items": items,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total": total,
                "pages": (total + page_size - 1) // page_size
            }
        }


class QueryOptimizer:
    """Optimizador de queries"""
    
    @staticmethod
    def optimize_filters(filters: dict) -> dict:
        """
        Optimiza filtros de query.
        
        Args:
            filters: Filtros originales
        
        Returns:
            Filtros optimizados
        """
        # Remover filtros vacíos
        return {k: v for k, v in filters.items() if v is not None and v != ""}
    
    @staticmethod
    def optimize_limit_offset(limit: int, offset: int) -> tuple:
        """
        Optimiza limit y offset.
        
        Args:
            limit: Límite
            offset: Offset
        
        Returns:
            (limit, offset) optimizados
        """
        # Limitar máximo
        limit = min(limit, 1000)
        limit = max(limit, 1)
        offset = max(offset, 0)
        return limit, offset















