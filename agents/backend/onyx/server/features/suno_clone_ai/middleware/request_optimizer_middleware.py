"""
Request Optimizer Middleware
Optimizaciones de requests HTTP
"""

import logging
import time
import asyncio
from typing import List
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class RequestOptimizerMiddleware(BaseHTTPMiddleware):
    """Middleware para optimizar requests"""
    
    def __init__(self, app, **kwargs):
        super().__init__(app)
        self.enable_prefetch = kwargs.get('enable_prefetch', True)
        self.enable_early_response = kwargs.get('enable_early_response', False)
    
    async def dispatch(self, request: Request, call_next):
        """Procesa request con optimizaciones"""
        start_time = time.time()
        
        # Optimizaciones de request
        request.state.start_time = start_time
        
        # Pre-fetch basado en path
        if self.enable_prefetch:
            await self._prefetch_related_data(request)
        
        # Procesar request
        response = await call_next(request)
        
        # Agregar headers de performance
        duration = time.time() - start_time
        response.headers["X-Process-Time"] = f"{duration:.4f}"
        response.headers["X-Request-ID"] = request.headers.get(
            "X-Request-ID",
            str(id(request))
        )
        
        return response
    
    async def _prefetch_related_data(self, request: Request):
        """Pre-carga datos relacionados"""
        try:
            from core.prefetch_optimizer import get_prefetch_optimizer
            
            path = request.url.path
            optimizer = get_prefetch_optimizer()
            
            # Registrar acceso
            optimizer.record_access(path)
            
            # Predecir y pre-cargar
            predictions = optimizer.predict_next(path)
            if predictions:
                # Pre-cargar en background
                asyncio.create_task(self._prefetch_data(predictions))
        except Exception as e:
            logger.debug(f"Prefetch error: {e}")
    
    async def _prefetch_data(self, keys: List[str]):
        """Pre-carga datos"""
        # Implementar según necesidades
        pass

