from fastapi import APIRouter, HTTPException, Request, Response, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Dict, List, Optional
import time
import json
import zlib
import threading
import mmh3
from .models import (
    KeyMessageRequest,
    GeneratedResponse,
    KeyMessageResponse,
    MessageType,
    MessageTone
)
from .services import KeyMessageService

router = APIRouter(prefix="/api/key-messages", tags=["key-messages"])

# Inicializar el servicio
service = KeyMessageService()

# Cache para respuestas comprimidas
response_cache: Dict[str, bytes] = {}
cache_lock = threading.Lock()

class CacheMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Generar clave de caché
        cache_key = f"{request.method}:{request.url.path}:{request.query_params}"
        
        # Verificar caché
        with cache_lock:
            if cache_key in response_cache:
                return Response(
                    content=response_cache[cache_key],
                    media_type="application/json",
                    headers={"X-Cache": "HIT"}
                )
        
        # Obtener respuesta
        response = await call_next(request)
        
        # Solo cachear respuestas exitosas
        if response.status_code == 200:
            # Obtener el contenido de la respuesta
            response_body = b""
            async for chunk in response.body_iterator:
                response_body += chunk
            
            # Comprimir respuesta
            compressed = zlib.compress(response_body)
            
            # Guardar en caché
            with cache_lock:
                response_cache[cache_key] = compressed
            
            # Devolver respuesta original
            return Response(
                content=response_body,
                media_type=response.media_type,
                headers=dict(response.headers)
            )
        
        return response

@router.post("/generate", response_model=KeyMessageResponse)
async def generate_response(request: KeyMessageRequest, background_tasks: BackgroundTasks):
    """Generate a key message response."""
    try:
        response = await service.generate_response(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze", response_model=KeyMessageResponse)
async def analyze_message(request: KeyMessageRequest, background_tasks: BackgroundTasks):
    """Analyze a message and return insights."""
    try:
        analysis = await service.analyze_message(request)
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/types", response_model=List[str])
async def get_message_types():
    """Get available message types."""
    return [mt.value for mt in MessageType]

@router.get("/tones", response_model=List[str])
async def get_message_tones():
    """Get available message tones."""
    return [mt.value for mt in MessageTone]

@router.delete("/cache")
async def clear_cache():
    """Clear all caches."""
    with cache_lock:
        response_cache.clear()
    await service.clear_cache()
    return {"status": "success", "message": "All caches cleared"} 