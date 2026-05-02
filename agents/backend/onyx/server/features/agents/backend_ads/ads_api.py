from fastapi import APIRouter, HTTPException, Body, Request as FastAPIRequest
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
from models import AdsIaRequest, AdsResponse, BrandKitResponse, ErrorResponse
from scraper import get_website_text
from llm_interface import (
    generate_ads_lcel,
    generate_brand_kit_lcel,
    generate_custom_content_lcel,
    generate_ads_lcel_streaming,
    generate_ads_lcel_streaming_parallel,
    DEEPSEEK_API_KEY
)
from config import settings
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/api/ads-ia",
          summary="Genera anuncios, brand kits o contenido personalizado desde una URL",
          responses={
              200: {"description": "Operación exitosa"},
              400: {"model": ErrorResponse, "description": "Solicitud incorrecta"},
              422: {"model": ErrorResponse, "description": "Error de validación de entrada"},
              500: {"model": ErrorResponse, "description": "Error interno del servidor"},
              503: {"model": ErrorResponse, "description": "Servicio DeepSeek no disponible o no configurado"}
          })
async def process_ads_ia_request(payload: AdsIaRequest, http_request: FastAPIRequest):
    if not DEEPSEEK_API_KEY:
        raise HTTPException(status_code=503, detail={
            "error": "Servicio DeepSeek no configurado",
            "details": "DEEPSEEK_API_KEY no está configurado en el backend."
        })
    website_text = get_website_text(str(payload.url))
    if not website_text:
        website_text = ""
    effective_website_text = website_text
    try:
        if payload.prompt:
            generated_content = await generate_custom_content_lcel(payload.prompt, effective_website_text)
            if not generated_content:
                raise HTTPException(status_code=500, detail={"error": "Error al generar contenido personalizado", "details": "Respuesta vacía o error de DeepSeek."})
            return AdsResponse(ads=generated_content)
        elif payload.type == "ads":
            if not effective_website_text:
                raise HTTPException(status_code=400, detail={"error": "Contenido web necesario para anuncios", "details": "No se pudo obtener contenido de la URL."})
            ads_list = await generate_ads_lcel(effective_website_text)
            if not ads_list:
                raise HTTPException(status_code=500, detail={"error": "Error al generar anuncios", "details": "Respuesta vacía o error de DeepSeek."})
            return AdsResponse(ads=ads_list)
        elif payload.type == "brand-kit":
            if not effective_website_text:
                raise HTTPException(status_code=400, detail={"error": "Contenido web necesario para brand kit", "details": "No se pudo obtener contenido de la URL."})
            brand_kit_string = await generate_brand_kit_lcel(effective_website_text)
            if not brand_kit_string:
                raise HTTPException(status_code=500, detail={"error": "Error al generar brand kit", "details": "Respuesta vacía o error de DeepSeek."})
            return BrandKitResponse(brandKit=brand_kit_string)
        else:
            raise HTTPException(status_code=400, detail={"error": "Tipo de solicitud no válido"})
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        return JSONResponse(status_code=500, content=ErrorResponse(error="Error interno del servidor", details=str(e)).model_dump())

@router.post("/api/ads-ia/stream")
async def process_ads_ia_stream(payload: AdsIaRequest, http_request: FastAPIRequest):
    website_content = getattr(payload, 'website_content', None)
    if not website_content:
        website_content = get_website_text(payload.url)
        if website_content:
            website_content = website_content[:800]  # Limitar a 800 caracteres para máxima velocidad
    else:
        website_content = website_content[:800]  # Limitar a 800 caracteres si viene directo
    async def ad_generator():
        async for ad in generate_ads_lcel_streaming_parallel(website_content or "", n_ads=1):
            yield ad
    return StreamingResponse(ad_generator(), media_type="text/event-stream") 