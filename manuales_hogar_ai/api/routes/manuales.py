"""
Rutas API para Manuales Hogar AI
==================================

Endpoints para generar manuales paso a paso tipo LEGO.
"""

import logging
import asyncio
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends, Query
from typing import Optional, List

from ...core.manual_generator import ManualGenerator
from ...infrastructure.openrouter_client import OpenRouterClient
from ...config.settings import get_settings
from ...utils.cache_manager import get_cache
from ...database.session import get_async_session
from ...services.manual.manual_service import ManualService
from ..models.manual_models import ManualTextRequest, ManualResponse, HealthResponse
from ..dependencies.manual_dependencies import get_manual_generator, get_openrouter_client
from ..handlers.image_handler import ImageHandler
from ..handlers.validation_handler import ValidationHandler

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["manuales-hogar"])

image_handler = ImageHandler()
validation_handler = ValidationHandler()


# Endpoints
@router.get("/health", response_model=HealthResponse)
async def health_check(client: OpenRouterClient = Depends(get_openrouter_client)):
    """
    Health check del servicio.
    """
    try:
        health = await client.health_check()
        return HealthResponse(**health)
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            status="error",
            healthy=False,
            error=str(e)
        )


@router.get("/models")
async def list_available_models(client: OpenRouterClient = Depends(get_openrouter_client)):
    """
    Listar modelos disponibles en OpenRouter.
    """
    try:
        models = await client.list_models()
        # Filtrar solo modelos relevantes
        relevant_models = [
            {
                "id": m.get("id"),
                "name": m.get("name"),
                "description": m.get("description"),
                "context_length": m.get("context_length"),
                "pricing": m.get("pricing")
            }
            for m in models
            if m.get("id") and (
                "claude" in m.get("id", "").lower() or
                "gpt-4" in m.get("id", "").lower() or
                "gemini" in m.get("id", "").lower() or
                "llama" in m.get("id", "").lower()
            )
        ]
        return {
            "success": True,
            "models": relevant_models,
            "total": len(relevant_models)
        }
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")


@router.post("/generate-from-text", response_model=ManualResponse)
async def generate_manual_from_text(
    request: ManualTextRequest,
    generator: ManualGenerator = Depends(get_manual_generator),
    save_to_db: bool = Query(True, description="Guardar en base de datos")
):
    """
    Generar manual paso a paso desde descripción de texto.
    
    - **problem_description**: Descripción detallada del problema
    - **category**: Categoría del oficio (plomeria, techos, carpinteria, etc.)
    - **model**: Modelo de IA a usar (opcional, usa default si no se especifica)
    - **include_safety**: Incluir advertencias de seguridad
    - **include_tools**: Incluir lista de herramientas
    - **include_materials**: Incluir lista de materiales
    """
    try:
        validation_handler.validate_category(request.category)
        
        result = await generator.generate_manual_from_text(
            problem_description=request.problem_description,
            category=request.category,
            model=request.model,
            include_safety=request.include_safety,
            include_tools=request.include_tools,
            include_materials=request.include_materials
        )
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result.get("error", "Error generando manual"))
        
        # Guardar en base de datos de forma asíncrona (no bloquea la respuesta)
        if save_to_db:
            async def save_to_db_async():
                try:
                    async for db_session in get_async_session():
                        service = ManualService(db_session)
                        await service.save_manual(
                            problem_description=request.problem_description,
                            category=result["category"],
                            manual_content=result["manual"],
                            model_used=result.get("model_used"),
                            tokens_used=result.get("tokens_used", 0)
                        )
                        break
                except Exception as e:
                    logger.warning(f"No se pudo guardar en BD (puede no estar configurada): {str(e)}")
            
            asyncio.create_task(save_to_db_async())
        
        return ManualResponse(**result)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generando manual desde texto: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generando manual: {str(e)}")


@router.post("/generate-from-image", response_model=ManualResponse)
async def generate_manual_from_image(
    file: UploadFile = File(..., description="Imagen del problema"),
    problem_description: Optional[str] = Form(None, description="Descripción adicional del problema"),
    category: Optional[str] = Form(None, description="Categoría del oficio (se detecta automáticamente si no se proporciona)"),
    model: Optional[str] = Form(None, description="Modelo de IA a usar"),
    include_safety: bool = Form(True, description="Incluir advertencias de seguridad"),
    include_tools: bool = Form(True, description="Incluir lista de herramientas"),
    include_materials: bool = Form(True, description="Incluir lista de materiales"),
    generator: ManualGenerator = Depends(get_manual_generator)
):
    """
    Generar manual paso a paso desde imagen.
    
    - **file**: Imagen del problema (JPG, PNG, etc.)
    - **problem_description**: Descripción adicional del problema (opcional)
    - **category**: Categoría del oficio (opcional, se detecta automáticamente)
    - **model**: Modelo de IA con visión a usar
    - **include_safety**: Incluir advertencias de seguridad
    - **include_tools**: Incluir lista de herramientas
    - **include_materials**: Incluir lista de materiales
    """
    try:
        processed_image = await image_handler.validate_and_process_image(file)
        image_bytes = processed_image["bytes"]
        
        if category:
            validation_handler.validate_category(category)
        
        result = await generator.generate_manual_from_image(
            image_bytes=image_bytes,
            problem_description=problem_description,
            category=category,
            model=model,
            include_safety=include_safety,
            include_tools=include_tools,
            include_materials=include_materials
        )
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result.get("error", "Error generando manual"))
        
        return ManualResponse(**result)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generando manual desde imagen: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generando manual: {str(e)}")


@router.post("/generate-combined", response_model=ManualResponse)
async def generate_manual_combined(
    problem_description: str = Form(..., description="Descripción del problema"),
    file: Optional[UploadFile] = File(None, description="Imagen del problema (opcional)"),
    category: str = Form("general", description="Categoría del oficio"),
    model: Optional[str] = Form(None, description="Modelo de IA a usar"),
    include_safety: bool = Form(True, description="Incluir advertencias de seguridad"),
    include_tools: bool = Form(True, description="Incluir lista de herramientas"),
    include_materials: bool = Form(True, description="Incluir lista de materiales"),
    generator: ManualGenerator = Depends(get_manual_generator)
):
    """
    Generar manual combinando texto e imagen (opcional).
    
    - **problem_description**: Descripción del problema (requerido)
    - **file**: Imagen del problema (opcional)
    - **category**: Categoría del oficio
    - **model**: Modelo de IA a usar
    - **include_safety**: Incluir advertencias de seguridad
    - **include_tools**: Incluir lista de herramientas
    - **include_materials**: Incluir lista de materiales
    """
    try:
        validation_handler.validate_category(category)
        
        image_bytes = None
        if file:
            processed_image = await image_handler.validate_and_process_image(file)
            image_bytes = processed_image["bytes"]
        
        result = await generator.generate_manual_combined(
            problem_description=problem_description,
            image_bytes=image_bytes,
            category=category,
            model=model,
            include_safety=include_safety,
            include_tools=include_tools,
            include_materials=include_materials
        )
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result.get("error", "Error generando manual"))
        
        return ManualResponse(**result)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generando manual combinado: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generando manual: {str(e)}")


@router.get("/categories")
async def get_supported_categories():
    """
    Obtener lista de categorías soportadas.
    """
    settings = get_settings()
    return {
        "success": True,
        "categories": settings.supported_categories,
        "category_names": {
            "plomeria": "Plomería",
            "techos": "Reparación de Techos",
            "carpinteria": "Carpintería",
            "electricidad": "Electricidad",
            "albanileria": "Albañilería",
            "pintura": "Pintura",
            "herreria": "Herrería",
            "jardineria": "Jardinería",
            "general": "Reparación General"
        }
    }


@router.post("/generate-from-multiple-images", response_model=ManualResponse)
async def generate_manual_from_multiple_images(
    files: List[UploadFile] = File(..., description="Múltiples imágenes del problema"),
    problem_description: Optional[str] = Form(None, description="Descripción adicional del problema"),
    category: Optional[str] = Form(None, description="Categoría del oficio"),
    model: Optional[str] = Form(None, description="Modelo de IA a usar"),
    include_safety: bool = Form(True, description="Incluir advertencias de seguridad"),
    include_tools: bool = Form(True, description="Incluir lista de herramientas"),
    include_materials: bool = Form(True, description="Incluir lista de materiales"),
    generator: ManualGenerator = Depends(get_manual_generator)
):
    """
    Generar manual desde múltiples imágenes.
    
    - **files**: Múltiples imágenes del problema (máximo 5)
    - **problem_description**: Descripción adicional del problema (opcional)
    - **category**: Categoría del oficio (opcional, se detecta automáticamente)
    - **model**: Modelo de IA con visión a usar
    - **include_safety**: Incluir advertencias de seguridad
    - **include_tools**: Incluir lista de herramientas
    - **include_materials**: Incluir lista de materiales
    """
    try:
        multiple_images = await image_handler.validate_and_process_multiple_images(files)
        
        if category:
            validation_handler.validate_category(category)
        
        from ...core.prompts.vision_prompt_builder import VisionPromptBuilder
        from ...utils.category_detector import CategoryDetector
        
        settings = get_settings()
        vision_builder = VisionPromptBuilder()
        category_detector = CategoryDetector()
        client = generator.client
        
        vision_prompt = vision_builder.build(problem_description)
        vision_prompt += "\n\nIMPORTANTE: Analiza TODAS las imágenes proporcionadas. Compara diferentes ángulos y vistas del problema."
        
        analysis_response = await client.generate_with_vision(
            prompt=vision_prompt,
            multiple_images=multiple_images,
            model=model or settings.vision_model,
            max_tokens=2000,
            temperature=0.5
        )
        
        analysis_text = analysis_response['choices'][0]['message']['content']
        
        if not category:
            detected_category, confidence = category_detector.detect_category(analysis_text)
            if confidence > 0.3:
                category = detected_category
        
        full_description = f"""ANÁLISIS DE LAS IMÁGENES (Múltiples vistas):
{analysis_text}

"""
        
        if problem_description:
            full_description += f"DESCRIPCIÓN ADICIONAL DEL USUARIO:\n{problem_description}\n\n"
        
        full_description += "Basándote en el análisis de todas las imágenes y la descripción, genera el manual de reparación."
        
        result = await generator.generate_manual_from_text(
            problem_description=full_description,
            category=category or "general",
            model=model,
            include_safety=include_safety,
            include_tools=include_tools,
            include_materials=include_materials
        )
        
        if result["success"]:
            result["image_analysis"] = analysis_text
            result["detected_category"] = category
            result["images_count"] = len(multiple_images)
        
        return ManualResponse(**result)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generando manual desde múltiples imágenes: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generando manual: {str(e)}")


@router.get("/cache/stats")
async def get_cache_stats():
    """
    Obtener estadísticas del cache.
    """
    try:
        cache = get_cache()
        stats = cache.stats()
        return {
            "success": True,
            "cache_stats": stats
        }
    except Exception as e:
        logger.error(f"Error obteniendo stats del cache: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/cache/clear")
async def clear_cache():
    """
    Limpiar el cache.
    """
    try:
        cache = get_cache()
        cache.clear()
        return {
            "success": True,
            "message": "Cache limpiado exitosamente"
        }
    except Exception as e:
        logger.error(f"Error limpiando cache: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

