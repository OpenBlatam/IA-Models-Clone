"""
Rutas de Machine Learning
==========================

Endpoints para funcionalidades de ML/DL.
"""

import logging
import io
import base64
from fastapi import APIRouter, HTTPException, Depends, Query, Form, File, UploadFile
from fastapi.responses import Response, StreamingResponse
from typing import Optional, List
from pydantic import BaseModel, Field
from PIL import Image

from ...ml.embeddings.embedding_service import EmbeddingService
from ...ml.image_generation.image_generator import ImageGenerator
from ...ml.models.manual_generator_model import ManualGeneratorModel
from ...ml.config.ml_config import get_ml_config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/ml", tags=["ml"])

# Instancias globales (singleton)
_embedding_service: Optional[EmbeddingService] = None
_image_generator: Optional[ImageGenerator] = None
_local_model: Optional[ManualGeneratorModel] = None


def get_embedding_service() -> EmbeddingService:
    """Obtener servicio de embeddings."""
    global _embedding_service
    if _embedding_service is None:
        config = get_ml_config()
        _embedding_service = EmbeddingService(
            model_name=config.embedding_model,
            device=config.device
        )
    return _embedding_service


def get_image_generator() -> ImageGenerator:
    """Obtener generador de imágenes."""
    global _image_generator
    if _image_generator is None:
        config = get_ml_config()
        _image_generator = ImageGenerator(
            model_name=config.image_model,
            use_xl=config.use_sd_xl,
            device=config.device
        )
    return _image_generator


# Modelos Pydantic
class EmbeddingRequest(BaseModel):
    """Request para generar embeddings."""
    texts: List[str] = Field(..., description="Lista de textos")


class SimilarityRequest(BaseModel):
    """Request para calcular similitud."""
    text1: str
    text2: str


class FindSimilarRequest(BaseModel):
    """Request para encontrar similares."""
    query: str
    texts: List[str]
    top_k: int = 5
    threshold: float = 0.5


class ImageGenerationRequest(BaseModel):
    """Request para generar imagen."""
    prompt: str
    negative_prompt: Optional[str] = None
    width: int = 512
    height: int = 512
    num_steps: int = 50
    guidance_scale: float = 7.5
    seed: Optional[int] = None


class ManualIllustrationRequest(BaseModel):
    """Request para ilustración de manual."""
    step_description: str
    category: str = "general"
    style: str = "lego_instruction"


# Endpoints
@router.post("/embeddings/generate")
async def generate_embeddings(
    request: EmbeddingRequest,
    embedding_service: EmbeddingService = Depends(get_embedding_service)
):
    """
    Generar embeddings para textos.
    
    - **texts**: Lista de textos
    """
    try:
        embeddings = embedding_service.encode(request.texts)
        
        return {
            "success": True,
            "embeddings": embeddings.tolist(),
            "dimension": embedding_service.get_embedding_dimension(),
            "count": len(request.texts)
        }
    
    except Exception as e:
        logger.error(f"Error generando embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generando embeddings: {str(e)}")


@router.post("/embeddings/similarity")
async def calculate_similarity(
    request: SimilarityRequest,
    embedding_service: EmbeddingService = Depends(get_embedding_service)
):
    """
    Calcular similitud entre dos textos.
    
    - **text1**: Primer texto
    - **text2**: Segundo texto
    """
    try:
        similarity = embedding_service.similarity(request.text1, request.text2)
        
        return {
            "success": True,
            "similarity": similarity,
            "text1": request.text1[:100] + "..." if len(request.text1) > 100 else request.text1,
            "text2": request.text2[:100] + "..." if len(request.text2) > 100 else request.text2
        }
    
    except Exception as e:
        logger.error(f"Error calculando similitud: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error calculando similitud: {str(e)}")


@router.post("/embeddings/find-similar")
async def find_similar_texts(
    request: FindSimilarRequest,
    embedding_service: EmbeddingService = Depends(get_embedding_service)
):
    """
    Encontrar textos similares.
    
    - **query**: Texto de consulta
    - **texts**: Lista de textos a buscar
    - **top_k**: Número de resultados
    - **threshold**: Umbral mínimo
    """
    try:
        results = embedding_service.find_similar(
            query=request.query,
            texts=request.texts,
            top_k=request.top_k,
            threshold=request.threshold
        )
        
        return {
            "success": True,
            "query": request.query,
            "results": [
                {
                    "index": idx,
                    "text": text[:200] + "..." if len(text) > 200 else text,
                    "similarity": float(sim)
                }
                for idx, text, sim in results
            ],
            "total_found": len(results)
        }
    
    except Exception as e:
        logger.error(f"Error buscando similares: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error buscando similares: {str(e)}")


@router.post("/images/generate")
async def generate_image(
    request: ImageGenerationRequest,
    image_generator: ImageGenerator = Depends(get_image_generator)
):
    """
    Generar imagen desde prompt.
    
    - **prompt**: Prompt de texto
    - **negative_prompt**: Prompt negativo (opcional)
    - **width**: Ancho (512, 768, 1024)
    - **height**: Alto (512, 768, 1024)
    - **num_steps**: Pasos de inferencia
    - **guidance_scale**: Escala de guía
    - **seed**: Semilla para reproducibilidad
    """
    try:
        image = image_generator.generate(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            num_inference_steps=request.num_steps,
            guidance_scale=request.guidance_scale,
            seed=request.seed
        )
        
        # Convertir a bytes
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="PNG")
        img_bytes.seek(0)
        
        return Response(
            content=img_bytes.getvalue(),
            media_type="image/png",
            headers={
                "Content-Disposition": "inline; filename=generated_image.png"
            }
        )
    
    except Exception as e:
        logger.error(f"Error generando imagen: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generando imagen: {str(e)}")


@router.post("/images/generate-manual-illustration")
async def generate_manual_illustration(
    request: ManualIllustrationRequest,
    image_generator: ImageGenerator = Depends(get_image_generator)
):
    """
    Generar ilustración para paso de manual.
    
    - **step_description**: Descripción del paso
    - **category**: Categoría del oficio
    - **style**: Estilo (lego_instruction, technical_diagram, realistic)
    """
    try:
        image = image_generator.generate_manual_illustration(
            step_description=request.step_description,
            category=request.category,
            style=request.style
        )
        
        # Convertir a bytes
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="PNG")
        img_bytes.seek(0)
        
        return Response(
            content=img_bytes.getvalue(),
            media_type="image/png",
            headers={
                "Content-Disposition": "inline; filename=manual_illustration.png"
            }
        )
    
    except Exception as e:
        logger.error(f"Error generando ilustración: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generando ilustración: {str(e)}")


@router.get("/embeddings/info")
async def get_embedding_info(
    embedding_service: EmbeddingService = Depends(get_embedding_service)
):
    """Obtener información del servicio de embeddings."""
    return {
        "success": True,
        "model_name": embedding_service.model_name,
        "embedding_dimension": embedding_service.get_embedding_dimension(),
        "device": embedding_service.device
    }


@router.get("/images/info")
async def get_image_generator_info(
    image_generator: ImageGenerator = Depends(get_image_generator)
):
    """Obtener información del generador de imágenes."""
    return {
        "success": True,
        "model_name": image_generator.model_name,
        "use_xl": image_generator.use_xl,
        "device": image_generator.device
    }




