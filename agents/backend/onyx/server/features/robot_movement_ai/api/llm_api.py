"""
LLM API
=======

API endpoints para procesamiento de lenguaje natural.
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..core.llm_processor import (
    get_llm_processor,
    LLMTask
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/llm", tags=["LLM"])


# Request/Response Models
class LoadModelRequest(BaseModel):
    """Request para cargar modelo."""
    model_name: str = Field(..., description="Nombre del modelo (HuggingFace)")
    task: str = Field(..., description="Tipo de tarea")
    device: str = Field(default="auto", description="Dispositivo")
    max_length: int = Field(default=512, description="Longitud máxima")
    temperature: float = Field(default=0.7, description="Temperatura")
    use_fp16: bool = Field(default=False, description="Usar FP16")


class GenerateRequest(BaseModel):
    """Request para generación."""
    model_id: str = Field(..., description="ID del modelo")
    prompt: str = Field(..., description="Prompt")
    max_length: Optional[int] = Field(None, description="Longitud máxima")
    temperature: Optional[float] = Field(None, description="Temperatura")
    top_p: Optional[float] = Field(None, description="Top-p")
    top_k: Optional[int] = Field(None, description="Top-k")


class ClassifyRequest(BaseModel):
    """Request para clasificación."""
    model_id: str = Field(..., description="ID del modelo")
    text: str = Field(..., description="Texto a clasificar")


class ParseCommandRequest(BaseModel):
    """Request para parsear comando."""
    model_id: str = Field(..., description="ID del modelo")
    command: str = Field(..., description="Comando de texto")


class QuestionAnswerRequest(BaseModel):
    """Request para Q&A."""
    model_id: str = Field(..., description="ID del modelo")
    question: str = Field(..., description="Pregunta")
    context: Optional[str] = Field(None, description="Contexto")


# Endpoints
@router.post("/models", response_model=Dict[str, Any])
async def load_model(request: LoadModelRequest):
    """Cargar modelo LLM."""
    try:
        processor = get_llm_processor()
        
        task = LLMTask(request.task)
        
        model_id = processor.load_model(
            request.model_name,
            task,
            config=None  # Se crea internamente
        )
        
        return {
            "model_id": model_id,
            "model_name": request.model_name,
            "task": request.task,
            "status": "loaded"
        }
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models", response_model=Dict[str, Any])
async def list_models():
    """Listar modelos LLM."""
    try:
        processor = get_llm_processor()
        stats = processor.get_statistics()
        
        models = []
        for model_id, config in processor.configs.items():
            models.append({
                "model_id": model_id,
                "model_name": config.model_name,
                "task": config.task.value
            })
        
        return {
            "models": models,
            "statistics": stats
        }
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate", response_model=Dict[str, Any])
async def generate_text(request: GenerateRequest):
    """Generar texto."""
    try:
        processor = get_llm_processor()
        
        response = processor.generate_text(
            request.model_id,
            request.prompt,
            max_length=request.max_length,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k
        )
        
        return {
            "response_id": response.response_id,
            "text": response.text,
            "confidence": response.confidence
        }
    except Exception as e:
        logger.error(f"Error generating text: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/classify", response_model=Dict[str, Any])
async def classify_text(request: ClassifyRequest):
    """Clasificar texto."""
    try:
        processor = get_llm_processor()
        
        response = processor.classify_text(
            request.model_id,
            request.text
        )
        
        return {
            "response_id": response.response_id,
            "label": response.text,
            "confidence": response.confidence
        }
    except Exception as e:
        logger.error(f"Error classifying text: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/parse-command", response_model=Dict[str, Any])
async def parse_command(request: ParseCommandRequest):
    """Parsear comando de robot."""
    try:
        processor = get_llm_processor()
        
        intent = processor.parse_command(
            request.model_id,
            request.command
        )
        
        return {
            "intent": intent.intent,
            "confidence": intent.confidence,
            "entities": intent.entities,
            "parameters": intent.parameters
        }
    except Exception as e:
        logger.error(f"Error parsing command: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/answer", response_model=Dict[str, Any])
async def answer_question(request: QuestionAnswerRequest):
    """Responder pregunta."""
    try:
        processor = get_llm_processor()
        
        response = processor.answer_question(
            request.model_id,
            request.question,
            request.context
        )
        
        return {
            "response_id": response.response_id,
            "answer": response.text,
            "confidence": response.confidence
        }
    except Exception as e:
        logger.error(f"Error answering question: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics", response_model=Dict[str, Any])
async def get_statistics():
    """Obtener estadísticas."""
    try:
        processor = get_llm_processor()
        return processor.get_statistics()
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))




