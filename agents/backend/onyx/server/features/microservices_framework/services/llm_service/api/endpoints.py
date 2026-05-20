"""
LLM Service API Endpoints
FastAPI endpoints using modular architecture.
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "shared"))

from ..core.service_core import LLMServiceCore
from shared.ml import (
    validate_generation_params,
    error_handler,
)


# Request/Response models
class TextGenerationRequest(BaseModel):
    """Request model for text generation."""
    prompt: str = Field(..., description="Input text prompt")
    model_name: str = Field(default="gpt2", description="Model identifier")
    max_length: int = Field(default=100, ge=1, le=2048)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=1)
    num_return_sequences: int = Field(default=1, ge=1, le=5)
    do_sample: bool = Field(default=True)
    repetition_penalty: float = Field(default=1.0, ge=1.0)


class TextGenerationResponse(BaseModel):
    """Response model for text generation."""
    generated_text: str
    model_name: str
    prompt: str
    generation_params: Dict[str, Any]


class EmbeddingRequest(BaseModel):
    """Request model for text embeddings."""
    texts: List[str] = Field(..., min_items=1, max_items=100)
    model_name: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    normalize: bool = Field(default=True)
    pooling: str = Field(default="mean", description="Pooling strategy: mean, cls, max")


class EmbeddingResponse(BaseModel):
    """Response model for embeddings."""
    embeddings: List[List[float]]
    model_name: str
    dimension: int


class ModelInfoResponse(BaseModel):
    """Model information response."""
    model_name: str
    loaded: bool
    device: str
    parameters: Optional[int] = None
    model_size_mb: Optional[float] = None
    error: Optional[str] = None


# Router
router = APIRouter(prefix="/api/v1", tags=["llm"])


# Dependency injection
def get_service_core() -> LLMServiceCore:
    """Get service core instance."""
    # In production, use dependency injection container
    from ..config import get_config
    config = get_config()
    return LLMServiceCore(config=config)


@router.get("/health")
async def health_check(service: LLMServiceCore = Depends(get_service_core)):
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "llm_service",
        "device": service.device,
        "cuda_available": torch.cuda.is_available(),
        "cached_models": len(service._inference_engines),
    }


@router.post("/generate", response_model=TextGenerationResponse)
async def generate_text(
    request: TextGenerationRequest,
    service: LLMServiceCore = Depends(get_service_core),
):
    """Generate text using language model."""
    try:
        # Validate parameters
        validate_generation_params(
            request.max_length,
            request.temperature,
            request.top_p,
            request.top_k,
        )
        
        # Generate text
        generated_text = service.generate_text(
            prompt=request.prompt,
            model_name=request.model_name,
            max_length=request.max_length,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
            do_sample=request.do_sample,
        )
        
        return TextGenerationResponse(
            generated_text=generated_text,
            model_name=request.model_name,
            prompt=request.prompt,
            generation_params={
                "max_length": request.max_length,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "top_k": request.top_k,
                "repetition_penalty": request.repetition_penalty,
                "do_sample": request.do_sample,
            },
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@router.post("/embeddings", response_model=EmbeddingResponse)
async def get_embeddings(
    request: EmbeddingRequest,
    service: LLMServiceCore = Depends(get_service_core),
):
    """Get embeddings for texts."""
    try:
        embeddings = service.get_embeddings(
            texts=request.texts,
            model_name=request.model_name,
            normalize=request.normalize,
            pooling=request.pooling,
        )
        
        dimension = len(embeddings[0]) if embeddings else 0
        
        return EmbeddingResponse(
            embeddings=embeddings,
            model_name=request.model_name,
            dimension=dimension,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")


@router.get("/models/{model_name}/info", response_model=ModelInfoResponse)
async def get_model_info(
    model_name: str,
    service: LLMServiceCore = Depends(get_service_core),
):
    """Get information about a model."""
    info = service.get_model_info(model_name)
    
    return ModelInfoResponse(
        model_name=model_name,
        loaded=info.get("loaded", False),
        device=info.get("device", service.device),
        parameters=info.get("total_parameters"),
        model_size_mb=info.get("model_size_mb"),
        error=info.get("error"),
    )


@router.delete("/models/{model_name}")
async def unload_model(
    model_name: str,
    service: LLMServiceCore = Depends(get_service_core),
):
    """Unload a model from memory."""
    success = service.unload_model(model_name)
    
    if success:
        return {"status": "unloaded", "model_name": model_name}
    else:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

