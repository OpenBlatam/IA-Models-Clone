"""
LLM Service endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.llm_service import LLMService, LLMConfig

router = APIRouter()
llm_service = LLMService()


@router.post("/load-model")
async def load_model(
    model_name: str,
    max_tokens: int = 512,
    temperature: float = 0.7
) -> Dict[str, Any]:
    """Cargar modelo LLM"""
    try:
        config = LLMConfig(
            model_name=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        success = llm_service.load_model(config)
        return {
            "model_name": model_name,
            "loaded": success,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate")
async def generate_text(
    prompt: str,
    model_name: str,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None
) -> Dict[str, Any]:
    """Generar texto con LLM"""
    try:
        response = llm_service.generate_text(prompt, model_name, max_tokens, temperature)
        return {
            "text": response.text,
            "tokens_used": response.tokens_used,
            "model_name": response.model_name,
            "generation_time": response.generation_time,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/fine-tune")
async def fine_tune_model(
    model_name: str,
    training_data: list,
    epochs: int = 3,
    learning_rate: float = 5e-5
) -> Dict[str, Any]:
    """Fine-tune modelo LLM"""
    try:
        result = llm_service.fine_tune_model(model_name, training_data, epochs, learning_rate)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/embeddings")
async def get_embeddings(
    text: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> Dict[str, Any]:
    """Obtener embeddings de texto"""
    try:
        embeddings = llm_service.get_embeddings(text, model_name)
        return {
            "text": text,
            "embeddings": embeddings,
            "dimension": len(embeddings),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/semantic-search")
async def semantic_search(
    query: str,
    documents: list,
    top_k: int = 5
) -> Dict[str, Any]:
    """Búsqueda semántica"""
    try:
        results = llm_service.semantic_search(query, documents, top_k)
        return {
            "query": query,
            "results": results,
            "total": len(results),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
