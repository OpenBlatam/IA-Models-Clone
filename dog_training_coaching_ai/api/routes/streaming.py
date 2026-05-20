"""
Streaming Endpoints
===================
Endpoints para streaming de respuestas de IA en tiempo real.
"""

from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse
from typing import AsyncIterator, Dict, Any
import json

from ...api.dependencies import get_coaching_service
from ...schemas import CoachingRequest, TrainingPlanRequest, ChatRequest
from ...services.coaching_service import (
    DogTrainingCoach,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    CHAT_MAX_TOKENS,
    CHAT_TEMPERATURE,
    TRAINING_PLAN_MAX_TOKENS,
    CONVERSATION_HISTORY_LIMIT
)
from ...core.exceptions import ServiceException, OpenRouterException
from ...utils.logger import get_logger
from ...utils.rate_limiter import limiter

router = APIRouter(prefix="/api/v1/stream", tags=["streaming"])
logger = get_logger(__name__)

# Constants
SSE_MEDIA_TYPE = "text/event-stream"
SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no"
}


def _create_streaming_response(stream: AsyncIterator[str]) -> StreamingResponse:
    """
    Crear StreamingResponse con headers estándar.
    
    Args:
        stream: Stream de datos
        
    Returns:
        StreamingResponse configurado
    """
    return StreamingResponse(
        stream,
        media_type=SSE_MEDIA_TYPE,
        headers=SSE_HEADERS
    )


async def stream_openrouter_response(
    prompt: str,
    system_prompt: str,
    max_tokens: int = 2000,
    temperature: float = 0.7
) -> AsyncIterator[str]:
    """
    Streamear respuesta de OpenRouter.
    
    Args:
        prompt: Prompt del usuario
        system_prompt: System prompt
        max_tokens: Tokens máximos
        temperature: Temperatura
        
    Yields:
        Chunks de texto en formato SSE
    """
    from ...infrastructure.openrouter.openrouter_client import OpenRouterClient
    
    client = OpenRouterClient()
    
    try:
        # Llamar a OpenRouter con streaming
        async for chunk in client.generate_text_stream(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature
        ):
            if chunk:
                yield f"data: {json.dumps({'content': chunk, 'done': False})}\n\n"
        
        yield f"data: {json.dumps({'done': True})}\n\n"
        
    except Exception as e:
        logger.error(f"Error in streaming: {e}")
        yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"


@router.post("/coach")
@limiter.limit("10/minute")
async def stream_coaching(
    request: Request,
    coaching_request: CoachingRequest,
    service: DogTrainingCoach = Depends(get_coaching_service)
):
    """
    Streamear coaching advice en tiempo real.
    """
    try:
        # Construir contexto usando el servicio
        context = service._build_context(
            dog_breed=coaching_request.dog_breed,
            dog_age=coaching_request.dog_age,
            dog_size=coaching_request.dog_size,
            training_goal=coaching_request.training_goal
        )
        
        user_message = service._build_user_message(
            instruction=f"User question: {coaching_request.question}",
            context=context,
            additional_context="Please provide expert coaching advice for this dog training question."
        )
        
        from ...core.prompts import COACHING_SYSTEM_PROMPT
        
        async def generate_stream() -> AsyncIterator[str]:
            async for chunk in stream_openrouter_response(
                prompt=user_message,
                system_prompt=COACHING_SYSTEM_PROMPT,
                max_tokens=DEFAULT_MAX_TOKENS,
                temperature=DEFAULT_TEMPERATURE
            ):
                yield chunk
        
        return _create_streaming_response(generate_stream())
        
    except (ServiceException, OpenRouterException) as e:
        logger.error(f"Error in stream coaching: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@router.post("/chat")
@limiter.limit("20/minute")
async def stream_chat(
    request: Request,
    chat_request: ChatRequest,
    service: DogTrainingCoach = Depends(get_coaching_service)
):
    """
    Streamear chat conversation en tiempo real.
    """
    try:
        from ...core.prompts import CHAT_SYSTEM_PROMPT
        
        # Construir mensaje con historial si existe
        messages = []
        if chat_request.dog_info:
            info_parts = [f"{k}: {v}" for k, v in chat_request.dog_info.items()]
            info_text = f"Dog information: {', '.join(info_parts)}"
            messages.append({"role": "system", "content": info_text})
        
        if chat_request.conversation_history:
            for msg in chat_request.conversation_history[-CONVERSATION_HISTORY_LIMIT:]:
                messages.append(msg)
        
        user_message = chat_request.message
        
        async def generate_stream() -> AsyncIterator[str]:
            async for chunk in stream_openrouter_response(
                prompt=user_message,
                system_prompt=CHAT_SYSTEM_PROMPT,
                max_tokens=CHAT_MAX_TOKENS,
                temperature=CHAT_TEMPERATURE
            ):
                yield chunk
        
        return _create_streaming_response(generate_stream())
        
    except (ServiceException, OpenRouterException) as e:
        logger.error(f"Error in stream chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@router.post("/training-plan")
@limiter.limit("5/minute")
async def stream_training_plan(
    request: Request,
    plan_request: TrainingPlanRequest,
    service: DogTrainingCoach = Depends(get_coaching_service)
):
    """
    Streamear training plan en tiempo real.
    """
    try:
        # Usar servicio para construir contexto
        context = service._build_context(
            dog_breed=plan_request.dog_breed,
            dog_age=plan_request.dog_age,
            dog_size=plan_request.dog_size,
            training_goals=", ".join(plan_request.training_goals)
        )
        
        user_message = service._build_user_message(
            instruction="Create a comprehensive training plan for:",
            context=context,
            additional_context="Provide a structured plan with phases, daily exercises, milestones, and estimated duration."
        )
        
        from ...core.prompts import TRAINING_PLAN_SYSTEM_PROMPT
        
        async def generate_stream() -> AsyncIterator[str]:
            async for chunk in stream_openrouter_response(
                prompt=user_message,
                system_prompt=TRAINING_PLAN_SYSTEM_PROMPT,
                max_tokens=TRAINING_PLAN_MAX_TOKENS,
                temperature=DEFAULT_TEMPERATURE
            ):
                yield chunk
        
        return _create_streaming_response(generate_stream())
        
    except (ServiceException, OpenRouterException) as e:
        logger.error(f"Error in stream training plan: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

