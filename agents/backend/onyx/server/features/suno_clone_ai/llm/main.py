"""
LLM Main - Funciones base y entry points del módulo de modelos de lenguaje

Rol en el Ecosistema IA:
- Integración con LLMs, generación de texto
- Core del sistema, generación de respuestas, embeddings
- Punto central de comunicación con modelos de IA
"""

from typing import Optional, List, Dict, Any
from .service import LLMService
from .provider import LLMProvider
from .generator import TextGenerator
from prompts.main import get_prompt_service
from tracing.main import get_tracing_service


# Instancia global del servicio
_llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """
    Obtiene la instancia global del servicio de LLM.
    
    Returns:
        LLMService: Servicio de LLM
    """
    global _llm_service
    if _llm_service is None:
        prompt_service = get_prompt_service()
        tracing_service = get_tracing_service()
        _llm_service = LLMService(
            prompt_service=prompt_service,
            tracing_service=tracing_service
        )
    return _llm_service


async def generate_text(prompt: str, **kwargs) -> str:
    """
    Genera texto a partir de un prompt.
    
    Args:
        prompt: Prompt de entrada
        **kwargs: Parámetros adicionales (temperature, max_tokens, etc.)
        
    Returns:
        Texto generado
    """
    service = get_llm_service()
    return await service.generate(prompt, **kwargs)


async def chat(messages: List[Dict[str, str]], **kwargs) -> str:
    """
    Genera respuesta en formato chat.
    
    Args:
        messages: Lista de mensajes [{"role": "user", "content": "..."}]
        **kwargs: Parámetros adicionales
        
    Returns:
        Respuesta del modelo
    """
    service = get_llm_service()
    return await service.chat(messages, **kwargs)


async def generate_batch(prompts: List[str], **kwargs) -> List[str]:
    """
    Genera texto para múltiples prompts en batch.
    
    Args:
        prompts: Lista de prompts
        **kwargs: Parámetros adicionales
        
    Returns:
        Lista de textos generados
    """
    service = get_llm_service()
    results = []
    for prompt in prompts:
        result = await service.generate(prompt, **kwargs)
        results.append(result)
    return results


def get_provider() -> LLMProvider:
    """
    Obtiene el proveedor de LLM.
    
    Returns:
        LLMProvider: Proveedor de LLM
    """
    service = get_llm_service()
    return service.provider


def initialize_llm() -> LLMService:
    """
    Inicializa el sistema de LLM.
    
    Returns:
        LLMService: Servicio inicializado
    """
    return get_llm_service()

