"""
LLM Module - Modelos de Lenguaje
Integración con LLMs, generación de texto, y manejo de modelos de IA.

Rol en el Ecosistema IA:
- Integración con LLMs, generación de texto
- Core del sistema, generación de respuestas, embeddings
- Punto central de comunicación con modelos de IA

Reglas de Importación:
- Puede importar: configs, prompts, tracing
- NO debe importar: chat, agents, server (evitar ciclos)
- Usar inyección de dependencias para servicios que dependen de LLM
"""

from .base import BaseLLM
from .service import LLMService
from .provider import LLMProvider
from .generator import TextGenerator
from .main import (
    get_llm_service,
    generate_text,
    chat,
    generate_batch,
    get_provider,
    initialize_llm,
)

__all__ = [
    # Clases principales
    "BaseLLM",
    "LLMService",
    "LLMProvider",
    "TextGenerator",
    # Funciones de acceso rápido
    "get_llm_service",
    "generate_text",
    "chat",
    "generate_batch",
    "get_provider",
    "initialize_llm",
]

