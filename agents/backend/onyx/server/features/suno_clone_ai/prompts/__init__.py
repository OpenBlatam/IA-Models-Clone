"""
Prompts Module - Gestión de Prompts
Gestión de prompts, templates, y construcción de prompts dinámicos.

Rol en el Ecosistema IA:
- Templates, construcción de prompts dinámicos
- Optimización de prompts, versionado, A/B testing
- Gestión centralizada de todos los prompts del sistema

Reglas de Importación:
- Puede importar: configs, utils
- NO debe importar: llm, chat, agents (evitar ciclos)
- Los módulos que usan prompts deben importar este módulo
"""

from .base import BasePrompt
from .service import PromptService
from .template import PromptTemplate
from .builder import PromptBuilder
from .main import (
    get_prompt_service,
    build_prompt,
    register_template,
    create_template,
    build_with_context,
    initialize_prompts,
)

__all__ = [
    # Clases principales
    "BasePrompt",
    "PromptService",
    "PromptTemplate",
    "PromptBuilder",
    # Funciones de acceso rápido
    "get_prompt_service",
    "build_prompt",
    "register_template",
    "create_template",
    "build_with_context",
    "initialize_prompts",
]

