"""
Prompts Main - Funciones base y entry points del módulo de prompts

Rol en el Ecosistema IA:
- Templates, construcción de prompts dinámicos
- Optimización de prompts, versionado, A/B testing
- Gestión centralizada de todos los prompts del sistema
"""

from typing import Optional, Dict, Any
from .service import PromptService
from .template import PromptTemplate
from .builder import PromptBuilder


# Instancia global del servicio
_prompt_service: Optional[PromptService] = None


def get_prompt_service() -> PromptService:
    """
    Obtiene la instancia global del servicio de prompts.
    
    Returns:
        PromptService: Servicio de prompts
    """
    global _prompt_service
    if _prompt_service is None:
        _prompt_service = PromptService()
    return _prompt_service


def build_prompt(template_name: str, **kwargs) -> str:
    """
    Construye un prompt desde un template.
    
    Args:
        template_name: Nombre del template
        **kwargs: Variables para el template
        
    Returns:
        Prompt construido
    """
    service = get_prompt_service()
    return service.build_prompt(template_name, **kwargs)


def register_template(name: str, template: str) -> None:
    """
    Registra un nuevo template de prompt.
    
    Args:
        name: Nombre del template
        template: String del template con {variables}
    """
    service = get_prompt_service()
    prompt_template = PromptTemplate(template)
    service.register_template(name, prompt_template)


def create_template(template: str) -> PromptTemplate:
    """
    Crea un nuevo template de prompt.
    
    Args:
        template: String del template con {variables}
        
    Returns:
        PromptTemplate: Template creado
    """
    return PromptTemplate(template)


def build_with_context(prompt: str, context: Dict[str, Any]) -> str:
    """
    Construye un prompt con contexto adicional.
    
    Args:
        prompt: Prompt base
        context: Diccionario con contexto
        
    Returns:
        Prompt con contexto
    """
    builder = PromptBuilder()
    return builder.build_with_context(prompt, context)


def initialize_prompts() -> PromptService:
    """
    Inicializa el sistema de prompts.
    
    Returns:
        PromptService: Servicio inicializado
    """
    return get_prompt_service()

