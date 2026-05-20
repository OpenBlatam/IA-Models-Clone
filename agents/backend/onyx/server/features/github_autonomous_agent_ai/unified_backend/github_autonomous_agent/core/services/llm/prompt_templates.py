"""
Prompt Templates - Sistema de templates para prompts de LLM.

Sigue principios de modularidad y reutilización de código.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from string import Template
import json

from config.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class PromptTemplate:
    """Template de prompt con variables."""
    name: str
    system_prompt: str
    user_prompt_template: str
    description: str = ""
    variables: List[str] = None
    default_temperature: float = 0.7
    default_max_tokens: Optional[int] = None
    
    def __post_init__(self):
        if self.variables is None:
            self.variables = []
    
    def render(self, **kwargs) -> Dict[str, str]:
        """
        Renderizar template con variables.
        
        Args:
            **kwargs: Variables para el template
            
        Returns:
            Diccionario con system_prompt y user_prompt
        """
        # Validar que todas las variables requeridas estén presentes
        template_vars = set(Template(self.user_prompt_template).get_identifiers())
        provided_vars = set(kwargs.keys())
        
        missing = template_vars - provided_vars
        if missing:
            logger.warning(f"Variables faltantes en template {self.name}: {missing}")
        
        # Renderizar user prompt
        user_template = Template(self.user_prompt_template)
        user_prompt = user_template.safe_substitute(**kwargs)
        
        # Renderizar system prompt si tiene variables
        if "$" in self.system_prompt:
            system_template = Template(self.system_prompt)
            system_prompt = system_template.safe_substitute(**kwargs)
        else:
            system_prompt = self.system_prompt
        
        return {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt
        }


class PromptTemplateRegistry:
    """Registry de templates de prompts."""
    
    def __init__(self):
        self.templates: Dict[str, PromptTemplate] = {}
        self._register_default_templates()
    
    def register(self, template: PromptTemplate) -> None:
        """Registrar un template."""
        self.templates[template.name] = template
        logger.debug(f"Template '{template.name}' registrado")
    
    def get(self, name: str) -> Optional[PromptTemplate]:
        """Obtener un template por nombre."""
        return self.templates.get(name)
    
    def render(self, name: str, **kwargs) -> Optional[Dict[str, str]]:
        """Renderizar un template por nombre."""
        template = self.get(name)
        if not template:
            logger.error(f"Template '{name}' no encontrado")
            return None
        return template.render(**kwargs)
    
    def _register_default_templates(self):
        """Registrar templates por defecto."""
        
        # Template de análisis de código
        self.register(PromptTemplate(
            name="code_analysis",
            system_prompt="Eres un experto analista de código. Analiza el código proporcionado y proporciona feedback constructivo sobre estructura, calidad y mejores prácticas.",
            user_prompt_template="""Analiza el siguiente código${language}:

```${language or ''}
${code}
```

Proporciona un análisis detallado enfocado en: ${analysis_type}. Incluye:
- Problemas identificados
- Sugerencias de mejora
- Ejemplos de código mejorado si es relevante""",
            description="Template para análisis de código",
            variables=["code", "language", "analysis_type"],
            default_temperature=0.3
        ))
        
        # Template de generación de documentación
        self.register(PromptTemplate(
            name="code_documentation",
            system_prompt="Eres un experto en documentación de código. Genera documentación completa, clara y profesional siguiendo las convenciones del lenguaje.",
            user_prompt_template="""Genera documentación para el siguiente código${language}:

```${language or ''}
${code}
```

Tipo de documentación: ${doc_type}""",
            description="Template para generar documentación",
            variables=["code", "language", "doc_type"],
            default_temperature=0.3
        ))
        
        # Template de refactorización
        self.register(PromptTemplate(
            name="code_refactor",
            system_prompt="Eres un experto en refactorización de código. Refactoriza manteniendo la funcionalidad pero mejorando calidad, rendimiento y mantenibilidad.",
            user_prompt_template="""Refactoriza el siguiente código${language}:

```${language or ''}
${code}
```

Tipo de refactorización: ${refactor_type}
Proporciona el código refactorizado y una explicación de los cambios.""",
            description="Template para refactorización",
            variables=["code", "language", "refactor_type"],
            default_temperature=0.4
        ))
        
        # Template de generación de tests
        self.register(PromptTemplate(
            name="code_tests",
            system_prompt="Eres un experto en testing. Genera tests completos, bien estructurados y comprehensivos.",
            user_prompt_template="""Genera tests ${test_type} para el siguiente código${language}:

```${language or ''}
${code}
```

Framework: ${test_framework}
Proporciona tests completos y bien estructurados.""",
            description="Template para generar tests",
            variables=["code", "language", "test_type", "test_framework"],
            default_temperature=0.3
        ))
        
        # Template de explicación
        self.register(PromptTemplate(
            name="code_explanation",
            system_prompt="Eres un experto en programación. Explica código de forma clara y educativa.",
            user_prompt_template="""Explica el siguiente código${language}:

```${language or ''}
${code}
```

Nivel de detalle: ${detail_level}""",
            description="Template para explicar código",
            variables=["code", "language", "detail_level"],
            default_temperature=0.5
        ))
        
        # Template de generación de código
        self.register(PromptTemplate(
            name="code_generation",
            system_prompt="Eres un experto programador. Genera código limpio, bien estructurado y siguiendo mejores prácticas.",
            user_prompt_template="""Genera código en ${language} que:

${description}${requirements}

El código debe ser funcional, bien documentado y seguir las convenciones del lenguaje.""",
            description="Template para generar código",
            variables=["language", "description", "requirements"],
            default_temperature=0.5
        ))
        
        logger.info(f"Registrados {len(self.templates)} templates por defecto")


# Instancia global del registry
_template_registry = PromptTemplateRegistry()


def get_template_registry() -> PromptTemplateRegistry:
    """Obtener el registry global de templates."""
    return _template_registry



