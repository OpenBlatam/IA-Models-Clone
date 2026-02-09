"""
Prompt Service - Servicio de prompts
"""

from typing import Dict, Any, Optional
from .base import BasePrompt
from .template import PromptTemplate
from .builder import PromptBuilder


class PromptService:
    """Servicio para gestionar prompts"""

    def __init__(self):
        """Inicializa el servicio de prompts"""
        self.builder = PromptBuilder()
        self._templates: Dict[str, PromptTemplate] = {}

    def register_template(self, name: str, template: PromptTemplate) -> None:
        """Registra un template de prompt"""
        self._templates[name] = template

    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """Obtiene un template por nombre"""
        return self._templates.get(name)

    def build_prompt(self, template_name: str, **kwargs) -> str:
        """Construye un prompt desde un template"""
        template = self.get_template(template_name)
        if not template:
            raise ValueError(f"Template '{template_name}' not found")
        return template.build(**kwargs)

