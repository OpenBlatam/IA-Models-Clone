"""
Prompt Template - Templates de prompts
"""

from typing import Dict, Any
from .base import BasePrompt


class PromptTemplate(BasePrompt):
    """Template de prompt con variables"""

    def __init__(self, template: str):
        """Inicializa el template"""
        self.template = template

    def build(self, **kwargs) -> str:
        """Construye el prompt con los parámetros dados"""
        return self.template.format(**kwargs)

    def validate(self, **kwargs) -> bool:
        """Valida los parámetros del prompt"""
        try:
            self.template.format(**kwargs)
            return True
        except KeyError:
            return False

