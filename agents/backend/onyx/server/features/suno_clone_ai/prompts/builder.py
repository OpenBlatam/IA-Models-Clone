"""
Prompt Builder - Constructor de prompts
"""

from typing import Dict, Any, List
from .base import BasePrompt


class PromptBuilder:
    """Constructor de prompts dinámicos"""

    def __init__(self):
        """Inicializa el constructor de prompts"""
        pass

    def build(self, parts: List[str], separator: str = "\n") -> str:
        """Construye un prompt desde partes"""
        return separator.join(parts)

    def build_with_context(self, prompt: str, context: Dict[str, Any]) -> str:
        """Construye un prompt con contexto"""
        context_str = "\n".join([f"{k}: {v}" for k, v in context.items()])
        return f"{context_str}\n\n{prompt}"

