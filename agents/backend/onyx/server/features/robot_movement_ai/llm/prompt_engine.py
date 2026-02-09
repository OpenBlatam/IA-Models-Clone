"""
Prompt Engine - Motor de prompts
"""
from typing import Dict, Any, Optional


class PromptEngine:
    """Motor de gestión de prompts"""
    
    def __init__(self):
        self.prompt_templates: Dict[str, str] = {}
    
    def register_template(self, name: str, template: str):
        """Registra una plantilla de prompt"""
        self.prompt_templates[name] = template
    
    def render(self, template_name: str, **kwargs) -> str:
        """Renderiza una plantilla de prompt"""
        template = self.prompt_templates.get(template_name, "")
        return template.format(**kwargs)
    
    def build_prompt(self, system: str, user: str, context: Optional[str] = None) -> str:
        """Construye un prompt estructurado"""
        parts = []
        if system:
            parts.append(f"System: {system}")
        if context:
            parts.append(f"Context: {context}")
        if user:
            parts.append(f"User: {user}")
        return "\n\n".join(parts)

