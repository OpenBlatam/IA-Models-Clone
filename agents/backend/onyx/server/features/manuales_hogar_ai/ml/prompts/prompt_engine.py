"""
Prompt Engine
=============

Motor de prompts mejorado con templates y variables.
"""

import logging
from typing import Dict, Any, Optional, List
from jinja2 import Template, Environment, FileSystemLoader
import os

logger = logging.getLogger(__name__)


class PromptEngine:
    """Motor de prompts con templates."""
    
    def __init__(self, templates_dir: Optional[str] = None):
        """
        Inicializar motor de prompts.
        
        Args:
            templates_dir: Directorio de templates
        """
        if templates_dir:
            self.env = Environment(loader=FileSystemLoader(templates_dir))
        else:
            self.env = Environment()
        
        self.templates: Dict[str, Template] = {}
        self._logger = logger
    
    def register_template(self, name: str, template_str: str):
        """
        Registrar template.
        
        Args:
            name: Nombre del template
            template_str: String del template
        """
        self.templates[name] = self.env.from_string(template_str)
        self._logger.info(f"Template '{name}' registrado")
    
    def render(
        self,
        template_name: str,
        context: Dict[str, Any]
    ) -> str:
        """
        Renderizar template.
        
        Args:
            template_name: Nombre del template
            context: Contexto para renderizar
        
        Returns:
            Prompt renderizado
        """
        try:
            if template_name not in self.templates:
                raise ValueError(f"Template '{template_name}' no encontrado")
            
            template = self.templates[template_name]
            return template.render(**context)
        
        except Exception as e:
            self._logger.error(f"Error renderizando template: {str(e)}")
            raise
    
    def get_default_manual_template(self) -> str:
        """Obtener template por defecto para manuales."""
        return """Genera un manual paso a paso tipo LEGO para {{ category_name }}.

PROBLEMA:
{{ problem_description }}

{% if context %}
CONTEXTO DE MANUALES SIMILARES:
{% for item in context %}
--- Manual Similar {{ loop.index }} ---
Problema: {{ item.problem }}
Solución: {{ item.content }}
{% endfor %}
{% endif %}

INSTRUCCIONES:
- Usa el formato paso a paso tipo LEGO
- Sé claro y conciso
- Incluye advertencias de seguridad cuando sea necesario
- Lista herramientas y materiales requeridos
- Proporciona estimación de tiempo

MANUAL:
"""
    
    def build_manual_prompt(
        self,
        problem_description: str,
        category: str = "general",
        context: Optional[List[Dict[str, Any]]] = None,
        include_safety: bool = True,
        include_tools: bool = True,
        include_materials: bool = True
    ) -> str:
        """
        Construir prompt para manual.
        
        Args:
            problem_description: Descripción del problema
            category: Categoría
            context: Contexto adicional
            include_safety: Incluir seguridad
            include_tools: Incluir herramientas
            include_materials: Incluir materiales
        
        Returns:
            Prompt completo
        """
        category_names = {
            "plomeria": "Plomería",
            "techos": "Reparación de Techos",
            "carpinteria": "Carpintería",
            "electricidad": "Electricidad",
            "albanileria": "Albañilería",
            "pintura": "Pintura",
            "herreria": "Herrería",
            "jardineria": "Jardinería",
            "general": "Reparación General"
        }
        
        context_data = []
        if context:
            for item in context:
                context_data.append({
                    "problem": item.get("problem", ""),
                    "content": item.get("content", "")[:500]
                })
        
        template_str = self.get_default_manual_template()
        template = self.env.from_string(template_str)
        
        return template.render(
            category_name=category_names.get(category, "Reparación General"),
            problem_description=problem_description,
            context=context_data,
            include_safety=include_safety,
            include_tools=include_tools,
            include_materials=include_materials
        )




