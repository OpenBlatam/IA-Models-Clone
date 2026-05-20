"""
Templates - Sistema de plantillas
==================================

Sistema de plantillas para mensajes y respuestas predefinidas.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from string import Template

logger = logging.getLogger(__name__)


@dataclass
class MessageTemplate:
    """Plantilla de mensaje."""
    id: str
    name: str
    content: str
    variables: List[str] = field(default_factory=list)
    category: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


class TemplateManager:
    """Gestor de plantillas."""
    
    def __init__(self):
        self.templates: Dict[str, MessageTemplate] = {}
        self._load_default_templates()
    
    def _load_default_templates(self):
        """Cargar plantillas por defecto."""
        default_templates = [
            MessageTemplate(
                id="greeting",
                name="Saludo",
                content="Hola! ¿En qué puedo ayudarte hoy?",
                category="greeting",
            ),
            MessageTemplate(
                id="farewell",
                name="Despedida",
                content="¡Gracias por usar nuestro servicio! Hasta pronto.",
                category="farewell",
            ),
            MessageTemplate(
                id="error",
                name="Error",
                content="Lo siento, ha ocurrido un error: ${error_message}",
                variables=["error_message"],
                category="error",
            ),
            MessageTemplate(
                id="help",
                name="Ayuda",
                content="Puedo ayudarte con: ${topics}. ¿Qué te gustaría saber?",
                variables=["topics"],
                category="help",
            ),
        ]
        
        for template in default_templates:
            self.templates[template.id] = template
    
    def register(self, template: MessageTemplate):
        """Registrar una plantilla."""
        self.templates[template.id] = template
        logger.info(f"Registered template: {template.id}")
    
    def get(self, template_id: str) -> Optional[MessageTemplate]:
        """Obtener una plantilla."""
        return self.templates.get(template_id)
    
    def render(self, template_id: str, variables: Optional[Dict[str, str]] = None) -> Optional[str]:
        """
        Renderizar una plantilla con variables.
        
        Args:
            template_id: ID de la plantilla
            variables: Variables para reemplazar
        
        Returns:
            Mensaje renderizado o None si no existe
        """
        template = self.get(template_id)
        if not template:
            return None
        
        variables = variables or {}
        
        try:
            template_obj = Template(template.content)
            return template_obj.safe_substitute(variables)
        except Exception as e:
            logger.error(f"Error rendering template {template_id}: {e}")
            return template.content
    
    def list_templates(self, category: Optional[str] = None) -> List[MessageTemplate]:
        """Listar plantillas."""
        templates = list(self.templates.values())
        
        if category:
            templates = [t for t in templates if t.category == category]
        
        return templates
    
    def delete(self, template_id: str) -> bool:
        """Eliminar una plantilla."""
        if template_id in self.templates:
            del self.templates[template_id]
            logger.info(f"Deleted template: {template_id}")
            return True
        return False
    
    def update(self, template_id: str, **kwargs) -> bool:
        """Actualizar una plantilla."""
        template = self.get(template_id)
        if not template:
            return False
        
        for key, value in kwargs.items():
            if hasattr(template, key):
                setattr(template, key, value)
        
        logger.info(f"Updated template: {template_id}")
        return True
































