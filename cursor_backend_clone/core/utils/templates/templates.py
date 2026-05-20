"""
Templates - Sistema de Plantillas
==================================

Sistema de plantillas para respuestas y mensajes consistentes.
"""

import logging
from typing import Dict, Any, Optional
from string import Template
from datetime import datetime

logger = logging.getLogger(__name__)


class ResponseTemplate:
    """Plantilla para respuestas"""
    
    def __init__(
        self,
        template: str,
        default_vars: Optional[Dict[str, Any]] = None
    ):
        """
        Inicializar plantilla.
        
        Args:
            template: Template string con placeholders ${var}
            default_vars: Variables por defecto
        """
        self.template = Template(template)
        self.default_vars = default_vars or {}
    
    def render(self, **kwargs) -> str:
        """
        Renderizar plantilla con variables.
        
        Args:
            **kwargs: Variables para la plantilla
            
        Returns:
            String renderizado
        """
        vars_dict = {**self.default_vars, **kwargs}
        return self.template.safe_substitute(vars_dict)


class TemplateManager:
    """Gestor de plantillas"""
    
    def __init__(self):
        self.templates: Dict[str, ResponseTemplate] = {}
        self._register_default_templates()
    
    def register(self, name: str, template: ResponseTemplate) -> None:
        """Registrar plantilla"""
        self.templates[name] = template
        logger.debug(f"📝 Template registered: {name}")
    
    def get(self, name: str) -> Optional[ResponseTemplate]:
        """Obtener plantilla"""
        return self.templates.get(name)
    
    def render(self, name: str, **kwargs) -> str:
        """
        Renderizar plantilla por nombre.
        
        Args:
            name: Nombre de la plantilla
            **kwargs: Variables
            
        Returns:
            String renderizado
        """
        template = self.get(name)
        if not template:
            logger.warning(f"Template '{name}' not found")
            return f"[Template '{name}' not found]"
        
        return template.render(**kwargs)
    
    def _register_default_templates(self) -> None:
        """Registrar plantillas por defecto"""
        
        # Plantilla de respuesta de tarea
        self.register("task_response", ResponseTemplate(
            "Task ${task_id}: ${status}\n"
            "Command: ${command}\n"
            "${if result}Result: ${result}${endif}\n"
            "${if error}Error: ${error}${endif}"
        ))
        
        # Plantilla de error
        self.register("error", ResponseTemplate(
            "❌ Error: ${message}\n"
            "${if details}Details: ${details}${endif}\n"
            "${if timestamp}Time: ${timestamp}${endif}"
        ))
        
        # Plantilla de éxito
        self.register("success", ResponseTemplate(
            "✅ Success: ${message}\n"
            "${if details}Details: ${details}${endif}"
        ))
        
        # Plantilla de estado del agente
        self.register("agent_status", ResponseTemplate(
            "Agent Status: ${status}\n"
            "Running: ${running}\n"
            "Tasks: ${tasks_total} total, ${tasks_pending} pending, "
            "${tasks_completed} completed, ${tasks_failed} failed"
        ))
        
        # Plantilla de métricas
        self.register("metrics", ResponseTemplate(
            "Metrics:\n"
            "Tasks per second: ${tasks_per_second:.2f}\n"
            "Success rate: ${success_rate:.1f}%\n"
            "Average execution time: ${avg_time:.3f}s"
        ))


# Instancia global
_template_manager = TemplateManager()


def get_template(name: str) -> Optional[ResponseTemplate]:
    """Obtener plantilla global"""
    return _template_manager.get(name)


def render_template(name: str, **kwargs) -> str:
    """Renderizar plantilla global"""
    return _template_manager.render(name, **kwargs)


def register_template(name: str, template: ResponseTemplate) -> None:
    """Registrar plantilla global"""
    _template_manager.register(name, template)
