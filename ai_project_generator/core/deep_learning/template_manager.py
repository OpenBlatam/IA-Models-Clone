"""
Template Manager
================

Gestor de templates personalizados para generación de código.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import json
import jinja2

logger = logging.getLogger(__name__)


@dataclass
class Template:
    """Template de código."""
    name: str
    content: str
    variables: List[str]
    description: str = ""
    category: str = "default"


class TemplateManager:
    """
    Gestor de templates personalizados.
    """
    
    def __init__(self, templates_dir: Optional[Path] = None):
        """
        Inicializar gestor de templates.
        
        Args:
            templates_dir: Directorio de templates (opcional)
        """
        self.templates: Dict[str, Template] = {}
        self.templates_dir = templates_dir or Path.cwd() / "templates"
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        
        # Cargar templates del directorio
        self._load_templates()
    
    def _load_templates(self) -> None:
        """Cargar templates del directorio."""
        if not self.templates_dir.exists():
            return
        
        for template_file in self.templates_dir.glob("*.json"):
            try:
                with open(template_file, 'r') as f:
                    data = json.load(f)
                
                template = Template(
                    name=data['name'],
                    content=data['content'],
                    variables=data.get('variables', []),
                    description=data.get('description', ''),
                    category=data.get('category', 'default')
                )
                
                self.templates[template.name] = template
                logger.info(f"Template cargado: {template.name}")
            except Exception as e:
                logger.warning(f"Error cargando template {template_file}: {e}")
    
    def register_template(
        self,
        name: str,
        content: str,
        variables: List[str],
        description: str = "",
        category: str = "default"
    ) -> None:
        """
        Registrar template.
        
        Args:
            name: Nombre del template
            content: Contenido del template (Jinja2)
            variables: Lista de variables del template
            description: Descripción
            category: Categoría
        """
        template = Template(
            name=name,
            content=content,
            variables=variables,
            description=description,
            category=category
        )
        
        self.templates[name] = template
        self._save_template(template)
        logger.info(f"Template registrado: {name}")
    
    def get_template(self, name: str) -> Optional[Template]:
        """
        Obtener template.
        
        Args:
            name: Nombre del template
            
        Returns:
            Template o None
        """
        return self.templates.get(name)
    
    def render_template(
        self,
        name: str,
        context: Dict[str, Any]
    ) -> str:
        """
        Renderizar template.
        
        Args:
            name: Nombre del template
            context: Contexto para renderizado
            
        Returns:
            Código renderizado
        """
        template = self.get_template(name)
        if not template:
            raise ValueError(f"Template '{name}' no encontrado")
        
        jinja_env = jinja2.Environment(
            loader=jinja2.DictLoader({name: template.content}),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        jinja_template = jinja_env.get_template(name)
        return jinja_template.render(**context)
    
    def list_templates(self, category: Optional[str] = None) -> List[Template]:
        """
        Listar templates.
        
        Args:
            category: Filtrar por categoría (opcional)
            
        Returns:
            Lista de templates
        """
        templates = list(self.templates.values())
        
        if category:
            templates = [t for t in templates if t.category == category]
        
        return templates
    
    def _save_template(self, template: Template) -> None:
        """Guardar template en disco."""
        try:
            template_file = self.templates_dir / f"{template.name}.json"
            data = {
                'name': template.name,
                'content': template.content,
                'variables': template.variables,
                'description': template.description,
                'category': template.category
            }
            
            with open(template_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Error guardando template: {e}")


# Instancia global
_global_template_manager: Optional[TemplateManager] = None


def get_template_manager(templates_dir: Optional[Path] = None) -> TemplateManager:
    """
    Obtener instancia global del gestor de templates.
    
    Args:
        templates_dir: Directorio de templates (solo primera llamada)
        
    Returns:
        Instancia del gestor
    """
    global _global_template_manager
    
    if _global_template_manager is None:
        _global_template_manager = TemplateManager(templates_dir)
    
    return _global_template_manager

