"""
Templates - Sistema de plantillas para comandos
================================================

Sistema de plantillas reutilizables para comandos comunes.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CommandTemplate:
    """Plantilla de comando"""
    id: str
    name: str
    description: str
    template: str
    variables: List[str] = field(default_factory=list)
    category: str = "general"
    created_at: datetime = field(default_factory=datetime.now)
    usage_count: int = 0


class TemplateManager:
    """Gestor de plantillas"""
    
    def __init__(self):
        self.templates: Dict[str, CommandTemplate] = {}
        self._load_default_templates()
    
    def _load_default_templates(self):
        """Cargar plantillas por defecto"""
        default_templates = [
            CommandTemplate(
                id="hello_world",
                name="Hello World",
                description="Comando simple de saludo",
                template="print('Hello, World!')",
                category="basic"
            ),
            CommandTemplate(
                id="list_files",
                name="List Files",
                description="Listar archivos en directorio",
                template="""from pathlib import Path
files = list(Path('{directory}').glob('*'))
for f in files[:10]:
    print(f"  {f.name}")""",
                variables=["directory"],
                category="file_operations"
            ),
            CommandTemplate(
                id="api_call",
                name="API Call",
                description="Llamar a una API",
                template="""import requests
response = requests.get('{url}')
print(response.json())""",
                variables=["url"],
                category="network"
            ),
            CommandTemplate(
                id="process_data",
                name="Process Data",
                description="Procesar datos",
                template="""data = {data}
result = [item * 2 for item in data]
print(f"Processed: {result}")""",
                variables=["data"],
                category="data_processing"
            )
        ]
        
        for template in default_templates:
            self.templates[template.id] = template
    
    def create_template(
        self,
        name: str,
        description: str,
        template: str,
        variables: Optional[List[str]] = None,
        category: str = "general"
    ) -> str:
        """Crear nueva plantilla"""
        template_id = f"template_{datetime.now().timestamp()}_{len(self.templates)}"
        
        cmd_template = CommandTemplate(
            id=template_id,
            name=name,
            description=description,
            template=template,
            variables=variables or [],
            category=category
        )
        
        self.templates[template_id] = cmd_template
        logger.info(f"📝 Template created: {name}")
        return template_id
    
    def render_template(
        self,
        template_id: str,
        variables: Dict[str, Any]
    ) -> Optional[str]:
        """Renderizar plantilla con variables"""
        if template_id not in self.templates:
            logger.error(f"Template not found: {template_id}")
            return None
        
        template = self.templates[template_id]
        
        try:
            # Renderizar plantilla
            rendered = template.template.format(**variables)
            
            # Incrementar contador de uso
            template.usage_count += 1
            
            return rendered
        except KeyError as e:
            logger.error(f"Missing variable in template: {e}")
            return None
        except Exception as e:
            logger.error(f"Error rendering template: {e}")
            return None
    
    def get_template(self, template_id: str) -> Optional[CommandTemplate]:
        """Obtener plantilla"""
        return self.templates.get(template_id)
    
    def list_templates(self, category: Optional[str] = None) -> List[Dict]:
        """Listar plantillas"""
        templates = list(self.templates.values())
        
        if category:
            templates = [t for t in templates if t.category == category]
        
        return [
            {
                "id": t.id,
                "name": t.name,
                "description": t.description,
                "category": t.category,
                "variables": t.variables,
                "usage_count": t.usage_count
            }
            for t in templates
        ]
    
    def delete_template(self, template_id: str) -> bool:
        """Eliminar plantilla"""
        if template_id in self.templates:
            del self.templates[template_id]
            logger.info(f"🗑️ Template deleted: {template_id}")
            return True
        return False



