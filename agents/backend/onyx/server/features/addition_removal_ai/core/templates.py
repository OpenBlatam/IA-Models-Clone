"""
Templates - Sistema de plantillas
"""

import logging
import re
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class Template:
    """Plantilla de contenido"""

    def __init__(self, name: str, content: str, variables: Optional[List[str]] = None):
        """
        Inicializar plantilla.

        Args:
            name: Nombre de la plantilla
            content: Contenido de la plantilla
            variables: Lista de variables
        """
        self.name = name
        self.content = content
        self.variables = variables or self._extract_variables(content)

    def _extract_variables(self, content: str) -> List[str]:
        """
        Extraer variables de la plantilla.

        Args:
            content: Contenido

        Returns:
            Lista de variables
        """
        # Buscar patrones {{variable}}
        pattern = r'\{\{(\w+)\}\}'
        variables = re.findall(pattern, content)
        return list(set(variables))

    def render(self, context: Dict[str, Any]) -> str:
        """
        Renderizar plantilla con contexto.

        Args:
            context: Contexto con valores

        Returns:
            Contenido renderizado
        """
        result = self.content
        
        for variable in self.variables:
            value = context.get(variable, f"{{{{{variable}}}}}")
            result = result.replace(f"{{{{{variable}}}}}", str(value))
        
        return result


class TemplateManager:
    """Gestor de plantillas"""

    def __init__(self, templates_dir: Optional[Path] = None):
        """
        Inicializar el gestor de plantillas.

        Args:
            templates_dir: Directorio de plantillas
        """
        if templates_dir is None:
            templates_dir = Path(__file__).parent.parent / "templates"
        
        self.templates_dir = Path(templates_dir)
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        self.templates: Dict[str, Template] = {}
        self._load_templates()

    def _load_templates(self):
        """Cargar plantillas del directorio"""
        if not self.templates_dir.exists():
            return
        
        for template_file in self.templates_dir.glob("*.txt"):
            try:
                with open(template_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                name = template_file.stem
                template = Template(name, content)
                self.templates[name] = template
                logger.info(f"Plantilla cargada: {name}")
            except Exception as e:
                logger.error(f"Error cargando plantilla {template_file}: {e}")

    def register_template(self, name: str, content: str) -> Template:
        """
        Registrar una plantilla.

        Args:
            name: Nombre
            content: Contenido

        Returns:
            Plantilla creada
        """
        template = Template(name, content)
        self.templates[name] = template
        
        # Guardar en disco
        template_path = self.templates_dir / f"{name}.txt"
        with open(template_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"Plantilla registrada: {name}")
        return template

    def get_template(self, name: str) -> Optional[Template]:
        """
        Obtener una plantilla.

        Args:
            name: Nombre de la plantilla

        Returns:
            Plantilla o None
        """
        return self.templates.get(name)

    def render_template(
        self,
        name: str,
        context: Dict[str, Any]
    ) -> Optional[str]:
        """
        Renderizar una plantilla.

        Args:
            name: Nombre de la plantilla
            context: Contexto

        Returns:
            Contenido renderizado o None
        """
        template = self.get_template(name)
        if not template:
            return None
        
        return template.render(context)

    def list_templates(self) -> List[Dict[str, Any]]:
        """
        Listar todas las plantillas.

        Returns:
            Lista de plantillas
        """
        return [
            {
                "name": name,
                "variables": template.variables,
                "content_preview": template.content[:100] + "..." if len(template.content) > 100 else template.content
            }
            for name, template in self.templates.items()
        ]

    def delete_template(self, name: str) -> bool:
        """
        Eliminar una plantilla.

        Args:
            name: Nombre de la plantilla

        Returns:
            True si se eliminó
        """
        if name not in self.templates:
            return False
        
        del self.templates[name]
        
        # Eliminar archivo
        template_path = self.templates_dir / f"{name}.txt"
        if template_path.exists():
            template_path.unlink()
        
        logger.info(f"Plantilla eliminada: {name}")
        return True






