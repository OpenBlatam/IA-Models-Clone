"""
Template Manager - Gestor de Templates
=======================================

Gestiona templates personalizados para proyectos.
Refactored with improved error handling and input validation.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

from .file_operations import read_json, write_json, FileOperationError

logger = logging.getLogger(__name__)


class TemplateManager:
    """Gestor de templates personalizados"""

    def __init__(self, templates_dir: Optional[Path] = None):
        """
        Inicializa el gestor de templates.

        Args:
            templates_dir: Directorio donde se almacenan los templates
            
        Raises:
            ValueError: If templates_dir is invalid
            FileOperationError: If directory cannot be created
        """
        if templates_dir is None:
            templates_dir = Path(__file__).parent.parent / "templates"
        
        if not isinstance(templates_dir, (str, Path)):
            raise ValueError("templates_dir must be a string or Path")
        
        self.templates_dir = Path(templates_dir)
        
        try:
            self.templates_dir.mkdir(parents=True, exist_ok=True)
        except (IOError, OSError) as e:
            raise FileOperationError(f"Cannot create templates directory: {e}") from e

    async def save_template(
        self,
        template_name: str,
        template_config: Dict[str, Any],
        description: str = "",
    ) -> Dict[str, Any]:
        """
        Guarda un template personalizado.

        Args:
            template_name: Nombre del template
            template_config: Configuración del template
            description: Descripción del template

        Returns:
            Información del template guardado
            
        Raises:
            ValueError: If inputs are invalid
            FileOperationError: If template cannot be saved
        """
        if not template_name or not isinstance(template_name, str):
            raise ValueError("template_name must be a non-empty string")
        
        if not isinstance(template_config, dict):
            raise ValueError("template_config must be a dictionary")
        
        if not isinstance(description, str):
            raise ValueError("description must be a string")
        
        try:
            template_file = self.templates_dir / f"{template_name}.json"
            
            from datetime import datetime
            template_data = {
                "name": template_name,
                "description": description,
                "config": template_config,
                "created_at": datetime.now().isoformat(),
            }

            write_json(template_file, template_data)

            logger.info(f"Template guardado: {template_name}")
            return {"success": True, "template_name": template_name}

        except FileOperationError as e:
            logger.error(f"Error guardando template: {e}")
            raise
        except Exception as e:
            logger.error(f"Error inesperado guardando template: {e}")
            raise FileOperationError(f"Error saving template: {e}") from e

    async def load_template(self, template_name: str) -> Optional[Dict[str, Any]]:
        """
        Carga un template.

        Args:
            template_name: Nombre del template

        Returns:
            Configuración del template o None
            
        Raises:
            ValueError: If template_name is invalid
        """
        if not template_name or not isinstance(template_name, str):
            raise ValueError("template_name must be a non-empty string")
        
        try:
            template_file = self.templates_dir / f"{template_name}.json"
            if not template_file.exists():
                return None

            return read_json(template_file)

        except FileOperationError as e:
            logger.error(f"Error cargando template: {e}")
            return None
        except Exception as e:
            logger.error(f"Error inesperado cargando template: {e}")
            return None

    async def list_templates(self) -> List[Dict[str, Any]]:
        """
        Lista todos los templates disponibles.

        Returns:
            Lista de templates
        """
        templates = []
        for template_file in self.templates_dir.glob("*.json"):
            try:
                template_data = read_json(template_file, default=None)
                if template_data is not None:
                    templates.append({
                        "name": template_data.get("name", template_file.stem),
                        "description": template_data.get("description", ""),
                    })
            except FileOperationError as e:
                logger.warning(f"Error reading template {template_file}: {e}")
                continue
            except Exception as e:
                logger.warning(f"Unexpected error reading template {template_file}: {e}")
                continue

        return templates

    async def delete_template(self, template_name: str) -> bool:
        """
        Elimina un template.

        Args:
            template_name: Nombre del template

        Returns:
            True si se eliminó exitosamente
            
        Raises:
            ValueError: If template_name is invalid
        """
        if not template_name or not isinstance(template_name, str):
            raise ValueError("template_name must be a non-empty string")
        
        try:
            template_file = self.templates_dir / f"{template_name}.json"
            if template_file.exists():
                template_file.unlink()
                logger.info(f"Template eliminado: {template_name}")
                return True
            return False
        except (IOError, OSError) as e:
            logger.error(f"Error eliminando template: {e}")
            return False
        except Exception as e:
            logger.error(f"Error inesperado eliminando template: {e}")
            return False

