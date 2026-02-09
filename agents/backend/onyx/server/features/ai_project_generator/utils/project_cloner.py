"""
Project Cloner - Clonador de Proyectos
=======================================

Permite clonar y duplicar proyectos existentes.
"""

import logging
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class ProjectCloner:
    """Clonador de proyectos"""

    def __init__(self):
        """Inicializa el clonador"""
        pass

    async def clone_project(
        self,
        source_path: Path,
        target_path: Optional[Path] = None,
        new_name: Optional[str] = None,
        update_config: bool = True,
    ) -> Dict[str, Any]:
        """
        Clona un proyecto existente.

        Args:
            source_path: Ruta del proyecto original
            target_path: Ruta destino (opcional)
            new_name: Nuevo nombre del proyecto (opcional)
            update_config: Si actualizar configuraciones

        Returns:
            Información del proyecto clonado
        """
        try:
            if not source_path.exists():
                raise ValueError(f"Proyecto fuente no existe: {source_path}")

            # Determinar ruta destino
            if target_path is None:
                if new_name:
                    target_path = source_path.parent / new_name
                else:
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    target_path = source_path.parent / f"{source_path.name}_clone_{timestamp}"

            # Copiar proyecto
            logger.info(f"Clonando proyecto de {source_path} a {target_path}")
            shutil.copytree(
                source_path,
                target_path,
                ignore=shutil.ignore_patterns(
                    '__pycache__', '*.pyc', '.git', 'node_modules',
                    '.pytest_cache', '.coverage', 'dist', 'build', 'venv', '.venv'
                )
            )

            # Actualizar configuraciones si es necesario
            if update_config:
                await self._update_project_config(target_path, new_name)

            # Leer project_info
            project_info_path = target_path / "project_info.json"
            if project_info_path.exists():
                project_info = json.loads(
                    project_info_path.read_text(encoding="utf-8")
                )
            else:
                project_info = {}

            project_info.update({
                "cloned_from": str(source_path),
                "cloned_at": datetime.now().isoformat(),
                "is_clone": True,
            })

            # Guardar project_info actualizado
            project_info_path.write_text(
                json.dumps(project_info, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )

            logger.info(f"Proyecto clonado exitosamente en: {target_path}")

            return {
                "success": True,
                "source_path": str(source_path),
                "target_path": str(target_path),
                "project_info": project_info,
            }

        except Exception as e:
            logger.error(f"Error clonando proyecto: {e}", exc_info=True)
            raise

    async def _update_project_config(
        self,
        project_path: Path,
        new_name: Optional[str],
    ):
        """Actualiza configuraciones del proyecto clonado"""
        if new_name:
            # Actualizar project_info.json
            info_path = project_path / "project_info.json"
            if info_path.exists():
                info = json.loads(info_path.read_text(encoding="utf-8"))
                info["name"] = new_name
                info_path.write_text(
                    json.dumps(info, indent=2, ensure_ascii=False),
                    encoding="utf-8"
                )

            # Actualizar package.json del frontend
            package_path = project_path / "frontend" / "package.json"
            if package_path.exists():
                package = json.loads(package_path.read_text(encoding="utf-8"))
                package["name"] = new_name
                package_path.write_text(
                    json.dumps(package, indent=2),
                    encoding="utf-8"
                )

            # Actualizar README.md
            readme_path = project_path / "README.md"
            if readme_path.exists():
                readme = readme_path.read_text(encoding="utf-8")
                readme = readme.replace(
                    readme.split('\n')[0],
                    f"# {new_name.replace('_', ' ').title()}"
                )
                readme_path.write_text(readme, encoding="utf-8")


