"""
Project Versioning - Sistema de Versionado de Proyectos
=======================================================

Gestiona versiones de proyectos generados.
Refactored to use safe file operations with proper error handling.
"""

import logging
import shutil
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
import hashlib

from .file_operations import read_json, write_json, FileOperationError

logger = logging.getLogger(__name__)


class ProjectVersioning:
    """Sistema de versionado de proyectos"""

    def __init__(self, versions_dir: Optional[Path] = None):
        """
        Inicializa el sistema de versionado.

        Args:
            versions_dir: Directorio para almacenar versiones
            
        Raises:
            ValueError: If versions_dir is invalid
            FileOperationError: If directory cannot be created
        """
        if versions_dir is None:
            versions_dir = Path("project_versions")
        
        if not isinstance(versions_dir, (str, Path)):
            raise ValueError("versions_dir must be a string or Path")
        
        self.versions_dir = Path(versions_dir)
        
        try:
            self.versions_dir.mkdir(parents=True, exist_ok=True)
        except (IOError, OSError) as e:
            raise FileOperationError(f"Cannot create versions directory: {e}") from e

    def create_version(
        self,
        project_id: str,
        project_path: Path,
        version: str = "1.0.0",
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Crea una nueva versión de un proyecto.

        Args:
            project_id: ID del proyecto
            project_path: Ruta del proyecto
            version: Versión (semver)
            description: Descripción de la versión
            metadata: Metadata adicional

        Returns:
            Información de la versión creada
        """
        version_dir = self.versions_dir / project_id / version
        version_dir.mkdir(parents=True, exist_ok=True)

        # Copiar proyecto
        if project_path.exists():
            dest_path = version_dir / "project"
            if dest_path.exists():
                shutil.rmtree(dest_path)
            shutil.copytree(project_path, dest_path)

        # Calcular hash
        project_hash = self._calculate_hash(project_path)

        version_info = {
            "project_id": project_id,
            "version": version,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "project_hash": project_hash,
            "metadata": metadata or {},
        }

        metadata_file = version_dir / "version_info.json"
        
        try:
            write_json(metadata_file, version_info)
        except FileOperationError as e:
            logger.error(f"Error saving version metadata: {e}")
            raise
        
        logger.info(f"Versión {version} creada para proyecto {project_id}")
        return version_info

    def list_versions(
        self,
        project_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Lista todas las versiones de un proyecto.

        Args:
            project_id: ID del proyecto

        Returns:
            Lista de versiones
        """
        project_versions_dir = self.versions_dir / project_id
        if not project_versions_dir.exists():
            return []

        versions = []
        for version_dir in project_versions_dir.iterdir():
            if not version_dir.is_dir():
                continue
                
            metadata_file = version_dir / "version_info.json"
            if not metadata_file.exists():
                continue
            
            try:
                version_info = read_json(metadata_file, default=None)
                if version_info is not None:
                    versions.append(version_info)
            except FileOperationError as e:
                logger.error(f"Error leyendo versión {version_dir}: {e}")

        # Ordenar por fecha (más recientes primero)
        versions.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return versions

    def get_version(
        self,
        project_id: str,
        version: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Obtiene información de una versión específica.

        Args:
            project_id: ID del proyecto
            version: Versión

        Returns:
            Información de la versión o None
        """
        version_dir = self.versions_dir / project_id / version
        metadata_file = version_dir / "version_info.json"

        if not metadata_file.exists():
            return None

        try:
            return read_json(metadata_file)
        except FileOperationError as e:
            logger.error(f"Error leyendo versión {version}: {e}")
            return None

    def restore_version(
        self,
        project_id: str,
        version: str,
        restore_to: Path,
    ) -> bool:
        """
        Restaura una versión de un proyecto.

        Args:
            project_id: ID del proyecto
            version: Versión a restaurar
            restore_to: Directorio donde restaurar

        Returns:
            True si se restauró exitosamente
        """
        version_dir = self.versions_dir / project_id / version
        project_source = version_dir / "project"

        if not project_source.exists():
            return False

        try:
            restore_to = Path(restore_to)
            if restore_to.exists():
                shutil.rmtree(restore_to)
            shutil.copytree(project_source, restore_to)
            logger.info(f"Versión {version} restaurada para proyecto {project_id}")
            return True
        except (IOError, OSError, shutil.Error) as e:
            logger.error(f"Error restaurando versión {version}: {e}")
            return False
        except Exception as e:
            logger.error(f"Error inesperado restaurando versión {version}: {e}")
            return False

    def compare_versions(
        self,
        project_id: str,
        version1: str,
        version2: str,
    ) -> Dict[str, Any]:
        """
        Compara dos versiones de un proyecto.

        Args:
            project_id: ID del proyecto
            version1: Primera versión
            version2: Segunda versión

        Returns:
            Comparación de versiones
        """
        v1_info = self.get_version(project_id, version1)
        v2_info = self.get_version(project_id, version2)

        if not v1_info or not v2_info:
            return {"error": "Una o ambas versiones no existen"}

        return {
            "version1": v1_info,
            "version2": v2_info,
            "same_hash": v1_info.get("project_hash") == v2_info.get("project_hash"),
            "differences": self._calculate_differences(v1_info, v2_info),
        }

    def _calculate_hash(self, project_path: Path) -> str:
        """
        Calcula hash de un proyecto.
        
        Args:
            project_path: Path to project
            
        Returns:
            MD5 hash string
        """
        if not project_path.exists():
            return ""

        hash_data = []
        try:
            for file_path in sorted(project_path.rglob("*")):
                if file_path.is_file():
                    try:
                        stat = file_path.stat()
                        hash_data.append(f"{file_path.name}:{stat.st_size}:{stat.st_mtime}")
                    except (IOError, OSError):
                        pass
        except (IOError, OSError) as e:
            logger.warning(f"Error calculating hash: {e}")
            return ""

        hash_str = "\n".join(hash_data)
        return hashlib.md5(hash_str.encode()).hexdigest()

    def _calculate_differences(
        self,
        v1_info: Dict[str, Any],
        v2_info: Dict[str, Any],
    ) -> List[str]:
        """Calcula diferencias entre versiones"""
        differences = []

        if v1_info.get("description") != v2_info.get("description"):
            differences.append("description changed")

        if v1_info.get("project_hash") != v2_info.get("project_hash"):
            differences.append("project files changed")

        metadata1 = v1_info.get("metadata", {})
        metadata2 = v2_info.get("metadata", {})
        if metadata1 != metadata2:
            differences.append("metadata changed")

        return differences


