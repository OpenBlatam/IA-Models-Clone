"""
Backup Manager - Gestor de Backups
==================================

Gestiona backups de proyectos y configuración.
"""

import logging
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import tarfile

from .file_operations import write_json, read_json, FileOperationError

logger = logging.getLogger(__name__)


class BackupManager:
    """Gestor de backups"""

    def __init__(self, backup_dir: Path = None):
        """
        Inicializa el gestor de backups.

        Args:
            backup_dir: Directorio para almacenar backups
        """
        if backup_dir is None:
            backup_dir = Path(__file__).parent.parent / "backups"
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    async def create_backup(
        self,
        projects_dir: Path,
        include_cache: bool = True,
        include_queue: bool = True,
    ) -> Dict[str, Any]:
        """
        Crea un backup completo.

        Args:
            projects_dir: Directorio de proyectos
            include_cache: Incluir cache
            include_queue: Incluir cola

        Returns:
            Información del backup creado
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_name = f"backup_{timestamp}"
            backup_path = self.backup_dir / f"{backup_name}.tar.gz"

            with tarfile.open(backup_path, 'w:gz') as tar:
                # Backup de proyectos
                if projects_dir.exists():
                    tar.add(projects_dir, arcname="projects")

                # Backup de cache
                if include_cache:
                    cache_dir = Path(__file__).parent.parent / "cache"
                    if cache_dir.exists():
                        tar.add(cache_dir, arcname="cache")

                # Backup de cola
                if include_queue:
                    queue_file = Path("project_queue.json")
                    if queue_file.exists():
                        tar.add(queue_file, arcname="project_queue.json")

            backup_info = {
                "backup_name": backup_name,
                "backup_path": str(backup_path),
                "size_bytes": backup_path.stat().st_size,
                "created_at": datetime.now().isoformat(),
                "includes": {
                    "projects": True,
                    "cache": include_cache,
                    "queue": include_queue,
                },
            }

            metadata_path = self.backup_dir / f"{backup_name}_info.json"
            
            try:
                write_json(metadata_path, backup_info)
            except FileOperationError as e:
                logger.error(f"Error saving backup metadata: {e}")
                raise

            logger.info(f"Backup creado: {backup_name}")
            return backup_info

        except Exception as e:
            logger.error(f"Error creando backup: {e}", exc_info=True)
            raise

    async def restore_backup(
        self,
        backup_path: Path,
        restore_to: Path,
    ) -> Dict[str, Any]:
        """
        Restaura un backup.

        Args:
            backup_path: Ruta del archivo de backup
            restore_to: Directorio donde restaurar

        Returns:
            Información de la restauración
        """
        try:
            restore_to = Path(restore_to)
            restore_to.mkdir(parents=True, exist_ok=True)

            with tarfile.open(backup_path, 'r:gz') as tar:
                tar.extractall(restore_to)

            logger.info(f"Backup restaurado desde: {backup_path}")
            return {
                "success": True,
                "restored_to": str(restore_to),
                "restored_at": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error restaurando backup: {e}", exc_info=True)
            raise

    async def list_backups(self) -> List[Dict[str, Any]]:
        """
        Lista todos los backups disponibles.

        Returns:
            Lista de backups
        """
        backups = []
        for backup_file in self.backup_dir.glob("backup_*.tar.gz"):
            try:
                metadata_file = backup_file.parent / f"{backup_file.stem}_info.json"
                if metadata_file.exists():
                    try:
                        metadata = read_json(metadata_file)
                        backups.append(metadata)
                    except FileOperationError as e:
                        logger.warning(f"Error reading backup metadata {metadata_file}: {e}")
                        continue
                else:
                    try:
                        backups.append({
                            "backup_name": backup_file.stem,
                            "backup_path": str(backup_file),
                            "size_bytes": backup_file.stat().st_size,
                            "created_at": "unknown",
                        })
                    except (IOError, OSError) as e:
                        logger.warning(f"Error getting backup file stats {backup_file}: {e}")
                        continue
            except Exception as e:
                logger.warning(f"Unexpected error processing backup {backup_file}: {e}")
                continue

        # Ordenar por fecha (más recientes primero)
        backups.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return backups

    async def delete_backup(self, backup_name: str) -> bool:
        """
        Elimina un backup.

        Args:
            backup_name: Nombre del backup

        Returns:
            True si se eliminó exitosamente
        """
        try:
            backup_file = self.backup_dir / f"{backup_name}.tar.gz"
            metadata_file = self.backup_dir / f"{backup_name}_info.json"

            if backup_file.exists():
                backup_file.unlink()
            if metadata_file.exists():
                metadata_file.unlink()

            logger.info(f"Backup eliminado: {backup_name}")
            return True
        except Exception as e:
            logger.error(f"Error eliminando backup: {e}")
            return False


