"""
Backup - Sistema de backups automáticos
"""

import logging
import json
import gzip
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from pathlib import Path
import shutil

logger = logging.getLogger(__name__)


class BackupManager:
    """Gestor de backups automáticos"""

    def __init__(
        self,
        backup_dir: Optional[Path] = None,
        max_backups: int = 30,
        backup_interval: int = 3600
    ):
        """
        Inicializar el gestor de backups.

        Args:
            backup_dir: Directorio para backups
            max_backups: Número máximo de backups a mantener
            backup_interval: Intervalo entre backups en segundos
        """
        if backup_dir is None:
            backup_dir = Path(__file__).parent.parent / "backups"
        
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.max_backups = max_backups
        self.backup_interval = backup_interval
        self.last_backup: Optional[datetime] = None

    def create_backup(
        self,
        data: Dict[str, Any],
        backup_type: str = "full"
    ) -> str:
        """
        Crear un backup.

        Args:
            data: Datos a respaldar
            backup_type: Tipo de backup (full, incremental)

        Returns:
            Ruta del archivo de backup creado
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"backup_{backup_type}_{timestamp}.json.gz"
        backup_path = self.backup_dir / backup_filename
        
        # Comprimir y guardar
        with gzip.open(backup_path, 'wt', encoding='utf-8') as f:
            json.dump({
                "backup_type": backup_type,
                "timestamp": datetime.utcnow().isoformat(),
                "data": data
            }, f, indent=2)
        
        self.last_backup = datetime.utcnow()
        logger.info(f"Backup creado: {backup_path}")
        
        # Limpiar backups antiguos
        self._cleanup_old_backups()
        
        return str(backup_path)

    def restore_backup(self, backup_path: str) -> Dict[str, Any]:
        """
        Restaurar desde un backup.

        Args:
            backup_path: Ruta al archivo de backup

        Returns:
            Datos restaurados
        """
        backup_path = Path(backup_path)
        
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup no encontrado: {backup_path}")
        
        with gzip.open(backup_path, 'rt', encoding='utf-8') as f:
            backup_data = json.load(f)
        
        logger.info(f"Backup restaurado desde: {backup_path}")
        return backup_data.get("data", {})

    def list_backups(self) -> List[Dict[str, Any]]:
        """
        Listar todos los backups disponibles.

        Returns:
            Lista de backups
        """
        backups = []
        
        for backup_file in self.backup_dir.glob("backup_*.json.gz"):
            try:
                stat = backup_file.stat()
                backups.append({
                    "filename": backup_file.name,
                    "path": str(backup_file),
                    "size": stat.st_size,
                    "created_at": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
            except Exception as e:
                logger.error(f"Error leyendo backup {backup_file}: {e}")
        
        # Ordenar por fecha (más recientes primero)
        backups.sort(key=lambda x: x["created_at"], reverse=True)
        return backups

    def _cleanup_old_backups(self):
        """Eliminar backups antiguos"""
        backups = self.list_backups()
        
        if len(backups) > self.max_backups:
            # Eliminar los más antiguos
            to_remove = backups[self.max_backups:]
            for backup in to_remove:
                try:
                    Path(backup["path"]).unlink()
                    logger.info(f"Backup antiguo eliminado: {backup['filename']}")
                except Exception as e:
                    logger.error(f"Error eliminando backup {backup['filename']}: {e}")

    def should_backup(self) -> bool:
        """
        Verificar si se debe crear un backup.

        Returns:
            True si se debe crear backup
        """
        if self.last_backup is None:
            return True
        
        time_since_last = (datetime.utcnow() - self.last_backup).total_seconds()
        return time_since_last >= self.backup_interval

    def backup_database(self, db_path: Path) -> str:
        """
        Hacer backup de la base de datos.

        Args:
            db_path: Ruta a la base de datos

        Returns:
            Ruta del backup creado
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"db_backup_{timestamp}.db"
        backup_path = self.backup_dir / backup_filename
        
        shutil.copy2(db_path, backup_path)
        
        # Comprimir
        with open(backup_path, 'rb') as f_in:
            with gzip.open(f"{backup_path}.gz", 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        backup_path.unlink()  # Eliminar sin comprimir
        backup_path = Path(f"{backup_path}.gz")
        
        logger.info(f"Backup de BD creado: {backup_path}")
        return str(backup_path)






