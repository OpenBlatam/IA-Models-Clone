"""
Backup Manager - Gestor de Backups
===================================

Sistema de backups automáticos programados para sesiones.
"""

import asyncio
import logging
import shutil
import json
from typing import Dict, List, Optional, Callable
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class BackupConfig:
    """Configuración de backup."""
    enabled: bool = True
    interval_hours: int = 24  # Backup cada 24 horas
    backup_directory: str = "backups"
    max_backups: int = 30  # Mantener últimos 30 backups
    compress: bool = True
    backup_sessions: bool = True
    backup_metrics: bool = True
    backup_cache: bool = False  # Cache generalmente no se respalda


class BackupManager:
    """Gestor de backups automáticos."""
    
    def __init__(
        self,
        config: BackupConfig,
        storage_path: Optional[str] = None,
        on_backup_complete: Optional[Callable] = None,
    ):
        """
        Inicializar gestor de backups.
        
        Args:
            config: Configuración de backup
            storage_path: Ruta donde se almacenan las sesiones
            on_backup_complete: Callback cuando se completa un backup
        """
        self.config = config
        self.storage_path = Path(storage_path) if storage_path else Path("sessions")
        self.backup_dir = Path(config.backup_directory)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.on_backup_complete = on_backup_complete
        
        self._backup_task: Optional[asyncio.Task] = None
        self._backup_history: List[Dict] = []
    
    async def start(self):
        """Iniciar backups automáticos."""
        if not self.config.enabled:
            logger.info("Backups are disabled")
            return
        
        if self._backup_task is None:
            self._backup_task = asyncio.create_task(self._backup_loop())
            logger.info(f"Backup manager started (interval: {self.config.interval_hours}h)")
    
    async def stop(self):
        """Detener backups automáticos."""
        if self._backup_task:
            self._backup_task.cancel()
            try:
                await self._backup_task
            except asyncio.CancelledError:
                pass
            self._backup_task = None
            logger.info("Backup manager stopped")
    
    async def _backup_loop(self):
        """Loop principal de backups."""
        while True:
            try:
                await asyncio.sleep(self.config.interval_hours * 3600)
                await self.create_backup()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in backup loop: {e}")
                await asyncio.sleep(3600)  # Reintentar en 1 hora
    
    async def create_backup(self) -> Optional[str]:
        """
        Crear backup manual.
        
        Returns:
            Ruta del backup creado
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.backup_dir / f"backup_{timestamp}"
            backup_path.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Creating backup: {backup_path}")
            
            # Backup de sesiones
            if self.config.backup_sessions and self.storage_path.exists():
                sessions_backup = backup_path / "sessions"
                shutil.copytree(self.storage_path, sessions_backup)
                logger.info(f"Backed up sessions to {sessions_backup}")
            
            # Backup de metadata
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "backup_config": {
                    "backup_sessions": self.config.backup_sessions,
                    "backup_metrics": self.config.backup_metrics,
                },
            }
            
            metadata_file = backup_path / "metadata.json"
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)
            
            # Comprimir si está habilitado
            if self.config.compress:
                compressed_path = f"{backup_path}.tar.gz"
                shutil.make_archive(str(backup_path), "gztar", backup_path)
                shutil.rmtree(backup_path)
                backup_path = Path(compressed_path)
                logger.info(f"Compressed backup to {backup_path}")
            
            # Limpiar backups antiguos
            await self._cleanup_old_backups()
            
            # Registrar en historial
            self._backup_history.append({
                "path": str(backup_path),
                "timestamp": timestamp,
                "size": backup_path.stat().st_size if backup_path.exists() else 0,
            })
            
            # Callback
            if self.on_backup_complete:
                await self.on_backup_complete(str(backup_path))
            
            logger.info(f"Backup completed: {backup_path}")
            return str(backup_path)
        
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            return None
    
    async def _cleanup_old_backups(self):
        """Limpiar backups antiguos."""
        try:
            # Obtener todos los backups
            backups = []
            for item in self.backup_dir.iterdir():
                if item.is_file() and (item.suffix == ".gz" or item.name.startswith("backup_")):
                    backups.append((item.stat().st_mtime, item))
                elif item.is_dir() and item.name.startswith("backup_"):
                    backups.append((item.stat().st_mtime, item))
            
            # Ordenar por fecha (más reciente primero)
            backups.sort(reverse=True)
            
            # Eliminar backups antiguos
            if len(backups) > self.config.max_backups:
                for _, backup in backups[self.config.max_backups:]:
                    if backup.is_file():
                        backup.unlink()
                    else:
                        shutil.rmtree(backup)
                    logger.info(f"Removed old backup: {backup}")
        
        except Exception as e:
            logger.error(f"Error cleaning up backups: {e}")
    
    async def restore_backup(self, backup_path: str, restore_to: Optional[str] = None):
        """
        Restaurar desde un backup.
        
        Args:
            backup_path: Ruta del backup
            restore_to: Ruta donde restaurar (default: storage_path original)
        """
        try:
            backup = Path(backup_path)
            restore_path = Path(restore_to) if restore_to else self.storage_path
            
            logger.info(f"Restoring backup from {backup_path} to {restore_path}")
            
            # Si es comprimido, descomprimir primero
            if backup.suffix == ".gz":
                import tempfile
                with tempfile.TemporaryDirectory() as tmpdir:
                    shutil.unpack_archive(backup, tmpdir)
                    extracted = Path(tmpdir) / backup.stem.replace(".tar", "")
                    if extracted.exists():
                        sessions_backup = extracted / "sessions"
                        if sessions_backup.exists():
                            restore_path.parent.mkdir(parents=True, exist_ok=True)
                            if restore_path.exists():
                                shutil.rmtree(restore_path)
                            shutil.copytree(sessions_backup, restore_path)
            else:
                # Backup sin comprimir
                sessions_backup = backup / "sessions"
                if sessions_backup.exists():
                    restore_path.parent.mkdir(parents=True, exist_ok=True)
                    if restore_path.exists():
                        shutil.rmtree(restore_path)
                    shutil.copytree(sessions_backup, restore_path)
            
            logger.info(f"Backup restored successfully to {restore_path}")
        
        except Exception as e:
            logger.error(f"Error restoring backup: {e}")
            raise
    
    def list_backups(self) -> List[Dict]:
        """Listar backups disponibles."""
        backups = []
        
        for item in self.backup_dir.iterdir():
            if item.is_file() and (item.suffix == ".gz" or item.name.startswith("backup_")):
                backups.append({
                    "path": str(item),
                    "size": item.stat().st_size,
                    "created": datetime.fromtimestamp(item.stat().st_mtime).isoformat(),
                })
            elif item.is_dir() and item.name.startswith("backup_"):
                backups.append({
                    "path": str(item),
                    "size": sum(f.stat().st_size for f in item.rglob("*") if f.is_file()),
                    "created": datetime.fromtimestamp(item.stat().st_mtime).isoformat(),
                })
        
        return sorted(backups, key=lambda x: x["created"], reverse=True)
    
    def get_backup_history(self) -> List[Dict]:
        """Obtener historial de backups."""
        return self._backup_history.copy()
































