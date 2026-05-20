"""
Sistema de Backup Automático de Base de Datos.
"""

import asyncio
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from config.logging_config import get_logger
from config.settings import settings
from core.database.connection_pool import get_pool

logger = get_logger(__name__)


class DatabaseBackup:
    """Sistema de backup automático de base de datos."""
    
    def __init__(
        self,
        backup_dir: Optional[Path] = None,
        max_backups: int = 30,
        backup_interval_hours: int = 24
    ):
        """
        Inicializar sistema de backup.
        
        Args:
            backup_dir: Directorio para backups
            max_backups: Número máximo de backups a mantener
            backup_interval_hours: Intervalo entre backups (horas)
        """
        if backup_dir is None:
            backup_dir = Path(settings.STORAGE_PATH) / "backups" / "database"
        self.backup_dir = backup_dir
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_backups = max_backups
        self.backup_interval_hours = backup_interval_hours
        self.running = False
        self.backup_task: Optional[asyncio.Task] = None
        self.stats = {
            "total_backups": 0,
            "successful_backups": 0,
            "failed_backups": 0,
            "last_backup": None,
            "last_error": None
        }
    
    async def create_backup(self, name: Optional[str] = None) -> Optional[Path]:
        """
        Crear backup de la base de datos.
        
        Args:
            name: Nombre del backup (opcional)
            
        Returns:
            Ruta al archivo de backup o None si falla
        """
        try:
            pool = get_pool()
            db_path = pool.db_path
            
            if not Path(db_path).exists():
                logger.warning(f"Base de datos no existe: {db_path}")
                return None
            
            # Generar nombre de backup
            if name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                name = f"backup_{timestamp}.db"
            
            backup_path = self.backup_dir / name
            
            # Copiar archivo de base de datos
            shutil.copy2(db_path, backup_path)
            
            # Copiar archivo WAL si existe
            wal_path = Path(f"{db_path}-wal")
            if wal_path.exists():
                backup_wal_path = self.backup_dir / f"{name}-wal"
                shutil.copy2(wal_path, backup_wal_path)
            
            # Copiar archivo SHM si existe
            shm_path = Path(f"{db_path}-shm")
            if shm_path.exists():
                backup_shm_path = self.backup_dir / f"{name}-shm"
                shutil.copy2(shm_path, backup_shm_path)
            
            self.stats["total_backups"] += 1
            self.stats["successful_backups"] += 1
            self.stats["last_backup"] = datetime.now().isoformat()
            
            logger.info(f"Backup creado: {backup_path}")
            return backup_path
            
        except Exception as e:
            logger.error(f"Error creando backup: {e}", exc_info=True)
            self.stats["total_backups"] += 1
            self.stats["failed_backups"] += 1
            self.stats["last_error"] = str(e)
            return None
    
    async def restore_backup(self, backup_name: str) -> bool:
        """
        Restaurar backup.
        
        Args:
            backup_name: Nombre del backup a restaurar
            
        Returns:
            True si se restauró exitosamente
        """
        try:
            backup_path = self.backup_dir / backup_name
            if not backup_path.exists():
                logger.error(f"Backup no encontrado: {backup_path}")
                return False
            
            pool = get_pool()
            db_path = pool.db_path
            
            # Cerrar pool antes de restaurar
            await pool.close()
            
            # Crear backup del archivo actual
            current_backup = f"{db_path}.before_restore_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            if Path(db_path).exists():
                shutil.copy2(db_path, current_backup)
            
            # Restaurar backup
            shutil.copy2(backup_path, db_path)
            
            # Restaurar archivos WAL y SHM si existen
            wal_backup = self.backup_dir / f"{backup_name}-wal"
            if wal_backup.exists():
                shutil.copy2(wal_backup, Path(f"{db_path}-wal"))
            
            shm_backup = self.backup_dir / f"{backup_name}-shm"
            if shm_backup.exists():
                shutil.copy2(shm_backup, Path(f"{db_path}-shm"))
            
            # Reinicializar pool
            await pool.initialize()
            
            logger.info(f"Backup restaurado: {backup_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error restaurando backup: {e}", exc_info=True)
            return False
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """
        Listar backups disponibles.
        
        Returns:
            Lista de backups con información
        """
        backups = []
        for backup_file in self.backup_dir.glob("backup_*.db"):
            stat = backup_file.stat()
            backups.append({
                "name": backup_file.name,
                "size": stat.st_size,
                "created": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "path": str(backup_file)
            })
        
        # Ordenar por fecha (más recientes primero)
        backups.sort(key=lambda x: x["created"], reverse=True)
        return backups
    
    async def cleanup_old_backups(self) -> int:
        """
        Limpiar backups antiguos.
        
        Returns:
            Número de backups eliminados
        """
        backups = self.list_backups()
        if len(backups) <= self.max_backups:
            return 0
        
        # Eliminar backups más antiguos
        to_remove = backups[self.max_backups:]
        removed = 0
        
        for backup in to_remove:
            try:
                backup_path = Path(backup["path"])
                backup_path.unlink()
                
                # Eliminar archivos relacionados
                wal_path = self.backup_dir / f"{backup['name']}-wal"
                if wal_path.exists():
                    wal_path.unlink()
                
                shm_path = self.backup_dir / f"{backup['name']}-shm"
                if shm_path.exists():
                    shm_path.unlink()
                
                removed += 1
            except Exception as e:
                logger.error(f"Error eliminando backup {backup['name']}: {e}")
        
        logger.info(f"Eliminados {removed} backups antiguos")
        return removed
    
    async def start_auto_backup(self) -> None:
        """Iniciar backups automáticos."""
        if self.running:
            return
        
        self.running = True
        
        async def backup_loop():
            while self.running:
                try:
                    await self.create_backup()
                    await self.cleanup_old_backups()
                    await asyncio.sleep(self.backup_interval_hours * 3600)
                except Exception as e:
                    logger.error(f"Error en backup loop: {e}", exc_info=True)
                    await asyncio.sleep(3600)  # Reintentar en 1 hora
        
        self.backup_task = asyncio.create_task(backup_loop())
        logger.info("Backup automático iniciado")
    
    async def stop_auto_backup(self) -> None:
        """Detener backups automáticos."""
        self.running = False
        if self.backup_task:
            self.backup_task.cancel()
            try:
                await self.backup_task
            except asyncio.CancelledError:
                pass
        logger.info("Backup automático detenido")
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        return {
            **self.stats,
            "backup_dir": str(self.backup_dir),
            "max_backups": self.max_backups,
            "backup_interval_hours": self.backup_interval_hours,
            "running": self.running,
            "total_backups_stored": len(self.list_backups())
        }



