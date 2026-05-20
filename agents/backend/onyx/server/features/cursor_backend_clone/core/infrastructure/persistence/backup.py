"""
Backup - Sistema de respaldo automático
========================================

Crea backups automáticos del estado del agente.
"""

import asyncio
import logging
import shutil
from typing import Optional, List
from pathlib import Path
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class BackupManager:
    """Gestor de backups"""
    
    def __init__(
        self,
        agent,
        backup_dir: str = "./data/backups",
        max_backups: int = 10,
        auto_backup_interval: int = 3600  # segundos (1 hora)
    ):
        self.agent = agent
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.max_backups = max_backups
        self.auto_backup_interval = auto_backup_interval
        self.running = False
        self._backup_task: Optional[asyncio.Task] = None
    
    async def start_auto_backup(self):
        """Iniciar backups automáticos"""
        if self.running:
            return
        
        self.running = True
        self._backup_task = asyncio.create_task(self._auto_backup_loop())
        logger.info(f"💾 Auto-backup started (interval: {self.auto_backup_interval}s)")
    
    async def stop_auto_backup(self):
        """Detener backups automáticos"""
        self.running = False
        if self._backup_task:
            self._backup_task.cancel()
            try:
                await self._backup_task
            except asyncio.CancelledError:
                pass
    
    async def _auto_backup_loop(self):
        """Loop de backups automáticos"""
        while self.running:
            try:
                await asyncio.sleep(self.auto_backup_interval)
                await self.create_backup()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in auto-backup loop: {e}")
                await asyncio.sleep(60)
    
    async def create_backup(self, name: Optional[str] = None) -> str:
        """Crear backup del estado del agente"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = name or f"backup_{timestamp}"
            backup_path = self.backup_dir / backup_name
            backup_path.mkdir(parents=True, exist_ok=True)
            
            # Backup del estado
            if self.agent.config.persistent_storage:
                state_file = Path(self.agent.config.storage_path)
                if state_file.exists():
                    backup_state = backup_path / "agent_state.json"
                    shutil.copy2(state_file, backup_state)
            
            # Backup de tareas
            tasks = await self.agent.get_tasks(limit=10000)
            tasks_file = backup_path / "tasks.json"
            
            # Usar orjson si está disponible
            try:
                import orjson
                tasks_file.write_bytes(orjson.dumps({
                    "backup_timestamp": timestamp,
                    "tasks": tasks
                }, option=orjson.OPT_INDENT_2))
            except ImportError:
                with open(tasks_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "backup_timestamp": timestamp,
                        "tasks": tasks
                    }, f, indent=2, ensure_ascii=False)
            
            # Backup de métricas si están disponibles
            if self.agent.metrics:
                metrics = self.agent.metrics.get_summary()
                metrics_file = backup_path / "metrics.json"
                
                try:
                    import orjson
                    metrics_file.write_bytes(orjson.dumps({
                        "backup_timestamp": timestamp,
                        "metrics": metrics
                    }, option=orjson.OPT_INDENT_2))
                except ImportError:
                    with open(metrics_file, 'w', encoding='utf-8') as f:
                        json.dump({
                            "backup_timestamp": timestamp,
                            "metrics": metrics
                        }, f, indent=2, ensure_ascii=False)
            
            # Limpiar backups antiguos
            await self._cleanup_old_backups()
            
            logger.info(f"💾 Backup created: {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            raise
    
    async def _cleanup_old_backups(self):
        """Limpiar backups antiguos"""
        try:
            backups = sorted(
                self.backup_dir.iterdir(),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            
            # Mantener solo los más recientes
            for backup in backups[self.max_backups:]:
                if backup.is_dir():
                    shutil.rmtree(backup)
                else:
                    backup.unlink()
                logger.info(f"🗑️ Deleted old backup: {backup.name}")
                
        except Exception as e:
            logger.error(f"Error cleaning up backups: {e}")
    
    async def restore_backup(self, backup_name: str) -> bool:
        """Restaurar desde un backup"""
        try:
            backup_path = self.backup_dir / backup_name
            
            if not backup_path.exists():
                raise FileNotFoundError(f"Backup not found: {backup_name}")
            
            # Restaurar estado
            state_file = backup_path / "agent_state.json"
            if state_file.exists():
                target_state = Path(self.agent.config.storage_path)
                shutil.copy2(state_file, target_state)
                logger.info("✅ Restored agent state")
            
            # Restaurar tareas
            tasks_file = backup_path / "tasks.json"
            if tasks_file.exists():
                # Cargar tareas del backup
                try:
                    import orjson
                    data = orjson.loads(tasks_file.read_bytes())
                except ImportError:
                    with open(tasks_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                
                # Las tareas se restaurarán cuando se cargue el estado
                logger.info(f"✅ Restored {len(data.get('tasks', []))} tasks")
            
            logger.info(f"✅ Backup restored: {backup_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error restoring backup: {e}")
            return False
    
    def list_backups(self) -> List[Dict]:
        """Listar todos los backups"""
        backups = []
        
        for backup_path in self.backup_dir.iterdir():
            if backup_path.is_dir():
                try:
                    stat = backup_path.stat()
                    backups.append({
                        "name": backup_path.name,
                        "created_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "size_mb": sum(
                            f.stat().st_size for f in backup_path.rglob('*') if f.is_file()
                        ) / (1024 * 1024)
                    })
                except Exception as e:
                    logger.error(f"Error reading backup {backup_path}: {e}")
        
        # Ordenar por fecha (más recientes primero)
        backups.sort(key=lambda x: x["created_at"], reverse=True)
        return backups
    
    def delete_backup(self, backup_name: str) -> bool:
        """Eliminar un backup"""
        try:
            backup_path = self.backup_dir / backup_name
            if backup_path.exists():
                if backup_path.is_dir():
                    shutil.rmtree(backup_path)
                else:
                    backup_path.unlink()
                logger.info(f"🗑️ Deleted backup: {backup_name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting backup: {e}")
            return False


