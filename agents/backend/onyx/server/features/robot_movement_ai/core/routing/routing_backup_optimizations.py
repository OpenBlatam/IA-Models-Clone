"""
Routing Backup and Recovery Optimizations
==========================================

Optimizaciones de backup y recuperación.
Incluye: Auto-backup, Snapshot management, Recovery procedures, etc.
"""

import logging
import json
import pickle
import time
from typing import Dict, Any, List, Optional
from pathlib import Path
import threading

logger = logging.getLogger(__name__)


class SnapshotManager:
    """Gestor de snapshots del sistema."""
    
    def __init__(self, snapshot_dir: str = "snapshots"):
        """
        Inicializar gestor de snapshots.
        
        Args:
            snapshot_dir: Directorio para snapshots
        """
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
    
    def create_snapshot(self, data: Dict[str, Any], name: Optional[str] = None) -> str:
        """
        Crear snapshot.
        
        Args:
            data: Datos a guardar
            name: Nombre del snapshot (opcional)
        
        Returns:
            Ruta del snapshot
        """
        if name is None:
            name = f"snapshot_{int(time.time())}"
        
        snapshot_path = self.snapshot_dir / f"{name}.pkl"
        
        with self.lock:
            try:
                with open(snapshot_path, 'wb') as f:
                    pickle.dump(data, f)
                logger.info(f"Snapshot created: {snapshot_path}")
                return str(snapshot_path)
            except Exception as e:
                logger.error(f"Error creating snapshot: {e}")
                raise
    
    def load_snapshot(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Cargar snapshot.
        
        Args:
            name: Nombre del snapshot
        
        Returns:
            Datos del snapshot o None
        """
        snapshot_path = self.snapshot_dir / f"{name}.pkl"
        
        if not snapshot_path.exists():
            logger.warning(f"Snapshot not found: {snapshot_path}")
            return None
        
        with self.lock:
            try:
                with open(snapshot_path, 'rb') as f:
                    data = pickle.load(f)
                logger.info(f"Snapshot loaded: {snapshot_path}")
                return data
            except Exception as e:
                logger.error(f"Error loading snapshot: {e}")
                return None
    
    def list_snapshots(self) -> List[str]:
        """Listar snapshots disponibles."""
        snapshots = []
        for path in self.snapshot_dir.glob("*.pkl"):
            snapshots.append(path.stem)
        return sorted(snapshots)
    
    def delete_snapshot(self, name: str) -> bool:
        """Eliminar snapshot."""
        snapshot_path = self.snapshot_dir / f"{name}.pkl"
        
        if not snapshot_path.exists():
            return False
        
        with self.lock:
            try:
                snapshot_path.unlink()
                logger.info(f"Snapshot deleted: {snapshot_path}")
                return True
            except Exception as e:
                logger.error(f"Error deleting snapshot: {e}")
                return False


class AutoBackup:
    """Sistema de backup automático."""
    
    def __init__(self, snapshot_manager: SnapshotManager, interval: float = 3600.0):
        """
        Inicializar auto-backup.
        
        Args:
            snapshot_manager: Gestor de snapshots
            interval: Intervalo de backup en segundos
        """
        self.snapshot_manager = snapshot_manager
        self.interval = interval
        self.last_backup = time.time()
        self.backup_enabled = False
        self.backup_thread: Optional[threading.Thread] = None
        self.backup_data_func: Optional[Any] = None
    
    def set_backup_data_func(self, func: Any):
        """Establecer función para obtener datos de backup."""
        self.backup_data_func = func
    
    def start_auto_backup(self):
        """Iniciar backup automático."""
        if self.backup_enabled:
            return
        
        self.backup_enabled = True
        self.backup_thread = threading.Thread(target=self._backup_loop, daemon=True)
        self.backup_thread.start()
        logger.info("Auto-backup started")
    
    def stop_auto_backup(self):
        """Detener backup automático."""
        self.backup_enabled = False
        if self.backup_thread:
            self.backup_thread.join(timeout=2.0)
        logger.info("Auto-backup stopped")
    
    def _backup_loop(self):
        """Loop de backup automático."""
        while self.backup_enabled:
            try:
                current_time = time.time()
                if (current_time - self.last_backup) >= self.interval:
                    if self.backup_data_func:
                        data = self.backup_data_func()
                        self.snapshot_manager.create_snapshot(data, f"auto_backup_{int(current_time)}")
                        self.last_backup = current_time
                
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Error in backup loop: {e}")
                time.sleep(60)
    
    def manual_backup(self) -> Optional[str]:
        """Realizar backup manual."""
        if not self.backup_data_func:
            logger.warning("No backup data function set")
            return None
        
        try:
            data = self.backup_data_func()
            snapshot_name = f"manual_backup_{int(time.time())}"
            return self.snapshot_manager.create_snapshot(data, snapshot_name)
        except Exception as e:
            logger.error(f"Error in manual backup: {e}")
            return None


class RecoveryManager:
    """Gestor de recuperación."""
    
    def __init__(self, snapshot_manager: SnapshotManager):
        """
        Inicializar gestor de recuperación.
        
        Args:
            snapshot_manager: Gestor de snapshots
        """
        self.snapshot_manager = snapshot_manager
    
    def recover_from_snapshot(self, name: str, restore_func: Any) -> bool:
        """
        Recuperar desde snapshot.
        
        Args:
            name: Nombre del snapshot
            restore_func: Función para restaurar datos
        
        Returns:
            True si se recuperó exitosamente
        """
        data = self.snapshot_manager.load_snapshot(name)
        if data is None:
            return False
        
        try:
            restore_func(data)
            logger.info(f"Recovered from snapshot: {name}")
            return True
        except Exception as e:
            logger.error(f"Error recovering from snapshot: {e}")
            return False
    
    def get_latest_snapshot(self) -> Optional[str]:
        """Obtener snapshot más reciente."""
        snapshots = self.snapshot_manager.list_snapshots()
        if not snapshots:
            return None
        
        # Buscar el más reciente por timestamp
        latest = None
        latest_time = 0
        
        for snapshot in snapshots:
            if snapshot.startswith("auto_backup_") or snapshot.startswith("manual_backup_"):
                try:
                    timestamp = int(snapshot.split("_")[-1])
                    if timestamp > latest_time:
                        latest_time = timestamp
                        latest = snapshot
                except:
                    pass
        
        return latest


class BackupOptimizer:
    """Optimizador completo de backup y recuperación."""
    
    def __init__(self, snapshot_dir: str = "snapshots", auto_backup_interval: float = 3600.0):
        """
        Inicializar optimizador de backup.
        
        Args:
            snapshot_dir: Directorio de snapshots
            auto_backup_interval: Intervalo de auto-backup en segundos
        """
        self.snapshot_manager = SnapshotManager(snapshot_dir)
        self.auto_backup = AutoBackup(self.snapshot_manager, auto_backup_interval)
        self.recovery_manager = RecoveryManager(self.snapshot_manager)
    
    def setup_auto_backup(self, backup_data_func: Any):
        """Configurar auto-backup."""
        self.auto_backup.set_backup_data_func(backup_data_func)
        self.auto_backup.start_auto_backup()
    
    def create_backup(self, data: Dict[str, Any], name: Optional[str] = None) -> str:
        """Crear backup manual."""
        return self.snapshot_manager.create_snapshot(data, name)
    
    def restore_backup(self, name: str, restore_func: Any) -> bool:
        """Restaurar desde backup."""
        return self.recovery_manager.recover_from_snapshot(name, restore_func)
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        snapshots = self.snapshot_manager.list_snapshots()
        return {
            'snapshot_count': len(snapshots),
            'auto_backup_enabled': self.auto_backup.backup_enabled,
            'latest_snapshot': self.recovery_manager.get_latest_snapshot()
        }

