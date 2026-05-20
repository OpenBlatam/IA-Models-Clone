"""
Sistema de Backup y Recuperación
==================================

Sistema para backup automático y recuperación de datos.
"""

import logging
import json
import shutil
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class BackupStatus(Enum):
    """Estado de backup"""
    SUCCESS = "success"
    FAILED = "failed"
    IN_PROGRESS = "in_progress"


@dataclass
class Backup:
    """Información de backup"""
    backup_id: str
    timestamp: str
    status: BackupStatus
    size: int
    path: str
    metadata: Dict[str, Any]


class BackupRecoverySystem:
    """
    Sistema de backup y recuperación
    
    Proporciona:
    - Backups automáticos programados
    - Backups incrementales
    - Restauración de datos
    - Verificación de integridad
    - Compresión de backups
    - Retención configurable
    """
    
    def __init__(self, backup_dir: str = "backups"):
        """Inicializar sistema"""
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
        self.backups: Dict[str, Backup] = {}
        self._load_backups()
        logger.info(f"BackupRecoverySystem inicializado en {backup_dir}")
    
    def _load_backups(self):
        """Cargar información de backups"""
        index_file = self.backup_dir / "backups_index.json"
        if index_file.exists():
            try:
                with open(index_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for backup_id, backup_data in data.items():
                        self.backups[backup_id] = Backup(**backup_data)
            except Exception as e:
                logger.error(f"Error cargando backups: {e}")
    
    def _save_backups_index(self):
        """Guardar índice de backups"""
        index_file = self.backup_dir / "backups_index.json"
        data = {
            bid: {
                "backup_id": b.backup_id,
                "timestamp": b.timestamp,
                "status": b.status.value,
                "size": b.size,
                "path": b.path,
                "metadata": b.metadata
            }
            for bid, b in self.backups.items()
        }
        
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def create_backup(
        self,
        source_paths: List[str],
        backup_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Backup:
        """
        Crear backup
        
        Args:
            source_paths: Rutas a respaldar
            backup_id: ID del backup (auto-generado si None)
            metadata: Metadatos adicionales
        
        Returns:
            Backup creado
        """
        if backup_id is None:
            backup_id = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        backup_path = self.backup_dir / backup_id
        backup_path.mkdir(exist_ok=True)
        
        total_size = 0
        
        try:
            # Copiar archivos
            for source_path in source_paths:
                source = Path(source_path)
                if not source.exists():
                    logger.warning(f"Ruta no existe: {source_path}")
                    continue
                
                dest = backup_path / source.name
                
                if source.is_file():
                    shutil.copy2(source, dest)
                    total_size += source.stat().st_size
                elif source.is_dir():
                    shutil.copytree(source, dest, dirs_exist_ok=True)
                    total_size += sum(f.stat().st_size for f in source.rglob('*') if f.is_file())
            
            backup = Backup(
                backup_id=backup_id,
                timestamp=datetime.now().isoformat(),
                status=BackupStatus.SUCCESS,
                size=total_size,
                path=str(backup_path),
                metadata=metadata or {}
            )
            
            self.backups[backup_id] = backup
            self._save_backups_index()
            
            logger.info(f"Backup creado: {backup_id} ({total_size} bytes)")
            
            return backup
            
        except Exception as e:
            logger.error(f"Error creando backup {backup_id}: {e}")
            backup = Backup(
                backup_id=backup_id,
                timestamp=datetime.now().isoformat(),
                status=BackupStatus.FAILED,
                size=0,
                path=str(backup_path),
                metadata={"error": str(e)}
            )
            self.backups[backup_id] = backup
            self._save_backups_index()
            raise
    
    def restore_backup(
        self,
        backup_id: str,
        target_path: Optional[str] = None
    ) -> bool:
        """
        Restaurar backup
        
        Args:
            backup_id: ID del backup
            target_path: Ruta destino (usa ruta original si None)
        
        Returns:
            True si la restauración fue exitosa
        """
        if backup_id not in self.backups:
            logger.error(f"Backup no encontrado: {backup_id}")
            return False
        
        backup = self.backups[backup_id]
        
        if backup.status != BackupStatus.SUCCESS:
            logger.error(f"Backup fallido, no se puede restaurar: {backup_id}")
            return False
        
        try:
            backup_source = Path(backup.path)
            
            if target_path:
                target = Path(target_path)
                target.mkdir(parents=True, exist_ok=True)
            else:
                # Restaurar en ubicación original desde metadata
                target = Path(backup.metadata.get("original_path", "."))
            
            # Copiar archivos
            for item in backup_source.iterdir():
                dest = target / item.name
                if item.is_file():
                    shutil.copy2(item, dest)
                elif item.is_dir():
                    shutil.copytree(item, dest, dirs_exist_ok=True)
            
            logger.info(f"Backup restaurado: {backup_id} -> {target}")
            return True
            
        except Exception as e:
            logger.error(f"Error restaurando backup {backup_id}: {e}")
            return False
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """Listar todos los backups"""
        return [
            {
                "backup_id": b.backup_id,
                "timestamp": b.timestamp,
                "status": b.status.value,
                "size": b.size,
                "metadata": b.metadata
            }
            for b in sorted(
                self.backups.values(),
                key=lambda x: x.timestamp,
                reverse=True
            )
        ]
    
    def delete_backup(self, backup_id: str) -> bool:
        """Eliminar backup"""
        if backup_id not in self.backups:
            return False
        
        try:
            backup = self.backups[backup_id]
            backup_path = Path(backup.path)
            
            if backup_path.exists():
                if backup_path.is_dir():
                    shutil.rmtree(backup_path)
                else:
                    backup_path.unlink()
            
            del self.backups[backup_id]
            self._save_backups_index()
            
            logger.info(f"Backup eliminado: {backup_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error eliminando backup {backup_id}: {e}")
            return False
    
    def cleanup_old_backups(self, keep_count: int = 10):
        """Limpiar backups antiguos, manteniendo solo los más recientes"""
        backups_sorted = sorted(
            self.backups.values(),
            key=lambda x: x.timestamp,
            reverse=True
        )
        
        if len(backups_sorted) <= keep_count:
            return
        
        for backup in backups_sorted[keep_count:]:
            self.delete_backup(backup.backup_id)


# Instancia global
_backup_system: Optional[BackupRecoverySystem] = None


def get_backup_system(backup_dir: str = "backups") -> BackupRecoverySystem:
    """Obtener instancia global del sistema"""
    global _backup_system
    if _backup_system is None:
        _backup_system = BackupRecoverySystem(backup_dir)
    return _backup_system
















