"""
Sistema de Backup y Recovery

Proporciona:
- Backup automático de datos
- Backup incremental
- Restauración de backups
- Verificación de integridad
"""

import logging
import json
import gzip
import shutil
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)


class BackupManager:
    """Gestor de backups"""
    
    def __init__(self, backup_dir: str = "./backups"):
        """
        Args:
            backup_dir: Directorio para almacenar backups
        """
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"BackupManager initialized with directory: {backup_dir}")
    
    def create_backup(
        self,
        data: Dict[str, Any],
        backup_type: str = "full",
        description: Optional[str] = None
    ) -> str:
        """
        Crea un backup
        
        Args:
            data: Datos a respaldar
            backup_type: Tipo de backup ("full", "incremental")
            description: Descripción del backup
        
        Returns:
            ID del backup
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_id = f"{backup_type}_{timestamp}"
        
        backup_data = {
            "id": backup_id,
            "type": backup_type,
            "timestamp": datetime.now().isoformat(),
            "description": description,
            "data": data,
            "checksum": self._calculate_checksum(data)
        }
        
        # Guardar backup
        backup_file = self.backup_dir / f"{backup_id}.json.gz"
        with gzip.open(backup_file, 'wt', encoding='utf-8') as f:
            json.dump(backup_data, f, indent=2, default=str)
        
        logger.info(f"Backup created: {backup_id}")
        return backup_id
    
    def list_backups(self, backup_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Lista todos los backups disponibles
        
        Args:
            backup_type: Filtrar por tipo (opcional)
        
        Returns:
            Lista de backups
        """
        backups = []
        
        for backup_file in self.backup_dir.glob("*.json.gz"):
            try:
                with gzip.open(backup_file, 'rt', encoding='utf-8') as f:
                    backup_data = json.load(f)
                
                if backup_type is None or backup_data.get("type") == backup_type:
                    backups.append({
                        "id": backup_data.get("id"),
                        "type": backup_data.get("type"),
                        "timestamp": backup_data.get("timestamp"),
                        "description": backup_data.get("description"),
                        "file": str(backup_file),
                        "size": backup_file.stat().st_size
                    })
            except Exception as e:
                logger.error(f"Error reading backup {backup_file}: {e}")
        
        return sorted(backups, key=lambda x: x["timestamp"], reverse=True)
    
    def restore_backup(self, backup_id: str) -> Dict[str, Any]:
        """
        Restaura un backup
        
        Args:
            backup_id: ID del backup a restaurar
        
        Returns:
            Datos restaurados
        """
        backup_file = self.backup_dir / f"{backup_id}.json.gz"
        
        if not backup_file.exists():
            raise FileNotFoundError(f"Backup {backup_id} not found")
        
        try:
            with gzip.open(backup_file, 'rt', encoding='utf-8') as f:
                backup_data = json.load(f)
            
            # Verificar checksum
            stored_checksum = backup_data.get("checksum")
            calculated_checksum = self._calculate_checksum(backup_data.get("data", {}))
            
            if stored_checksum != calculated_checksum:
                raise ValueError("Backup checksum verification failed - data may be corrupted")
            
            logger.info(f"Backup {backup_id} restored successfully")
            return backup_data.get("data", {})
        except Exception as e:
            logger.error(f"Error restoring backup {backup_id}: {e}")
            raise
    
    def delete_backup(self, backup_id: str) -> bool:
        """
        Elimina un backup
        
        Args:
            backup_id: ID del backup a eliminar
        
        Returns:
            True si se eliminó correctamente
        """
        backup_file = self.backup_dir / f"{backup_id}.json.gz"
        
        if backup_file.exists():
            backup_file.unlink()
            logger.info(f"Backup {backup_id} deleted")
            return True
        
        return False
    
    def cleanup_old_backups(self, days: int = 30) -> int:
        """
        Elimina backups más antiguos que X días
        
        Args:
            days: Número de días
        
        Returns:
            Número de backups eliminados
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        deleted = 0
        
        for backup_file in self.backup_dir.glob("*.json.gz"):
            try:
                file_time = datetime.fromtimestamp(backup_file.stat().st_mtime)
                if file_time < cutoff_date:
                    backup_file.unlink()
                    deleted += 1
            except Exception as e:
                logger.error(f"Error deleting backup {backup_file}: {e}")
        
        logger.info(f"Cleaned up {deleted} old backups")
        return deleted
    
    def _calculate_checksum(self, data: Any) -> str:
        """Calcula checksum de los datos"""
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def verify_backup(self, backup_id: str) -> Dict[str, Any]:
        """
        Verifica la integridad de un backup
        
        Args:
            backup_id: ID del backup
        
        Returns:
            Resultado de la verificación
        """
        backup_file = self.backup_dir / f"{backup_id}.json.gz"
        
        if not backup_file.exists():
            return {
                "valid": False,
                "error": "Backup file not found"
            }
        
        try:
            with gzip.open(backup_file, 'rt', encoding='utf-8') as f:
                backup_data = json.load(f)
            
            stored_checksum = backup_data.get("checksum")
            calculated_checksum = self._calculate_checksum(backup_data.get("data", {}))
            
            valid = stored_checksum == calculated_checksum
            
            return {
                "valid": valid,
                "stored_checksum": stored_checksum,
                "calculated_checksum": calculated_checksum,
                "timestamp": backup_data.get("timestamp"),
                "type": backup_data.get("type")
            }
        except Exception as e:
            return {
                "valid": False,
                "error": str(e)
            }


# Instancia global
_backup_manager: Optional[BackupManager] = None


def get_backup_manager(backup_dir: str = "./backups") -> BackupManager:
    """Obtiene la instancia global del gestor de backups"""
    global _backup_manager
    if _backup_manager is None:
        _backup_manager = BackupManager(backup_dir=backup_dir)
    return _backup_manager

