"""
Backup and Restore Utilities
=============================
Utilidades para backup y restore de datos.
"""

import json
import pickle
from typing import Any, Dict, List, Optional
from datetime import datetime
from pathlib import Path
import gzip
import shutil

from .logger import get_logger
from .file_helpers import ensure_directory, write_json_file, read_json_file

logger = get_logger(__name__)


class BackupManager:
    """Manager para backups."""
    
    def __init__(self, backup_dir: str = "backups"):
        """
        Inicializar backup manager.
        
        Args:
            backup_dir: Directorio de backups
        """
        self.backup_dir = Path(backup_dir)
        ensure_directory(str(self.backup_dir))
    
    def create_backup(
        self,
        data: Any,
        name: str,
        compress: bool = True,
        format: str = "json"
    ) -> str:
        """
        Crear backup.
        
        Args:
            data: Datos a respaldar
            name: Nombre del backup
            compress: Comprimir backup
            format: Formato (json, pickle)
            
        Returns:
            Ruta del backup creado
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}"
        
        if format == "json":
            filename += ".json"
            filepath = self.backup_dir / filename
            
            if isinstance(data, (dict, list)):
                write_json_file(str(filepath), data)
            else:
                with open(filepath, 'w') as f:
                    json.dump({"data": data}, f, default=str)
        
        elif format == "pickle":
            filename += ".pkl"
            filepath = self.backup_dir / filename
            
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Comprimir si se solicita
        if compress:
            compressed_path = filepath.with_suffix(filepath.suffix + '.gz')
            with open(filepath, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            filepath.unlink()  # Eliminar archivo sin comprimir
            filepath = compressed_path
        
        logger.info(f"Backup created: {filepath}")
        return str(filepath)
    
    def restore_backup(self, backup_path: str) -> Any:
        """
        Restaurar backup.
        
        Args:
            backup_path: Ruta del backup
            
        Returns:
            Datos restaurados
        """
        path = Path(backup_path)
        
        # Detectar si está comprimido
        if path.suffix == '.gz':
            # Descomprimir
            temp_path = path.with_suffix('')
            with gzip.open(path, 'rb') as f_in:
                with open(temp_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            path = temp_path
        
        # Cargar según formato
        if path.suffix == '.json':
            data = read_json_file(str(path))
        elif path.suffix == '.pkl':
            with open(path, 'rb') as f:
                data = pickle.load(f)
        else:
            raise ValueError(f"Unsupported backup format: {path.suffix}")
        
        # Limpiar archivo temporal si se descomprimió
        if backup_path.endswith('.gz') and temp_path.exists():
            temp_path.unlink()
        
        logger.info(f"Backup restored from: {backup_path}")
        return data
    
    def list_backups(self, name_pattern: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Listar backups disponibles.
        
        Args:
            name_pattern: Patrón de nombre (opcional)
            
        Returns:
            Lista de backups
        """
        backups = []
        
        for filepath in self.backup_dir.glob("*"):
            if filepath.is_file():
                # Extraer nombre y timestamp
                name = filepath.stem
                if name.endswith('.gz'):
                    name = name[:-3]  # Remover .gz
                
                # Separar nombre y timestamp
                parts = name.rsplit('_', 2)
                if len(parts) >= 2:
                    backup_name = '_'.join(parts[:-2]) if len(parts) > 2 else parts[0]
                    timestamp_str = '_'.join(parts[-2:])
                    
                    # Filtrar por patrón si se especifica
                    if name_pattern and name_pattern not in backup_name:
                        continue
                    
                    try:
                        timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                    except ValueError:
                        timestamp = None
                    
                    backups.append({
                        "name": backup_name,
                        "path": str(filepath),
                        "timestamp": timestamp.isoformat() if timestamp else None,
                        "size": filepath.stat().st_size,
                        "compressed": filepath.suffix == '.gz'
                    })
        
        # Ordenar por timestamp (más reciente primero)
        backups.sort(key=lambda x: x["timestamp"] or "", reverse=True)
        
        return backups
    
    def cleanup_old_backups(self, days: int = 30, keep_latest: int = 10):
        """
        Limpiar backups antiguos.
        
        Args:
            days: Días de antigüedad para eliminar
            keep_latest: Mantener N más recientes
        """
        backups = self.list_backups()
        
        if len(backups) <= keep_latest:
            return
        
        # Ordenar por timestamp
        backups.sort(key=lambda x: x["timestamp"] or "", reverse=True)
        
        # Mantener los más recientes
        to_keep = backups[:keep_latest]
        to_check = backups[keep_latest:]
        
        cutoff_date = datetime.now().timestamp() - (days * 24 * 60 * 60)
        
        deleted = 0
        for backup in to_check:
            if backup["timestamp"]:
                backup_time = datetime.fromisoformat(backup["timestamp"]).timestamp()
                if backup_time < cutoff_date:
                    try:
                        Path(backup["path"]).unlink()
                        deleted += 1
                        logger.info(f"Deleted old backup: {backup['path']}")
                    except Exception as e:
                        logger.error(f"Error deleting backup {backup['path']}: {e}")
        
        logger.info(f"Cleanup completed: {deleted} backups deleted")



