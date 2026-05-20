"""
Backup Service - Sistema de backup y restore
============================================

Sistema para hacer backup y restore de datos.
"""

import logging
import json
import gzip
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class BackupService:
    """Servicio de backup"""
    
    def __init__(self, backup_dir: str = "backups"):
        """Inicializar servicio"""
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
        logger.info(f"BackupService initialized with directory: {backup_dir}")
    
    def create_backup(
        self,
        data: Dict[str, Any],
        backup_name: Optional[str] = None
    ) -> str:
        """Crear backup de datos"""
        if not backup_name:
            backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        backup_path = self.backup_dir / f"{backup_name}.json.gz"
        
        # Comprimir y guardar
        with gzip.open(backup_path, 'wt', encoding='utf-8') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "data": data,
            }, f, indent=2)
        
        logger.info(f"Backup created: {backup_path}")
        return str(backup_path)
    
    def restore_backup(self, backup_path: str) -> Dict[str, Any]:
        """Restaurar desde backup"""
        path = Path(backup_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_path}")
        
        # Leer y descomprimir
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            backup_data = json.load(f)
        
        logger.info(f"Backup restored from: {backup_path}")
        return backup_data.get("data", {})
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """Listar backups disponibles"""
        backups = []
        
        for backup_file in self.backup_dir.glob("*.json.gz"):
            try:
                with gzip.open(backup_file, 'rt', encoding='utf-8') as f:
                    backup_data = json.load(f)
                    backups.append({
                        "filename": backup_file.name,
                        "path": str(backup_file),
                        "timestamp": backup_data.get("timestamp"),
                        "size": backup_file.stat().st_size,
                    })
            except Exception as e:
                logger.error(f"Error reading backup {backup_file}: {e}")
        
        # Ordenar por timestamp
        backups.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return backups
    
    def delete_backup(self, backup_name: str) -> bool:
        """Eliminar backup"""
        backup_path = self.backup_dir / backup_name
        if backup_path.exists():
            backup_path.unlink()
            logger.info(f"Backup deleted: {backup_name}")
            return True
        return False




