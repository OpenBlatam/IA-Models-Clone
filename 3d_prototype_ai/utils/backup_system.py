"""
Backup System - Sistema de backup y recuperación
=================================================
"""

import logging
import json
import shutil
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime, timedelta
from zipfile import ZipFile
import hashlib

logger = logging.getLogger(__name__)


class BackupSystem:
    """Sistema de backup y recuperación"""
    
    def __init__(self, backup_dir: Optional[str] = None):
        self.backup_dir = Path(backup_dir) if backup_dir else Path("backups")
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.backup_metadata_file = self.backup_dir / "backup_metadata.json"
        self._load_metadata()
    
    def _load_metadata(self):
        """Carga metadatos de backups"""
        if self.backup_metadata_file.exists():
            try:
                with open(self.backup_metadata_file, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
            except:
                self.metadata = {"backups": []}
        else:
            self.metadata = {"backups": []}
    
    def _save_metadata(self):
        """Guarda metadatos de backups"""
        with open(self.backup_metadata_file, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False, default=str)
    
    def create_backup(self, source_paths: List[str], 
                     backup_name: Optional[str] = None,
                     description: Optional[str] = None) -> str:
        """
        Crea un backup de archivos/directorios
        
        Returns:
            Backup ID
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = backup_name or f"backup_{timestamp}"
        backup_id = f"backup_{timestamp}"
        
        backup_file = self.backup_dir / f"{backup_name}.zip"
        
        # Crear archivo ZIP
        with ZipFile(backup_file, 'w') as zipf:
            for source_path in source_paths:
                source = Path(source_path)
                if source.exists():
                    if source.is_file():
                        zipf.write(source, source.name)
                    elif source.is_dir():
                        for file_path in source.rglob('*'):
                            if file_path.is_file():
                                arcname = file_path.relative_to(source.parent)
                                zipf.write(file_path, arcname)
        
        # Calcular hash
        with open(backup_file, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        
        # Guardar metadatos
        backup_info = {
            "id": backup_id,
            "name": backup_name,
            "description": description,
            "file_path": str(backup_file),
            "file_size": backup_file.stat().st_size,
            "file_hash": file_hash,
            "source_paths": source_paths,
            "created_at": datetime.now().isoformat(),
            "status": "completed"
        }
        
        self.metadata["backups"].append(backup_info)
        self._save_metadata()
        
        logger.info(f"Backup creado: {backup_id} ({backup_file.stat().st_size} bytes)")
        return backup_id
    
    def restore_backup(self, backup_id: str, target_dir: str,
                      overwrite: bool = False) -> bool:
        """Restaura un backup"""
        backup_info = None
        for backup in self.metadata["backups"]:
            if backup["id"] == backup_id:
                backup_info = backup
                break
        
        if not backup_info:
            logger.error(f"Backup {backup_id} no encontrado")
            return False
        
        backup_file = Path(backup_info["file_path"])
        if not backup_file.exists():
            logger.error(f"Archivo de backup no existe: {backup_file}")
            return False
        
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)
        
        # Verificar hash
        with open(backup_file, 'rb') as f:
            current_hash = hashlib.sha256(f.read()).hexdigest()
        
        if current_hash != backup_info["file_hash"]:
            logger.error(f"Hash del backup no coincide. Posible corrupción.")
            return False
        
        # Extraer backup
        try:
            with ZipFile(backup_file, 'r') as zipf:
                zipf.extractall(target)
            
            logger.info(f"Backup {backup_id} restaurado en {target_dir}")
            return True
        except Exception as e:
            logger.error(f"Error restaurando backup: {e}")
            return False
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """Lista todos los backups"""
        return [
            {
                "id": b["id"],
                "name": b["name"],
                "description": b.get("description"),
                "created_at": b["created_at"],
                "file_size": b["file_size"],
                "status": b["status"]
            }
            for b in self.metadata["backups"]
        ]
    
    def delete_backup(self, backup_id: str) -> bool:
        """Elimina un backup"""
        backup_info = None
        index = None
        
        for i, backup in enumerate(self.metadata["backups"]):
            if backup["id"] == backup_id:
                backup_info = backup
                index = i
                break
        
        if not backup_info:
            return False
        
        # Eliminar archivo
        backup_file = Path(backup_info["file_path"])
        if backup_file.exists():
            backup_file.unlink()
        
        # Eliminar de metadatos
        self.metadata["backups"].pop(index)
        self._save_metadata()
        
        logger.info(f"Backup {backup_id} eliminado")
        return True
    
    def cleanup_old_backups(self, days: int = 30):
        """Elimina backups antiguos"""
        cutoff = datetime.now() - timedelta(days=days)
        backups_to_delete = []
        
        for backup in self.metadata["backups"]:
            created_at = datetime.fromisoformat(backup["created_at"])
            if created_at < cutoff:
                backups_to_delete.append(backup["id"])
        
        for backup_id in backups_to_delete:
            self.delete_backup(backup_id)
        
        logger.info(f"Eliminados {len(backups_to_delete)} backups antiguos")
        return len(backups_to_delete)
    
    def get_backup_info(self, backup_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene información de un backup"""
        for backup in self.metadata["backups"]:
            if backup["id"] == backup_id:
                return backup
        return None




