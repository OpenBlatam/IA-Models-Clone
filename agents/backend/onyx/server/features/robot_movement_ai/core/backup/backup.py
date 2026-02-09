"""
Backup and Recovery System
===========================

Sistema de backup y recuperación.
"""

import json
import shutil
import zipfile
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class BackupManager:
    """
    Gestor de backups.
    
    Gestiona backups del sistema y recuperación.
    """
    
    def __init__(self, backup_directory: str = "backups"):
        """
        Inicializar gestor de backups.
        
        Args:
            backup_directory: Directorio para backups
        """
        self.backup_directory = Path(backup_directory)
        self.backup_directory.mkdir(parents=True, exist_ok=True)
        self.backup_history: List[Dict[str, Any]] = []
    
    def create_backup(
        self,
        name: Optional[str] = None,
        include_config: bool = True,
        include_cache: bool = False,
        include_logs: bool = False,
        include_data: bool = True
    ) -> Dict[str, Any]:
        """
        Crear backup del sistema.
        
        Args:
            name: Nombre del backup (opcional)
            include_config: Incluir configuración
            include_cache: Incluir caché
            include_logs: Incluir logs
            include_data: Incluir datos
            
        Returns:
            Información del backup creado
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = name or f"backup_{timestamp}"
        backup_path = self.backup_directory / f"{backup_name}.zip"
        
        logger.info(f"Creating backup: {backup_name}")
        
        try:
            with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Backup de configuración
                if include_config:
                    self._backup_config(zipf)
                
                # Backup de caché
                if include_cache:
                    self._backup_cache(zipf)
                
                # Backup de logs
                if include_logs:
                    self._backup_logs(zipf)
                
                # Backup de datos
                if include_data:
                    self._backup_data(zipf)
                
                # Metadata del backup
                metadata = {
                    "name": backup_name,
                    "timestamp": timestamp,
                    "created_at": datetime.now().isoformat(),
                    "includes": {
                        "config": include_config,
                        "cache": include_cache,
                        "logs": include_logs,
                        "data": include_data
                    }
                }
                
                zipf.writestr("backup_metadata.json", json.dumps(metadata, indent=2))
            
            backup_info = {
                "name": backup_name,
                "path": str(backup_path),
                "size": backup_path.stat().st_size,
                "created_at": metadata["created_at"],
                "includes": metadata["includes"]
            }
            
            self.backup_history.append(backup_info)
            logger.info(f"Backup created successfully: {backup_name} ({backup_info['size']} bytes)")
            
            return backup_info
        
        except Exception as e:
            logger.error(f"Error creating backup: {e}", exc_info=True)
            raise
    
    def restore_backup(
        self,
        backup_name: str,
        restore_config: bool = True,
        restore_cache: bool = False,
        restore_logs: bool = False,
        restore_data: bool = True
    ) -> Dict[str, Any]:
        """
        Restaurar backup.
        
        Args:
            backup_name: Nombre del backup
            restore_config: Restaurar configuración
            restore_cache: Restaurar caché
            restore_logs: Restaurar logs
            restore_data: Restaurar datos
            
        Returns:
            Información de restauración
        """
        backup_path = self.backup_directory / f"{backup_name}.zip"
        
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup not found: {backup_name}")
        
        logger.info(f"Restoring backup: {backup_name}")
        
        try:
            with zipfile.ZipFile(backup_path, 'r') as zipf:
                # Leer metadata
                metadata_str = zipf.read("backup_metadata.json").decode('utf-8')
                metadata = json.loads(metadata_str)
                
                # Restaurar configuración
                if restore_config and metadata["includes"]["config"]:
                    self._restore_config(zipf)
                
                # Restaurar caché
                if restore_cache and metadata["includes"]["cache"]:
                    self._restore_cache(zipf)
                
                # Restaurar logs
                if restore_logs and metadata["includes"]["logs"]:
                    self._restore_logs(zipf)
                
                # Restaurar datos
                if restore_data and metadata["includes"]["data"]:
                    self._restore_data(zipf)
            
            logger.info(f"Backup restored successfully: {backup_name}")
            
            return {
                "backup_name": backup_name,
                "restored_at": datetime.now().isoformat(),
                "restored": {
                    "config": restore_config,
                    "cache": restore_cache,
                    "logs": restore_logs,
                    "data": restore_data
                }
            }
        
        except Exception as e:
            logger.error(f"Error restoring backup: {e}", exc_info=True)
            raise
    
    def _backup_config(self, zipf: zipfile.ZipFile) -> None:
        """Backup de configuración."""
        config_path = Path("config")
        if config_path.exists():
            for file in config_path.rglob("*"):
                if file.is_file():
                    zipf.write(file, f"config/{file.relative_to(config_path)}")
    
    def _backup_cache(self, zipf: zipfile.ZipFile) -> None:
        """Backup de caché."""
        cache_path = Path("cache")
        if cache_path.exists():
            for file in cache_path.rglob("*"):
                if file.is_file():
                    zipf.write(file, f"cache/{file.relative_to(cache_path)}")
    
    def _backup_logs(self, zipf: zipfile.ZipFile) -> None:
        """Backup de logs."""
        logs_path = Path("logs")
        if logs_path.exists():
            for file in logs_path.rglob("*.log"):
                zipf.write(file, f"logs/{file.relative_to(logs_path)}")
    
    def _backup_data(self, zipf: zipfile.ZipFile) -> None:
        """Backup de datos."""
        data_path = Path("data")
        if data_path.exists():
            for file in data_path.rglob("*"):
                if file.is_file():
                    zipf.write(file, f"data/{file.relative_to(data_path)}")
    
    def _restore_config(self, zipf: zipfile.ZipFile) -> None:
        """Restaurar configuración."""
        config_path = Path("config")
        config_path.mkdir(parents=True, exist_ok=True)
        
        for member in zipf.namelist():
            if member.startswith("config/") and not member.endswith("/"):
                zipf.extract(member, ".")
    
    def _restore_cache(self, zipf: zipfile.ZipFile) -> None:
        """Restaurar caché."""
        cache_path = Path("cache")
        cache_path.mkdir(parents=True, exist_ok=True)
        
        for member in zipf.namelist():
            if member.startswith("cache/") and not member.endswith("/"):
                zipf.extract(member, ".")
    
    def _restore_logs(self, zipf: zipfile.ZipFile) -> None:
        """Restaurar logs."""
        logs_path = Path("logs")
        logs_path.mkdir(parents=True, exist_ok=True)
        
        for member in zipf.namelist():
            if member.startswith("logs/") and not member.endswith("/"):
                zipf.extract(member, ".")
    
    def _restore_data(self, zipf: zipfile.ZipFile) -> None:
        """Restaurar datos."""
        data_path = Path("data")
        data_path.mkdir(parents=True, exist_ok=True)
        
        for member in zipf.namelist():
            if member.startswith("data/") and not member.endswith("/"):
                zipf.extract(member, ".")
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """Listar todos los backups."""
        backups = []
        
        for backup_file in self.backup_directory.glob("*.zip"):
            try:
                with zipfile.ZipFile(backup_file, 'r') as zipf:
                    metadata_str = zipf.read("backup_metadata.json").decode('utf-8')
                    metadata = json.loads(metadata_str)
                    
                    backups.append({
                        "name": backup_file.stem,
                        "path": str(backup_file),
                        "size": backup_file.stat().st_size,
                        "created_at": metadata.get("created_at"),
                        "includes": metadata.get("includes", {})
                    })
            except Exception as e:
                logger.warning(f"Error reading backup {backup_file}: {e}")
        
        return sorted(backups, key=lambda x: x.get("created_at", ""), reverse=True)
    
    def delete_backup(self, backup_name: str) -> bool:
        """
        Eliminar backup.
        
        Args:
            backup_name: Nombre del backup
            
        Returns:
            True si se eliminó exitosamente
        """
        backup_path = self.backup_directory / f"{backup_name}.zip"
        
        if backup_path.exists():
            backup_path.unlink()
            logger.info(f"Backup deleted: {backup_name}")
            return True
        
        return False


# Instancia global
_backup_manager: Optional[BackupManager] = None


def get_backup_manager(backup_directory: str = "backups") -> BackupManager:
    """Obtener instancia global del gestor de backups."""
    global _backup_manager
    if _backup_manager is None:
        _backup_manager = BackupManager(backup_directory)
    return _backup_manager






