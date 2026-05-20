"""
Sistema de backup y recuperación
"""

import json
import shutil
from typing import Optional, List
from datetime import datetime
from pathlib import Path
import zipfile


class BackupManager:
    """Gestor de backups del sistema"""
    
    def __init__(self, backup_dir: str = "backups"):
        """
        Inicializa el gestor de backups
        
        Args:
            backup_dir: Directorio para backups
        """
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
    
    def create_backup(self, include_history: bool = True,
                     include_products: bool = True,
                     include_cache: bool = False) -> str:
        """
        Crea un backup del sistema
        
        Args:
            include_history: Incluir historial
            include_products: Incluir base de datos de productos
            include_cache: Incluir cache
            
        Returns:
            Path del archivo de backup creado
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = self.backup_dir / f"backup_{timestamp}.zip"
        
        with zipfile.ZipFile(backup_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Backup de base de datos
            db_file = Path("dermatology_history.db")
            if db_file.exists():
                zipf.write(db_file, "dermatology_history.db")
            
            # Backup de historial (archivos)
            history_dir = Path("history")
            if include_history and history_dir.exists():
                for file in history_dir.glob("*.json"):
                    zipf.write(file, f"history/{file.name}")
            
            # Backup de productos
            products_file = Path("products_database.json")
            if include_products and products_file.exists():
                zipf.write(products_file, "products_database.json")
            
            # Backup de cache (opcional)
            if include_cache:
                cache_dir = Path("cache")
                if cache_dir.exists():
                    for file in cache_dir.glob("*.pkl"):
                        zipf.write(file, f"cache/{file.name}")
            
            # Metadata del backup
            metadata = {
                "timestamp": timestamp,
                "created_at": datetime.now().isoformat(),
                "includes": {
                    "history": include_history,
                    "products": include_products,
                    "cache": include_cache
                }
            }
            
            zipf.writestr("backup_metadata.json", json.dumps(metadata, indent=2))
        
        return str(backup_file)
    
    def list_backups(self) -> List[Dict]:
        """
        Lista todos los backups disponibles
        
        Returns:
            Lista de backups con información
        """
        backups = []
        
        for backup_file in self.backup_dir.glob("backup_*.zip"):
            try:
                with zipfile.ZipFile(backup_file, 'r') as zipf:
                    if "backup_metadata.json" in zipf.namelist():
                        metadata_str = zipf.read("backup_metadata.json").decode('utf-8')
                        metadata = json.loads(metadata_str)
                        
                        backups.append({
                            "filename": backup_file.name,
                            "path": str(backup_file),
                            "size": backup_file.stat().st_size,
                            "created_at": metadata.get("created_at"),
                            "includes": metadata.get("includes", {})
                        })
            except Exception:
                # Si no se puede leer metadata, usar info del archivo
                backups.append({
                    "filename": backup_file.name,
                    "path": str(backup_file),
                    "size": backup_file.stat().st_size,
                    "created_at": datetime.fromtimestamp(backup_file.stat().st_mtime).isoformat()
                })
        
        # Ordenar por fecha (más reciente primero)
        backups.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        
        return backups
    
    def restore_backup(self, backup_file: str, 
                      restore_history: bool = True,
                      restore_products: bool = True,
                      restore_cache: bool = False) -> bool:
        """
        Restaura un backup
        
        Args:
            backup_file: Path al archivo de backup
            restore_history: Restaurar historial
            restore_products: Restaurar productos
            restore_cache: Restaurar cache
            
        Returns:
            True si se restauró correctamente
        """
        backup_path = Path(backup_file)
        
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup no encontrado: {backup_file}")
        
        try:
            with zipfile.ZipFile(backup_path, 'r') as zipf:
                # Restaurar base de datos
                if "dermatology_history.db" in zipf.namelist():
                    zipf.extract("dermatology_history.db", ".")
                
                # Restaurar historial
                if restore_history:
                    history_dir = Path("history")
                    history_dir.mkdir(exist_ok=True)
                    for name in zipf.namelist():
                        if name.startswith("history/") and name.endswith(".json"):
                            zipf.extract(name, ".")
                
                # Restaurar productos
                if restore_products and "products_database.json" in zipf.namelist():
                    zipf.extract("products_database.json", ".")
                
                # Restaurar cache
                if restore_cache:
                    cache_dir = Path("cache")
                    cache_dir.mkdir(exist_ok=True)
                    for name in zipf.namelist():
                        if name.startswith("cache/") and name.endswith(".pkl"):
                            zipf.extract(name, ".")
            
            return True
        
        except Exception as e:
            print(f"Error restaurando backup: {e}")
            return False
    
    def delete_backup(self, backup_file: str) -> bool:
        """
        Elimina un backup
        
        Args:
            backup_file: Nombre o path del backup
            
        Returns:
            True si se eliminó correctamente
        """
        backup_path = self.backup_dir / backup_file if not Path(backup_file).is_absolute() else Path(backup_file)
        
        if backup_path.exists() and backup_path.suffix == ".zip":
            backup_path.unlink()
            return True
        
        return False
    
    def cleanup_old_backups(self, keep_last: int = 10) -> int:
        """
        Limpia backups antiguos, manteniendo solo los últimos N
        
        Args:
            keep_last: Número de backups a mantener
            
        Returns:
            Número de backups eliminados
        """
        backups = self.list_backups()
        
        if len(backups) <= keep_last:
            return 0
        
        # Ordenar por fecha (más antiguos primero)
        backups.sort(key=lambda x: x.get("created_at", ""))
        
        # Eliminar los más antiguos
        to_delete = backups[:-keep_last]
        deleted = 0
        
        for backup in to_delete:
            if self.delete_backup(backup["filename"]):
                deleted += 1
        
        return deleted






