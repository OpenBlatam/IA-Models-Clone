"""
Backup Service
==============

Servicio de backup y exportación de datos.
"""

import logging
import json
import zipfile
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class BackupService:
    """Servicio de backup y exportación."""
    
    def __init__(self, backup_dir: str = "backups"):
        """
        Inicializar servicio de backup.
        
        Args:
            backup_dir: Directorio para backups
        """
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self._logger = logger
    
    def create_backup(
        self,
        artist_id: str,
        data: Dict[str, Any],
        format: str = "json"
    ) -> str:
        """
        Crear backup de datos.
        
        Args:
            artist_id: ID del artista
            data: Datos a respaldar
            format: Formato del backup (json, zip)
        
        Returns:
            Ruta del archivo de backup
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"backup_{artist_id}_{timestamp}"
        
        if format == "json":
            filepath = self.backup_dir / f"{filename}.json"
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        
        elif format == "zip":
            filepath = self.backup_dir / f"{filename}.zip"
            with zipfile.ZipFile(filepath, "w", zipfile.ZIP_DEFLATED) as zipf:
                # Agregar datos como JSON
                json_data = json.dumps(data, indent=2, ensure_ascii=False, default=str)
                zipf.writestr(f"{filename}.json", json_data)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self._logger.info(f"Created backup: {filepath}")
        return str(filepath)
    
    def export_to_json(self, artist_id: str, data: Dict[str, Any]) -> str:
        """
        Exportar datos a JSON.
        
        Args:
            artist_id: ID del artista
            data: Datos a exportar
        
        Returns:
            Ruta del archivo JSON
        """
        return self.create_backup(artist_id, data, format="json")
    
    def export_to_csv(self, artist_id: str, data: Dict[str, Any], entity_type: str) -> str:
        """
        Exportar datos a CSV.
        
        Args:
            artist_id: ID del artista
            data: Datos a exportar
            entity_type: Tipo de entidad (events, routines, etc.)
        
        Returns:
            Ruta del archivo CSV
        """
        import csv
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"export_{artist_id}_{entity_type}_{timestamp}.csv"
        filepath = self.backup_dir / filename
        
        # Obtener lista de entidades
        entities = data.get(entity_type, [])
        if not entities:
            raise ValueError(f"No {entity_type} data to export")
        
        # Escribir CSV
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            if entities:
                writer = csv.DictWriter(f, fieldnames=entities[0].keys())
                writer.writeheader()
                writer.writerows(entities)
        
        self._logger.info(f"Exported {entity_type} to CSV: {filepath}")
        return str(filepath)
    
    def restore_from_backup(self, backup_path: str) -> Dict[str, Any]:
        """
        Restaurar datos desde backup.
        
        Args:
            backup_path: Ruta del archivo de backup
        
        Returns:
            Datos restaurados
        """
        backup_file = Path(backup_path)
        
        if backup_file.suffix == ".zip":
            with zipfile.ZipFile(backup_file, "r") as zipf:
                # Buscar archivo JSON dentro del ZIP
                json_files = [f for f in zipf.namelist() if f.endswith(".json")]
                if not json_files:
                    raise ValueError("No JSON file found in ZIP")
                
                json_content = zipf.read(json_files[0])
                data = json.loads(json_content)
        else:
            with open(backup_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        
        self._logger.info(f"Restored data from backup: {backup_path}")
        return data
    
    def list_backups(self, artist_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Listar backups disponibles.
        
        Args:
            artist_id: Filtrar por artista (opcional)
        
        Returns:
            Lista de backups
        """
        backups = []
        
        for filepath in self.backup_dir.glob("backup_*.json"):
            if artist_id and artist_id not in filepath.name:
                continue
            
            stat = filepath.stat()
            backups.append({
                "filename": filepath.name,
                "path": str(filepath),
                "size": stat.st_size,
                "created_at": datetime.fromtimestamp(stat.st_mtime).isoformat()
            })
        
        for filepath in self.backup_dir.glob("backup_*.zip"):
            if artist_id and artist_id not in filepath.name:
                continue
            
            stat = filepath.stat()
            backups.append({
                "filename": filepath.name,
                "path": str(filepath),
                "size": stat.st_size,
                "created_at": datetime.fromtimestamp(stat.st_mtime).isoformat()
            })
        
        return sorted(backups, key=lambda x: x["created_at"], reverse=True)
    
    def delete_backup(self, backup_path: str) -> bool:
        """
        Eliminar backup.
        
        Args:
            backup_path: Ruta del archivo de backup
        
        Returns:
            True si se eliminó
        """
        try:
            backup_file = Path(backup_path)
            if backup_file.exists():
                backup_file.unlink()
                self._logger.info(f"Deleted backup: {backup_path}")
                return True
            return False
        except Exception as e:
            self._logger.error(f"Error deleting backup: {str(e)}")
            return False




