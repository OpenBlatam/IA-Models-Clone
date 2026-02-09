"""
Backup Service - Sistema de backup y sincronización
"""

import logging
import json
import shutil
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class BackupService:
    """Servicio para backup y sincronización"""
    
    def __init__(self, backup_path: str = "backups"):
        self.backup_path = Path(backup_path)
        self.backup_path.mkdir(parents=True, exist_ok=True)
    
    def create_backup(
        self,
        designs: List[Dict[str, Any]],
        backup_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Crear backup de diseños"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = backup_name or f"backup_{timestamp}"
        backup_file = self.backup_path / f"{backup_name}.json"
        
        backup_data = {
            "backup_id": backup_name,
            "created_at": datetime.now().isoformat(),
            "designs_count": len(designs),
            "designs": designs
        }
        
        try:
            with open(backup_file, "w", encoding="utf-8") as f:
                json.dump(backup_data, f, indent=2, default=str, ensure_ascii=False)
            
            logger.info(f"Backup creado: {backup_name} con {len(designs)} diseños")
            
            return {
                "backup_id": backup_name,
                "file_path": str(backup_file),
                "created_at": backup_data["created_at"],
                "designs_count": len(designs),
                "size_bytes": backup_file.stat().st_size
            }
        except Exception as e:
            logger.error(f"Error creando backup: {e}")
            raise
    
    def restore_backup(
        self,
        backup_id: str
    ) -> Dict[str, Any]:
        """Restaurar backup"""
        
        backup_file = self.backup_path / f"{backup_id}.json"
        
        if not backup_file.exists():
            raise FileNotFoundError(f"Backup {backup_id} no encontrado")
        
        try:
            with open(backup_file, "r", encoding="utf-8") as f:
                backup_data = json.load(f)
            
            logger.info(f"Backup restaurado: {backup_id}")
            
            return {
                "backup_id": backup_id,
                "restored_at": datetime.now().isoformat(),
                "designs_count": backup_data.get("designs_count", 0),
                "backup_created_at": backup_data.get("created_at"),
                "designs": backup_data.get("designs", [])
            }
        except Exception as e:
            logger.error(f"Error restaurando backup: {e}")
            raise
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """Listar todos los backups"""
        backups = []
        
        for backup_file in self.backup_path.glob("*.json"):
            try:
                with open(backup_file, "r", encoding="utf-8") as f:
                    backup_data = json.load(f)
                
                backups.append({
                    "backup_id": backup_data.get("backup_id", backup_file.stem),
                    "created_at": backup_data.get("created_at"),
                    "designs_count": backup_data.get("designs_count", 0),
                    "file_path": str(backup_file),
                    "size_bytes": backup_file.stat().st_size
                })
            except Exception as e:
                logger.warning(f"Error leyendo backup {backup_file}: {e}")
        
        # Ordenar por fecha (más reciente primero)
        backups.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        
        return backups
    
    def delete_backup(self, backup_id: str) -> bool:
        """Eliminar backup"""
        backup_file = self.backup_path / f"{backup_id}.json"
        
        if backup_file.exists():
            backup_file.unlink()
            logger.info(f"Backup eliminado: {backup_id}")
            return True
        
        return False
    
    def export_backup(
        self,
        backup_id: str,
        export_path: str
    ) -> str:
        """Exportar backup a ubicación externa"""
        backup_file = self.backup_path / f"{backup_id}.json"
        
        if not backup_file.exists():
            raise FileNotFoundError(f"Backup {backup_id} no encontrado")
        
        export_path_obj = Path(export_path)
        export_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        shutil.copy2(backup_file, export_path_obj)
        
        logger.info(f"Backup exportado: {backup_id} a {export_path}")
        return str(export_path_obj)
    
    def sync_designs(
        self,
        source_designs: List[Dict[str, Any]],
        target_designs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Sincronizar diseños entre fuentes"""
        
        source_ids = {d.get("store_id") for d in source_designs}
        target_ids = {d.get("store_id") for d in target_designs}
        
        to_add = [d for d in source_designs if d.get("store_id") not in target_ids]
        to_update = [
            d for d in source_designs
            if d.get("store_id") in target_ids
        ]
        to_remove = [d for d in target_designs if d.get("store_id") not in source_ids]
        
        return {
            "to_add": len(to_add),
            "to_update": len(to_update),
            "to_remove": len(to_remove),
            "additions": to_add,
            "updates": to_update,
            "removals": to_remove
        }




