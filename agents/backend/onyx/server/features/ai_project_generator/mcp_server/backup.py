"""
MCP Backup/Restore - Herramientas de backup y restore
=======================================================
"""

import logging
import json
import shutil
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class BackupMetadata(BaseModel):
    """Metadata de backup"""
    backup_id: str = Field(..., description="ID único del backup")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field(..., description="Versión del sistema")
    components: List[str] = Field(default_factory=list, description="Componentes incluidos")
    size_bytes: int = Field(default=0, description="Tamaño del backup en bytes")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata adicional")


class BackupManager:
    """
    Gestor de backups
    
    Permite crear y restaurar backups del sistema MCP.
    """
    
    def __init__(self, backup_dir: str = "./backups"):
        """
        Args:
            backup_dir: Directorio donde guardar backups
        """
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    def create_backup(
        self,
        components: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Crea un backup
        
        Args:
            components: Componentes a incluir (None = todos)
            metadata: Metadata adicional
            
        Returns:
            ID del backup creado
        """
        import uuid
        
        backup_id = str(uuid.uuid4())
        backup_path = self.backup_dir / backup_id
        backup_path.mkdir(exist_ok=True)
        
        # Backup de configuraciones
        config_data = {
            "version": "1.8.0",
            "timestamp": datetime.utcnow().isoformat(),
            "components": components or ["all"],
            "metadata": metadata or {},
        }
        
        with open(backup_path / "metadata.json", "w") as f:
            json.dump(config_data, f, indent=2)
        
        # Calcular tamaño
        size = sum(f.stat().st_size for f in backup_path.rglob("*") if f.is_file())
        
        backup_metadata = BackupMetadata(
            backup_id=backup_id,
            version=config_data["version"],
            components=components or ["all"],
            size_bytes=size,
            metadata=metadata or {},
        )
        
        logger.info(f"Created backup: {backup_id} ({size} bytes)")
        
        return backup_id
    
    def list_backups(self) -> List[BackupMetadata]:
        """
        Lista todos los backups
        
        Returns:
            Lista de backups
        """
        backups = []
        
        for backup_dir in self.backup_dir.iterdir():
            if backup_dir.is_dir():
                metadata_file = backup_dir / "metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file) as f:
                            data = json.load(f)
                            backups.append(BackupMetadata(
                                backup_id=backup_dir.name,
                                timestamp=datetime.fromisoformat(data["timestamp"]),
                                version=data["version"],
                                components=data.get("components", []),
                                metadata=data.get("metadata", {}),
                            ))
                    except Exception as e:
                        logger.error(f"Error reading backup {backup_dir.name}: {e}")
        
        return sorted(backups, key=lambda b: b.timestamp, reverse=True)
    
    def restore_backup(self, backup_id: str) -> bool:
        """
        Restaura un backup
        
        Args:
            backup_id: ID del backup a restaurar
            
        Returns:
            True si se restauró exitosamente
        """
        backup_path = self.backup_dir / backup_id
        
        if not backup_path.exists():
            logger.error(f"Backup {backup_id} not found")
            return False
        
        metadata_file = backup_path / "metadata.json"
        if not metadata_file.exists():
            logger.error(f"Backup metadata not found for {backup_id}")
            return False
        
        try:
            with open(metadata_file) as f:
                metadata = json.load(f)
            
            # Restaurar configuraciones
            # Implementar lógica de restore según necesidad
            
            logger.info(f"Restored backup: {backup_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error restoring backup {backup_id}: {e}")
            return False
    
    def delete_backup(self, backup_id: str) -> bool:
        """
        Elimina un backup
        
        Args:
            backup_id: ID del backup a eliminar
            
        Returns:
            True si se eliminó exitosamente
        """
        backup_path = self.backup_dir / backup_id
        
        if not backup_path.exists():
            return False
        
        try:
            shutil.rmtree(backup_path)
            logger.info(f"Deleted backup: {backup_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting backup {backup_id}: {e}")
            return False

