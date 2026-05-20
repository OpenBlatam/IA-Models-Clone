"""
Document Backup - Sistema de Backup y Recuperación
===================================================

Sistema de backup y recuperación de análisis y documentos.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field
import json
import os
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class BackupMetadata:
    """Metadatos de backup."""
    backup_id: str
    timestamp: datetime
    backup_type: str  # 'full', 'incremental'
    document_count: int
    size_bytes: int
    location: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class BackupManager:
    """Gestor de backup."""
    
    def __init__(self, analyzer, backup_dir: str = "backups"):
        """Inicializar gestor."""
        self.analyzer = analyzer
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.backups: List[BackupMetadata] = []
    
    async def create_backup(
        self,
        backup_type: str = "full",
        include_analyses: bool = True,
        include_versions: bool = True
    ) -> BackupMetadata:
        """
        Crear backup.
        
        Args:
            backup_type: Tipo de backup ('full', 'incremental')
            include_analyses: Incluir análisis
            include_versions: Incluir versiones
        
        Returns:
            BackupMetadata con información del backup
        """
        backup_id = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_path = self.backup_dir / backup_id
        backup_path.mkdir(exist_ok=True)
        
        backup_data = {
            "backup_id": backup_id,
            "timestamp": datetime.now().isoformat(),
            "backup_type": backup_type,
            "analyses": [],
            "versions": []
        }
        
        # Backup de análisis si está disponible
        if include_analyses and hasattr(self.analyzer, 'database'):
            # En producción, exportar desde base de datos
            backup_data["analyses"] = []
        
        # Backup de versiones si está disponible
        if include_versions and hasattr(self.analyzer, 'version_manager'):
            # En producción, exportar versiones
            backup_data["versions"] = []
        
        # Guardar metadata
        metadata_file = backup_path / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(backup_data, f, indent=2, default=str)
        
        # Calcular tamaño
        size_bytes = sum(
            f.stat().st_size for f in backup_path.rglob('*') if f.is_file()
        )
        
        metadata = BackupMetadata(
            backup_id=backup_id,
            timestamp=datetime.now(),
            backup_type=backup_type,
            document_count=len(backup_data.get("analyses", [])),
            size_bytes=size_bytes,
            location=str(backup_path),
            metadata=backup_data
        )
        
        self.backups.append(metadata)
        
        logger.info(f"Backup creado: {backup_id} ({size_bytes} bytes)")
        
        return metadata
    
    async def restore_backup(
        self,
        backup_id: str,
        restore_analyses: bool = True,
        restore_versions: bool = True
    ) -> Dict[str, Any]:
        """
        Restaurar backup.
        
        Args:
            backup_id: ID del backup
            restore_analyses: Restaurar análisis
            restore_versions: Restaurar versiones
        
        Returns:
            Diccionario con resultado de restauración
        """
        # Buscar backup
        backup_metadata = None
        for backup in self.backups:
            if backup.backup_id == backup_id:
                backup_metadata = backup
                break
        
        if not backup_metadata:
            raise ValueError(f"Backup {backup_id} no encontrado")
        
        backup_path = Path(backup_metadata.location)
        metadata_file = backup_path / "metadata.json"
        
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata de backup no encontrado: {metadata_file}")
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            backup_data = json.load(f)
        
        restored = {
            "backup_id": backup_id,
            "analyses_restored": 0,
            "versions_restored": 0,
            "errors": []
        }
        
        # Restaurar análisis
        if restore_analyses and hasattr(self.analyzer, 'database'):
            # En producción, restaurar desde backup
            restored["analyses_restored"] = len(backup_data.get("analyses", []))
        
        # Restaurar versiones
        if restore_versions and hasattr(self.analyzer, 'version_manager'):
            # En producción, restaurar versiones
            restored["versions_restored"] = len(backup_data.get("versions", []))
        
        logger.info(f"Backup restaurado: {backup_id}")
        
        return restored
    
    def list_backups(self) -> List[BackupMetadata]:
        """Listar backups disponibles."""
        return self.backups
    
    def get_backup_info(self, backup_id: str) -> Optional[BackupMetadata]:
        """Obtener información de backup."""
        for backup in self.backups:
            if backup.backup_id == backup_id:
                return backup
        return None
    
    async def delete_backup(self, backup_id: str) -> bool:
        """Eliminar backup."""
        backup_metadata = self.get_backup_info(backup_id)
        
        if not backup_metadata:
            return False
        
        backup_path = Path(backup_metadata.location)
        
        if backup_path.exists():
            import shutil
            shutil.rmtree(backup_path)
        
        self.backups = [b for b in self.backups if b.backup_id != backup_id]
        
        logger.info(f"Backup eliminado: {backup_id}")
        
        return True


__all__ = [
    "BackupManager",
    "BackupMetadata"
]
















