"""
Incremental Backup - Backup Incremental
========================================

Sistema de backup incremental con deduplicación y restauración selectiva.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import hashlib
import json

logger = logging.getLogger(__name__)


class BackupType(Enum):
    """Tipo de backup."""
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"


@dataclass
class BackupEntry:
    """Entrada de backup."""
    backup_id: str
    backup_type: BackupType
    data_hash: str
    data: bytes
    created_at: datetime
    parent_backup_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BackupSet:
    """Conjunto de backups."""
    set_id: str
    name: str
    entries: List[str] = field(default_factory=list)  # backup_ids
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class IncrementalBackup:
    """Sistema de backup incremental."""
    
    def __init__(self, base_path: str = "./storage/backups"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        self.backups: Dict[str, BackupEntry] = {}
        self.backup_sets: Dict[str, BackupSet] = {}
        self.data_store: Dict[str, bytes] = {}  # hash -> data (deduplicación)
        self._lock = asyncio.Lock()
    
    def _calculate_hash(self, data: bytes) -> str:
        """Calcular hash de datos."""
        return hashlib.sha256(data).hexdigest()
    
    async def create_backup(
        self,
        backup_id: str,
        data: bytes,
        backup_type: BackupType = BackupType.INCREMENTAL,
        parent_backup_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Crear backup."""
        data_hash = self._calculate_hash(data)
        
        # Verificar si ya existe (deduplicación)
        if data_hash in self.data_store:
            logger.info(f"Data already backed up (hash: {data_hash[:16]}...)")
        
        # Guardar datos
        self.data_store[data_hash] = data
        
        backup_entry = BackupEntry(
            backup_id=backup_id,
            backup_type=backup_type,
            data_hash=data_hash,
            data=data,
            created_at=datetime.now(),
            parent_backup_id=parent_backup_id,
            metadata=metadata or {},
        )
        
        async with self._lock:
            self.backups[backup_id] = backup_entry
        
        logger.info(f"Created backup: {backup_id} ({backup_type.value})")
        return backup_id
    
    async def restore_backup(self, backup_id: str) -> Optional[bytes]:
        """Restaurar backup."""
        backup = self.backups.get(backup_id)
        if not backup:
            return None
        
        # Si es incremental, necesitamos restaurar desde el parent
        if backup.backup_type == BackupType.INCREMENTAL and backup.parent_backup_id:
            parent_data = await self.restore_backup(backup.parent_backup_id)
            if parent_data:
                # En producción, aplicar diferencias
                return backup.data
        else:
            return backup.data
    
    async def create_backup_set(
        self,
        set_id: str,
        name: str,
        backup_ids: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Crear conjunto de backups."""
        backup_set = BackupSet(
            set_id=set_id,
            name=name,
            entries=backup_ids,
            metadata=metadata or {},
        )
        
        async with self._lock:
            self.backup_sets[set_id] = backup_set
        
        logger.info(f"Created backup set: {set_id} - {name}")
        return set_id
    
    def get_backup(self, backup_id: str) -> Optional[Dict[str, Any]]:
        """Obtener información de backup."""
        backup = self.backups.get(backup_id)
        if not backup:
            return None
        
        return {
            "backup_id": backup.backup_id,
            "backup_type": backup.backup_type.value,
            "data_hash": backup.data_hash,
            "size": len(backup.data),
            "created_at": backup.created_at.isoformat(),
            "parent_backup_id": backup.parent_backup_id,
            "metadata": backup.metadata,
        }
    
    def list_backups(
        self,
        backup_type: Optional[BackupType] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Listar backups."""
        backups = list(self.backups.values())
        
        if backup_type:
            backups = [b for b in backups if b.backup_type == backup_type]
        
        backups.sort(key=lambda b: b.created_at, reverse=True)
        
        return [self.get_backup(b.backup_id) for b in backups[:limit] if self.get_backup(b.backup_id)]
    
    def get_backup_summary(self) -> Dict[str, Any]:
        """Obtener resumen de backups."""
        by_type: Dict[str, int] = defaultdict(int)
        total_size = 0
        unique_data_size = 0
        
        for backup in self.backups.values():
            by_type[backup.backup_type.value] += 1
            total_size += len(backup.data)
        
        unique_data_size = sum(len(data) for data in self.data_store.values())
        
        return {
            "total_backups": len(self.backups),
            "backups_by_type": dict(by_type),
            "total_size_bytes": total_size,
            "unique_data_size_bytes": unique_data_size,
            "deduplication_ratio": unique_data_size / total_size if total_size > 0 else 1.0,
            "total_backup_sets": len(self.backup_sets),
        }















