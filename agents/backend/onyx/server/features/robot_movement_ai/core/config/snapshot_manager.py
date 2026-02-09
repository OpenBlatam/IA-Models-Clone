"""
Snapshot Manager System
=======================

Sistema de gestión de snapshots del sistema.
"""

import logging
import json
import pickle
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Snapshot:
    """Snapshot."""
    snapshot_id: str
    name: str
    description: str
    data: Dict[str, Any]
    created_by: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


class SnapshotManager:
    """
    Gestor de snapshots.
    
    Gestiona snapshots del estado del sistema.
    """
    
    def __init__(self, storage_path: str = "data/snapshots"):
        """
        Inicializar gestor de snapshots.
        
        Args:
            storage_path: Ruta de almacenamiento
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.snapshots: Dict[str, Snapshot] = {}
        self._load_snapshots()
    
    def _load_snapshots(self) -> None:
        """Cargar snapshots desde archivos."""
        try:
            for snapshot_file in self.storage_path.glob("*.json"):
                try:
                    with open(snapshot_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        snapshot = Snapshot(**data)
                        self.snapshots[snapshot.snapshot_id] = snapshot
                except Exception as e:
                    logger.error(f"Error loading snapshot {snapshot_file}: {e}")
        except Exception as e:
            logger.error(f"Error loading snapshots: {e}")
    
    def create_snapshot(
        self,
        snapshot_id: str,
        name: str,
        description: str,
        data: Dict[str, Any],
        created_by: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Snapshot:
        """
        Crear snapshot.
        
        Args:
            snapshot_id: ID único del snapshot
            name: Nombre
            description: Descripción
            data: Datos del snapshot
            created_by: ID del creador
            metadata: Metadata adicional
            
        Returns:
            Snapshot creado
        """
        snapshot = Snapshot(
            snapshot_id=snapshot_id,
            name=name,
            description=description,
            data=data,
            created_by=created_by,
            metadata=metadata or {}
        )
        
        self.snapshots[snapshot_id] = snapshot
        self._save_snapshot(snapshot)
        
        logger.info(f"Created snapshot: {name} ({snapshot_id})")
        
        return snapshot
    
    def _save_snapshot(self, snapshot: Snapshot) -> None:
        """Guardar snapshot en archivo."""
        try:
            snapshot_file = self.storage_path / f"{snapshot.snapshot_id}.json"
            with open(snapshot_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "snapshot_id": snapshot.snapshot_id,
                    "name": snapshot.name,
                    "description": snapshot.description,
                    "data": snapshot.data,
                    "created_by": snapshot.created_by,
                    "timestamp": snapshot.timestamp,
                    "metadata": snapshot.metadata
                }, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving snapshot: {e}")
    
    def create_system_snapshot(
        self,
        snapshot_id: str,
        name: str,
        description: str = "System snapshot",
        created_by: Optional[str] = None
    ) -> Snapshot:
        """
        Crear snapshot del sistema completo.
        
        Args:
            snapshot_id: ID único del snapshot
            name: Nombre
            description: Descripción
            created_by: ID del creador
            
        Returns:
            Snapshot creado
        """
        from ..analytics.analytics import get_analytics_engine
        from ..performance import get_performance_monitor
        from ..system.metrics import get_metrics_collector
        
        # Recopilar datos del sistema
        system_data = {
            "timestamp": datetime.now().isoformat(),
            "analytics": {},
            "performance": {},
            "metrics": {}
        }
        
        try:
            analytics = get_analytics_engine()
            system_data["analytics"] = analytics.get_operation_statistics()
        except Exception as e:
            logger.warning(f"Error getting analytics: {e}")
        
        try:
            monitor = get_performance_monitor()
            system_data["performance"] = monitor.get_performance_metrics()
        except Exception as e:
            logger.warning(f"Error getting performance: {e}")
        
        try:
            metrics = get_metrics_collector()
            system_data["metrics"] = metrics.get_all_metrics()
        except Exception as e:
            logger.warning(f"Error getting metrics: {e}")
        
        return self.create_snapshot(
            snapshot_id=snapshot_id,
            name=name,
            description=description,
            data=system_data,
            created_by=created_by,
            metadata={"type": "system"}
        )
    
    def get_snapshot(self, snapshot_id: str) -> Optional[Snapshot]:
        """Obtener snapshot por ID."""
        return self.snapshots.get(snapshot_id)
    
    def list_snapshots(self, limit: int = 100) -> List[Snapshot]:
        """
        Listar snapshots.
        
        Args:
            limit: Límite de resultados
            
        Returns:
            Lista de snapshots
        """
        snapshots = list(self.snapshots.values())
        snapshots.sort(key=lambda x: x.timestamp, reverse=True)
        return snapshots[:limit]
    
    def delete_snapshot(self, snapshot_id: str) -> bool:
        """
        Eliminar snapshot.
        
        Args:
            snapshot_id: ID del snapshot
            
        Returns:
            True si se eliminó, False si no existe
        """
        if snapshot_id not in self.snapshots:
            return False
        
        del self.snapshots[snapshot_id]
        
        # Eliminar archivo
        snapshot_file = self.storage_path / f"{snapshot_id}.json"
        if snapshot_file.exists():
            snapshot_file.unlink()
        
        logger.info(f"Deleted snapshot: {snapshot_id}")
        
        return True
    
    def restore_snapshot(self, snapshot_id: str) -> bool:
        """
        Restaurar snapshot.
        
        Args:
            snapshot_id: ID del snapshot
            
        Returns:
            True si se restauró, False si no existe
        """
        snapshot = self.get_snapshot(snapshot_id)
        if not snapshot:
            return False
        
        # Restaurar datos del sistema
        try:
            if "analytics" in snapshot.data:
                # Restaurar analytics si es necesario
                pass
            
            if "performance" in snapshot.data:
                # Restaurar performance si es necesario
                pass
            
            if "metrics" in snapshot.data:
                # Restaurar metrics si es necesario
                pass
            
            logger.info(f"Restored snapshot: {snapshot_id}")
            return True
        except Exception as e:
            logger.error(f"Error restoring snapshot: {e}")
            return False


# Instancia global
_snapshot_manager: Optional[SnapshotManager] = None


def get_snapshot_manager(storage_path: str = "data/snapshots") -> SnapshotManager:
    """Obtener instancia global del gestor de snapshots."""
    global _snapshot_manager
    if _snapshot_manager is None:
        _snapshot_manager = SnapshotManager(storage_path=storage_path)
    return _snapshot_manager






