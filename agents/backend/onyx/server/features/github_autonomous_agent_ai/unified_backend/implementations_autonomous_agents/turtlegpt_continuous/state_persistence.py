"""
State Persistence Module
========================

Persistencia y serialización del estado del agente.
Proporciona funcionalidades para guardar y cargar el estado completo del agente.
"""

import json
import pickle
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class SerializationFormat(Enum):
    """Formato de serialización."""
    JSON = "json"
    PICKLE = "pickle"
    YAML = "yaml"


@dataclass
class StateSnapshot:
    """Snapshot del estado del agente."""
    timestamp: datetime
    agent_name: str
    state: Dict[str, Any]
    metrics: Optional[Dict[str, Any]] = None
    config: Optional[Dict[str, Any]] = None
    version: str = "1.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "agent_name": self.agent_name,
            "state": self.state,
            "metrics": self.metrics,
            "config": self.config,
            "version": self.version
        }


class StatePersistence:
    """
    Gestor de persistencia de estado.
    
    Proporciona funcionalidades para:
    - Serializar estado del agente
    - Guardar estado a archivo
    - Cargar estado desde archivo
    - Múltiples formatos (JSON, Pickle, YAML)
    - Versionado de estado
    - Snapshots y restauración
    """
    
    def __init__(
        self,
        base_path: Optional[Path] = None,
        format: SerializationFormat = SerializationFormat.JSON,
        enable_versioning: bool = True
    ):
        """
        Inicializar gestor de persistencia.
        
        Args:
            base_path: Ruta base para guardar estados
            format: Formato de serialización
            enable_versioning: Habilitar versionado
        """
        self.base_path = base_path or Path("state")
        self.format = format
        self.enable_versioning = enable_versioning
        self.snapshots: List[StateSnapshot] = []
        self.max_snapshots = 50
        
        # Crear directorio si no existe
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def save_state(
        self,
        agent_name: str,
        state: Dict[str, Any],
        metrics: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        filename: Optional[str] = None
    ) -> Path:
        """
        Guardar estado del agente.
        
        Args:
            agent_name: Nombre del agente
            state: Estado del agente
            metrics: Métricas (opcional)
            config: Configuración (opcional)
            filename: Nombre del archivo (opcional)
            
        Returns:
            Ruta del archivo guardado
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{agent_name}_state_{timestamp}.{self.format.value}"
        
        file_path = self.base_path / filename
        
        # Crear snapshot
        snapshot = StateSnapshot(
            timestamp=datetime.now(),
            agent_name=agent_name,
            state=state,
            metrics=metrics,
            config=config
        )
        
        # Serializar según formato
        try:
            if self.format == SerializationFormat.JSON:
                self._save_json(file_path, snapshot)
            elif self.format == SerializationFormat.PICKLE:
                self._save_pickle(file_path, snapshot)
            elif self.format == SerializationFormat.YAML:
                self._save_yaml(file_path, snapshot)
            else:
                raise ValueError(f"Unsupported format: {self.format}")
            
            # Agregar a snapshots
            self.snapshots.append(snapshot)
            if len(self.snapshots) > self.max_snapshots:
                self.snapshots = self.snapshots[-self.max_snapshots:]
            
            logger.info(f"State saved to {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Error saving state: {e}", exc_info=True)
            raise
    
    def load_state(self, file_path: Path) -> StateSnapshot:
        """
        Cargar estado del agente.
        
        Args:
            file_path: Ruta del archivo
            
        Returns:
            Snapshot del estado
        """
        if not file_path.exists():
            raise FileNotFoundError(f"State file not found: {file_path}")
        
        try:
            if self.format == SerializationFormat.JSON:
                return self._load_json(file_path)
            elif self.format == SerializationFormat.PICKLE:
                return self._load_pickle(file_path)
            elif self.format == SerializationFormat.YAML:
                return self._load_yaml(file_path)
            else:
                raise ValueError(f"Unsupported format: {self.format}")
                
        except Exception as e:
            logger.error(f"Error loading state: {e}", exc_info=True)
            raise
    
    def save_snapshot(
        self,
        agent_name: str,
        state: Dict[str, Any],
        metrics: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> StateSnapshot:
        """
        Guardar snapshot del estado.
        
        Args:
            agent_name: Nombre del agente
            state: Estado del agente
            metrics: Métricas (opcional)
            config: Configuración (opcional)
            
        Returns:
            Snapshot creado
        """
        snapshot = StateSnapshot(
            timestamp=datetime.now(),
            agent_name=agent_name,
            state=state.copy(),
            metrics=metrics.copy() if metrics else None,
            config=config.copy() if config else None
        )
        
        self.snapshots.append(snapshot)
        
        # Limitar tamaño
        if len(self.snapshots) > self.max_snapshots:
            self.snapshots = self.snapshots[-self.max_snapshots:]
        
        return snapshot
    
    def get_snapshots(
        self,
        agent_name: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[StateSnapshot]:
        """
        Obtener snapshots.
        
        Args:
            agent_name: Filtrar por nombre de agente
            limit: Límite de resultados
            
        Returns:
            Lista de snapshots
        """
        filtered = self.snapshots
        
        if agent_name:
            filtered = [s for s in filtered if s.agent_name == agent_name]
        
        if limit:
            filtered = filtered[-limit:]
        
        return filtered
    
    def restore_from_snapshot(
        self,
        snapshot: StateSnapshot
    ) -> Dict[str, Any]:
        """
        Restaurar estado desde snapshot.
        
        Args:
            snapshot: Snapshot a restaurar
            
        Returns:
            Estado restaurado
        """
        return snapshot.state.copy()
    
    def get_latest_snapshot(self, agent_name: Optional[str] = None) -> Optional[StateSnapshot]:
        """
        Obtener snapshot más reciente.
        
        Args:
            agent_name: Filtrar por nombre de agente
            
        Returns:
            Snapshot más reciente o None
        """
        filtered = self.snapshots
        
        if agent_name:
            filtered = [s for s in filtered if s.agent_name == agent_name]
        
        if not filtered:
            return None
        
        return max(filtered, key=lambda s: s.timestamp)
    
    def list_state_files(self, agent_name: Optional[str] = None) -> List[Path]:
        """
        Listar archivos de estado.
        
        Args:
            agent_name: Filtrar por nombre de agente
            
        Returns:
            Lista de archivos de estado
        """
        pattern = f"*_state_*.{self.format.value}"
        if agent_name:
            pattern = f"{agent_name}_state_*.{self.format.value}"
        
        files = list(self.base_path.glob(pattern))
        return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)
    
    def delete_state_file(self, file_path: Path) -> bool:
        """
        Eliminar archivo de estado.
        
        Args:
            file_path: Ruta del archivo
            
        Returns:
            True si se eliminó exitosamente
        """
        try:
            if file_path.exists():
                file_path.unlink()
                logger.info(f"State file deleted: {file_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting state file: {e}", exc_info=True)
            return False
    
    def cleanup_old_states(self, keep_count: int = 10) -> int:
        """
        Limpiar estados antiguos, manteniendo solo los más recientes.
        
        Args:
            keep_count: Número de estados a mantener
            
        Returns:
            Número de archivos eliminados
        """
        files = self.list_state_files()
        
        if len(files) <= keep_count:
            return 0
        
        files_to_delete = files[keep_count:]
        deleted = 0
        
        for file_path in files_to_delete:
            if self.delete_state_file(file_path):
                deleted += 1
        
        logger.info(f"Cleaned up {deleted} old state files")
        return deleted
    
    def _save_json(self, file_path: Path, snapshot: StateSnapshot) -> None:
        """Guardar en formato JSON."""
        data = snapshot.to_dict()
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
    
    def _load_json(self, file_path: Path) -> StateSnapshot:
        """Cargar desde formato JSON."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return StateSnapshot(
            timestamp=datetime.fromisoformat(data['timestamp']),
            agent_name=data['agent_name'],
            state=data['state'],
            metrics=data.get('metrics'),
            config=data.get('config'),
            version=data.get('version', '1.0')
        )
    
    def _save_pickle(self, file_path: Path, snapshot: StateSnapshot) -> None:
        """Guardar en formato Pickle."""
        with open(file_path, 'wb') as f:
            pickle.dump(snapshot, f)
    
    def _load_pickle(self, file_path: Path) -> StateSnapshot:
        """Cargar desde formato Pickle."""
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    
    def _save_yaml(self, file_path: Path, snapshot: StateSnapshot) -> None:
        """Guardar en formato YAML."""
        try:
            import yaml
            data = snapshot.to_dict()
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        except ImportError:
            raise ImportError("PyYAML is required for YAML format. Install with: pip install pyyaml")
    
    def _load_yaml(self, file_path: Path) -> StateSnapshot:
        """Cargar desde formato YAML."""
        try:
            import yaml
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            return StateSnapshot(
                timestamp=datetime.fromisoformat(data['timestamp']),
                agent_name=data['agent_name'],
                state=data['state'],
                metrics=data.get('metrics'),
                config=data.get('config'),
                version=data.get('version', '1.0')
            )
        except ImportError:
            raise ImportError("PyYAML is required for YAML format. Install with: pip install pyyaml")
    
    def export_state_summary(self, file_path: Path) -> bool:
        """
        Exportar resumen de estados.
        
        Args:
            file_path: Ruta del archivo
            
        Returns:
            True si se exportó exitosamente
        """
        try:
            summary = {
                "total_snapshots": len(self.snapshots),
                "snapshots": [s.to_dict() for s in self.snapshots],
                "state_files": [str(f) for f in self.list_state_files()],
                "export_timestamp": datetime.now().isoformat()
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, default=str)
            
            return True
            
        except Exception as e:
            logger.error(f"Error exporting state summary: {e}", exc_info=True)
            return False


def create_state_persistence(
    base_path: Optional[Path] = None,
    format: SerializationFormat = SerializationFormat.JSON,
    enable_versioning: bool = True
) -> StatePersistence:
    """
    Factory function para crear StatePersistence.
    
    Args:
        base_path: Ruta base para guardar estados
        format: Formato de serialización
        enable_versioning: Habilitar versionado
        
    Returns:
        Instancia de StatePersistence
    """
    return StatePersistence(
        base_path=base_path,
        format=format,
        enable_versioning=enable_versioning
    )
