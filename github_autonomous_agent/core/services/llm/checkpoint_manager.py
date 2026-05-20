"""
Checkpoint Manager - Gestión de checkpoints y estados.

Sigue principios de checkpointing en deep learning.
"""

from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import json
import pickle
import hashlib

from config.logging_config import get_logger

logger = get_logger(__name__)


class CheckpointManager:
    """
    Gestor de checkpoints para estados y configuraciones.
    
    Sigue principios de checkpointing (como PyTorch).
    """
    
    def __init__(self, checkpoint_dir: str = "./storage/checkpoints"):
        """
        Inicializar CheckpointManager.
        
        Args:
            checkpoint_dir: Directorio para checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(
        self,
        name: str,
        state: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        format: str = "json"
    ) -> str:
        """
        Guardar checkpoint.
        
        Args:
            name: Nombre del checkpoint
            state: Estado a guardar
            metadata: Metadatos adicionales
            format: Formato (json o pickle)
            
        Returns:
            Ruta del checkpoint guardado
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"{name}_{timestamp}"
        
        if format == "json":
            filepath = self.checkpoint_dir / f"{checkpoint_name}.json"
            checkpoint_data = {
                "name": name,
                "timestamp": timestamp,
                "state": state,
                "metadata": metadata or {}
            }
            with open(filepath, 'w') as f:
                json.dump(checkpoint_data, f, indent=2, default=str)
        elif format == "pickle":
            filepath = self.checkpoint_dir / f"{checkpoint_name}.pkl"
            checkpoint_data = {
                "name": name,
                "timestamp": timestamp,
                "state": state,
                "metadata": metadata or {}
            }
            with open(filepath, 'wb') as f:
                pickle.dump(checkpoint_data, f)
        else:
            raise ValueError(f"Formato no soportado: {format}")
        
        logger.info(f"Checkpoint guardado: {filepath}")
        return str(filepath)
    
    def load_checkpoint(
        self,
        filepath: str,
        format: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Cargar checkpoint.
        
        Args:
            filepath: Ruta al checkpoint
            format: Formato (auto-detecta si None)
            
        Returns:
            Diccionario con estado y metadatos
        """
        path = Path(filepath)
        
        if format is None:
            format = "json" if path.suffix == ".json" else "pickle"
        
        if format == "json":
            with open(path, 'r') as f:
                data = json.load(f)
        elif format == "pickle":
            with open(path, 'rb') as f:
                data = pickle.load(f)
        else:
            raise ValueError(f"Formato no soportado: {format}")
        
        logger.info(f"Checkpoint cargado: {filepath}")
        return data
    
    def list_checkpoints(
        self,
        name_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Listar checkpoints disponibles.
        
        Args:
            name_filter: Filtrar por nombre
            
        Returns:
            Lista de checkpoints
        """
        checkpoints = []
        
        for filepath in self.checkpoint_dir.glob("*.json"):
            try:
                data = self.load_checkpoint(str(filepath))
                if name_filter is None or name_filter in data.get("name", ""):
                    checkpoints.append({
                        "filepath": str(filepath),
                        "name": data.get("name"),
                        "timestamp": data.get("timestamp"),
                        "metadata": data.get("metadata", {})
                    })
            except Exception as e:
                logger.error(f"Error cargando checkpoint {filepath}: {e}")
        
        # Ordenar por timestamp (más reciente primero)
        checkpoints.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        return checkpoints
    
    def get_latest_checkpoint(
        self,
        name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Obtener el checkpoint más reciente por nombre.
        
        Args:
            name: Nombre del checkpoint
            
        Returns:
            Checkpoint más reciente o None
        """
        checkpoints = self.list_checkpoints(name_filter=name)
        return checkpoints[0] if checkpoints else None
    
    def delete_checkpoint(self, filepath: str) -> bool:
        """
        Eliminar checkpoint.
        
        Args:
            filepath: Ruta al checkpoint
            
        Returns:
            True si se eliminó, False si no existía
        """
        path = Path(filepath)
        if path.exists():
            path.unlink()
            logger.info(f"Checkpoint eliminado: {filepath}")
            return True
        return False
    
    def create_state_hash(self, state: Dict[str, Any]) -> str:
        """
        Crear hash de un estado para verificación.
        
        Args:
            state: Estado a hashear
            
        Returns:
            Hash SHA256 del estado
        """
        state_str = json.dumps(state, sort_keys=True, default=str)
        return hashlib.sha256(state_str.encode()).hexdigest()


# Instancia global
_checkpoint_manager = CheckpointManager()


def get_checkpoint_manager() -> CheckpointManager:
    """Obtener instancia global del checkpoint manager."""
    return _checkpoint_manager



