"""
Checkpoint Manager
==================

Gestor de checkpoints de modelos.
"""

import logging
import pickle
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Gestor de checkpoints."""
    
    def __init__(self, checkpoints_dir: str = "checkpoints"):
        """
        Inicializar gestor de checkpoints.
        
        Args:
            checkpoints_dir: Directorio para checkpoints
        """
        self.checkpoints_dir = Path(checkpoints_dir)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self._logger = logger
    
    def save_checkpoint(
        self,
        model: Any,
        epoch: int,
        metrics: Dict[str, float],
        optimizer_state: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Guardar checkpoint.
        
        Args:
            model: Modelo a guardar
            epoch: Época actual
            metrics: Métricas
            optimizer_state: Estado del optimizador
            metadata: Metadata adicional
        
        Returns:
            Ruta del checkpoint
        """
        checkpoint_data = {
            "epoch": epoch,
            "model_state": model.state_dict() if hasattr(model, "state_dict") else model,
            "metrics": metrics,
            "optimizer_state": optimizer_state,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        }
        
        filename = f"checkpoint_epoch_{epoch}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        filepath = self.checkpoints_dir / filename
        
        with open(filepath, "wb") as f:
            pickle.dump(checkpoint_data, f)
        
        self._logger.info(f"Saved checkpoint: {filepath}")
        return str(filepath)
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Cargar checkpoint.
        
        Args:
            checkpoint_path: Ruta del checkpoint
        
        Returns:
            Datos del checkpoint
        """
        filepath = Path(checkpoint_path)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        with open(filepath, "rb") as f:
            checkpoint_data = pickle.load(f)
        
        self._logger.info(f"Loaded checkpoint: {checkpoint_path}")
        return checkpoint_data
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """
        Obtener checkpoint más reciente.
        
        Returns:
            Ruta del checkpoint o None
        """
        checkpoints = list(self.checkpoints_dir.glob("checkpoint_*.pkl"))
        
        if not checkpoints:
            return None
        
        # Ordenar por tiempo de modificación
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        return str(latest)
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        Listar checkpoints.
        
        Returns:
            Lista de checkpoints
        """
        checkpoints = []
        
        for filepath in self.checkpoints_dir.glob("checkpoint_*.pkl"):
            stat = filepath.stat()
            checkpoints.append({
                "path": str(filepath),
                "filename": filepath.name,
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
            })
        
        return sorted(checkpoints, key=lambda x: x["modified"], reverse=True)




