"""
Advanced Checkpointing
======================

Sistema avanzado de checkpointing para modelos.
"""

import logging
import os
from typing import Dict, Any, Optional, List
from pathlib import Path
import json

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Gestor avanzado de checkpoints.
    
    Maneja guardado/carga de modelos con metadata.
    """
    
    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        """
        Inicializar gestor.
        
        Args:
            checkpoint_dir: Directorio de checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints: List[Dict[str, Any]] = []
    
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: int = 0,
        step: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        is_best: bool = False,
        checkpoint_name: Optional[str] = None
    ) -> str:
        """
        Guardar checkpoint.
        
        Args:
            model: Modelo
            optimizer: Optimizador (opcional)
            epoch: Época
            step: Paso
            metrics: Métricas
            metadata: Metadata adicional
            is_best: Es el mejor modelo
            checkpoint_name: Nombre del checkpoint
            
        Returns:
            Ruta del checkpoint guardado
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        
        checkpoint_data = {
            "epoch": epoch,
            "step": step,
            "model_state_dict": model.state_dict(),
            "metrics": metrics or {},
            "metadata": metadata or {}
        }
        
        if optimizer:
            checkpoint_data["optimizer_state_dict"] = optimizer.state_dict()
        
        # Nombre del checkpoint
        if checkpoint_name is None:
            checkpoint_name = f"checkpoint_epoch_{epoch}_step_{step}.pt"
        
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        torch.save(checkpoint_data, checkpoint_path)
        
        # Guardar mejor modelo
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint_data, best_path)
            logger.info(f"Saved best model to {best_path}")
        
        # Guardar último checkpoint
        latest_path = self.checkpoint_dir / "latest.pt"
        torch.save(checkpoint_data, latest_path)
        
        # Registrar checkpoint
        checkpoint_info = {
            "name": checkpoint_name,
            "path": str(checkpoint_path),
            "epoch": epoch,
            "step": step,
            "metrics": metrics or {},
            "is_best": is_best
        }
        self.checkpoints.append(checkpoint_info)
        
        # Guardar índice
        self._save_index()
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        return str(checkpoint_path)
    
    def load_checkpoint(
        self,
        model: nn.Module,
        checkpoint_path: Optional[str] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        load_best: bool = False
    ) -> Dict[str, Any]:
        """
        Cargar checkpoint.
        
        Args:
            model: Modelo
            checkpoint_path: Ruta del checkpoint (None para latest)
            optimizer: Optimizador (opcional)
            load_best: Cargar mejor modelo
            
        Returns:
            Datos del checkpoint
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        
        if checkpoint_path is None:
            if load_best:
                checkpoint_path = self.checkpoint_dir / "best_model.pt"
            else:
                checkpoint_path = self.checkpoint_dir / "latest.pt"
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint_data = torch.load(checkpoint_path, map_location="cpu")
        
        # Cargar modelo
        model.load_state_dict(checkpoint_data["model_state_dict"])
        
        # Cargar optimizador
        if optimizer and "optimizer_state_dict" in checkpoint_data:
            optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint_data
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """Listar checkpoints."""
        return self.checkpoints.copy()
    
    def get_best_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Obtener mejor checkpoint."""
        best = [c for c in self.checkpoints if c.get("is_best", False)]
        if best:
            return max(best, key=lambda x: x.get("metrics", {}).get("accuracy", 0))
        return None
    
    def _save_index(self):
        """Guardar índice de checkpoints."""
        index_path = self.checkpoint_dir / "checkpoints_index.json"
        with open(index_path, "w") as f:
            json.dump(self.checkpoints, f, indent=2)
    
    def cleanup_old_checkpoints(self, keep_last_n: int = 5):
        """
        Limpiar checkpoints antiguos.
        
        Args:
            keep_last_n: Mantener últimos N checkpoints
        """
        if len(self.checkpoints) <= keep_last_n:
            return
        
        # Ordenar por step
        sorted_checkpoints = sorted(self.checkpoints, key=lambda x: x["step"], reverse=True)
        
        # Eliminar antiguos
        for checkpoint in sorted_checkpoints[keep_last_n:]:
            try:
                os.remove(checkpoint["path"])
                self.checkpoints.remove(checkpoint)
            except Exception as e:
                logger.warning(f"Could not remove checkpoint {checkpoint['path']}: {e}")
        
        self._save_index()
        logger.info(f"Cleaned up old checkpoints, kept {keep_last_n}")

