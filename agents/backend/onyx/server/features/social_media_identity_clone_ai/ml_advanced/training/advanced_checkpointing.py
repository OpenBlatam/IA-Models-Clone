"""
Checkpointing avanzado con estrategias inteligentes
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class AdvancedCheckpointer:
    """Checkpointer avanzado con estrategias inteligentes"""
    
    def __init__(
        self,
        checkpoint_dir: str = "./checkpoints",
        max_checkpoints: int = 5,
        save_best: bool = True,
        save_every_n_epochs: int = 1,
        monitor_metric: str = "val_loss",
        mode: str = "min"  # "min" or "max"
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.save_best = save_best
        self.save_every_n_epochs = save_every_n_epochs
        self.monitor_metric = monitor_metric
        self.mode = mode
        
        self.best_metric = float('inf') if mode == "min" else float('-inf')
        self.checkpoint_history = []
        self.metadata_file = self.checkpoint_dir / "checkpoints.json"
        self._load_metadata()
    
    def _load_metadata(self):
        """Carga metadata de checkpoints"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.checkpoint_history = json.load(f)
        else:
            self.checkpoint_history = []
    
    def _save_metadata(self):
        """Guarda metadata"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.checkpoint_history, f, indent=2)
    
    def save(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Guarda checkpoint
        
        Args:
            model: Modelo
            optimizer: Optimizador
            scheduler: Scheduler
            epoch: Época actual
            metrics: Métricas
            is_best: Si es el mejor modelo
            additional_info: Información adicional
            
        Returns:
            Path del checkpoint
        """
        checkpoint_name = f"checkpoint_epoch_{epoch}.pt"
        if is_best:
            checkpoint_name = "best_model.pt"
        
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "metrics": metrics or {},
            "timestamp": datetime.utcnow().isoformat(),
            "additional_info": additional_info or {}
        }
        
        if optimizer:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        
        if scheduler:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        
        # Guardar checkpoint
        torch.save(checkpoint, checkpoint_path)
        
        # Actualizar metadata
        checkpoint_info = {
            "path": str(checkpoint_path),
            "epoch": epoch,
            "metrics": metrics or {},
            "is_best": is_best,
            "timestamp": checkpoint["timestamp"]
        }
        
        self.checkpoint_history.append(checkpoint_info)
        
        # Limpiar checkpoints antiguos
        self._cleanup_old_checkpoints()
        
        # Guardar metadata
        self._save_metadata()
        
        logger.info(f"Checkpoint guardado: {checkpoint_path}")
        return str(checkpoint_path)
    
    def _cleanup_old_checkpoints(self):
        """Limpia checkpoints antiguos"""
        # Ordenar por época
        self.checkpoint_history.sort(key=lambda x: x["epoch"], reverse=True)
        
        # Mantener solo los mejores y los más recientes
        best_checkpoints = [c for c in self.checkpoint_history if c.get("is_best", False)]
        regular_checkpoints = [c for c in self.checkpoint_history if not c.get("is_best", False)]
        
        # Mantener solo los N más recientes
        regular_checkpoints = regular_checkpoints[:self.max_checkpoints]
        
        # Eliminar checkpoints antiguos
        all_checkpoints = best_checkpoints + regular_checkpoints
        checkpoint_paths = {c["path"] for c in all_checkpoints}
        
        for checkpoint_info in self.checkpoint_history:
            if checkpoint_info["path"] not in checkpoint_paths:
                try:
                    Path(checkpoint_info["path"]).unlink()
                    logger.info(f"Checkpoint eliminado: {checkpoint_info['path']}")
                except:
                    pass
        
        self.checkpoint_history = all_checkpoints
    
    def load(
        self,
        checkpoint_path: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Carga checkpoint
        
        Args:
            checkpoint_path: Path del checkpoint
            model: Modelo
            optimizer: Optimizador
            scheduler: Scheduler
            
        Returns:
            Información del checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        model.load_state_dict(checkpoint["model_state_dict"])
        
        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if scheduler and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        logger.info(f"Checkpoint cargado: {checkpoint_path}")
        return checkpoint
    
    def load_best(self, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None) -> Dict[str, Any]:
        """Carga mejor checkpoint"""
        best_path = self.checkpoint_dir / "best_model.pt"
        if best_path.exists():
            return self.load(str(best_path), model, optimizer)
        else:
            logger.warning("No se encontró mejor checkpoint")
            return {}




