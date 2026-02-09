"""
Model Checkpointer - Sistema avanzado de checkpoints
=====================================================
"""

import logging
import torch
import os
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import shutil

logger = logging.getLogger(__name__)


@dataclass
class Checkpoint:
    """Checkpoint individual"""
    epoch: int
    step: int
    model_state: Dict[str, Any]
    optimizer_state: Optional[Dict[str, Any]] = None
    scheduler_state: Optional[Dict[str, Any]] = None
    scaler_state: Optional[Dict[str, Any]] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    is_best: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario"""
        return {
            "epoch": self.epoch,
            "step": self.step,
            "metrics": self.metrics,
            "timestamp": self.timestamp.isoformat(),
            "is_best": self.is_best
        }


class ModelCheckpointer:
    """Gestor avanzado de checkpoints"""
    
    def __init__(self, checkpoint_dir: str = "./checkpoints", max_to_keep: int = 5):
        self.checkpoint_dir = checkpoint_dir
        self.max_to_keep = max_to_keep
        self.checkpoints: List[Checkpoint] = []
        self.best_checkpoint: Optional[Checkpoint] = None
        self.best_metric_value = float('-inf')
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save_checkpoint(
        self,
        epoch: int,
        step: int,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False,
        metric_name: str = "val_loss",
        higher_is_better: bool = False
    ) -> str:
        """Guarda un checkpoint"""
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"checkpoint_epoch_{epoch}_step_{step}.pt"
        )
        
        checkpoint_data = {
            "epoch": epoch,
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "scaler_state_dict": scaler.state_dict() if scaler else None,
            "metrics": metrics or {},
            "timestamp": datetime.now().isoformat()
        }
        
        torch.save(checkpoint_data, checkpoint_path)
        
        checkpoint = Checkpoint(
            epoch=epoch,
            step=step,
            model_state=checkpoint_data["model_state_dict"],
            optimizer_state=checkpoint_data["optimizer_state_dict"],
            scheduler_state=checkpoint_data["scheduler_state_dict"],
            scaler_state=checkpoint_data["scaler_state_dict"],
            metrics=metrics or {},
            is_best=is_best
        )
        
        self.checkpoints.append(checkpoint)
        
        # Actualizar mejor checkpoint
        if metrics and metric_name in metrics:
            metric_value = metrics[metric_name]
            is_better = (
                (higher_is_better and metric_value > self.best_metric_value) or
                (not higher_is_better and metric_value < self.best_metric_value)
            )
            
            if is_better:
                self.best_metric_value = metric_value
                self.best_checkpoint = checkpoint
                
                # Guardar como best
                best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
                shutil.copy(checkpoint_path, best_path)
                checkpoint.is_best = True
                logger.info(f"Mejor modelo guardado: {metric_name}={metric_value:.4f}")
        
        # Limpiar checkpoints antiguos
        self._cleanup_old_checkpoints()
        
        logger.info(f"Checkpoint guardado: {checkpoint_path}")
        return checkpoint_path
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
        map_location: str = "cpu"
    ) -> Dict[str, Any]:
        """Carga un checkpoint"""
        checkpoint_data = torch.load(checkpoint_path, map_location=map_location)
        
        model.load_state_dict(checkpoint_data["model_state_dict"])
        
        if optimizer and checkpoint_data.get("optimizer_state_dict"):
            optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
        
        if scheduler and checkpoint_data.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint_data["scheduler_state_dict"])
        
        if scaler and checkpoint_data.get("scaler_state_dict"):
            scaler.load_state_dict(checkpoint_data["scaler_state_dict"])
        
        logger.info(f"Checkpoint cargado: {checkpoint_path}")
        return checkpoint_data
    
    def load_best_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None
    ) -> Optional[Dict[str, Any]]:
        """Carga el mejor checkpoint"""
        best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
        if os.path.exists(best_path):
            return self.load_checkpoint(best_path, model, optimizer, scheduler, scaler)
        return None
    
    def _cleanup_old_checkpoints(self):
        """Limpia checkpoints antiguos"""
        if len(self.checkpoints) <= self.max_to_keep:
            return
        
        # Ordenar por epoch (más antiguos primero)
        sorted_checkpoints = sorted(self.checkpoints, key=lambda c: c.epoch)
        
        # Mantener solo los más recientes y el mejor
        to_keep = sorted_checkpoints[-self.max_to_keep:]
        if self.best_checkpoint and self.best_checkpoint not in to_keep:
            to_keep.append(self.best_checkpoint)
        
        # Eliminar checkpoints no mantenidos
        for checkpoint in self.checkpoints:
            if checkpoint not in to_keep and not checkpoint.is_best:
                checkpoint_path = os.path.join(
                    self.checkpoint_dir,
                    f"checkpoint_epoch_{checkpoint.epoch}_step_{checkpoint.step}.pt"
                )
                if os.path.exists(checkpoint_path):
                    os.remove(checkpoint_path)
        
        self.checkpoints = to_keep
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """Lista todos los checkpoints"""
        return [c.to_dict() for c in self.checkpoints]
    
    def get_checkpoint_info(self) -> Dict[str, Any]:
        """Obtiene información de checkpoints"""
        return {
            "total_checkpoints": len(self.checkpoints),
            "best_checkpoint": self.best_checkpoint.to_dict() if self.best_checkpoint else None,
            "best_metric_value": self.best_metric_value,
            "checkpoints": self.list_checkpoints()
        }




