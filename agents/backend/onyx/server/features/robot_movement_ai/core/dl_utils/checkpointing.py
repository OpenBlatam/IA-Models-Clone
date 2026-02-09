"""
Checkpoint Management
=====================

Gestión de checkpoints de modelos.
"""

import logging
import uuid
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

logger = logging.getLogger(__name__)


@dataclass
class Checkpoint:
    """Checkpoint de modelo."""
    checkpoint_id: str
    model_id: str
    epoch: int
    loss: float
    metrics: Dict[str, float]
    model_state: Dict[str, Any]
    optimizer_state: Optional[Dict[str, Any]] = None
    scheduler_state: Optional[Dict[str, Any]] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


class CheckpointManager:
    """
    Gestor de checkpoints.
    
    Maneja guardado y carga de checkpoints.
    """
    
    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        """
        Inicializar gestor.
        
        Args:
            checkpoint_dir: Directorio para checkpoints
        """
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available. Checkpoint features will be limited.")
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints: Dict[str, Checkpoint] = {}
    
    def save(
        self,
        model,
        epoch: int,
        loss: float,
        metrics: Dict[str, float],
        optimizer: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        checkpoint_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Guardar checkpoint.
        
        Args:
            model: Modelo
            epoch: Época
            loss: Pérdida
            metrics: Métricas
            optimizer: Optimizador (opcional)
            scheduler: Scheduler (opcional)
            checkpoint_id: ID del checkpoint (opcional)
            metadata: Metadata adicional (opcional)
            
        Returns:
            ID del checkpoint
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for checkpointing")
        
        checkpoint_id = checkpoint_id or str(uuid.uuid4())
        
        checkpoint = Checkpoint(
            checkpoint_id=checkpoint_id,
            model_id=getattr(model, 'model_id', 'unknown'),
            epoch=epoch,
            loss=loss,
            metrics=metrics,
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict() if optimizer else None,
            scheduler_state=scheduler.state_dict() if scheduler else None,
            metadata=metadata or {}
        )
        
        # Guardar archivo
        file_path = self.checkpoint_dir / f"checkpoint_{checkpoint_id}.pt"
        torch.save({
            "checkpoint_id": checkpoint.checkpoint_id,
            "model_id": checkpoint.model_id,
            "epoch": checkpoint.epoch,
            "loss": checkpoint.loss,
            "metrics": checkpoint.metrics,
            "model_state": checkpoint.model_state,
            "optimizer_state": checkpoint.optimizer_state,
            "scheduler_state": checkpoint.scheduler_state,
            "created_at": checkpoint.created_at,
            "metadata": checkpoint.metadata
        }, file_path)
        
        self.checkpoints[checkpoint_id] = checkpoint
        logger.info(f"Saved checkpoint {checkpoint_id} to {file_path}")
        
        return checkpoint_id
    
    def load(self, checkpoint_id: str, model, optimizer: Optional[Any] = None, scheduler: Optional[Any] = None) -> Checkpoint:
        """
        Cargar checkpoint.
        
        Args:
            checkpoint_id: ID del checkpoint
            model: Modelo a cargar
            optimizer: Optimizador (opcional)
            scheduler: Scheduler (opcional)
            
        Returns:
            Checkpoint cargado
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for checkpointing")
        
        file_path = self.checkpoint_dir / f"checkpoint_{checkpoint_id}.pt"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {file_path}")
        
        data = torch.load(file_path, map_location="cpu")
        
        # Cargar estados
        model.load_state_dict(data["model_state"])
        
        if optimizer and data.get("optimizer_state"):
            optimizer.load_state_dict(data["optimizer_state"])
        
        if scheduler and data.get("scheduler_state"):
            scheduler.load_state_dict(data["scheduler_state"])
        
        checkpoint = Checkpoint(
            checkpoint_id=data["checkpoint_id"],
            model_id=data["model_id"],
            epoch=data["epoch"],
            loss=data["loss"],
            metrics=data["metrics"],
            model_state=data["model_state"],
            optimizer_state=data.get("optimizer_state"),
            scheduler_state=data.get("scheduler_state"),
            created_at=data["created_at"],
            metadata=data.get("metadata", {})
        )
        
        self.checkpoints[checkpoint_id] = checkpoint
        logger.info(f"Loaded checkpoint {checkpoint_id} from {file_path}")
        
        return checkpoint
    
    def list_checkpoints(self) -> list:
        """Listar todos los checkpoints."""
        return list(self.checkpoints.keys())
    
    def get_best_checkpoint(self, metric_name: str = "val_loss", minimize: bool = True) -> Optional[str]:
        """
        Obtener mejor checkpoint según métrica.
        
        Args:
            metric_name: Nombre de la métrica
            minimize: Minimizar (True) o maximizar (False)
            
        Returns:
            ID del mejor checkpoint o None
        """
        if not self.checkpoints:
            return None
        
        best_id = None
        best_value = float('inf') if minimize else float('-inf')
        
        for checkpoint_id, checkpoint in self.checkpoints.items():
            value = checkpoint.metrics.get(metric_name)
            if value is None:
                continue
            
            if minimize:
                if value < best_value:
                    best_value = value
                    best_id = checkpoint_id
            else:
                if value > best_value:
                    best_value = value
                    best_id = checkpoint_id
        
        return best_id

