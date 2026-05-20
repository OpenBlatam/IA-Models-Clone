"""
Checkpoint Manager - Gestor de checkpoints
===========================================

Sistema avanzado para gestión de checkpoints de modelos.
Sigue mejores prácticas de checkpointing.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import shutil
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class CheckpointMetadata:
    """Metadatos de checkpoint"""
    epoch: int
    step: int
    loss: float
    metric: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    model_config: Optional[Dict[str, Any]] = None
    training_config: Optional[Dict[str, Any]] = None
    optimizer_state: bool = False
    scheduler_state: bool = False
    scaler_state: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class CheckpointManager:
    """Gestor de checkpoints"""
    
    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        """
        Inicializar gestor de checkpoints.
        
        Args:
            checkpoint_dir: Directorio donde guardar checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints: Dict[str, CheckpointMetadata] = {}
        logger.info(f"CheckpointManager initialized in {checkpoint_dir}")
    
    def save_checkpoint(
        self,
        checkpoint_id: str,
        model: nn.Module,
        epoch: int,
        step: int,
        loss: float,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        metric: Optional[float] = None,
        model_config: Optional[Dict[str, Any]] = None,
        training_config: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        is_best: bool = False
    ) -> str:
        """
        Guardar checkpoint completo.
        
        Args:
            checkpoint_id: ID único del checkpoint
            model: Modelo a guardar
            epoch: Época actual
            step: Step actual
            loss: Pérdida actual
            optimizer: Optimizador (opcional)
            scheduler: Scheduler (opcional)
            scaler: GradScaler (opcional)
            metric: Métrica adicional (opcional)
            model_config: Configuración del modelo (opcional)
            training_config: Configuración de entrenamiento (opcional)
            metadata: Metadatos adicionales (opcional)
            is_best: Si es el mejor modelo hasta ahora
        
        Returns:
            Ruta al checkpoint guardado
        """
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.pt"
        best_path = self.checkpoint_dir / "best.pt"
        
        # Prepare checkpoint data
        checkpoint_data = {
            "epoch": epoch,
            "step": step,
            "loss": loss,
            "metric": metric,
            "model_state_dict": model.state_dict(),
            "model_config": model_config,
            "training_config": training_config,
            "metadata": metadata or {},
        }
        
        # Add optimizer state
        if optimizer is not None:
            checkpoint_data["optimizer_state_dict"] = optimizer.state_dict()
        
        # Add scheduler state
        if scheduler is not None:
            checkpoint_data["scheduler_state_dict"] = scheduler.state_dict()
        
        # Add scaler state
        if scaler is not None:
            checkpoint_data["scaler_state_dict"] = scaler.state_dict()
        
        # Save checkpoint
        try:
            torch.save(checkpoint_data, checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
            
            # Save as best if indicated
            if is_best:
                shutil.copy(checkpoint_path, best_path)
                logger.info(f"Best checkpoint saved: {best_path}")
            
            # Store metadata
            metadata_obj = CheckpointMetadata(
                epoch=epoch,
                step=step,
                loss=loss,
                metric=metric,
                model_config=model_config,
                training_config=training_config,
                optimizer_state=optimizer is not None,
                scheduler_state=scheduler is not None,
                scaler_state=scaler is not None,
                metadata=metadata or {},
            )
            self.checkpoints[checkpoint_id] = metadata_obj
            
            return str(checkpoint_path)
        
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}", exc_info=True)
            raise
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        map_location: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Cargar checkpoint.
        
        Args:
            checkpoint_path: Ruta al checkpoint
            model: Modelo donde cargar pesos
            optimizer: Optimizador (opcional)
            scheduler: Scheduler (opcional)
            scaler: GradScaler (opcional)
            map_location: Ubicación donde mapear (None = auto)
        
        Returns:
            Diccionario con información del checkpoint
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location=map_location)
            
            # Load model state
            model.load_state_dict(checkpoint["model_state_dict"])
            
            # Load optimizer state
            if optimizer is not None and "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            
            # Load scheduler state
            if scheduler is not None and "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            
            # Load scaler state
            if scaler is not None and "scaler_state_dict" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
            
            result = {
                "epoch": checkpoint.get("epoch", 0),
                "step": checkpoint.get("step", 0),
                "loss": checkpoint.get("loss", 0.0),
                "metric": checkpoint.get("metric"),
                "model_config": checkpoint.get("model_config"),
                "training_config": checkpoint.get("training_config"),
                "metadata": checkpoint.get("metadata", {}),
                "loaded": True,
            }
            
            logger.info(f"Checkpoint loaded: {checkpoint_path}")
            return result
        
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}", exc_info=True)
            return {"loaded": False, "error": str(e)}
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        Listar todos los checkpoints disponibles.
        
        Returns:
            Lista de información de checkpoints
        """
        checkpoints = []
        
        for checkpoint_file in self.checkpoint_dir.glob("*.pt"):
            try:
                checkpoint = torch.load(checkpoint_file, map_location="cpu")
                checkpoints.append({
                    "path": str(checkpoint_file),
                    "name": checkpoint_file.stem,
                    "epoch": checkpoint.get("epoch", 0),
                    "step": checkpoint.get("step", 0),
                    "loss": checkpoint.get("loss", 0.0),
                    "metric": checkpoint.get("metric"),
                    "size_mb": checkpoint_file.stat().st_size / (1024 ** 2),
                })
            except Exception as e:
                logger.warning(f"Error reading checkpoint {checkpoint_file}: {e}")
        
        # Sort by epoch, then step
        checkpoints.sort(key=lambda x: (x["epoch"], x["step"]), reverse=True)
        
        return checkpoints
    
    def get_best_checkpoint(self) -> Optional[str]:
        """
        Obtener ruta al mejor checkpoint.
        
        Returns:
            Ruta al mejor checkpoint o None
        """
        best_path = self.checkpoint_dir / "best.pt"
        if best_path.exists():
            return str(best_path)
        return None
    
    def cleanup_old_checkpoints(
        self,
        keep_last_n: int = 5,
        keep_best: bool = True
    ) -> int:
        """
        Limpiar checkpoints antiguos, manteniendo los últimos N.
        
        Args:
            keep_last_n: Número de checkpoints a mantener
            keep_best: Si mantener el mejor checkpoint
        
        Returns:
            Número de checkpoints eliminados
        """
        checkpoints = self.list_checkpoints()
        
        # Filter out best checkpoint if keeping it
        if keep_best:
            best_path = self.get_best_checkpoint()
            checkpoints = [c for c in checkpoints if c["path"] != best_path]
        
        # Keep only last N
        if len(checkpoints) <= keep_last_n:
            return 0
        
        # Sort and get checkpoints to delete
        checkpoints_to_delete = checkpoints[keep_last_n:]
        
        deleted = 0
        for checkpoint in checkpoints_to_delete:
            try:
                Path(checkpoint["path"]).unlink()
                deleted += 1
                logger.info(f"Deleted old checkpoint: {checkpoint['path']}")
            except Exception as e:
                logger.warning(f"Error deleting checkpoint {checkpoint['path']}: {e}")
        
        logger.info(f"Cleaned up {deleted} old checkpoints")
        return deleted




