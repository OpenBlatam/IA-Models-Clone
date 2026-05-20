"""
Checkpoint Utilities - Utilidades de checkpointing
===================================================

Funciones avanzadas para manejo de checkpoints.
"""

import logging
import os
import shutil
from typing import Dict, List, Optional, Any
from pathlib import Path
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: int = 0,
    loss: Optional[float] = None,
    metrics: Optional[Dict[str, float]] = None,
    filepath: str = "checkpoint.pt",
    is_best: bool = False,
    best_filepath: Optional[str] = None
) -> bool:
    """
    Guardar checkpoint completo.
    
    Args:
        model: Modelo
        optimizer: Optimizador (opcional)
        scheduler: Scheduler (opcional)
        epoch: Época actual
        loss: Pérdida actual (opcional)
        metrics: Métricas adicionales (opcional)
        filepath: Ruta del checkpoint
        is_best: Si True, también guardar como mejor modelo
        best_filepath: Ruta para mejor modelo (opcional)
    
    Returns:
        True si se guardó correctamente
    """
    try:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "loss": loss,
            "metrics": metrics or {},
        }
        
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        
        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        
        # Guardar checkpoint
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved to {filepath}")
        
        # Guardar como mejor modelo si es necesario
        if is_best:
            best_path = best_filepath or filepath.replace(".pt", "_best.pt")
            shutil.copyfile(filepath, best_path)
            logger.info(f"Best model saved to {best_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error saving checkpoint: {e}")
        return False


def load_checkpoint(
    model: nn.Module,
    filepath: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    map_location: Optional[str] = None,
    strict: bool = True
) -> Dict[str, Any]:
    """
    Cargar checkpoint completo.
    
    Args:
        model: Modelo
        filepath: Ruta del checkpoint
        optimizer: Optimizador (opcional)
        scheduler: Scheduler (opcional)
        map_location: Dispositivo para mapear (opcional)
        strict: Si True, requiere coincidencia exacta de parámetros
    
    Returns:
        Diccionario con información del checkpoint
    """
    try:
        if not os.path.exists(filepath):
            return {"error": f"Checkpoint file not found: {filepath}"}
        
        checkpoint = torch.load(filepath, map_location=map_location)
        
        # Cargar estado del modelo
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        
        # Cargar estado del optimizador
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Cargar estado del scheduler
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        result = {
            "epoch": checkpoint.get("epoch", 0),
            "loss": checkpoint.get("loss"),
            "metrics": checkpoint.get("metrics", {}),
            "loaded": True,
        }
        
        logger.info(f"Checkpoint loaded from {filepath}")
        return result
        
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        return {"loaded": False, "error": str(e)}


def list_checkpoints(checkpoint_dir: str) -> List[Dict[str, Any]]:
    """
    Listar todos los checkpoints en un directorio.
    
    Args:
        checkpoint_dir: Directorio con checkpoints
    
    Returns:
        Lista de información de checkpoints
    """
    checkpoints = []
    checkpoint_path = Path(checkpoint_dir)
    
    if not checkpoint_path.exists():
        return checkpoints
    
    for filepath in checkpoint_path.glob("*.pt"):
        try:
            checkpoint = torch.load(filepath, map_location="cpu")
            checkpoints.append({
                "filepath": str(filepath),
                "epoch": checkpoint.get("epoch", 0),
                "loss": checkpoint.get("loss"),
                "metrics": checkpoint.get("metrics", {}),
                "size_mb": filepath.stat().st_size / (1024 ** 2),
            })
        except Exception as e:
            logger.warning(f"Error reading checkpoint {filepath}: {e}")
    
    # Ordenar por época
    checkpoints.sort(key=lambda x: x["epoch"])
    
    return checkpoints


def cleanup_old_checkpoints(
    checkpoint_dir: str,
    keep_last_n: int = 5,
    keep_best: bool = True
) -> int:
    """
    Limpiar checkpoints antiguos, manteniendo los últimos N y el mejor.
    
    Args:
        checkpoint_dir: Directorio con checkpoints
        keep_last_n: Número de checkpoints recientes a mantener
        keep_best: Si True, mantener el mejor checkpoint
    
    Returns:
        Número de checkpoints eliminados
    """
    checkpoints = list_checkpoints(checkpoint_dir)
    
    if len(checkpoints) <= keep_last_n:
        return 0
    
    # Identificar mejor checkpoint
    best_checkpoint = None
    if keep_best:
        best_checkpoint = min(
            checkpoints,
            key=lambda x: x["loss"] if x["loss"] is not None else float('inf')
        )
    
    # Mantener últimos N y el mejor
    to_keep = checkpoints[-keep_last_n:]
    if best_checkpoint and best_checkpoint not in to_keep:
        to_keep.append(best_checkpoint)
    
    # Eliminar el resto
    to_delete = [c for c in checkpoints if c not in to_keep]
    
    deleted_count = 0
    for checkpoint in to_delete:
        try:
            os.remove(checkpoint["filepath"])
            deleted_count += 1
            logger.info(f"Deleted old checkpoint: {checkpoint['filepath']}")
        except Exception as e:
            logger.warning(f"Error deleting checkpoint {checkpoint['filepath']}: {e}")
    
    return deleted_count




