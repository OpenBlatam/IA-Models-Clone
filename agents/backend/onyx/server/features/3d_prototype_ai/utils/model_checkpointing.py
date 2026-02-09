"""
Model Checkpointing System - Sistema de checkpointing y versionado de modelos
============================================================================
Gestión completa de checkpoints, versionado y restauración de modelos
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
import json
import shutil

logger = logging.getLogger(__name__)


class ModelCheckpointer:
    """Sistema de checkpointing de modelos"""
    
    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints: Dict[str, Dict[str, Any]] = {}
    
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        checkpoint_name: Optional[str] = None,
        is_best: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Guarda checkpoint del modelo"""
        if checkpoint_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_name = f"checkpoint_epoch_{epoch}_{timestamp}"
        
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.pt"
        
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "model_config": self._get_model_config(model),
            "timestamp": datetime.now().isoformat()
        }
        
        if optimizer:
            checkpoint_data["optimizer_state_dict"] = optimizer.state_dict()
        
        if scheduler:
            checkpoint_data["scheduler_state_dict"] = scheduler.state_dict()
        
        if metrics:
            checkpoint_data["metrics"] = metrics
        
        if metadata:
            checkpoint_data["metadata"] = metadata
        
        # Guardar checkpoint
        torch.save(checkpoint_data, checkpoint_path)
        
        # Guardar metadata JSON
        metadata_path = self.checkpoint_dir / f"{checkpoint_name}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump({
                "epoch": epoch,
                "metrics": metrics or {},
                "timestamp": checkpoint_data["timestamp"],
                "metadata": metadata or {}
            }, f, indent=2)
        
        # Si es el mejor, crear symlink
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            if best_path.exists():
                best_path.unlink()
            shutil.copy(checkpoint_path, best_path)
            logger.info(f"Saved best model checkpoint: {checkpoint_name}")
        
        self.checkpoints[checkpoint_name] = {
            "path": str(checkpoint_path),
            "epoch": epoch,
            "metrics": metrics or {},
            "is_best": is_best
        }
        
        logger.info(f"Saved checkpoint: {checkpoint_name} at epoch {epoch}")
        return str(checkpoint_path)
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        map_location: Optional[str] = None
    ) -> Dict[str, Any]:
        """Carga checkpoint"""
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        if map_location is None:
            map_location = "cuda" if torch.cuda.is_available() else "cpu"
        
        checkpoint_data = torch.load(checkpoint_path, map_location=map_location)
        
        if model:
            model.load_state_dict(checkpoint_data["model_state_dict"])
            logger.info("Model state loaded")
        
        if optimizer and "optimizer_state_dict" in checkpoint_data:
            optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
            logger.info("Optimizer state loaded")
        
        if scheduler and "scheduler_state_dict" in checkpoint_data:
            scheduler.load_state_dict(checkpoint_data["scheduler_state_dict"])
            logger.info("Scheduler state loaded")
        
        logger.info(f"Loaded checkpoint from epoch {checkpoint_data.get('epoch', 'unknown')}")
        return checkpoint_data
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """Lista todos los checkpoints"""
        checkpoints = []
        
        for checkpoint_file in self.checkpoint_dir.glob("*.pt"):
            if checkpoint_file.name == "best_model.pt":
                continue
            
            try:
                checkpoint_data = torch.load(checkpoint_file, map_location="cpu")
                checkpoints.append({
                    "name": checkpoint_file.stem,
                    "path": str(checkpoint_file),
                    "epoch": checkpoint_data.get("epoch", 0),
                    "metrics": checkpoint_data.get("metrics", {}),
                    "timestamp": checkpoint_data.get("timestamp", "")
                })
            except Exception as e:
                logger.warning(f"Failed to load checkpoint {checkpoint_file}: {e}")
        
        return sorted(checkpoints, key=lambda x: x["epoch"], reverse=True)
    
    def delete_checkpoint(self, checkpoint_name: str):
        """Elimina un checkpoint"""
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.pt"
        metadata_path = self.checkpoint_dir / f"{checkpoint_name}_metadata.json"
        
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        
        if metadata_path.exists():
            metadata_path.unlink()
        
        if checkpoint_name in self.checkpoints:
            del self.checkpoints[checkpoint_name]
        
        logger.info(f"Deleted checkpoint: {checkpoint_name}")
    
    def _get_model_config(self, model: nn.Module) -> Dict[str, Any]:
        """Extrae configuración del modelo"""
        config = {
            "model_type": type(model).__name__,
            "num_parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)
        }
        
        # Intentar extraer config si el modelo tiene atributo config
        if hasattr(model, "config"):
            config["model_config"] = str(model.config)
        
        return config




