"""
Routing Model Checkpointing
============================

Sistema profesional de checkpointing para modelos de routing.
Implementa guardado/restauración, versionado, y gestión de checkpoints.
"""

import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import json
import shutil
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Checkpointing features will be disabled.")


class CheckpointManager:
    """Gestor profesional de checkpoints."""
    
    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        max_checkpoints: int = 10,
        keep_best: bool = True
    ):
        """
        Inicializar gestor de checkpoints.
        
        Args:
            checkpoint_dir: Directorio para checkpoints
            max_checkpoints: Número máximo de checkpoints a mantener
            keep_best: Mantener siempre el mejor checkpoint
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.keep_best = keep_best
        self.best_metric: Optional[float] = None
        self.best_checkpoint: Optional[str] = None
    
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: Any,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Guardar checkpoint.
        
        Args:
            model: Modelo PyTorch
            optimizer: Optimizador
            epoch: Época actual
            metrics: Métricas del checkpoint
            is_best: Si es el mejor checkpoint
            additional_info: Información adicional
        
        Returns:
            Ruta del checkpoint guardado
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for checkpointing")
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
            'additional_info': additional_info or {}
        }
        
        # Nombre del checkpoint
        if is_best:
            checkpoint_name = "best_model.pt"
        else:
            checkpoint_name = f"checkpoint_epoch_{epoch:04d}.pt"
        
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Guardar checkpoint
        torch.save(checkpoint, checkpoint_path)
        
        # Guardar metadata
        metadata_path = checkpoint_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump({
                'epoch': epoch,
                'metrics': metrics,
                'timestamp': checkpoint['timestamp'],
                'is_best': is_best
            }, f, indent=2)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Actualizar mejor checkpoint
        if is_best:
            val_metric = metrics.get('val_loss', metrics.get('val_accuracy', None))
            if val_metric is not None:
                if self.best_metric is None or val_metric < self.best_metric:
                    self.best_metric = val_metric
                    self.best_checkpoint = str(checkpoint_path)
        
        # Limpiar checkpoints antiguos
        self._cleanup_old_checkpoints()
        
        return str(checkpoint_path)
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: nn.Module,
        optimizer: Optional[Any] = None,
        device: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Cargar checkpoint.
        
        Args:
            checkpoint_path: Ruta del checkpoint
            model: Modelo a cargar
            optimizer: Optimizador a cargar (opcional)
            device: Dispositivo
        
        Returns:
            Información del checkpoint
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for checkpointing")
        
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Cargar estado del modelo
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Cargar estado del optimizador
        if optimizer and checkpoint.get('optimizer_state_dict'):
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
        logger.info(f"Epoch: {checkpoint['epoch']}, Metrics: {checkpoint.get('metrics', {})}")
        
        return checkpoint
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """Listar todos los checkpoints disponibles."""
        checkpoints = []
        
        for checkpoint_file in self.checkpoint_dir.glob("*.pt"):
            metadata_file = checkpoint_file.with_suffix('.json')
            
            info = {
                'path': str(checkpoint_file),
                'name': checkpoint_file.name,
                'size': checkpoint_file.stat().st_size,
                'modified': datetime.fromtimestamp(checkpoint_file.stat().st_mtime).isoformat()
            }
            
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                        info.update(metadata)
                except:
                    pass
            
            checkpoints.append(info)
        
        # Ordenar por época
        checkpoints.sort(key=lambda x: x.get('epoch', 0), reverse=True)
        
        return checkpoints
    
    def get_best_checkpoint(self) -> Optional[str]:
        """Obtener ruta del mejor checkpoint."""
        if self.best_checkpoint and Path(self.best_checkpoint).exists():
            return self.best_checkpoint
        
        # Buscar checkpoint "best"
        best_path = self.checkpoint_dir / "best_model.pt"
        if best_path.exists():
            return str(best_path)
        
        return None
    
    def _cleanup_old_checkpoints(self):
        """Limpiar checkpoints antiguos."""
        checkpoints = [
            f for f in self.checkpoint_dir.glob("checkpoint_epoch_*.pt")
            if f.name != "best_model.pt"
        ]
        
        if len(checkpoints) <= self.max_checkpoints:
            return
        
        # Ordenar por época
        checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
        
        # Eliminar los más antiguos
        to_remove = checkpoints[:-self.max_checkpoints]
        for checkpoint in to_remove:
            checkpoint.unlink()
            # Eliminar metadata también
            metadata = checkpoint.with_suffix('.json')
            if metadata.exists():
                metadata.unlink()
            
            logger.info(f"Removed old checkpoint: {checkpoint}")

