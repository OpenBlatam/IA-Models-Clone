"""
Checkpoint Manager - Modular Checkpoint Management
==================================================

Gestión modular de checkpoints de modelos.
"""

import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import torch
import torch.nn as nn
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Gestor de checkpoints modular.
    
    Maneja guardado, carga y gestión de checkpoints
    de modelos de manera organizada.
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        max_checkpoints: int = 5,
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
        
        self.checkpoints: List[Dict[str, Any]] = []
        self.best_checkpoint: Optional[Dict[str, Any]] = None
        self.best_metric: Optional[float] = None
        
        logger.info(f"Checkpoint Manager initialized: {checkpoint_dir}")
    
    def save(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Guardar checkpoint.
        
        Args:
            model: Modelo a guardar
            optimizer: Optimizador (opcional)
            scheduler: Scheduler (opcional)
            epoch: Época actual
            metrics: Métricas (opcional)
            is_best: Si es el mejor checkpoint
            metadata: Metadata adicional (opcional)
            
        Returns:
            Ruta del checkpoint guardado
        """
        checkpoint_name = f"checkpoint_epoch_{epoch}.pt"
        if is_best:
            checkpoint_name = "best_model.pt"
        
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'metrics': metrics or {},
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        if optimizer:
            checkpoint_data['optimizer_state_dict'] = optimizer.state_dict()
        
        if scheduler:
            checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
        
        # Guardar checkpoint
        torch.save(checkpoint_data, checkpoint_path)
        
        # Registrar checkpoint
        checkpoint_info = {
            'path': str(checkpoint_path),
            'epoch': epoch,
            'metrics': metrics or {},
            'is_best': is_best,
            'timestamp': checkpoint_data['timestamp']
        }
        
        self.checkpoints.append(checkpoint_info)
        
        # Actualizar mejor checkpoint
        if is_best or (metrics and self._is_better(metrics)):
            self.best_checkpoint = checkpoint_info
            if metrics:
                # Usar val_loss o la primera métrica disponible
                self.best_metric = metrics.get('val_loss', list(metrics.values())[0])
        
        # Limpiar checkpoints antiguos
        self._cleanup_checkpoints()
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        return str(checkpoint_path)
    
    def load(
        self,
        checkpoint_path: Optional[str] = None,
        load_best: bool = False
    ) -> Dict[str, Any]:
        """
        Cargar checkpoint.
        
        Args:
            checkpoint_path: Ruta del checkpoint (opcional)
            load_best: Cargar el mejor checkpoint
            
        Returns:
            Datos del checkpoint
        """
        if load_best and self.best_checkpoint:
            checkpoint_path = self.best_checkpoint['path']
        elif not checkpoint_path:
            # Cargar el más reciente
            if self.checkpoints:
                checkpoint_path = sorted(
                    self.checkpoints,
                    key=lambda x: x['epoch']
                )[-1]['path']
            else:
                raise ValueError("No checkpoints available")
        
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
        
        return checkpoint_data
    
    def load_model(
        self,
        model: nn.Module,
        checkpoint_path: Optional[str] = None,
        load_best: bool = False,
        strict: bool = True
    ) -> nn.Module:
        """
        Cargar estado del modelo desde checkpoint.
        
        Args:
            model: Modelo a cargar
            checkpoint_path: Ruta del checkpoint (opcional)
            load_best: Cargar el mejor checkpoint
            strict: Carga estricta
            
        Returns:
            Modelo con estado cargado
        """
        checkpoint_data = self.load(checkpoint_path, load_best)
        model.load_state_dict(checkpoint_data['model_state_dict'], strict=strict)
        logger.info("Model state loaded")
        return model
    
    def _is_better(self, metrics: Dict[str, float]) -> bool:
        """Verificar si las métricas son mejores."""
        if self.best_metric is None:
            return True
        
        # Usar val_loss o la primera métrica
        current_metric = metrics.get('val_loss', list(metrics.values())[0])
        
        # Asumir que menor es mejor (val_loss)
        return current_metric < self.best_metric
    
    def _cleanup_checkpoints(self):
        """Limpiar checkpoints antiguos."""
        if len(self.checkpoints) <= self.max_checkpoints:
            return
        
        # Ordenar por época (más antiguos primero)
        sorted_checkpoints = sorted(self.checkpoints, key=lambda x: x['epoch'])
        
        # Mantener el mejor y los más recientes
        to_remove = []
        for checkpoint in sorted_checkpoints[:-self.max_checkpoints]:
            if not checkpoint.get('is_best', False) or not self.keep_best:
                to_remove.append(checkpoint)
        
        # Eliminar checkpoints
        for checkpoint in to_remove:
            checkpoint_path = Path(checkpoint['path'])
            if checkpoint_path.exists():
                checkpoint_path.unlink()
            self.checkpoints.remove(checkpoint)
        
        logger.info(f"Cleaned up {len(to_remove)} old checkpoints")
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """Listar todos los checkpoints."""
        return self.checkpoints.copy()
    
    def get_best_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Obtener información del mejor checkpoint."""
        return self.best_checkpoint
    
    def get_checkpoint_info(self, checkpoint_path: str) -> Dict[str, Any]:
        """Obtener información de un checkpoint."""
        checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
        return {
            'epoch': checkpoint_data.get('epoch', 0),
            'metrics': checkpoint_data.get('metrics', {}),
            'timestamp': checkpoint_data.get('timestamp', ''),
            'metadata': checkpoint_data.get('metadata', {})
        }








