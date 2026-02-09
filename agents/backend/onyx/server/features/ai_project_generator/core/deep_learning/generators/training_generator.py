"""
Training Generator - Generador de código de entrenamiento (optimizado)
======================================================================

Genera código de entrenamiento siguiendo mejores prácticas de PyTorch.
Incluye mixed precision, gradient accumulation, early stopping, etc.
"""

import logging
from pathlib import Path
from typing import Dict, Any

from .base_generator import BaseGenerator
from ..utils.code_templates import (
    get_training_loop_template,
    get_dataloader_template,
    TemplateType
)

logger = logging.getLogger(__name__)


class TrainingGenerator(BaseGenerator):
    """
    Generador de código de entrenamiento (optimizado).
    
    Genera código para entrenamiento siguiendo mejores prácticas de Deep Learning.
    """
    
    def __init__(self):
        """Inicializar generador de entrenamiento"""
        super().__init__(
            name="training",
            description="Generates training code following Deep Learning best practices"
        )
    
    def should_generate(self, keywords: Dict[str, Any]) -> bool:
        """Determinar si se debe generar código de entrenamiento"""
        return keywords.get("requires_training", True)
    
    def generate(
        self,
        target_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any]
    ) -> None:
        """
        Generar código de entrenamiento (optimizado).
        
        Args:
            target_dir: Directorio destino
            keywords: Keywords extraídos
            project_info: Información del proyecto
        """
        # Generar training loop
        training_file = target_dir / "training_loop.py"
        training_code = get_training_loop_template(
            mixed_precision=project_info.get("requires_mixed_precision", True),
            gradient_accumulation=project_info.get("requires_gradient_accumulation", True),
            multi_gpu=project_info.get("requires_multi_gpu", False)
        )
        training_file.write_text(training_code)
        self.logger.info(f"Generated training loop at {training_file}")
        
        # Generar DataLoader
        dataloader_file = target_dir / "dataloader.py"
        dataloader_code = get_dataloader_template(
            pin_memory=True,
            num_workers=4,
            prefetch_factor=2
        )
        dataloader_file.write_text(dataloader_code)
        self.logger.info(f"Generated DataLoader at {dataloader_file}")
        
        # Generar trainer principal
        trainer_file = target_dir / "trainer.py"
        trainer_code = self._generate_trainer_code(project_info)
        trainer_file.write_text(trainer_code)
        self.logger.info(f"Generated trainer at {trainer_file}")
    
    def _generate_trainer_code(self, project_info: Dict[str, Any]) -> str:
        """
        Generar código de trainer principal (optimizado).
        
        Args:
            project_info: Información del proyecto
            
        Returns:
            Código del trainer
        """
        return '''"""
Trainer - Entrenador principal (optimizado)
===========================================

Entrenador que orquesta el proceso de entrenamiento completo.
Incluye validación, checkpointing, y experiment tracking.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
from pathlib import Path
import logging

from .training_loop import TrainingLoop

logger = logging.getLogger(__name__)


class Trainer:
    """
    Entrenador principal (optimizado).
    
    Orquesta el proceso completo de entrenamiento con todas las mejores prácticas.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        criterion: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Inicializar trainer (optimizado).
        
        Args:
            model: Modelo PyTorch
            train_loader: DataLoader de entrenamiento
            val_loader: DataLoader de validación (opcional)
            optimizer: Optimizador (opcional, se crea si no se proporciona)
            criterion: Función de pérdida (opcional)
            device: Dispositivo (opcional, se detecta automáticamente)
            config: Configuración (opcional)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config or {}
        
        # Crear optimizer si no se proporciona
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.config.get("learning_rate", 1e-4),
                weight_decay=self.config.get("weight_decay", 0.01)
            )
        else:
            self.optimizer = optimizer
        
        # Crear criterion si no se proporciona
        if criterion is None:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = criterion
        
        # Crear training loop
        self.training_loop = TrainingLoop(
            model=model,
            optimizer=self.optimizer,
            criterion=self.criterion,
            device=self.device,
            mixed_precision=self.config.get("mixed_precision", True),
            gradient_accumulation_steps=self.config.get("gradient_accumulation_steps", 1),
            max_grad_norm=self.config.get("max_grad_norm", 1.0),
            multi_gpu=self.config.get("multi_gpu", False)
        )
        
        # Estado de entrenamiento
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.training_history = []
    
    def train(self, num_epochs: int) -> Dict[str, Any]:
        """
        Entrenar modelo (optimizado).
        
        Args:
            num_epochs: Número de épocas
            
        Returns:
            Historial de entrenamiento
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Training
            train_metrics = self._train_epoch()
            
            # Validation
            val_metrics = {}
            if self.val_loader:
                val_metrics = self.training_loop.validate(self.val_loader)
                
                # Checkpointing
                val_loss = val_metrics.get("val_loss", float('inf'))
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint(epoch, is_best=True)
            
            # Logging
            metrics = {**train_metrics, **val_metrics, "epoch": epoch}
            self.training_history.append(metrics)
            
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs}: "
                f"Train Loss: {train_metrics.get('loss', 0):.4f}, "
                f"Val Loss: {val_metrics.get('val_loss', 0):.4f}"
            )
        
        logger.info("Training completed")
        return {"history": self.training_history}
    
    def _train_epoch(self) -> Dict[str, float]:
        """Entrenar una época"""
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            metrics = self.training_loop.train_step(batch, self.global_step)
            total_loss += metrics["loss"]
            num_batches += 1
            self.global_step += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {"loss": avg_loss}
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Guardar checkpoint"""
        checkpoint_dir = Path(self.config.get("checkpoint_dir", "./checkpoints"))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
        }
        
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model at epoch {epoch}")
'''
