"""
Trainer - Entrenador de Modelos
=================================

Sistema de entrenamiento para fine-tuning de modelos.
"""

import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import wandb
from typing import Dict, Any, Optional, Callable
import os

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Entrenador de modelos con best practices"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: Optional[str] = None,
        use_wandb: bool = False,
        project_name: str = "community-manager-ai"
    ):
        """
        Inicializar entrenador
        
        Args:
            model: Modelo a entrenar
            train_loader: DataLoader de entrenamiento
            val_loader: DataLoader de validación (opcional)
            device: Dispositivo
            use_wandb: Usar Weights & Biases para tracking
            project_name: Nombre del proyecto en wandb
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(project=project_name)
    
    def train_epoch(
        self,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        use_amp: bool = True
    ) -> Dict[str, float]:
        """
        Entrenar una época
        
        Args:
            optimizer: Optimizador
            criterion: Función de pérdida
            gradient_accumulation_steps: Pasos de acumulación de gradiente
            max_grad_norm: Norma máxima de gradiente (clipping)
            use_amp: Usar mixed precision training
            
        Returns:
            Dict con métricas
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        scaler = torch.cuda.amp.GradScaler() if use_amp and self.device == "cuda" else None
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Mover batch a dispositivo
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass con mixed precision
            with torch.cuda.amp.autocast() if use_amp and self.device == "cuda" else torch.no_grad():
                outputs = self.model(**batch)
                loss = criterion(outputs.logits, batch["labels"])
                loss = loss / gradient_accumulation_steps
            
            # Backward pass
            if use_amp and scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Actualizar pesos
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                if use_amp and scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    optimizer.step()
                
                optimizer.zero_grad()
            
            total_loss += loss.item() * gradient_accumulation_steps
            num_batches += 1
            
            # Actualizar progress bar
            progress_bar.set_postfix({"loss": loss.item() * gradient_accumulation_steps})
        
        avg_loss = total_loss / num_batches
        
        return {"train_loss": avg_loss}
    
    def validate(self, criterion: nn.Module) -> Dict[str, float]:
        """
        Validar modelo
        
        Args:
            criterion: Función de pérdida
            
        Returns:
            Dict con métricas de validación
        """
        if not self.val_loader:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = self.model(**batch)
                loss = criterion(outputs.logits, batch["labels"])
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        return {"val_loss": avg_loss}
    
    def train(
        self,
        num_epochs: int,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
        save_dir: str = "checkpoints",
        save_best: bool = True
    ):
        """
        Entrenar modelo completo
        
        Args:
            num_epochs: Número de épocas
            learning_rate: Learning rate
            weight_decay: Weight decay
            warmup_steps: Pasos de warmup
            save_dir: Directorio para guardar checkpoints
            save_best: Guardar mejor modelo
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Optimizador
        optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Scheduler
        total_steps = len(self.train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Criterion
        criterion = nn.CrossEntropyLoss()
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            logger.info(f"Época {epoch + 1}/{num_epochs}")
            
            # Entrenar
            train_metrics = self.train_epoch(optimizer, criterion)
            
            # Validar
            val_metrics = self.validate(criterion)
            
            # Actualizar learning rate
            scheduler.step()
            
            # Logging
            metrics = {**train_metrics, **val_metrics, "epoch": epoch + 1}
            logger.info(f"Métricas: {metrics}")
            
            if self.use_wandb:
                wandb.log(metrics)
            
            # Guardar checkpoint
            if save_best and val_metrics.get("val_loss", float('inf')) < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                self.save_checkpoint(save_dir, epoch, metrics)
    
    def save_checkpoint(self, save_dir: str, epoch: int, metrics: Dict[str, float]):
        """Guardar checkpoint"""
        checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "metrics": metrics
        }, checkpoint_path)
        logger.info(f"Checkpoint guardado: {checkpoint_path}")




