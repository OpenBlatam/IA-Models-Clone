"""
Fast Trainer
============

Entrenador optimizado para máxima velocidad.
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from typing import Optional, Dict, Any
import logging
from tqdm import tqdm

from .trainer import RouteTrainer, TrainingConfig
from ..routing_optimization import FastDataLoader, MemoryOptimizer

logger = logging.getLogger(__name__)


class FastRouteTrainer(RouteTrainer):
    """
    Entrenador optimizado para velocidad.
    """
    
    def __init__(self, *args, **kwargs):
        """Inicializar entrenador rápido."""
        super(FastRouteTrainer, self).__init__(*args, **kwargs)
        
        # Optimizaciones adicionales
        self._enable_optimizations()
    
    def _enable_optimizations(self):
        """Habilitar optimizaciones."""
        # Optimizaciones de memoria
        MemoryOptimizer.enable_memory_efficient_attention()
        
        # Optimizaciones de cuDNN
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
        
        logger.info("Optimizaciones de velocidad habilitadas")
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Entrenar época (optimizado).
        
        Returns:
            Métricas
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Usar tqdm sin actualización frecuente para mejor rendimiento
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}",
            mininterval=1.0  # Actualizar cada segundo
        )
        
        for batch_features, batch_targets, batch_metadata in progress_bar:
            batch_features = batch_features.to(
                self.model.device,
                non_blocking=True  # Transferencia asíncrona
            )
            batch_targets = batch_targets.to(
                self.model.device,
                non_blocking=True
            )
            
            self.optimizer.zero_grad(set_to_none=True)  # Más rápido
            
            # Forward pass con mixed precision
            if self.scaler:
                with autocast():
                    outputs = self.model(batch_features)
                    loss = self.criterion(outputs, batch_targets)
                
                self.scaler.scale(loss).backward()
                
                if self.config.gradient_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_norm
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(batch_features)
                loss = self.criterion(outputs, batch_targets)
                loss.backward()
                
                if self.config.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_norm
                    )
                
                self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Actualizar progress bar menos frecuentemente
            if num_batches % 10 == 0:
                progress_bar.set_postfix({"loss": loss.item()})
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {
            "train_loss": avg_loss,
            "learning_rate": self.optimizer.param_groups[0]['lr']
        }
    
    def validate(self) -> Dict[str, float]:
        """
        Validar (optimizado).
        
        Returns:
            Métricas
        """
        if not self.val_loader:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            for batch_features, batch_targets, batch_metadata in self.val_loader:
                batch_features = batch_features.to(
                    self.model.device,
                    non_blocking=True
                )
                batch_targets = batch_targets.to(
                    self.model.device,
                    non_blocking=True
                )
                
                if self.scaler:
                    with autocast():
                        outputs = self.model(batch_features)
                        loss = self.criterion(outputs, batch_targets)
                else:
                    outputs = self.model(batch_features)
                    loss = self.criterion(outputs, batch_targets)
                
                total_loss += loss.item()
                num_batches += 1
                
                all_outputs.append(outputs.cpu())
                all_targets.append(batch_targets.cpu())
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # Calcular métricas
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        metrics = self.metrics_calculator.calculate(all_outputs, all_targets)
        metrics["val_loss"] = avg_loss
        
        return metrics

