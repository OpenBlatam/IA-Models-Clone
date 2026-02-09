"""
Learning Rate Finder - Encuentra el learning rate óptimo
==========================================================
"""

import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LRFinderConfig:
    """Configuración de LR Finder"""
    start_lr: float = 1e-8
    end_lr: float = 10.0
    num_iterations: int = 100
    step_mode: str = "exp"  # "exp" or "linear"
    smooth_f: float = 0.05
    diverge_th: float = 5.0


class LearningRateFinder:
    """Encuentra el learning rate óptimo"""
    
    def __init__(self, config: LRFinderConfig):
        self.config = config
        self.history: Dict[str, List[float]] = {
            "lr": [],
            "loss": []
        }
    
    def find_lr(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer_class=optim.Adam,
        loss_fn: Optional[Callable] = None,
        device: str = "cuda"
    ) -> Tuple[float, Dict[str, List[float]]]:
        """Encuentra el learning rate óptimo"""
        device = torch.device(device)
        model = model.to(device)
        model.train()
        
        # Crear optimizador temporal
        optimizer = optimizer_class(model.parameters(), lr=self.config.start_lr)
        
        # Calcular step size
        if self.config.step_mode == "exp":
            lr_lambda = lambda x: np.exp(
                x * np.log(self.config.end_lr / self.config.start_lr) / self.config.num_iterations
            )
        else:
            lr_lambda = lambda x: self.config.start_lr + (
                self.config.end_lr - self.config.start_lr
            ) * x / self.config.num_iterations
        
        # Iterar sobre datos
        iterator = iter(train_loader)
        best_loss = float('inf')
        best_lr = self.config.start_lr
        
        for iteration in range(self.config.num_iterations):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)
            
            # Actualizar learning rate
            lr = lr_lambda(iteration) * self.config.start_lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            # Forward pass
            optimizer.zero_grad()
            
            if isinstance(batch, dict):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                outputs = model(**batch)
            else:
                batch = batch.to(device)
                outputs = model(batch)
            
            # Calcular pérdida
            if loss_fn:
                if isinstance(batch, dict):
                    loss = loss_fn(outputs, batch)
                else:
                    loss = loss_fn(outputs, batch)
            elif hasattr(outputs, 'loss'):
                loss = outputs.loss
            else:
                logger.warning("No se pudo calcular pérdida")
                continue
            
            # Backward
            loss.backward()
            optimizer.step()
            
            # Guardar historial
            smoothed_loss = self._smooth_loss(loss.item())
            self.history["lr"].append(lr)
            self.history["loss"].append(smoothed_loss)
            
            # Verificar divergencia
            if smoothed_loss > self.config.diverge_th * best_loss:
                logger.warning(f"Pérdida divergente en iteration {iteration}")
                break
            
            # Actualizar mejor
            if smoothed_loss < best_loss:
                best_loss = smoothed_loss
                best_lr = lr
        
        # Encontrar LR sugerido (punto de mayor descenso)
        suggested_lr = self._suggest_lr()
        
        logger.info(f"LR sugerido: {suggested_lr:.2e}")
        return suggested_lr, self.history
    
    def _smooth_loss(self, loss: float) -> float:
        """Suaviza la pérdida"""
        if not self.history["loss"]:
            return loss
        
        return self.config.smooth_f * loss + (1 - self.config.smooth_f) * self.history["loss"][-1]
    
    def _suggest_lr(self) -> float:
        """Sugiere un learning rate óptimo"""
        if not self.history["loss"]:
            return self.config.start_lr
        
        losses = np.array(self.history["loss"])
        lrs = np.array(self.history["lr"])
        
        # Encontrar punto de mayor descenso
        gradients = np.gradient(losses)
        min_grad_idx = np.argmin(gradients)
        
        if min_grad_idx > 0:
            suggested_lr = lrs[min_grad_idx]
        else:
            # Fallback: usar LR donde la pérdida es mínima
            min_loss_idx = np.argmin(losses)
            suggested_lr = lrs[min_loss_idx]
        
        return float(suggested_lr)
    
    def plot(self, save_path: Optional[str] = None):
        """Grafica el historial de LR"""
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 6))
            plt.plot(self.history["lr"], self.history["loss"])
            plt.xscale('log')
            plt.xlabel('Learning Rate')
            plt.ylabel('Loss')
            plt.title('Learning Rate Finder')
            plt.grid(True)
            
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Gráfico guardado en {save_path}")
            else:
                plt.show()
        except ImportError:
            logger.warning("matplotlib no disponible para graficar")

