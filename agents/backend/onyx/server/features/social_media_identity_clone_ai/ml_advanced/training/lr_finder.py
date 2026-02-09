"""
Learning Rate Finder
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, List
import numpy as np
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class LearningRateFinder:
    """Encuentra learning rate óptimo"""
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = "cuda"
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        
        # Guardar estado original
        self.original_state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
    
    def find(
        self,
        train_loader: DataLoader,
        init_lr: float = 1e-8,
        final_lr: float = 10.0,
        num_iter: int = 100,
        smooth_f: float = 0.05
    ) -> Dict[str, Any]:
        """
        Encuentra learning rate óptimo
        
        Args:
            train_loader: DataLoader de entrenamiento
            init_lr: Learning rate inicial
            final_lr: Learning rate final
            num_iter: Número de iteraciones
            smooth_f: Factor de suavizado
            
        Returns:
            Learning rates y losses
        """
        # Restaurar estado original
        self.model.load_state_dict(self.original_state['model'])
        self.optimizer.load_state_dict(self.original_state['optimizer'])
        
        # Configurar learning rates
        lr_mult = (final_lr / init_lr) ** (1 / num_iter)
        lrs = []
        losses = []
        
        # Iterador de datos
        data_iter = iter(train_loader)
        
        # Learning rate inicial
        current_lr = init_lr
        
        self.model.train()
        progress_bar = tqdm(range(num_iter), desc="Finding LR")
        
        for i in progress_bar:
            # Actualizar learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr
            
            lrs.append(current_lr)
            
            # Obtener batch
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            
            # Forward y backward
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            
            self.optimizer.zero_grad()
            outputs = self.model(**inputs)
            
            labels = inputs.get('labels')
            if labels is None:
                # Asumir que el segundo elemento es labels
                labels = list(inputs.values())[1] if len(inputs) > 1 else None
            
            if labels is not None:
                loss = self.criterion(outputs.logits if hasattr(outputs, 'logits') else outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                # Suavizar loss
                if losses:
                    smoothed_loss = smooth_f * loss.item() + (1 - smooth_f) * losses[-1]
                else:
                    smoothed_loss = loss.item()
                
                losses.append(smoothed_loss)
                
                progress_bar.set_postfix({'lr': f'{current_lr:.2e}', 'loss': f'{smoothed_loss:.4f}'})
            
            # Incrementar learning rate
            current_lr *= lr_mult
        
        # Encontrar mejor learning rate (punto de descenso más pronunciado)
        best_lr = self._find_best_lr(lrs, losses)
        
        return {
            "learning_rates": lrs,
            "losses": losses,
            "best_lr": best_lr,
            "suggested_lr": best_lr / 10  # Usar 10x menor que el óptimo
        }
    
    def _find_best_lr(self, lrs: List[float], losses: List[float]) -> float:
        """Encuentra mejor learning rate"""
        if not losses:
            return 1e-3
        
        # Convertir a numpy
        lrs_np = np.array(lrs)
        losses_np = np.array(losses)
        
        # Encontrar punto de descenso más pronunciado
        # Usar gradiente de loss respecto a log(lr)
        log_lrs = np.log10(lrs_np)
        
        # Calcular gradiente
        gradients = np.gradient(losses_np, log_lrs)
        
        # Encontrar mínimo de gradiente (máximo descenso)
        min_grad_idx = np.argmin(gradients)
        
        return float(lrs_np[min_grad_idx])




