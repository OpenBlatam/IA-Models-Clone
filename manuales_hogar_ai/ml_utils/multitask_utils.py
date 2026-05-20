"""
Multi-Task Learning Utils - Utilidades de Aprendizaje Multi-Tarea
=================================================================

Utilidades para aprendizaje multi-tarea.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Callable, Tuple
from collections import defaultdict
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class MultiTaskHead(nn.Module):
    """
    Head para una tarea específica en multi-task learning.
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: Optional[List[int]] = None,
        dropout: float = 0.1
    ):
        """
        Inicializar task head.
        
        Args:
            input_size: Tamaño de entrada
            output_size: Tamaño de salida
            hidden_sizes: Tamaños de capas ocultas
            dropout: Dropout rate
        """
        super().__init__()
        
        if hidden_sizes is None:
            hidden_sizes = []
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        self.head = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        return self.head(x)


class MultiTaskModel(nn.Module):
    """
    Modelo multi-tarea con backbone compartido.
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        task_configs: Dict[str, Dict],
        shared_output_size: int
    ):
        """
        Inicializar modelo multi-tarea.
        
        Args:
            backbone: Backbone compartido
            task_configs: Configuración de tareas {task_name: {output_size, hidden_sizes, ...}}
            shared_output_size: Tamaño de salida del backbone
        """
        super().__init__()
        self.backbone = backbone
        self.task_heads = nn.ModuleDict()
        
        for task_name, config in task_configs.items():
            self.task_heads[task_name] = MultiTaskHead(
                input_size=shared_output_size,
                output_size=config.get('output_size', 1),
                hidden_sizes=config.get('hidden_sizes', None),
                dropout=config.get('dropout', 0.1)
            )
    
    def forward(
        self,
        x: torch.Tensor,
        task: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor
            task: Tarea específica (opcional, si None retorna todas)
            
        Returns:
            Diccionario con outputs por tarea
        """
        shared_features = self.backbone(x)
        
        if task is not None:
            return {task: self.task_heads[task](shared_features)}
        
        outputs = {}
        for task_name, head in self.task_heads.items():
            outputs[task_name] = head(shared_features)
        
        return outputs


class MultiTaskLoss(nn.Module):
    """
    Loss combinado para multi-task learning.
    """
    
    def __init__(
        self,
        task_losses: Dict[str, nn.Module],
        task_weights: Optional[Dict[str, float]] = None,
        uncertainty_weighting: bool = False
    ):
        """
        Inicializar multi-task loss.
        
        Args:
            task_losses: Diccionario de losses por tarea
            task_weights: Pesos por tarea (opcional)
            uncertainty_weighting: Usar weighting basado en incertidumbre
        """
        super().__init__()
        self.task_losses = nn.ModuleDict(task_losses)
        self.task_weights = task_weights or {task: 1.0 for task in task_losses.keys()}
        self.uncertainty_weighting = uncertainty_weighting
        
        if uncertainty_weighting:
            # Log variances para weighting adaptativo
            self.log_vars = nn.ParameterDict({
                task: nn.Parameter(torch.zeros(1))
                for task in task_losses.keys()
            })
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calcular multi-task loss.
        
        Args:
            predictions: Predicciones por tarea
            targets: Targets por tarea
            
        Returns:
            Tupla (total_loss, losses_por_tarea)
        """
        task_losses = {}
        total_loss = 0.0
        
        for task_name in predictions.keys():
            loss_fn = self.task_losses[task_name]
            loss = loss_fn(predictions[task_name], targets[task_name])
            
            if self.uncertainty_weighting:
                # Weighting basado en incertidumbre
                precision = torch.exp(-self.log_vars[task_name])
                weighted_loss = precision * loss + self.log_vars[task_name]
                task_losses[task_name] = weighted_loss
                total_loss += weighted_loss
            else:
                # Weighting fijo
                weighted_loss = self.task_weights[task_name] * loss
                task_losses[task_name] = loss
                total_loss += weighted_loss
        
        return total_loss, task_losses


class MultiTaskTrainer:
    """
    Trainer para modelos multi-tarea.
    """
    
    def __init__(
        self,
        model: MultiTaskModel,
        loss_fn: MultiTaskLoss,
        device: str = "cuda"
    ):
        """
        Inicializar trainer.
        
        Args:
            model: Modelo multi-tarea
            loss_fn: Función de pérdida multi-tarea
            device: Dispositivo
        """
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.device = device
    
    def train_step(
        self,
        inputs: torch.Tensor,
        targets: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """
        Realizar paso de entrenamiento.
        
        Args:
            inputs: Inputs
            targets: Targets por tarea
            optimizer: Optimizador
            
        Returns:
            Métricas
        """
        self.model.train()
        optimizer.zero_grad()
        
        inputs = inputs.to(self.device)
        targets = {k: v.to(self.device) for k, v in targets.items()}
        
        # Forward pass
        predictions = self.model(inputs)
        
        # Calcular loss
        total_loss, task_losses = self.loss_fn(predictions, targets)
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # Métricas
        metrics = {'total_loss': total_loss.item()}
        for task_name, loss in task_losses.items():
            metrics[f'{task_name}_loss'] = loss.item()
        
        return metrics
    
    def train(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        epochs: int = 10,
        val_loader: Optional[DataLoader] = None
    ) -> Dict[str, list]:
        """
        Entrenar modelo multi-tarea.
        
        Args:
            train_loader: DataLoader de entrenamiento
            optimizer: Optimizador
            epochs: Número de épocas
            val_loader: DataLoader de validación (opcional)
            
        Returns:
            Historial
        """
        history = {'train_loss': []}
        
        for epoch in range(epochs):
            # Entrenamiento
            epoch_metrics = defaultdict(float)
            num_batches = 0
            
            for batch in train_loader:
                if isinstance(batch, dict):
                    inputs = batch['inputs']
                    targets = {k: v for k, v in batch.items() if k != 'inputs'}
                else:
                    inputs, targets = batch[0], batch[1]
                
                metrics = self.train_step(inputs, targets, optimizer)
                
                for key, value in metrics.items():
                    epoch_metrics[key] += value
                num_batches += 1
            
            # Promediar métricas
            avg_metrics = {k: v / num_batches for k, v in epoch_metrics.items()}
            history['train_loss'].append(avg_metrics.get('total_loss', 0))
            
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            for key, value in avg_metrics.items():
                logger.info(f"  {key}: {value:.4f}")
            
            # Validación
            if val_loader:
                val_metrics = self.evaluate(val_loader)
                logger.info("Validation:")
                for key, value in val_metrics.items():
                    logger.info(f"  {key}: {value:.4f}")
        
        return history
    
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluar modelo.
        
        Args:
            val_loader: DataLoader de validación
            
        Returns:
            Métricas
        """
        self.model.eval()
        metrics = defaultdict(float)
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, dict):
                    inputs = batch['inputs']
                    targets = {k: v for k, v in batch.items() if k != 'inputs'}
                else:
                    inputs, targets = batch[0], batch[1]
                
                inputs = inputs.to(self.device)
                targets = {k: v.to(self.device) for k, v in targets.items()}
                
                predictions = self.model(inputs)
                total_loss, task_losses = self.loss_fn(predictions, targets)
                
                metrics['total_loss'] += total_loss.item()
                for task_name, loss in task_losses.items():
                    metrics[f'{task_name}_loss'] += loss.item()
                
                num_batches += 1
        
        return {k: v / num_batches for k, v in metrics.items()}

