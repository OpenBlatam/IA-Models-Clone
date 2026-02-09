"""
Routing ML-Specific Optimizations
==================================

Optimizaciones específicas para modelos de Machine Learning.
Incluye: Pruning, Knowledge Distillation, Model Ensembling, etc.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import threading
import time

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.nn.utils.prune as prune
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class ModelPruner:
    """Pruner de modelos para reducir tamaño y acelerar inferencia."""
    
    def __init__(self):
        """Inicializar pruner."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
    
    def prune_model(
        self,
        model: nn.Module,
        pruning_method: str = "l1_unstructured",
        amount: float = 0.2
    ) -> nn.Module:
        """
        Podar modelo para reducir parámetros.
        
        Args:
            model: Modelo PyTorch
            pruning_method: Método de pruning ('l1_unstructured', 'l2_structured', 'random')
            amount: Cantidad a podar (0.0-1.0)
        
        Returns:
            Modelo podado
        """
        model.eval()
        
        # Aplicar pruning a capas lineales y convolucionales
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                if pruning_method == "l1_unstructured":
                    prune.l1_unstructured(module, name='weight', amount=amount)
                elif pruning_method == "l2_structured":
                    prune.ln_structured(module, name='weight', amount=amount, n=2, dim=0)
                elif pruning_method == "random":
                    prune.random_unstructured(module, name='weight', amount=amount)
        
        # Hacer pruning permanente
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                prune.remove(module, 'weight')
        
        logger.info(f"Model pruned with {pruning_method}, amount={amount}")
        return model
    
    def get_sparsity(self, model: nn.Module) -> float:
        """
        Calcular sparsity del modelo.
        
        Args:
            model: Modelo PyTorch
        
        Returns:
            Sparsity (0.0-1.0)
        """
        total_params = 0
        zero_params = 0
        
        for param in model.parameters():
            total_params += param.numel()
            zero_params += (param == 0).sum().item()
        
        return zero_params / total_params if total_params > 0 else 0.0


class KnowledgeDistiller:
    """Distiller de conocimiento para comprimir modelos."""
    
    def __init__(self, temperature: float = 3.0, alpha: float = 0.5):
        """
        Inicializar distiller.
        
        Args:
            temperature: Temperatura para softmax
            alpha: Peso entre loss del estudiante y distiller
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
        
        self.temperature = temperature
        self.alpha = alpha
    
    def distill_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Calcular loss de knowledge distillation.
        
        Args:
            student_logits: Logits del modelo estudiante
            teacher_logits: Logits del modelo profesor
            labels: Labels verdaderos
        
        Returns:
            Loss combinado
        """
        # Softmax con temperatura
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        
        # Distillation loss
        distillation_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean')
        distillation_loss *= (self.temperature ** 2)
        
        # Student loss
        student_loss = F.cross_entropy(student_logits, labels)
        
        # Combined loss
        total_loss = self.alpha * distillation_loss + (1 - self.alpha) * student_loss
        
        return total_loss


class ModelEnsembler:
    """Ensembler de modelos para mejor precisión."""
    
    def __init__(self, ensemble_method: str = "average"):
        """
        Inicializar ensembler.
        
        Args:
            ensemble_method: Método de ensemble ('average', 'weighted', 'voting')
        """
        self.ensemble_method = ensemble_method
        self.models: List[nn.Module] = []
        self.weights: List[float] = []
    
    def add_model(self, model: nn.Module, weight: float = 1.0):
        """
        Agregar modelo al ensemble.
        
        Args:
            model: Modelo a agregar
            weight: Peso del modelo
        """
        self.models.append(model)
        self.weights.append(weight)
        logger.info(f"Model added to ensemble (weight={weight})")
    
    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Predecir con ensemble.
        
        Args:
            inputs: Inputs del modelo
        
        Returns:
            Predicciones del ensemble
        """
        if not self.models:
            raise ValueError("No models in ensemble")
        
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(inputs)
                predictions.append(pred)
        
        if self.ensemble_method == "average":
            # Promedio simple
            return torch.stack(predictions).mean(dim=0)
        elif self.ensemble_method == "weighted":
            # Promedio ponderado
            weights_tensor = torch.tensor(self.weights, device=inputs.device)
            weights_tensor = weights_tensor / weights_tensor.sum()
            weighted_preds = [pred * w for pred, w in zip(predictions, weights_tensor)]
            return torch.stack(weighted_preds).sum(dim=0)
        else:
            # Voting (para clasificación)
            return torch.stack(predictions).mode(dim=0)[0]


class AdaptiveLearningRate:
    """Learning rate adaptativo basado en performance."""
    
    def __init__(self, initial_lr: float = 0.001):
        """
        Inicializar learning rate adaptativo.
        
        Args:
            initial_lr: Learning rate inicial
        """
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.best_loss = float('inf')
        self.patience = 0
        self.patience_limit = 5
        self.factor = 0.5
        self.min_lr = 1e-6
    
    def update(self, loss: float) -> float:
        """
        Actualizar learning rate basado en loss.
        
        Args:
            loss: Loss actual
        
        Returns:
            Nuevo learning rate
        """
        if loss < self.best_loss:
            self.best_loss = loss
            self.patience = 0
        else:
            self.patience += 1
            if self.patience >= self.patience_limit:
                self.current_lr *= self.factor
                self.current_lr = max(self.current_lr, self.min_lr)
                self.patience = 0
                logger.info(f"Learning rate reduced to {self.current_lr}")
        
        return self.current_lr


class BatchNormFreezer:
    """Freezer de BatchNorm para inferencia más rápida."""
    
    def __init__(self):
        """Inicializar freezer."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
    
    def freeze_bn(self, model: nn.Module):
        """
        Congelar BatchNorm layers para inferencia.
        
        Args:
            model: Modelo PyTorch
        """
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.eval()
                # Deshabilitar tracking de running stats
                module.track_running_stats = False
        
        logger.info("BatchNorm layers frozen for inference")
    
    def unfreeze_bn(self, model: nn.Module):
        """
        Descongelar BatchNorm layers para entrenamiento.
        
        Args:
            model: Modelo PyTorch
        """
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.train()
                module.track_running_stats = True
        
        logger.info("BatchNorm layers unfrozen for training")


class MLPerformanceRouter:
    """Router con optimizaciones específicas de ML."""
    
    def __init__(self):
        """Inicializar router ML."""
        self.pruner = ModelPruner() if TORCH_AVAILABLE else None
        self.distiller = KnowledgeDistiller() if TORCH_AVAILABLE else None
        self.ensembler = ModelEnsembler() if TORCH_AVAILABLE else None
        self.lr_adapter = AdaptiveLearningRate() if TORCH_AVAILABLE else None
        self.bn_freezer = BatchNormFreezer() if TORCH_AVAILABLE else None
    
    def optimize_model_for_production(
        self,
        model: nn.Module,
        prune_amount: float = 0.3,
        freeze_bn: bool = True
    ) -> nn.Module:
        """
        Optimizar modelo para producción.
        
        Args:
            model: Modelo PyTorch
            prune_amount: Cantidad de pruning
            freeze_bn: Congelar BatchNorm
        
        Returns:
            Modelo optimizado
        """
        optimized = model
        
        # 1. Pruning
        if self.pruner and prune_amount > 0:
            try:
                optimized = self.pruner.prune_model(optimized, amount=prune_amount)
                sparsity = self.pruner.get_sparsity(optimized)
                logger.info(f"Model pruned, sparsity: {sparsity:.2%}")
            except Exception as e:
                logger.warning(f"Pruning failed: {e}")
        
        # 2. Freeze BatchNorm
        if freeze_bn and self.bn_freezer:
            try:
                self.bn_freezer.freeze_bn(optimized)
            except Exception as e:
                logger.warning(f"BatchNorm freezing failed: {e}")
        
        return optimized
    
    def create_ensemble(
        self,
        models: List[nn.Module],
        weights: Optional[List[float]] = None,
        method: str = "weighted"
    ) -> ModelEnsembler:
        """
        Crear ensemble de modelos.
        
        Args:
            models: Lista de modelos
            weights: Pesos de los modelos
            method: Método de ensemble
        
        Returns:
            ModelEnsembler configurado
        """
        if not self.ensembler:
            raise RuntimeError("Ensembler not available")
        
        self.ensembler.ensemble_method = method
        
        if weights is None:
            weights = [1.0] * len(models)
        
        for model, weight in zip(models, weights):
            self.ensembler.add_model(model, weight)
        
        return self.ensembler

