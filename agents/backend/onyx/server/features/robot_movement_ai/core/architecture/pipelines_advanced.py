"""
Advanced Deep Learning Features Module
=======================================

Funcionalidades avanzadas de deep learning.
Incluye reinforcement learning, meta-learning, y técnicas de vanguardia.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None

logger = logging.getLogger(__name__)


class ReinforcementLearningWrapper:
    """
    Wrapper para Reinforcement Learning con PyTorch.
    
    Soporta algoritmos básicos de RL para control de robots.
    """
    
    def __init__(
        self,
        policy_network: nn.Module,
        value_network: Optional[nn.Module] = None,
        learning_rate: float = 3e-4,
        gamma: float = 0.99
    ):
        """
        Inicializar wrapper de RL.
        
        Args:
            policy_network: Red de política
            value_network: Red de valor (opcional)
            learning_rate: Learning rate
            gamma: Factor de descuento
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        
        self.policy_network = policy_network
        self.value_network = value_network
        self.gamma = gamma
        
        self.policy_optimizer = torch.optim.Adam(
            policy_network.parameters(),
            lr=learning_rate
        )
        
        if value_network:
            self.value_optimizer = torch.optim.Adam(
                value_network.parameters(),
                lr=learning_rate
            )
        
        logger.info("ReinforcementLearningWrapper initialized")
    
    def compute_returns(
        self,
        rewards: List[float],
        dones: Optional[List[bool]] = None
    ) -> torch.Tensor:
        """
        Calcular returns (recompensas descontadas).
        
        Args:
            rewards: Lista de recompensas
            dones: Lista de flags de terminación
            
        Returns:
            Returns descontados
        """
        returns = []
        G = 0
        
        for i in reversed(range(len(rewards))):
            if dones and dones[i]:
                G = 0
            G = rewards[i] + self.gamma * G
            returns.insert(0, G)
        
        return torch.FloatTensor(returns)
    
    def policy_gradient_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns: torch.Tensor,
        advantages: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Paso de policy gradient.
        
        Args:
            states: Estados
            actions: Acciones tomadas
            returns: Returns calculados
            advantages: Ventajas (opcional)
            
        Returns:
            Dict con losses
        """
        self.policy_network.train()
        
        # Calcular log probabilities
        action_probs = self.policy_network(states)
        if action_probs.ndim > 1:
            log_probs = F.log_softmax(action_probs, dim=1)
            selected_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        else:
            selected_log_probs = F.log_softmax(action_probs, dim=0)[actions]
        
        # Usar advantages si están disponibles, sino usar returns
        weights = advantages if advantages is not None else returns
        
        # Policy loss (negativo porque queremos maximizar)
        policy_loss = -(selected_log_probs * weights).mean()
        
        # Entropy bonus (para exploración)
        entropy = -(action_probs * F.log_softmax(action_probs, dim=-1)).sum(dim=-1).mean()
        entropy_bonus = 0.01 * entropy
        
        total_loss = policy_loss - entropy_bonus
        
        # Backward pass
        self.policy_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 0.5)
        self.policy_optimizer.step()
        
        return {
            "policy_loss": policy_loss.item(),
            "entropy": entropy.item(),
            "total_loss": total_loss.item()
        }


class MetaLearningWrapper:
    """
    Wrapper para Meta-Learning (MAML-like).
    
    Permite adaptación rápida a nuevas tareas.
    """
    
    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        outer_lr: float = 1e-3,
        num_inner_steps: int = 1
    ):
        """
        Inicializar meta-learning.
        
        Args:
            model: Modelo base
            inner_lr: Learning rate interno (para adaptación rápida)
            outer_lr: Learning rate externo (para meta-optimización)
            num_inner_steps: Número de pasos internos
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        
        self.model = model
        self.inner_lr = inner_lr
        self.num_inner_steps = num_inner_steps
        
        self.meta_optimizer = torch.optim.Adam(
            model.parameters(),
            lr=outer_lr
        )
        
        logger.info("MetaLearningWrapper initialized")
    
    def adapt(
        self,
        support_data: torch.Tensor,
        support_labels: torch.Tensor,
        loss_fn: Callable = nn.CrossEntropyLoss()
    ) -> nn.Module:
        """
        Adaptar modelo a nueva tarea (inner loop).
        
        Args:
            support_data: Datos de soporte
            support_labels: Labels de soporte
            loss_fn: Función de pérdida
            
        Returns:
            Modelo adaptado
        """
        # Crear copia del modelo para adaptación
        adapted_model = type(self.model)(**self._get_model_config())
        adapted_model.load_state_dict(self.model.state_dict())
        
        # Inner loop: adaptación rápida
        inner_optimizer = torch.optim.SGD(
            adapted_model.parameters(),
            lr=self.inner_lr
        )
        
        for _ in range(self.num_inner_steps):
            inner_optimizer.zero_grad()
            predictions = adapted_model(support_data)
            loss = loss_fn(predictions, support_labels)
            loss.backward()
            inner_optimizer.step()
        
        return adapted_model
    
    def meta_update(
        self,
        query_data: torch.Tensor,
        query_labels: torch.Tensor,
        adapted_model: nn.Module,
        loss_fn: Callable = nn.CrossEntropyLoss()
    ) -> float:
        """
        Meta-update (outer loop).
        
        Args:
            query_data: Datos de query
            query_labels: Labels de query
            adapted_model: Modelo adaptado
            loss_fn: Función de pérdida
            
        Returns:
            Meta-loss
        """
        # Evaluar modelo adaptado en query set
        predictions = adapted_model(query_data)
        meta_loss = loss_fn(predictions, query_labels)
        
        # Meta-update
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()
        
        return meta_loss.item()
    
    def _get_model_config(self) -> Dict[str, Any]:
        """Obtener configuración del modelo."""
        # Simplificado - en producción usaría inspect o configuración explícita
        return {}


class ContinualLearning:
    """
    Sistema de Continual Learning.
    
    Permite aprender nuevas tareas sin olvidar las anteriores.
    """
    
    def __init__(
        self,
        model: nn.Module,
        method: str = "ewc"
    ):
        """
        Inicializar continual learning.
        
        Args:
            model: Modelo base
            method: Método ("ewc", "replay", "packnet")
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        
        self.model = model
        self.method = method
        self.fisher_information: Dict[str, torch.Tensor] = {}
        self.optimal_params: Dict[str, torch.Tensor] = {}
        logger.info(f"ContinualLearning initialized with method: {method}")
    
    def compute_fisher_information(
        self,
        data_loader,
        num_samples: int = 100
    ):
        """
        Calcular información de Fisher (para EWC).
        
        Args:
            data_loader: DataLoader con datos de tarea anterior
            num_samples: Número de muestras para estimar Fisher
        """
        self.model.eval()
        self.fisher_information = {}
        
        for name, param in self.model.named_parameters():
            self.fisher_information[name] = torch.zeros_like(param)
        
        sample_count = 0
        for batch in data_loader:
            if sample_count >= num_samples:
                break
            
            inputs = batch["input"]
            self.model.zero_grad()
            
            outputs = self.model(inputs)
            # Asumir que outputs son logits para clasificación
            if outputs.ndim > 1:
                sample_idx = torch.randint(0, outputs.size(1), (1,)).item()
                loss = outputs[:, sample_idx].mean()
            else:
                loss = outputs.mean()
            
            loss.backward()
            
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    self.fisher_information[name] += param.grad.data ** 2
            
            sample_count += len(inputs)
        
        # Normalizar
        for name in self.fisher_information:
            self.fisher_information[name] /= sample_count
        
        logger.info("Fisher information computed")
    
    def ewc_loss(
        self,
        current_loss: torch.Tensor,
        lambda_ewc: float = 0.4
    ) -> torch.Tensor:
        """
        Calcular EWC loss (Elastic Weight Consolidation).
        
        Args:
            current_loss: Loss de la tarea actual
            lambda_ewc: Peso del término EWC
            
        Returns:
            Loss total con penalización EWC
        """
        if not self.fisher_information or not self.optimal_params:
            return current_loss
        
        ewc_penalty = 0.0
        
        for name, param in self.model.named_parameters():
            if name in self.fisher_information:
                optimal_param = self.optimal_params[name]
                fisher = self.fisher_information[name]
                ewc_penalty += (fisher * (param - optimal_param) ** 2).sum()
        
        total_loss = current_loss + lambda_ewc * ewc_penalty
        return total_loss
    
    def save_task_params(self):
        """Guardar parámetros óptimos de la tarea actual."""
        self.optimal_params = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
        }
        logger.info("Task parameters saved")


class AdvancedRegularization:
    """
    Técnicas avanzadas de regularización.
    
    Incluye dropout avanzado, batch normalization adaptativo, etc.
    """
    
    @staticmethod
    def apply_dropout_schedule(
        model: nn.Module,
        current_epoch: int,
        max_epochs: int,
        initial_dropout: float = 0.5,
        final_dropout: float = 0.1
    ):
        """
        Aplicar schedule de dropout (empezar alto, reducir gradualmente).
        
        Args:
            model: Modelo
            current_epoch: Época actual
            max_epochs: Número total de épocas
            initial_dropout: Dropout inicial
            final_dropout: Dropout final
        """
        progress = current_epoch / max_epochs
        current_dropout = initial_dropout * (1 - progress) + final_dropout * progress
        
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = current_dropout
        
        logger.debug(f"Dropout updated to {current_dropout:.3f}")
    
    @staticmethod
    def apply_label_smoothing(
        targets: torch.Tensor,
        num_classes: int,
        smoothing: float = 0.1
    ) -> torch.Tensor:
        """
        Aplicar label smoothing.
        
        Args:
            targets: Targets one-hot o índices
            num_classes: Número de clases
            smoothing: Factor de smoothing
            
        Returns:
            Targets con smoothing aplicado
        """
        if targets.ndim == 1:
            # Convertir índices a one-hot
            targets_one_hot = F.one_hot(targets, num_classes).float()
        else:
            targets_one_hot = targets.float()
        
        # Aplicar smoothing
        smoothed = targets_one_hot * (1 - smoothing) + smoothing / num_classes
        return smoothed
    
    @staticmethod
    def mixup_data(
        x: torch.Tensor,
        y: torch.Tensor,
        alpha: float = 0.2
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Aplicar Mixup augmentation.
        
        Args:
            x: Inputs
            y: Targets
            alpha: Parámetro Beta
            
        Returns:
            Mixed inputs, mixed targets, lambda
        """
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam

