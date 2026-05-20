"""
Adversarial Utils - Utilidades de Adversarial Training
=======================================================

Utilidades para adversarial training y generación de ejemplos adversarios.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class FGSMAttack:
    """
    Fast Gradient Sign Method (FGSM) attack.
    
    Paper: https://arxiv.org/abs/1412.6572
    """
    
    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.03,
        targeted: bool = False
    ):
        """
        Inicializar ataque FGSM.
        
        Args:
            model: Modelo a atacar
            epsilon: Magnitud del ataque
            targeted: Ataque dirigido
        """
        self.model = model
        self.epsilon = epsilon
        self.targeted = targeted
    
    def generate(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        target_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generar ejemplos adversarios.
        
        Args:
            inputs: Inputs originales
            labels: Labels originales
            target_labels: Labels objetivo (para ataque dirigido)
            
        Returns:
            Inputs adversarios
        """
        inputs.requires_grad_(True)
        
        outputs = self.model(inputs)
        loss = F.cross_entropy(outputs, labels)
        
        if self.targeted and target_labels is not None:
            loss = -loss  # Maximizar pérdida para ataque dirigido
        
        self.model.zero_grad()
        loss.backward()
        
        # Perturbación
        perturbation = self.epsilon * inputs.grad.sign()
        adversarial_inputs = inputs + perturbation
        
        # Clipping para mantener en rango válido
        adversarial_inputs = torch.clamp(adversarial_inputs, 0, 1)
        
        return adversarial_inputs.detach()


class PGDAttack:
    """
    Projected Gradient Descent (PGD) attack.
    
    Paper: https://arxiv.org/abs/1706.06083
    """
    
    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.03,
        alpha: float = 0.01,
        num_iterations: int = 10,
        random_start: bool = True
    ):
        """
        Inicializar ataque PGD.
        
        Args:
            model: Modelo a atacar
            epsilon: Magnitud máxima del ataque
            alpha: Tamaño de paso
            num_iterations: Número de iteraciones
            random_start: Iniciar con perturbación aleatoria
        """
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_iterations = num_iterations
        self.random_start = random_start
    
    def generate(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Generar ejemplos adversarios.
        
        Args:
            inputs: Inputs originales
            labels: Labels originales
            
        Returns:
            Inputs adversarios
        """
        if self.random_start:
            # Iniciar con perturbación aleatoria
            perturbation = torch.randn_like(inputs) * self.epsilon
            adversarial_inputs = inputs + perturbation
            adversarial_inputs = torch.clamp(adversarial_inputs, 0, 1)
        else:
            adversarial_inputs = inputs.clone()
        
        for _ in range(self.num_iterations):
            adversarial_inputs.requires_grad_(True)
            
            outputs = self.model(adversarial_inputs)
            loss = F.cross_entropy(outputs, labels)
            
            self.model.zero_grad()
            loss.backward()
            
            # Actualizar perturbación
            perturbation = self.alpha * adversarial_inputs.grad.sign()
            adversarial_inputs = adversarial_inputs + perturbation
            
            # Proyectar a epsilon-ball
            perturbation = torch.clamp(
                adversarial_inputs - inputs,
                -self.epsilon,
                self.epsilon
            )
            adversarial_inputs = inputs + perturbation
            
            # Clipping
            adversarial_inputs = torch.clamp(adversarial_inputs, 0, 1)
            adversarial_inputs = adversarial_inputs.detach()
        
        return adversarial_inputs


class AdversarialTrainer:
    """
    Trainer con adversarial training.
    """
    
    def __init__(
        self,
        model: nn.Module,
        attack: Callable,
        alpha: float = 0.5
    ):
        """
        Inicializar adversarial trainer.
        
        Args:
            model: Modelo a entrenar
            attack: Función de ataque
            alpha: Peso de loss adversarial
        """
        self.model = model
        self.attack = attack
        self.alpha = alpha
    
    def train_step(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        optimizer: torch.optim.Optimizer
    ) -> Tuple[float, float]:
        """
        Realizar paso de entrenamiento.
        
        Args:
            inputs: Inputs
            labels: Labels
            optimizer: Optimizador
            
        Returns:
            Tupla (loss normal, loss adversarial)
        """
        self.model.train()
        optimizer.zero_grad()
        
        # Loss normal
        normal_outputs = self.model(inputs)
        normal_loss = F.cross_entropy(normal_outputs, labels)
        
        # Generar ejemplos adversarios
        adversarial_inputs = self.attack(inputs, labels)
        
        # Loss adversarial
        adversarial_outputs = self.model(adversarial_inputs)
        adversarial_loss = F.cross_entropy(adversarial_outputs, labels)
        
        # Loss combinado
        total_loss = (1 - self.alpha) * normal_loss + self.alpha * adversarial_loss
        
        total_loss.backward()
        optimizer.step()
        
        return normal_loss.item(), adversarial_loss.item()


class AdversarialRobustness:
    """
    Evaluador de robustez adversarial.
    """
    
    def __init__(
        self,
        model: nn.Module,
        attack: Callable
    ):
        """
        Inicializar evaluador.
        
        Args:
            model: Modelo a evaluar
            attack: Función de ataque
        """
        self.model = model
        self.attack = attack
    
    def evaluate(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, float]:
        """
        Evaluar robustez.
        
        Args:
            inputs: Inputs
            labels: Labels
            
        Returns:
            Métricas de robustez
        """
        self.model.eval()
        
        # Accuracy normal
        with torch.no_grad():
            normal_outputs = self.model(inputs)
            normal_preds = normal_outputs.argmax(dim=1)
            normal_accuracy = (normal_preds == labels).float().mean().item()
        
        # Generar ejemplos adversarios
        adversarial_inputs = self.attack(inputs, labels)
        
        # Accuracy adversarial
        with torch.no_grad():
            adversarial_outputs = self.model(adversarial_inputs)
            adversarial_preds = adversarial_outputs.argmax(dim=1)
            adversarial_accuracy = (adversarial_preds == labels).float().mean().item()
        
        return {
            'normal_accuracy': normal_accuracy,
            'adversarial_accuracy': adversarial_accuracy,
            'robustness_gap': normal_accuracy - adversarial_accuracy
        }




