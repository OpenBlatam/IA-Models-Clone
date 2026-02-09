"""
Adversarial Training para robustez
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class FGSMAttack:
    """Fast Gradient Sign Method Attack"""
    
    def __init__(self, model: nn.Module, epsilon: float = 0.03):
        self.model = model
        self.epsilon = epsilon
    
    def generate(
        self,
        inputs: Dict[str, torch.Tensor],
        labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Genera ejemplos adversariales
        
        Args:
            inputs: Inputs originales
            labels: Labels verdaderos
            epsilon: Magnitud de perturbación
            
        Returns:
            Inputs adversariales
        """
        inputs_adv = {}
        
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor) and value.requires_grad:
                value.requires_grad_(True)
        
        # Forward pass
        outputs = self.model(**inputs)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        loss = F.cross_entropy(logits, labels)
        
        # Backward
        self.model.zero_grad()
        loss.backward()
        
        # Generar perturbación
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor) and value.grad is not None:
                # FGSM: signo del gradiente
                perturbation = self.epsilon * torch.sign(value.grad)
                inputs_adv[key] = value + perturbation
            else:
                inputs_adv[key] = value
        
        return inputs_adv


class PGDAttack:
    """Projected Gradient Descent Attack"""
    
    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.03,
        alpha: float = 0.01,
        num_iter: int = 7
    ):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_iter = num_iter
    
    def generate(
        self,
        inputs: Dict[str, torch.Tensor],
        labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Genera ejemplos adversariales con PGD
        
        Args:
            inputs: Inputs originales
            labels: Labels verdaderos
            
        Returns:
            Inputs adversariales
        """
        inputs_adv = {}
        
        # Inicializar con ruido aleatorio
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                noise = torch.randn_like(value) * self.epsilon
                inputs_adv[key] = value + noise
            else:
                inputs_adv[key] = value
        
        # Iteraciones PGD
        for _ in range(self.num_iter):
            for key, value in inputs_adv.items():
                if isinstance(value, torch.Tensor):
                    value.requires_grad_(True)
            
            # Forward
            outputs = self.model(**inputs_adv)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            loss = F.cross_entropy(logits, labels)
            
            # Backward
            self.model.zero_grad()
            loss.backward()
            
            # Actualizar con gradiente
            for key, value in inputs_adv.items():
                if isinstance(value, torch.Tensor) and value.grad is not None:
                    # Paso de gradiente
                    perturbation = self.alpha * torch.sign(value.grad)
                    inputs_adv[key] = value + perturbation
                    
                    # Proyectar a epsilon-ball
                    delta = inputs_adv[key] - inputs[key]
                    delta = torch.clamp(delta, -self.epsilon, self.epsilon)
                    inputs_adv[key] = inputs[key] + delta
        
        return inputs_adv


class AdversarialTrainer:
    """Trainer con adversarial training"""
    
    def __init__(
        self,
        model: nn.Module,
        attack_type: str = "fgsm",  # "fgsm" or "pgd"
        epsilon: float = 0.03
    ):
        self.model = model
        
        if attack_type == "fgsm":
            self.attack = FGSMAttack(model, epsilon)
        elif attack_type == "pgd":
            self.attack = PGDAttack(model, epsilon)
        else:
            raise ValueError(f"Attack type {attack_type} no soportado")
    
    def adversarial_loss(
        self,
        inputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        alpha: float = 0.5
    ) -> torch.Tensor:
        """
        Calcula loss combinado (normal + adversarial)
        
        Args:
            inputs: Inputs originales
            labels: Labels
            alpha: Peso de loss adversarial
            
        Returns:
            Loss combinado
        """
        # Loss normal
        outputs = self.model(**inputs)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        normal_loss = F.cross_entropy(logits, labels)
        
        # Generar ejemplos adversariales
        inputs_adv = self.attack.generate(inputs, labels)
        
        # Loss adversarial
        outputs_adv = self.model(**inputs_adv)
        logits_adv = outputs_adv.logits if hasattr(outputs_adv, 'logits') else outputs_adv
        adv_loss = F.cross_entropy(logits_adv, labels)
        
        # Loss combinado
        total_loss = (1 - alpha) * normal_loss + alpha * adv_loss
        
        return total_loss




