"""
Model Security - Seguridad de modelos (Adversarial attacks, robustness)
=========================================================================
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AdversarialAttackResult:
    """Resultado de ataque adversarial"""
    attack_type: str
    success: bool
    original_prediction: int
    adversarial_prediction: int
    perturbation_norm: float
    robustness_score: float
    timestamp: datetime = field(default_factory=datetime.now)


class ModelSecurity:
    """Sistema de seguridad de modelos"""
    
    def __init__(self):
        self.attack_results: List[AdversarialAttackResult] = []
    
    def fgsm_attack(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        true_label: torch.Tensor,
        epsilon: float = 0.1,
        device: str = "cuda"
    ) -> Tuple[torch.Tensor, bool]:
        """Fast Gradient Sign Method attack"""
        device = torch.device(device)
        model = model.to(device)
        model.eval()
        
        input_tensor = input_tensor.to(device).requires_grad_(True)
        true_label = true_label.to(device)
        
        # Forward pass
        output = model(input_tensor)
        loss = F.cross_entropy(output, true_label)
        
        # Backward
        model.zero_grad()
        loss.backward()
        
        # Create adversarial example
        perturbation = epsilon * input_tensor.grad.sign()
        adversarial_input = input_tensor + perturbation
        
        # Clip to valid range
        adversarial_input = torch.clamp(adversarial_input, 0, 1)
        
        # Check if attack succeeded
        with torch.no_grad():
            adv_output = model(adversarial_input)
            adv_pred = adv_output.argmax(dim=-1)
            success = (adv_pred != true_label).item()
        
        return adversarial_input, success
    
    def pgd_attack(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        true_label: torch.Tensor,
        epsilon: float = 0.1,
        alpha: float = 0.01,
        num_iterations: int = 10,
        device: str = "cuda"
    ) -> Tuple[torch.Tensor, bool]:
        """Projected Gradient Descent attack"""
        device = torch.device(device)
        model = model.to(device)
        model.eval()
        
        adversarial_input = input_tensor.clone().to(device).requires_grad_(True)
        true_label = true_label.to(device)
        
        for _ in range(num_iterations):
            output = model(adversarial_input)
            loss = F.cross_entropy(output, true_label)
            
            model.zero_grad()
            loss.backward()
            
            # Update adversarial input
            with torch.no_grad():
                perturbation = alpha * adversarial_input.grad.sign()
                adversarial_input = adversarial_input + perturbation
                
                # Project to epsilon ball
                perturbation = torch.clamp(
                    adversarial_input - input_tensor,
                    -epsilon, epsilon
                )
                adversarial_input = input_tensor + perturbation
                adversarial_input = torch.clamp(adversarial_input, 0, 1)
                adversarial_input.requires_grad_(True)
        
        # Check if attack succeeded
        with torch.no_grad():
            adv_output = model(adversarial_input)
            adv_pred = adv_output.argmax(dim=-1)
            success = (adv_pred != true_label).item()
        
        return adversarial_input, success
    
    def test_robustness(
        self,
        model: nn.Module,
        test_loader: Any,
        attack_type: str = "fgsm",
        epsilon: float = 0.1,
        device: str = "cuda"
    ) -> Dict[str, float]:
        """Prueba robustez del modelo"""
        device = torch.device(device)
        model = model.to(device)
        model.eval()
        
        total = 0
        successful_attacks = 0
        total_perturbation_norm = 0.0
        
        for batch in test_loader:
            if isinstance(batch, dict):
                inputs = batch.get("input_ids") or batch.get("inputs")
                labels = batch.get("labels") or batch.get("targets")
            else:
                inputs, labels = batch
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Original prediction
            with torch.no_grad():
                original_output = model(inputs)
                original_pred = original_output.argmax(dim=-1)
            
            # Attack
            if attack_type == "fgsm":
                adv_input, success = self.fgsm_attack(
                    model, inputs, labels, epsilon, device
                )
            elif attack_type == "pgd":
                adv_input, success = self.pgd_attack(
                    model, inputs, labels, epsilon, device=device
                )
            else:
                continue
            
            # Calculate perturbation norm
            perturbation = (adv_input - inputs).norm().item()
            total_perturbation_norm += perturbation
            
            if success:
                successful_attacks += 1
            
            total += 1
        
        attack_success_rate = successful_attacks / total if total > 0 else 0
        avg_perturbation = total_perturbation_norm / total if total > 0 else 0
        robustness_score = 1.0 - attack_success_rate
        
        return {
            "attack_success_rate": attack_success_rate,
            "robustness_score": robustness_score,
            "avg_perturbation_norm": avg_perturbation,
            "total_samples": total
        }
    
    def adversarial_training_step(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        epsilon: float = 0.1,
        device: str = "cuda"
    ) -> torch.Tensor:
        """Paso de entrenamiento adversarial"""
        device = torch.device(device)
        model = model.to(device)
        model.train()
        
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Generate adversarial examples
        adv_inputs, _ = self.fgsm_attack(model, inputs, labels, epsilon, device)
        
        # Train on adversarial examples
        outputs = model(adv_inputs)
        loss = F.cross_entropy(outputs, labels)
        
        return loss




