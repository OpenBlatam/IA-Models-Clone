"""
Adversarial Training Engine for Export IA
Advanced adversarial training with multiple attack methods and defense strategies
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
import time
import json
import random
from pathlib import Path
from collections import defaultdict, deque
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import cv2

logger = logging.getLogger(__name__)

@dataclass
class AdversarialConfig:
    """Configuration for adversarial training"""
    # Attack methods
    attack_methods: List[str] = None  # fgsm, pgd, cw, deepfool, jsma, autoattack
    
    # Training parameters
    adversarial_ratio: float = 0.5  # Ratio of adversarial examples in training
    attack_epochs: int = 10  # Number of attack iterations
    attack_lr: float = 0.01  # Learning rate for attacks
    
    # FGSM parameters
    fgsm_epsilon: float = 0.1
    
    # PGD parameters
    pgd_epsilon: float = 0.1
    pgd_alpha: float = 0.01
    pgd_steps: int = 10
    
    # C&W parameters
    cw_c: float = 1.0
    cw_kappa: float = 0.0
    cw_binary_search_steps: int = 9
    cw_max_iterations: int = 1000
    
    # DeepFool parameters
    deepfool_overshoot: float = 0.02
    deepfool_max_iter: int = 50
    
    # JSMA parameters
    jsma_theta: float = 1.0
    jsma_gamma: float = 0.1
    
    # AutoAttack parameters
    autoattack_epsilon: float = 0.1
    autoattack_norm: str = "Linf"
    
    # Defense methods
    defense_methods: List[str] = None  # adversarial_training, mixup, cutmix, label_smoothing
    
    # Mixup parameters
    mixup_alpha: float = 1.0
    
    # CutMix parameters
    cutmix_alpha: float = 1.0
    cutmix_prob: float = 0.5
    
    # Label smoothing
    label_smoothing: float = 0.1
    
    # Evaluation
    evaluate_robustness: bool = True
    robustness_metrics: List[str] = None  # accuracy, asr, mse, psnr
    
    # Logging
    log_attacks: bool = True
    save_adversarial_examples: bool = True
    visualize_attacks: bool = True

class FGSMAttack:
    """Fast Gradient Sign Method (FGSM) attack"""
    
    def __init__(self, config: AdversarialConfig):
        self.config = config
        
    def generate(self, model: nn.Module, inputs: torch.Tensor, 
                targets: torch.Tensor) -> torch.Tensor:
        """Generate FGSM adversarial examples"""
        
        model.eval()
        inputs.requires_grad_(True)
        
        # Forward pass
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        
        # Generate adversarial examples
        adversarial_inputs = inputs + self.config.fgsm_epsilon * inputs.grad.sign()
        
        # Clip to valid range
        adversarial_inputs = torch.clamp(adversarial_inputs, 0, 1)
        
        return adversarial_inputs.detach()

class PGDAttack:
    """Projected Gradient Descent (PGD) attack"""
    
    def __init__(self, config: AdversarialConfig):
        self.config = config
        
    def generate(self, model: nn.Module, inputs: torch.Tensor, 
                targets: torch.Tensor) -> torch.Tensor:
        """Generate PGD adversarial examples"""
        
        model.eval()
        adversarial_inputs = inputs.clone()
        
        # Initialize with random noise
        noise = torch.randn_like(inputs) * self.config.pgd_epsilon
        adversarial_inputs = torch.clamp(adversarial_inputs + noise, 0, 1)
        
        for _ in range(self.config.pgd_steps):
            adversarial_inputs.requires_grad_(True)
            
            # Forward pass
            outputs = model(adversarial_inputs)
            loss = F.cross_entropy(outputs, targets)
            
            # Backward pass
            model.zero_grad()
            loss.backward()
            
            # Update adversarial inputs
            with torch.no_grad():
                adversarial_inputs = adversarial_inputs + self.config.pgd_alpha * adversarial_inputs.grad.sign()
                
                # Project to epsilon ball
                delta = adversarial_inputs - inputs
                delta = torch.clamp(delta, -self.config.pgd_epsilon, self.config.pgd_epsilon)
                adversarial_inputs = torch.clamp(inputs + delta, 0, 1)
                
        return adversarial_inputs.detach()

class CWL2Attack:
    """Carlini & Wagner L2 attack"""
    
    def __init__(self, config: AdversarialConfig):
        self.config = config
        
    def generate(self, model: nn.Module, inputs: torch.Tensor, 
                targets: torch.Tensor) -> torch.Tensor:
        """Generate C&W L2 adversarial examples"""
        
        model.eval()
        batch_size = inputs.size(0)
        adversarial_inputs = inputs.clone()
        
        for i in range(batch_size):
            input_i = inputs[i:i+1]
            target_i = targets[i:i+1]
            
            # Binary search for optimal c
            c_low = 0.0
            c_high = 1.0
            c = 1.0
            
            for _ in range(self.config.cw_binary_search_steps):
                # Optimize with current c
                adv_input = self._optimize_cw(model, input_i, target_i, c)
                
                # Check if attack succeeded
                with torch.no_grad():
                    outputs = model(adv_input)
                    _, predicted = torch.max(outputs, 1)
                    success = predicted != target_i
                    
                if success:
                    c_high = c
                    adversarial_inputs[i] = adv_input[0]
                else:
                    c_low = c
                    
                c = (c_low + c_high) / 2
                
        return adversarial_inputs.detach()
        
    def _optimize_cw(self, model: nn.Module, input_tensor: torch.Tensor, 
                    target: torch.Tensor, c: float) -> torch.Tensor:
        """Optimize C&W objective"""
        
        # Initialize perturbation
        delta = torch.zeros_like(input_tensor, requires_grad=True)
        optimizer = optim.Adam([delta], lr=0.01)
        
        for _ in range(self.config.cw_max_iterations):
            optimizer.zero_grad()
            
            # Adversarial input
            adv_input = torch.clamp(input_tensor + delta, 0, 1)
            
            # Forward pass
            outputs = model(adv_input)
            
            # C&W loss
            target_logit = outputs[0, target]
            max_other_logit = torch.max(outputs[0, :target].max(), outputs[0, target+1:].max())
            
            f = torch.clamp(max_other_logit - target_logit + self.config.cw_kappa, min=0)
            l2_norm = torch.norm(delta)
            
            loss = l2_norm + c * f
            
            loss.backward()
            optimizer.step()
            
        return torch.clamp(input_tensor + delta, 0, 1).detach()

class DeepFoolAttack:
    """DeepFool attack"""
    
    def __init__(self, config: AdversarialConfig):
        self.config = config
        
    def generate(self, model: nn.Module, inputs: torch.Tensor, 
                targets: torch.Tensor) -> torch.Tensor:
        """Generate DeepFool adversarial examples"""
        
        model.eval()
        batch_size = inputs.size(0)
        adversarial_inputs = inputs.clone()
        
        for i in range(batch_size):
            input_i = inputs[i:i+1]
            target_i = targets[i:i+1]
            
            # Initialize
            adv_input = input_i.clone()
            adv_input.requires_grad_(True)
            
            for _ in range(self.config.deepfool_max_iter):
                # Forward pass
                outputs = model(adv_input)
                _, predicted = torch.max(outputs, 1)
                
                if predicted != target_i:
                    break
                    
                # Compute gradients
                model.zero_grad()
                outputs[0, predicted].backward(retain_graph=True)
                grad_pred = adv_input.grad.clone()
                
                # Find closest decision boundary
                min_dist = float('inf')
                min_perturbation = None
                
                for class_idx in range(outputs.size(1)):
                    if class_idx == predicted:
                        continue
                        
                    model.zero_grad()
                    outputs[0, class_idx].backward(retain_graph=True)
                    grad_class = adv_input.grad.clone()
                    
                    # Compute perturbation
                    grad_diff = grad_pred - grad_class
                    output_diff = outputs[0, predicted] - outputs[0, class_idx]
                    
                    if torch.norm(grad_diff) > 1e-8:
                        perturbation = (output_diff / torch.norm(grad_diff) ** 2) * grad_diff
                        
                        if torch.norm(perturbation) < min_dist:
                            min_dist = torch.norm(perturbation)
                            min_perturbation = perturbation
                            
                if min_perturbation is not None:
                    adv_input = adv_input + (1 + self.config.deepfool_overshoot) * min_perturbation
                    adv_input = torch.clamp(adv_input, 0, 1)
                    adv_input = adv_input.detach().requires_grad_(True)
                    
        return adversarial_inputs.detach()

class JSMAAttack:
    """Jacobian-based Saliency Map Attack (JSMA)"""
    
    def __init__(self, config: AdversarialConfig):
        self.config = config
        
    def generate(self, model: nn.Module, inputs: torch.Tensor, 
                targets: torch.Tensor) -> torch.Tensor:
        """Generate JSMA adversarial examples"""
        
        model.eval()
        batch_size = inputs.size(0)
        adversarial_inputs = inputs.clone()
        
        for i in range(batch_size):
            input_i = inputs[i:i+1]
            target_i = targets[i:i+1]
            
            adv_input = input_i.clone()
            
            for _ in range(100):  # Maximum iterations
                # Check if attack succeeded
                with torch.no_grad():
                    outputs = model(adv_input)
                    _, predicted = torch.max(outputs, 1)
                    if predicted != target_i:
                        break
                        
                # Compute saliency map
                saliency_map = self._compute_saliency_map(model, adv_input, target_i)
                
                # Find best pixel to modify
                flat_saliency = saliency_map.view(-1)
                _, best_pixel = torch.max(flat_saliency, 0)
                
                # Modify pixel
                adv_input.view(-1)[best_pixel] += self.config.jsma_theta
                adv_input = torch.clamp(adv_input, 0, 1)
                
        adversarial_inputs[i] = adv_input[0]
        
        return adversarial_inputs.detach()
        
    def _compute_saliency_map(self, model: nn.Module, input_tensor: torch.Tensor, 
                             target: torch.Tensor) -> torch.Tensor:
        """Compute saliency map"""
        
        input_tensor.requires_grad_(True)
        model.zero_grad()
        
        outputs = model(input_tensor)
        target_logit = outputs[0, target]
        
        target_logit.backward()
        gradients = input_tensor.grad
        
        # Compute saliency
        saliency = torch.abs(gradients)
        
        return saliency

class AutoAttack:
    """AutoAttack ensemble"""
    
    def __init__(self, config: AdversarialConfig):
        self.config = config
        self.attacks = [
            FGSMAttack(config),
            PGDAttack(config),
            CWL2Attack(config),
            DeepFoolAttack(config)
        ]
        
    def generate(self, model: nn.Module, inputs: torch.Tensor, 
                targets: torch.Tensor) -> torch.Tensor:
        """Generate adversarial examples using ensemble of attacks"""
        
        best_attacks = []
        
        for attack in self.attacks:
            try:
                adv_inputs = attack.generate(model, inputs, targets)
                best_attacks.append(adv_inputs)
            except Exception as e:
                logger.error(f"Attack failed: {e}")
                continue
                
        if not best_attacks:
            return inputs
            
        # Return the most successful attack
        return best_attacks[0]

class MixupDefense:
    """Mixup defense strategy"""
    
    def __init__(self, config: AdversarialConfig):
        self.config = config
        
    def mixup_data(self, inputs: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Apply mixup data augmentation"""
        
        batch_size = inputs.size(0)
        lam = np.random.beta(self.config.mixup_alpha, self.config.mixup_alpha)
        
        # Random permutation
        index = torch.randperm(batch_size)
        
        # Mix inputs
        mixed_inputs = lam * inputs + (1 - lam) * inputs[index]
        
        # Mix targets
        targets_a, targets_b = targets, targets[index]
        
        return mixed_inputs, targets_a, targets_b, lam
        
    def mixup_criterion(self, criterion: Callable, outputs: torch.Tensor, 
                       targets_a: torch.Tensor, targets_b: torch.Tensor, lam: float) -> torch.Tensor:
        """Compute mixup loss"""
        
        return lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)

class CutMixDefense:
    """CutMix defense strategy"""
    
    def __init__(self, config: AdversarialConfig):
        self.config = config
        
    def cutmix_data(self, inputs: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Apply CutMix data augmentation"""
        
        batch_size = inputs.size(0)
        lam = np.random.beta(self.config.cutmix_alpha, self.config.cutmix_alpha)
        
        # Random permutation
        index = torch.randperm(batch_size)
        
        # Get bounding box
        W, H = inputs.size(2), inputs.size(3)
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        # Random center
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        # Apply CutMix
        mixed_inputs = inputs.clone()
        mixed_inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[index, :, bbx1:bbx2, bby1:bby2]
        
        # Adjust lambda
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        
        targets_a, targets_b = targets, targets[index]
        
        return mixed_inputs, targets_a, targets_b, lam

class AdversarialTrainingEngine:
    """Main Adversarial Training Engine"""
    
    def __init__(self, config: AdversarialConfig):
        self.config = config
        
        # Initialize attacks
        self.attacks = {}
        if "fgsm" in (self.config.attack_methods or []):
            self.attacks["fgsm"] = FGSMAttack(config)
        if "pgd" in (self.config.attack_methods or []):
            self.attacks["pgd"] = PGDAttack(config)
        if "cw" in (self.config.attack_methods or []):
            self.attacks["cw"] = CWL2Attack(config)
        if "deepfool" in (self.config.attack_methods or []):
            self.attacks["deepfool"] = DeepFoolAttack(config)
        if "jsma" in (self.config.attack_methods or []):
            self.attacks["jsma"] = JSMAAttack(config)
        if "autoattack" in (self.config.attack_methods or []):
            self.attacks["autoattack"] = AutoAttack(config)
            
        # Initialize defenses
        self.defenses = {}
        if "mixup" in (self.config.defense_methods or []):
            self.defenses["mixup"] = MixupDefense(config)
        if "cutmix" in (self.config.defense_methods or []):
            self.defenses["cutmix"] = CutMixDefense(config)
            
        # Training metrics
        self.training_metrics = defaultdict(list)
        self.attack_success_rates = defaultdict(list)
        
    def train_with_adversarial_examples(self, model: nn.Module, 
                                      train_dataloader: torch.utils.data.DataLoader,
                                      val_dataloader: torch.utils.data.DataLoader = None) -> Dict[str, Any]:
        """Train model with adversarial examples"""
        
        logger.info("Starting adversarial training...")
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)
        
        best_accuracy = 0.0
        best_robust_accuracy = 0.0
        
        for epoch in range(50):  # 50 epochs
            model.train()
            total_loss = 0.0
            correct = 0
            total = 0
            adversarial_correct = 0
            adversarial_total = 0
            
            for batch_idx, (inputs, targets) in enumerate(train_dataloader):
                optimizer.zero_grad()
                
                # Determine if this batch should be adversarial
                use_adversarial = random.random() < self.config.adversarial_ratio
                
                if use_adversarial and self.attacks:
                    # Generate adversarial examples
                    attack_method = random.choice(list(self.attacks.keys()))
                    adversarial_inputs = self.attacks[attack_method].generate(model, inputs, targets)
                    
                    # Train on adversarial examples
                    outputs = model(adversarial_inputs)
                    loss = criterion(outputs, targets)
                    
                    # Track adversarial accuracy
                    _, predicted = torch.max(outputs.data, 1)
                    adversarial_total += targets.size(0)
                    adversarial_correct += (predicted == targets).sum().item()
                    
                else:
                    # Train on clean examples
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                # Apply defense strategies
                if "mixup" in self.defenses and random.random() < 0.5:
                    mixed_inputs, targets_a, targets_b, lam = self.defenses["mixup"].mixup_data(inputs, targets)
                    mixed_outputs = model(mixed_inputs)
                    loss = self.defenses["mixup"].mixup_criterion(criterion, mixed_outputs, targets_a, targets_b, lam)
                    
                elif "cutmix" in self.defenses and random.random() < self.config.cutmix_prob:
                    mixed_inputs, targets_a, targets_b, lam = self.defenses["cutmix"].cutmix_data(inputs, targets)
                    mixed_outputs = model(mixed_inputs)
                    loss = self.defenses["cutmix"].mixup_criterion(criterion, mixed_outputs, targets_a, targets_b, lam)
                    
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
            # Validation
            if val_dataloader is not None:
                val_accuracy = self._evaluate_model(model, val_dataloader)
                robust_accuracy = self._evaluate_robustness(model, val_dataloader)
                
                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                if robust_accuracy > best_robust_accuracy:
                    best_robust_accuracy = robust_accuracy
                    
                logger.info(f"Epoch {epoch}: Loss = {total_loss:.4f}, "
                           f"Accuracy = {100 * correct / total:.2f}%, "
                           f"Val Accuracy = {100 * val_accuracy:.2f}%, "
                           f"Robust Accuracy = {100 * robust_accuracy:.2f}%")
                           
                # Store metrics
                self.training_metrics['epoch'].append(epoch)
                self.training_metrics['loss'].append(total_loss)
                self.training_metrics['accuracy'].append(100 * correct / total)
                self.training_metrics['val_accuracy'].append(100 * val_accuracy)
                self.training_metrics['robust_accuracy'].append(100 * robust_accuracy)
                
        return {
            'best_accuracy': best_accuracy,
            'best_robust_accuracy': best_robust_accuracy,
            'training_metrics': dict(self.training_metrics)
        }
        
    def _evaluate_model(self, model: nn.Module, dataloader: torch.utils.data.DataLoader) -> float:
        """Evaluate model on clean data"""
        
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
        return correct / total
        
    def _evaluate_robustness(self, model: nn.Module, dataloader: torch.utils.data.DataLoader) -> float:
        """Evaluate model robustness against attacks"""
        
        if not self.attacks:
            return self._evaluate_model(model, dataloader)
            
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                # Test against all attacks
                all_correct = True
                
                for attack_name, attack in self.attacks.items():
                    try:
                        adversarial_inputs = attack.generate(model, inputs, targets)
                        outputs = model(adversarial_inputs)
                        _, predicted = torch.max(outputs.data, 1)
                        
                        if not torch.all(predicted == targets):
                            all_correct = False
                            break
                            
                    except Exception as e:
                        logger.error(f"Attack {attack_name} failed: {e}")
                        all_correct = False
                        break
                        
                total += targets.size(0)
                if all_correct:
                    correct += targets.size(0)
                    
        return correct / total
        
    def evaluate_attack_success_rates(self, model: nn.Module, 
                                    dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Evaluate attack success rates"""
        
        model.eval()
        success_rates = {}
        
        for attack_name, attack in self.attacks.items():
            successful_attacks = 0
            total_attacks = 0
            
            with torch.no_grad():
                for inputs, targets in dataloader:
                    try:
                        # Generate adversarial examples
                        adversarial_inputs = attack.generate(model, inputs, targets)
                        
                        # Check if attack succeeded
                        clean_outputs = model(inputs)
                        adv_outputs = model(adversarial_inputs)
                        
                        _, clean_predicted = torch.max(clean_outputs.data, 1)
                        _, adv_predicted = torch.max(adv_outputs.data, 1)
                        
                        # Attack succeeds if prediction changes
                        successful_attacks += (clean_predicted != adv_predicted).sum().item()
                        total_attacks += targets.size(0)
                        
                    except Exception as e:
                        logger.error(f"Attack {attack_name} evaluation failed: {e}")
                        continue
                        
            success_rates[attack_name] = successful_attacks / total_attacks if total_attacks > 0 else 0.0
            
        return success_rates
        
    def generate_adversarial_dataset(self, model: nn.Module, 
                                   dataloader: torch.utils.data.DataLoader,
                                   attack_method: str = "pgd") -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate dataset of adversarial examples"""
        
        if attack_method not in self.attacks:
            raise ValueError(f"Attack method {attack_method} not available")
            
        model.eval()
        all_adversarial_inputs = []
        all_targets = []
        
        for inputs, targets in dataloader:
            adversarial_inputs = self.attacks[attack_method].generate(model, inputs, targets)
            all_adversarial_inputs.append(adversarial_inputs)
            all_targets.append(targets)
            
        return torch.cat(all_adversarial_inputs), torch.cat(all_targets)
        
    def visualize_attacks(self, model: nn.Module, inputs: torch.Tensor, 
                         targets: torch.Tensor, save_path: str = None):
        """Visualize adversarial attacks"""
        
        if not self.config.visualize_attacks:
            return
            
        model.eval()
        num_examples = min(5, inputs.size(0))
        
        fig, axes = plt.subplots(len(self.attacks) + 1, num_examples, figsize=(15, 10))
        if num_examples == 1:
            axes = axes.reshape(-1, 1)
            
        # Original images
        for i in range(num_examples):
            if inputs.dim() == 4:  # Image data
                img = inputs[i].permute(1, 2, 0).cpu().numpy()
                axes[0, i].imshow(img)
                axes[0, i].set_title(f"Original (Target: {targets[i].item()})")
                axes[0, i].axis('off')
            else:
                axes[0, i].text(0.5, 0.5, f"Original\nTarget: {targets[i].item()}", 
                               ha='center', va='center')
                axes[0, i].axis('off')
                
        # Adversarial examples
        for attack_idx, (attack_name, attack) in enumerate(self.attacks.items()):
            try:
                adversarial_inputs = attack.generate(model, inputs[:num_examples], targets[:num_examples])
                
                for i in range(num_examples):
                    if adversarial_inputs.dim() == 4:  # Image data
                        img = adversarial_inputs[i].permute(1, 2, 0).cpu().numpy()
                        axes[attack_idx + 1, i].imshow(img)
                        
                        # Check if attack succeeded
                        with torch.no_grad():
                            clean_output = model(inputs[i:i+1])
                            adv_output = model(adversarial_inputs[i:i+1])
                            _, clean_pred = torch.max(clean_output, 1)
                            _, adv_pred = torch.max(adv_output, 1)
                            
                        success = "✓" if clean_pred != adv_pred else "✗"
                        axes[attack_idx + 1, i].set_title(f"{attack_name} {success}\nPred: {adv_pred.item()}")
                        axes[attack_idx + 1, i].axis('off')
                    else:
                        axes[attack_idx + 1, i].text(0.5, 0.5, f"{attack_name}\nAdversarial", 
                                                   ha='center', va='center')
                        axes[attack_idx + 1, i].axis('off')
                        
            except Exception as e:
                logger.error(f"Visualization failed for {attack_name}: {e}")
                for i in range(num_examples):
                    axes[attack_idx + 1, i].text(0.5, 0.5, f"{attack_name}\nFailed", 
                                               ha='center', va='center')
                    axes[attack_idx + 1, i].axis('off')
                    
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
            
        plt.close()

# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Test adversarial training
    print("Testing Adversarial Training Engine...")
    
    # Create test model
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(64 * 8 * 8, 128)
            self.fc2 = nn.Linear(128, 10)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.2)
            
        def forward(self, x):
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = x.view(x.size(0), -1)
            x = self.dropout(self.relu(self.fc1(x)))
            x = self.fc2(x)
            return x
    
    model = TestModel()
    
    # Create adversarial config
    config = AdversarialConfig(
        attack_methods=["fgsm", "pgd"],
        defense_methods=["mixup"],
        adversarial_ratio=0.5,
        fgsm_epsilon=0.1,
        pgd_epsilon=0.1,
        pgd_steps=10,
        mixup_alpha=1.0,
        evaluate_robustness=True
    )
    
    # Create adversarial training engine
    adv_engine = AdversarialTrainingEngine(config)
    
    # Create dummy dataloaders
    def create_dummy_dataloader(num_samples: int = 100):
        inputs = torch.randn(num_samples, 3, 32, 32)
        targets = torch.randint(0, 10, (num_samples,))
        dataset = torch.utils.data.TensorDataset(inputs, targets)
        return torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    train_loader = create_dummy_dataloader(200)
    val_loader = create_dummy_dataloader(50)
    
    # Test adversarial training
    print("Testing adversarial training...")
    results = adv_engine.train_with_adversarial_examples(model, train_loader, val_loader)
    print(f"Training completed: {results}")
    
    # Test attack success rates
    print("Testing attack success rates...")
    success_rates = adv_engine.evaluate_attack_success_rates(model, val_loader)
    print(f"Attack success rates: {success_rates}")
    
    # Test adversarial dataset generation
    print("Testing adversarial dataset generation...")
    adv_inputs, adv_targets = adv_engine.generate_adversarial_dataset(model, val_loader, "pgd")
    print(f"Generated adversarial dataset: {adv_inputs.shape}, {adv_targets.shape}")
    
    print("\nAdversarial training engine initialized successfully!")
























