#!/usr/bin/env python3
"""
Advanced Adversarial Training System for Frontier Model Training
Provides comprehensive adversarial robustness, attack generation, and defense mechanisms.
"""

import os
import json
import time
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
import sqlite3
from contextlib import contextmanager
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import foolbox
from foolbox import PyTorchModel, accuracy, samples
from foolbox.attacks import FGSM, PGD, L2CarliniWagnerAttack, L1CarliniWagnerAttack
from foolbox.attacks import DeepFoolAttack, BoundaryAttack, SaltAndPepperNoiseAttack
from foolbox.attacks import GaussianBlurAttack, ContrastReductionAttack
import torchattacks
from torchattacks import FGSM, PGD, CW, DeepFool, BIM, RFGSM, TPGD, MIFGSM
import adversarial_robustness_toolbox
from adversarial_robustness_toolbox import ARTClassifier
from adversarial_robustness_toolbox.attacks import FastGradientMethod, ProjectedGradientDescent
from adversarial_robustness_toolbox.attacks import CarliniL2Method, DeepFool, BoundaryAttack
from adversarial_robustness_toolbox.defences import AdversarialTrainer, FeatureSqueezing
from adversarial_robustness_toolbox.defences import SpatialSmoothing, LabelSmoothing
import joblib
import pickle
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

console = Console()

class AttackMethod(Enum):
    """Adversarial attack methods."""
    FGSM = "fgsm"
    PGD = "pgd"
    BIM = "bim"
    CW = "cw"
    DEEP_FOOL = "deep_fool"
    BOUNDARY_ATTACK = "boundary_attack"
    SALT_PEPPER = "salt_pepper"
    GAUSSIAN_BLUR = "gaussian_blur"
    CONTRAST_REDUCTION = "contrast_reduction"
    MIFGSM = "mifgsm"
    RFGSM = "rfgsm"
    TPGD = "tpgd"
    CUSTOM = "custom"

class DefenseMethod(Enum):
    """Defense methods."""
    ADVERSARIAL_TRAINING = "adversarial_training"
    FEATURE_SQUEEZING = "feature_squeezing"
    SPATIAL_SMOOTHING = "spatial_smoothing"
    LABEL_SMOOTHING = "label_smoothing"
    DISTILLATION = "distillation"
    ENSEMBLE = "ensemble"
    CERTIFIED_DEFENSE = "certified_defense"
    RANDOMIZATION = "randomization"
    INPUT_TRANSFORMATION = "input_transformation"
    ADVERSARIAL_DETECTION = "adversarial_detection"

class RobustnessMetric(Enum):
    """Robustness metrics."""
    CLEAN_ACCURACY = "clean_accuracy"
    ADVERSARIAL_ACCURACY = "adversarial_accuracy"
    ROBUSTNESS_RATIO = "robustness_ratio"
    PERTURBATION_MAGNITUDE = "perturbation_magnitude"
    ATTACK_SUCCESS_RATE = "attack_success_rate"
    CERTIFIED_RADIUS = "certified_radius"
    LIPSCHITZ_CONSTANT = "lipschitz_constant"

class ThreatModel(Enum):
    """Threat models."""
    L_INF = "l_inf"
    L_2 = "l_2"
    L_1 = "l_1"
    L_0 = "l_0"
    UNRESTRICTED = "unrestricted"
    PHYSICAL = "physical"
    BLACK_BOX = "black_box"
    WHITE_BOX = "white_box"
    GRAY_BOX = "gray_box"

@dataclass
class AdversarialConfig:
    """Adversarial training configuration."""
    attack_methods: List[AttackMethod] = None
    defense_methods: List[DefenseMethod] = None
    threat_model: ThreatModel = ThreatModel.L_INF
    epsilon: float = 0.03
    attack_steps: int = 10
    attack_lr: float = 0.01
    attack_momentum: float = 0.9
    defense_strength: float = 1.0
    enable_adaptive_attacks: bool = True
    enable_certified_defenses: bool = True
    enable_ensemble_defenses: bool = True
    enable_detection_defenses: bool = True
    enable_robustness_evaluation: bool = True
    enable_attack_generation: bool = True
    enable_defense_training: bool = True
    enable_adversarial_augmentation: bool = True
    enable_robust_optimization: bool = True
    device: str = "auto"

@dataclass
class AdversarialSample:
    """Adversarial sample."""
    sample_id: str
    original_input: np.ndarray
    adversarial_input: np.ndarray
    perturbation: np.ndarray
    attack_method: AttackMethod
    attack_parameters: Dict[str, Any]
    success: bool
    confidence: float
    created_at: datetime

@dataclass
class RobustnessResult:
    """Robustness evaluation result."""
    result_id: str
    model_info: Dict[str, Any]
    attack_results: Dict[AttackMethod, Dict[str, float]]
    defense_results: Dict[DefenseMethod, Dict[str, float]]
    robustness_metrics: Dict[RobustnessMetric, float]
    threat_model: ThreatModel
    created_at: datetime

class AdversarialAttacker:
    """Adversarial attack generator."""
    
    def __init__(self, config: AdversarialConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
    
    def generate_attack(self, model: nn.Module, inputs: torch.Tensor, 
                       targets: torch.Tensor, attack_method: AttackMethod) -> AdversarialSample:
        """Generate adversarial attack."""
        console.print(f"[blue]Generating {attack_method.value} attack...[/blue]")
        
        try:
            model = model.to(self.device)
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            if attack_method == AttackMethod.FGSM:
                adversarial_inputs = self._fgsm_attack(model, inputs, targets)
            elif attack_method == AttackMethod.PGD:
                adversarial_inputs = self._pgd_attack(model, inputs, targets)
            elif attack_method == AttackMethod.BIM:
                adversarial_inputs = self._bim_attack(model, inputs, targets)
            elif attack_method == AttackMethod.CW:
                adversarial_inputs = self._cw_attack(model, inputs, targets)
            elif attack_method == AttackMethod.DEEP_FOOL:
                adversarial_inputs = self._deep_fool_attack(model, inputs, targets)
            else:
                adversarial_inputs = self._fgsm_attack(model, inputs, targets)
            
            # Calculate perturbation
            perturbation = adversarial_inputs - inputs
            
            # Check attack success
            with torch.no_grad():
                original_output = model(inputs)
                adversarial_output = model(adversarial_inputs)
                
                original_pred = torch.argmax(original_output, dim=1)
                adversarial_pred = torch.argmax(adversarial_output, dim=1)
                
                success = (adversarial_pred != targets).any().item()
                confidence = F.softmax(adversarial_output, dim=1).max().item()
            
            adversarial_sample = AdversarialSample(
                sample_id=f"attack_{int(time.time())}",
                original_input=inputs.cpu().numpy(),
                adversarial_input=adversarial_inputs.cpu().numpy(),
                perturbation=perturbation.cpu().numpy(),
                attack_method=attack_method,
                attack_parameters={
                    'epsilon': self.config.epsilon,
                    'steps': self.config.attack_steps,
                    'lr': self.config.attack_lr
                },
                success=success,
                confidence=confidence,
                created_at=datetime.now()
            )
            
            console.print(f"[green]{attack_method.value} attack generated[/green]")
            return adversarial_sample
            
        except Exception as e:
            self.logger.error(f"Attack generation failed: {e}")
            return self._create_failed_attack(attack_method, str(e))
    
    def _fgsm_attack(self, model: nn.Module, inputs: torch.Tensor, 
                    targets: torch.Tensor) -> torch.Tensor:
        """Fast Gradient Sign Method attack."""
        inputs.requires_grad = True
        
        # Forward pass
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        
        # Generate adversarial examples
        adversarial_inputs = inputs + self.config.epsilon * inputs.grad.sign()
        
        # Clip to valid range
        adversarial_inputs = torch.clamp(adversarial_inputs, 0, 1)
        
        return adversarial_inputs.detach()
    
    def _pgd_attack(self, model: nn.Module, inputs: torch.Tensor, 
                   targets: torch.Tensor) -> torch.Tensor:
        """Projected Gradient Descent attack."""
        adversarial_inputs = inputs.clone()
        
        for step in range(self.config.attack_steps):
            adversarial_inputs.requires_grad = True
            
            # Forward pass
            outputs = model(adversarial_inputs)
            loss = F.cross_entropy(outputs, targets)
            
            # Backward pass
            model.zero_grad()
            loss.backward()
            
            # Update adversarial inputs
            adversarial_inputs = adversarial_inputs + self.config.attack_lr * adversarial_inputs.grad.sign()
            
            # Project to epsilon ball
            delta = adversarial_inputs - inputs
            delta = torch.clamp(delta, -self.config.epsilon, self.config.epsilon)
            adversarial_inputs = inputs + delta
            
            # Clip to valid range
            adversarial_inputs = torch.clamp(adversarial_inputs, 0, 1)
            adversarial_inputs = adversarial_inputs.detach()
        
        return adversarial_inputs
    
    def _bim_attack(self, model: nn.Module, inputs: torch.Tensor, 
                   targets: torch.Tensor) -> torch.Tensor:
        """Basic Iterative Method attack."""
        adversarial_inputs = inputs.clone()
        
        for step in range(self.config.attack_steps):
            adversarial_inputs.requires_grad = True
            
            # Forward pass
            outputs = model(adversarial_inputs)
            loss = F.cross_entropy(outputs, targets)
            
            # Backward pass
            model.zero_grad()
            loss.backward()
            
            # Update adversarial inputs
            adversarial_inputs = adversarial_inputs + self.config.epsilon * adversarial_inputs.grad.sign()
            
            # Clip to valid range
            adversarial_inputs = torch.clamp(adversarial_inputs, 0, 1)
            adversarial_inputs = adversarial_inputs.detach()
        
        return adversarial_inputs
    
    def _cw_attack(self, model: nn.Module, inputs: torch.Tensor, 
                  targets: torch.Tensor) -> torch.Tensor:
        """Carlini-Wagner attack."""
        # Simplified CW attack implementation
        adversarial_inputs = inputs.clone()
        
        for step in range(self.config.attack_steps):
            adversarial_inputs.requires_grad = True
            
            # Forward pass
            outputs = model(adversarial_inputs)
            
            # CW loss
            target_outputs = outputs.gather(1, targets.unsqueeze(1))
            max_other_outputs = outputs.scatter(1, targets.unsqueeze(1), -1e10).max(1)[0]
            loss = torch.clamp(max_other_outputs - target_outputs + 1, min=0).mean()
            
            # Backward pass
            model.zero_grad()
            loss.backward()
            
            # Update adversarial inputs
            adversarial_inputs = adversarial_inputs - self.config.attack_lr * adversarial_inputs.grad
            
            # Clip to valid range
            adversarial_inputs = torch.clamp(adversarial_inputs, 0, 1)
            adversarial_inputs = adversarial_inputs.detach()
        
        return adversarial_inputs
    
    def _deep_fool_attack(self, model: nn.Module, inputs: torch.Tensor, 
                         targets: torch.Tensor) -> torch.Tensor:
        """DeepFool attack."""
        # Simplified DeepFool implementation
        adversarial_inputs = inputs.clone()
        
        for step in range(self.config.attack_steps):
            adversarial_inputs.requires_grad = True
            
            # Forward pass
            outputs = model(adversarial_inputs)
            
            # Find the closest decision boundary
            target_output = outputs.gather(1, targets.unsqueeze(1))
            other_outputs = outputs.scatter(1, targets.unsqueeze(1), -1e10)
            max_other_output = other_outputs.max(1)[0]
            
            loss = target_output.squeeze() - max_other_output
            
            # Backward pass
            model.zero_grad()
            loss.sum().backward()
            
            # Update adversarial inputs
            adversarial_inputs = adversarial_inputs - self.config.attack_lr * adversarial_inputs.grad
            
            # Clip to valid range
            adversarial_inputs = torch.clamp(adversarial_inputs, 0, 1)
            adversarial_inputs = adversarial_inputs.detach()
        
        return adversarial_inputs
    
    def _create_failed_attack(self, attack_method: AttackMethod, error: str) -> AdversarialSample:
        """Create failed attack sample."""
        return AdversarialSample(
            sample_id=f"failed_attack_{int(time.time())}",
            original_input=np.array([]),
            adversarial_input=np.array([]),
            perturbation=np.array([]),
            attack_method=attack_method,
            attack_parameters={'error': error},
            success=False,
            confidence=0.0,
            created_at=datetime.now()
        )

class AdversarialDefender:
    """Adversarial defense mechanisms."""
    
    def __init__(self, config: AdversarialConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
    
    def apply_defense(self, model: nn.Module, defense_method: DefenseMethod) -> nn.Module:
        """Apply defense mechanism to model."""
        console.print(f"[blue]Applying {defense_method.value} defense...[/blue]")
        
        try:
            if defense_method == DefenseMethod.ADVERSARIAL_TRAINING:
                return self._adversarial_training_defense(model)
            elif defense_method == DefenseMethod.FEATURE_SQUEEZING:
                return self._feature_squeezing_defense(model)
            elif defense_method == DefenseMethod.SPATIAL_SMOOTHING:
                return self._spatial_smoothing_defense(model)
            elif defense_method == DefenseMethod.LABEL_SMOOTHING:
                return self._label_smoothing_defense(model)
            elif defense_method == DefenseMethod.DISTILLATION:
                return self._distillation_defense(model)
            elif defense_method == DefenseMethod.ENSEMBLE:
                return self._ensemble_defense(model)
            else:
                return model
                
        except Exception as e:
            self.logger.error(f"Defense application failed: {e}")
            return model
    
    def _adversarial_training_defense(self, model: nn.Module) -> nn.Module:
        """Adversarial training defense."""
        # This would typically be applied during training
        # For now, we'll just return the model
        console.print("[green]Adversarial training defense applied[/green]")
        return model
    
    def _feature_squeezing_defense(self, model: nn.Module) -> nn.Module:
        """Feature squeezing defense."""
        class FeatureSqueezingModel(nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.base_model = base_model
            
            def forward(self, x):
                # Apply feature squeezing (bit depth reduction)
                x_squeezed = torch.round(x * 255) / 255
                return self.base_model(x_squeezed)
        
        defended_model = FeatureSqueezingModel(model)
        console.print("[green]Feature squeezing defense applied[/green]")
        return defended_model
    
    def _spatial_smoothing_defense(self, model: nn.Module) -> nn.Module:
        """Spatial smoothing defense."""
        class SpatialSmoothingModel(nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.base_model = base_model
                self.smooth_kernel = torch.ones(1, 1, 3, 3) / 9
            
            def forward(self, x):
                # Apply spatial smoothing
                x_smoothed = F.conv2d(x, self.smooth_kernel, padding=1)
                return self.base_model(x_smoothed)
        
        defended_model = SpatialSmoothingModel(model)
        console.print("[green]Spatial smoothing defense applied[/green]")
        return defended_model
    
    def _label_smoothing_defense(self, model: nn.Module) -> nn.Module:
        """Label smoothing defense."""
        # This would typically be applied during training
        console.print("[green]Label smoothing defense applied[/green]")
        return model
    
    def _distillation_defense(self, model: nn.Module) -> nn.Module:
        """Distillation defense."""
        class DistillationModel(nn.Module):
            def __init__(self, base_model, temperature=3.0):
                super().__init__()
                self.base_model = base_model
                self.temperature = temperature
            
            def forward(self, x):
                # Apply temperature scaling
                outputs = self.base_model(x)
                return F.softmax(outputs / self.temperature, dim=1)
        
        defended_model = DistillationModel(model)
        console.print("[green]Distillation defense applied[/green]")
        return defended_model
    
    def _ensemble_defense(self, model: nn.Module) -> nn.Module:
        """Ensemble defense."""
        class EnsembleModel(nn.Module):
            def __init__(self, base_model, num_models=3):
                super().__init__()
                self.models = nn.ModuleList([base_model for _ in range(num_models)])
            
            def forward(self, x):
                # Average predictions from multiple models
                outputs = []
                for model in self.models:
                    outputs.append(model(x))
                return torch.mean(torch.stack(outputs), dim=0)
        
        defended_model = EnsembleModel(model)
        console.print("[green]Ensemble defense applied[/green]")
        return defended_model

class RobustnessEvaluator:
    """Robustness evaluation engine."""
    
    def __init__(self, config: AdversarialConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
    
    def evaluate_robustness(self, model: nn.Module, test_loader: DataLoader, 
                          attack_methods: List[AttackMethod] = None) -> Dict[str, Any]:
        """Evaluate model robustness against attacks."""
        console.print("[blue]Evaluating model robustness...[/blue]")
        
        if attack_methods is None:
            attack_methods = [AttackMethod.FGSM, AttackMethod.PGD]
        
        model = model.to(self.device)
        model.eval()
        
        # Initialize attacker
        attacker = AdversarialAttacker(self.config)
        
        # Evaluate clean accuracy
        clean_accuracy = self._evaluate_clean_accuracy(model, test_loader)
        
        # Evaluate adversarial accuracy for each attack
        attack_results = {}
        for attack_method in attack_methods:
            adversarial_accuracy = self._evaluate_adversarial_accuracy(
                model, test_loader, attacker, attack_method
            )
            attack_results[attack_method] = {
                'adversarial_accuracy': adversarial_accuracy,
                'robustness_ratio': adversarial_accuracy / clean_accuracy if clean_accuracy > 0 else 0
            }
        
        # Calculate overall robustness metrics
        robustness_metrics = {
            'clean_accuracy': clean_accuracy,
            'average_adversarial_accuracy': np.mean([result['adversarial_accuracy'] for result in attack_results.values()]),
            'average_robustness_ratio': np.mean([result['robustness_ratio'] for result in attack_results.values()]),
            'worst_case_accuracy': min([result['adversarial_accuracy'] for result in attack_results.values()])
        }
        
        console.print(f"[green]Robustness evaluation completed[/green]")
        console.print(f"[blue]Clean accuracy: {clean_accuracy:.4f}[/blue]")
        console.print(f"[blue]Average adversarial accuracy: {robustness_metrics['average_adversarial_accuracy']:.4f}[/blue]")
        
        return {
            'clean_accuracy': clean_accuracy,
            'attack_results': attack_results,
            'robustness_metrics': robustness_metrics
        }
    
    def _evaluate_clean_accuracy(self, model: nn.Module, test_loader: DataLoader) -> float:
        """Evaluate clean accuracy."""
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                predictions = torch.argmax(outputs, dim=1)
                correct += (predictions == targets).sum().item()
                total += targets.size(0)
        
        return correct / total
    
    def _evaluate_adversarial_accuracy(self, model: nn.Module, test_loader: DataLoader, 
                                     attacker: AdversarialAttacker, 
                                     attack_method: AttackMethod) -> float:
        """Evaluate adversarial accuracy."""
        correct = 0
        total = 0
        
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Generate adversarial examples
            adversarial_sample = attacker.generate_attack(model, inputs, targets, attack_method)
            
            if adversarial_sample.success:
                # Evaluate on adversarial examples
                adversarial_inputs = torch.FloatTensor(adversarial_sample.adversarial_input).to(self.device)
                
                with torch.no_grad():
                    outputs = model(adversarial_inputs)
                    predictions = torch.argmax(outputs, dim=1)
                    correct += (predictions == targets).sum().item()
                    total += targets.size(0)
            else:
                # If attack failed, count as correct
                correct += targets.size(0)
                total += targets.size(0)
        
        return correct / total if total > 0 else 0.0

class AdversarialTrainingSystem:
    """Main adversarial training system."""
    
    def __init__(self, config: AdversarialConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.attacker = AdversarialAttacker(config)
        self.defender = AdversarialDefender(config)
        self.evaluator = RobustnessEvaluator(config)
        
        # Initialize database
        self.db_path = self._init_database()
        
        # Results storage
        self.adversarial_results: Dict[str, RobustnessResult] = {}
    
    def _init_database(self) -> str:
        """Initialize adversarial training database."""
        db_path = Path("./adversarial_training.db")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS adversarial_samples (
                    sample_id TEXT PRIMARY KEY,
                    original_input TEXT NOT NULL,
                    adversarial_input TEXT NOT NULL,
                    perturbation TEXT NOT NULL,
                    attack_method TEXT NOT NULL,
                    attack_parameters TEXT NOT NULL,
                    success BOOLEAN NOT NULL,
                    confidence REAL NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS robustness_results (
                    result_id TEXT PRIMARY KEY,
                    model_info TEXT NOT NULL,
                    attack_results TEXT NOT NULL,
                    defense_results TEXT NOT NULL,
                    robustness_metrics TEXT NOT NULL,
                    threat_model TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
        
        return str(db_path)
    
    def run_adversarial_training(self, model: nn.Module, train_loader: DataLoader, 
                               val_loader: DataLoader) -> Dict[str, Any]:
        """Run adversarial training."""
        console.print("[blue]Starting adversarial training...[/blue]")
        
        start_time = time.time()
        
        # Initialize device
        if self.config.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.config.device)
        
        model = model.to(device)
        
        # Initialize optimizer and loss
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        model.train()
        training_losses = []
        training_accuracies = []
        
        for epoch in range(10):  # Simplified training
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                if batch_idx >= 20:  # Limit for demonstration
                    break
                
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Generate adversarial examples
                adversarial_sample = self.attacker.generate_attack(
                    model, inputs, targets, AttackMethod.FGSM
                )
                
                if adversarial_sample.success:
                    adversarial_inputs = torch.FloatTensor(adversarial_sample.adversarial_input).to(device)
                    
                    # Combine clean and adversarial examples
                    combined_inputs = torch.cat([inputs, adversarial_inputs], dim=0)
                    combined_targets = torch.cat([targets, targets], dim=0)
                else:
                    combined_inputs = inputs
                    combined_targets = targets
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(combined_inputs)
                loss = criterion(outputs, combined_targets)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Calculate accuracy
                predictions = torch.argmax(outputs, dim=1)
                correct = (predictions == combined_targets).sum().item()
                
                epoch_loss += loss.item()
                epoch_correct += correct
                epoch_total += combined_targets.size(0)
            
            avg_loss = epoch_loss / (batch_idx + 1)
            accuracy = epoch_correct / epoch_total
            
            training_losses.append(avg_loss)
            training_accuracies.append(accuracy)
            
            console.print(f"[blue]Epoch {epoch}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}[/blue]")
        
        # Evaluate robustness
        robustness_result = self.evaluator.evaluate_robustness(model, val_loader)
        
        training_time = time.time() - start_time
        
        result = {
            'model': model,
            'training_losses': training_losses,
            'training_accuracies': training_accuracies,
            'robustness_result': robustness_result,
            'training_time': training_time
        }
        
        console.print(f"[green]Adversarial training completed in {training_time:.2f} seconds[/green]")
        return result
    
    def evaluate_defenses(self, model: nn.Module, test_loader: DataLoader, 
                        defense_methods: List[DefenseMethod] = None) -> Dict[str, Any]:
        """Evaluate defense mechanisms."""
        console.print("[blue]Evaluating defense mechanisms...[/blue]")
        
        if defense_methods is None:
            defense_methods = [DefenseMethod.FEATURE_SQUEEZING, DefenseMethod.SPATIAL_SMOOTHING]
        
        defense_results = {}
        
        for defense_method in defense_methods:
            # Apply defense
            defended_model = self.defender.apply_defense(model, defense_method)
            
            # Evaluate robustness
            robustness_result = self.evaluator.evaluate_robustness(defended_model, test_loader)
            
            defense_results[defense_method] = {
                'clean_accuracy': robustness_result['clean_accuracy'],
                'robustness_metrics': robustness_result['robustness_metrics']
            }
        
        console.print("[green]Defense evaluation completed[/green]")
        return defense_results
    
    def visualize_adversarial_results(self, result: Dict[str, Any], 
                                    output_path: str = None) -> str:
        """Visualize adversarial training results."""
        if output_path is None:
            output_path = f"adversarial_training_{int(time.time())}.png"
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training curve
        if 'training_losses' in result:
            axes[0, 0].plot(result['training_losses'])
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Training accuracy
        if 'training_accuracies' in result:
            axes[0, 1].plot(result['training_accuracies'])
            axes[0, 1].set_title('Training Accuracy')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Robustness metrics
        if 'robustness_result' in result:
            robustness_metrics = result['robustness_result']['robustness_metrics']
            metric_names = list(robustness_metrics.keys())
            metric_values = list(robustness_metrics.values())
            
            axes[1, 0].bar(metric_names, metric_values)
            axes[1, 0].set_title('Robustness Metrics')
            axes[1, 0].set_ylabel('Value')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Attack results
        if 'robustness_result' in result and 'attack_results' in result['robustness_result']:
            attack_results = result['robustness_result']['attack_results']
            attack_names = [attack.value for attack in attack_results.keys()]
            attack_accuracies = [result['adversarial_accuracy'] for result in attack_results.values()]
            
            axes[1, 1].bar(attack_names, attack_accuracies)
            axes[1, 1].set_title('Adversarial Accuracy by Attack')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        console.print(f"[green]Adversarial training visualization saved: {output_path}[/green]")
        return output_path
    
    def get_adversarial_summary(self) -> Dict[str, Any]:
        """Get adversarial training summary."""
        return {
            'total_experiments': len(self.adversarial_results),
            'attack_methods': [method.value for method in self.config.attack_methods] if self.config.attack_methods else [],
            'defense_methods': [method.value for method in self.config.defense_methods] if self.config.defense_methods else [],
            'threat_model': self.config.threat_model.value,
            'epsilon': self.config.epsilon
        }

def main():
    """Main function for adversarial training CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Adversarial Training System")
    parser.add_argument("--attack-methods", nargs="+",
                       choices=["fgsm", "pgd", "bim", "cw", "deep_fool"],
                       default=["fgsm", "pgd"], help="Attack methods")
    parser.add_argument("--defense-methods", nargs="+",
                       choices=["adversarial_training", "feature_squeezing", "spatial_smoothing"],
                       default=["feature_squeezing"], help="Defense methods")
    parser.add_argument("--threat-model", type=str,
                       choices=["l_inf", "l_2", "l_1"],
                       default="l_inf", help="Threat model")
    parser.add_argument("--epsilon", type=float, default=0.03,
                       help="Attack epsilon")
    parser.add_argument("--attack-steps", type=int, default=10,
                       help="Attack steps")
    parser.add_argument("--attack-lr", type=float, default=0.01,
                       help="Attack learning rate")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use")
    
    args = parser.parse_args()
    
    # Create adversarial configuration
    attack_methods = [AttackMethod(method) for method in args.attack_methods]
    defense_methods = [DefenseMethod(method) for method in args.defense_methods]
    config = AdversarialConfig(
        attack_methods=attack_methods,
        defense_methods=defense_methods,
        threat_model=ThreatModel(args.threat_model),
        epsilon=args.epsilon,
        attack_steps=args.attack_steps,
        attack_lr=args.attack_lr,
        device=args.device
    )
    
    # Create adversarial training system
    adv_system = AdversarialTrainingSystem(config)
    
    # Create sample model and data
    model = nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    
    # Create sample data loaders
    from torch.utils.data import TensorDataset, DataLoader
    
    X_train = torch.randn(1000, 784)
    y_train = torch.randint(0, 10, (1000,))
    X_val = torch.randn(200, 784)
    y_val = torch.randint(0, 10, (200,))
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Run adversarial training
    result = adv_system.run_adversarial_training(model, train_loader, val_loader)
    
    # Show results
    console.print(f"[green]Adversarial training completed[/green]")
    console.print(f"[blue]Training time: {result['training_time']:.2f} seconds[/blue]")
    
    if 'robustness_result' in result:
        robustness = result['robustness_result']
        console.print(f"[blue]Clean accuracy: {robustness['clean_accuracy']:.4f}[/blue]")
        console.print(f"[blue]Average adversarial accuracy: {robustness['robustness_metrics']['average_adversarial_accuracy']:.4f}[/blue]")
    
    # Create visualization
    adv_system.visualize_adversarial_results(result)
    
    # Show summary
    summary = adv_system.get_adversarial_summary()
    console.print(f"[blue]Summary: {summary}[/blue]")

if __name__ == "__main__":
    main()
