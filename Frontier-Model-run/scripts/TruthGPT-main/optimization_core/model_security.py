"""
Advanced Model Security System for TruthGPT Optimization Core
Complete model security with adversarial defense, privacy protection, and security analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
import logging
from dataclasses import dataclass, field
from enum import Enum
import time
import json
from pathlib import Path
import pickle
from abc import ABC, abstractmethod
import math
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    """Security levels"""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    MAXIMUM = "maximum"

class SecurityType(Enum):
    """Security types"""
    ADVERSARIAL_DEFENSE = "adversarial_defense"
    PRIVACY_PROTECTION = "privacy_protection"
    MODEL_WATERMARKING = "model_watermarking"
    INPUT_VALIDATION = "input_validation"
    OUTPUT_SANITIZATION = "output_sanitization"
    ACCESS_CONTROL = "access_control"

class DefenseStrategy(Enum):
    """Defense strategies"""
    ADVERSARIAL_TRAINING = "adversarial_training"
    INPUT_PREPROCESSING = "input_preprocessing"
    FEATURE_SQUEEZING = "feature_squeezing"
    DEFENSIVE_DISTILLATION = "defensive_distillation"
    CERTIFIED_DEFENSE = "certified_defense"
    ENSEMBLE_DEFENSE = "ensemble_defense"

class ModelSecurityConfig:
    """Configuration for model security system"""
    # Basic settings
    security_level: SecurityLevel = SecurityLevel.INTERMEDIATE
    security_type: SecurityType = SecurityType.ADVERSARIAL_DEFENSE
    defense_strategy: DefenseStrategy = DefenseStrategy.ADVERSARIAL_TRAINING
    
    # Adversarial defense settings
    adversarial_training_ratio: float = 0.3
    adversarial_epsilon: float = 0.03
    adversarial_steps: int = 10
    adversarial_step_size: float = 0.01
    
    # Privacy protection settings
    differential_privacy_epsilon: float = 1.0
    differential_privacy_delta: float = 1e-5
    noise_multiplier: float = 1.1
    max_grad_norm: float = 1.0
    
    # Model watermarking settings
    watermark_strength: float = 0.1
    watermark_pattern: str = "random"
    watermark_layers: List[int] = field(default_factory=lambda: [0, 1, 2])
    
    # Input validation settings
    input_range_min: float = 0.0
    input_range_max: float = 1.0
    input_shape_validation: bool = True
    input_type_validation: bool = True
    
    # Output sanitization settings
    output_clipping: bool = True
    output_normalization: bool = True
    output_threshold: float = 0.5
    
    # Access control settings
    authentication_required: bool = True
    authorization_levels: List[str] = field(default_factory=lambda: ["read", "write", "admin"])
    rate_limiting: bool = True
    max_requests_per_minute: int = 100
    
    # Advanced features
    enable_adversarial_defense: bool = True
    enable_privacy_protection: bool = True
    enable_model_watermarking: bool = True
    enable_input_validation: bool = True
    enable_output_sanitization: bool = True
    enable_access_control: bool = True
    
    def __post_init__(self):
        """Validate security configuration"""
        if not (0 <= self.adversarial_training_ratio <= 1):
            raise ValueError("Adversarial training ratio must be between 0 and 1")
        if self.adversarial_epsilon <= 0:
            raise ValueError("Adversarial epsilon must be positive")
        if self.adversarial_steps <= 0:
            raise ValueError("Adversarial steps must be positive")
        if self.adversarial_step_size <= 0:
            raise ValueError("Adversarial step size must be positive")
        if self.differential_privacy_epsilon <= 0:
            raise ValueError("Differential privacy epsilon must be positive")
        if not (0 <= self.differential_privacy_delta <= 1):
            raise ValueError("Differential privacy delta must be between 0 and 1")
        if self.noise_multiplier <= 0:
            raise ValueError("Noise multiplier must be positive")
        if self.max_grad_norm <= 0:
            raise ValueError("Max gradient norm must be positive")
        if self.watermark_strength <= 0:
            raise ValueError("Watermark strength must be positive")
        if self.input_range_min >= self.input_range_max:
            raise ValueError("Input range min must be less than max")
        if self.output_threshold <= 0:
            raise ValueError("Output threshold must be positive")
        if self.max_requests_per_minute <= 0:
            raise ValueError("Max requests per minute must be positive")

class AdversarialDefender:
    """Adversarial defense system"""
    
    def __init__(self, config: ModelSecurityConfig):
        self.config = config
        self.defense_history = []
        logger.info("✅ Adversarial Defender initialized")
    
    def generate_adversarial_examples(self, model: nn.Module, input_data: torch.Tensor,
                                   target: torch.Tensor) -> torch.Tensor:
        """Generate adversarial examples"""
        logger.info("🔍 Generating adversarial examples")
        
        model.eval()
        adversarial_input = input_data.clone().detach().requires_grad_(True)
        
        for step in range(self.config.adversarial_steps):
            # Forward pass
            output = model(adversarial_input)
            loss = F.cross_entropy(output, target)
            
            # Backward pass
            model.zero_grad()
            loss.backward()
            
            # Update adversarial input
            with torch.no_grad():
                adversarial_input += self.config.adversarial_step_size * adversarial_input.grad.sign()
                adversarial_input = torch.clamp(adversarial_input, 
                                               input_data - self.config.adversarial_epsilon,
                                               input_data + self.config.adversarial_epsilon)
                adversarial_input = torch.clamp(adversarial_input, 0, 1)
            
            adversarial_input.grad.zero_()
        
        return adversarial_input.detach()
    
    def adversarial_training(self, model: nn.Module, input_data: torch.Tensor,
                           target: torch.Tensor) -> Dict[str, Any]:
        """Adversarial training"""
        logger.info("🔍 Performing adversarial training")
        
        # Generate adversarial examples
        adversarial_input = self.generate_adversarial_examples(model, input_data, target)
        
        # Mix clean and adversarial data
        if random.random() < self.config.adversarial_training_ratio:
            training_input = adversarial_input
            training_target = target
        else:
            training_input = input_data
            training_target = target
        
        # Forward pass
        model.train()
        output = model(training_input)
        loss = F.cross_entropy(output, training_target)
        
        # Calculate accuracy
        with torch.no_grad():
            clean_output = model(input_data)
            clean_accuracy = accuracy_score(target.cpu().numpy(), 
                                          torch.argmax(clean_output, dim=1).cpu().numpy())
            
            adv_output = model(adversarial_input)
            adv_accuracy = accuracy_score(target.cpu().numpy(), 
                                        torch.argmax(adv_output, dim=1).cpu().numpy())
        
        defense_results = {
            'loss': loss.item(),
            'clean_accuracy': clean_accuracy,
            'adversarial_accuracy': adv_accuracy,
            'robustness_gap': clean_accuracy - adv_accuracy,
            'adversarial_input': adversarial_input,
            'training_input': training_input
        }
        
        # Store defense history
        self.defense_history.append(defense_results)
        
        return defense_results
    
    def input_preprocessing(self, input_data: torch.Tensor) -> torch.Tensor:
        """Input preprocessing for defense"""
        logger.info("🔍 Applying input preprocessing")
        
        # Feature squeezing
        squeezed_input = self._feature_squeezing(input_data)
        
        # Input normalization
        normalized_input = self._input_normalization(squeezed_input)
        
        return normalized_input
    
    def _feature_squeezing(self, input_data: torch.Tensor) -> torch.Tensor:
        """Feature squeezing defense"""
        # Reduce bit depth
        squeezed = torch.round(input_data * 255) / 255
        
        # Apply median filter (simplified)
        if len(squeezed.shape) == 4:  # Batch of images
            squeezed = F.avg_pool2d(squeezed, kernel_size=2, stride=1, padding=1)
        
        return squeezed
    
    def _input_normalization(self, input_data: torch.Tensor) -> torch.Tensor:
        """Input normalization"""
        # Normalize to [0, 1]
        normalized = (input_data - input_data.min()) / (input_data.max() - input_data.min() + 1e-8)
        
        return normalized

class PrivacyProtector:
    """Privacy protection system"""
    
    def __init__(self, config: ModelSecurityConfig):
        self.config = config
        self.privacy_history = []
        logger.info("✅ Privacy Protector initialized")
    
    def add_differential_privacy_noise(self, gradients: List[torch.Tensor]) -> List[torch.Tensor]:
        """Add differential privacy noise to gradients"""
        logger.info("🔍 Adding differential privacy noise")
        
        noisy_gradients = []
        
        for grad in gradients:
            if grad is not None:
                # Calculate noise scale
                noise_scale = self.config.noise_multiplier * self.config.max_grad_norm
                
                # Add Gaussian noise
                noise = torch.normal(0, noise_scale, size=grad.shape, device=grad.device)
                noisy_grad = grad + noise
                
                # Clip gradients
                noisy_grad = torch.clamp(noisy_grad, -self.config.max_grad_norm, 
                                       self.config.max_grad_norm)
                
                noisy_gradients.append(noisy_grad)
            else:
                noisy_gradients.append(None)
        
        return noisy_gradients
    
    def calculate_privacy_budget(self, num_samples: int, num_epochs: int) -> Dict[str, float]:
        """Calculate privacy budget"""
        logger.info("🔍 Calculating privacy budget")
        
        # Simplified privacy budget calculation
        total_epsilon = self.config.differential_privacy_epsilon * num_samples * num_epochs
        total_delta = self.config.differential_privacy_delta * num_samples * num_epochs
        
        privacy_budget = {
            'total_epsilon': total_epsilon,
            'total_delta': total_delta,
            'per_sample_epsilon': self.config.differential_privacy_epsilon,
            'per_sample_delta': self.config.differential_privacy_delta,
            'noise_multiplier': self.config.noise_multiplier
        }
        
        # Store privacy history
        self.privacy_history.append(privacy_budget)
        
        return privacy_budget
    
    def federated_learning_privacy(self, local_models: List[nn.Module]) -> nn.Module:
        """Federated learning with privacy protection"""
        logger.info("🔍 Applying federated learning privacy protection")
        
        # Aggregate models with privacy protection
        aggregated_model = self._aggregate_models_with_privacy(local_models)
        
        return aggregated_model
    
    def _aggregate_models_with_privacy(self, local_models: List[nn.Module]) -> nn.Module:
        """Aggregate models with privacy protection"""
        if not local_models:
            return None
        
        # Use first model as base
        aggregated_model = local_models[0]
        
        # Add noise to aggregated parameters
        for name, param in aggregated_model.named_parameters():
            # Calculate average parameter
            avg_param = torch.stack([model.state_dict()[name] for model in local_models]).mean(dim=0)
            
            # Add privacy noise
            noise_scale = self.config.noise_multiplier * self.config.max_grad_norm
            noise = torch.normal(0, noise_scale, size=avg_param.shape, device=avg_param.device)
            
            # Update parameter
            param.data = avg_param + noise
        
        return aggregated_model

class ModelWatermarker:
    """Model watermarking system"""
    
    def __init__(self, config: ModelSecurityConfig):
        self.config = config
        self.watermark_history = []
        logger.info("✅ Model Watermarker initialized")
    
    def embed_watermark(self, model: nn.Module, watermark_data: torch.Tensor = None) -> nn.Module:
        """Embed watermark in model"""
        logger.info("🔍 Embedding watermark in model")
        
        watermarked_model = model
        
        if watermark_data is None:
            watermark_data = self._generate_watermark_pattern()
        
        # Embed watermark in specified layers
        for layer_idx in self.config.watermark_layers:
            watermarked_model = self._embed_watermark_in_layer(watermarked_model, 
                                                            layer_idx, watermark_data)
        
        # Store watermark history
        self.watermark_history.append({
            'watermark_data': watermark_data,
            'watermark_layers': self.config.watermark_layers,
            'watermark_strength': self.config.watermark_strength
        })
        
        return watermarked_model
    
    def _generate_watermark_pattern(self) -> torch.Tensor:
        """Generate watermark pattern"""
        if self.config.watermark_pattern == "random":
            watermark = torch.randn(100) * self.config.watermark_strength
        elif self.config.watermark_pattern == "zeros":
            watermark = torch.zeros(100)
        elif self.config.watermark_pattern == "ones":
            watermark = torch.ones(100) * self.config.watermark_strength
        else:
            watermark = torch.randn(100) * self.config.watermark_strength
        
        return watermark
    
    def _embed_watermark_in_layer(self, model: nn.Module, layer_idx: int, 
                                 watermark_data: torch.Tensor) -> nn.Module:
        """Embed watermark in specific layer"""
        layer_count = 0
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                if layer_count == layer_idx:
                    # Add watermark to weights
                    with torch.no_grad():
                        watermark_reshaped = watermark_data[:module.weight.numel()].reshape(
                            module.weight.shape)
                        module.weight.data += watermark_reshaped * self.config.watermark_strength
                    break
                layer_count += 1
        
        return model
    
    def detect_watermark(self, model: nn.Module, original_model: nn.Module) -> Dict[str, Any]:
        """Detect watermark in model"""
        logger.info("🔍 Detecting watermark in model")
        
        watermark_detection = {}
        
        for name, (param, orig_param) in zip(model.named_parameters(), 
                                            original_model.named_parameters()):
            # Calculate difference
            diff = torch.abs(param - orig_param)
            
            watermark_detection[name] = {
                'mean_difference': torch.mean(diff).item(),
                'max_difference': torch.max(diff).item(),
                'std_difference': torch.std(diff).item(),
                'watermark_detected': torch.mean(diff).item() > self.config.watermark_strength * 0.1
            }
        
        return watermark_detection

class InputValidator:
    """Input validation system"""
    
    def __init__(self, config: ModelSecurityConfig):
        self.config = config
        self.validation_history = []
        logger.info("✅ Input Validator initialized")
    
    def validate_input(self, input_data: torch.Tensor) -> Dict[str, Any]:
        """Validate input data"""
        logger.info("🔍 Validating input data")
        
        validation_results = {
            'is_valid': True,
            'issues': [],
            'warnings': []
        }
        
        # Shape validation
        if self.config.input_shape_validation:
            if len(input_data.shape) < 2:
                validation_results['issues'].append("Input must have at least 2 dimensions")
                validation_results['is_valid'] = False
        
        # Type validation
        if self.config.input_type_validation:
            if not isinstance(input_data, torch.Tensor):
                validation_results['issues'].append("Input must be a torch.Tensor")
                validation_results['is_valid'] = False
        
        # Range validation
        if input_data.min().item() < self.config.input_range_min:
            validation_results['warnings'].append(f"Input values below minimum ({self.config.input_range_min})")
        
        if input_data.max().item() > self.config.input_range_max:
            validation_results['warnings'].append(f"Input values above maximum ({self.config.input_range_max})")
        
        # NaN and Inf check
        if torch.isnan(input_data).any():
            validation_results['issues'].append("Input contains NaN values")
            validation_results['is_valid'] = False
        
        if torch.isinf(input_data).any():
            validation_results['issues'].append("Input contains infinite values")
            validation_results['is_valid'] = False
        
        # Store validation history
        self.validation_history.append(validation_results)
        
        return validation_results
    
    def sanitize_input(self, input_data: torch.Tensor) -> torch.Tensor:
        """Sanitize input data"""
        logger.info("🔍 Sanitizing input data")
        
        # Clamp to valid range
        sanitized = torch.clamp(input_data, self.config.input_range_min, 
                              self.config.input_range_max)
        
        # Replace NaN and Inf
        sanitized = torch.nan_to_num(sanitized, nan=0.0, posinf=1.0, neginf=0.0)
        
        return sanitized

class OutputSanitizer:
    """Output sanitization system"""
    
    def __init__(self, config: ModelSecurityConfig):
        self.config = config
        self.sanitization_history = []
        logger.info("✅ Output Sanitizer initialized")
    
    def sanitize_output(self, output: torch.Tensor) -> torch.Tensor:
        """Sanitize output data"""
        logger.info("🔍 Sanitizing output data")
        
        sanitized_output = output.clone()
        
        # Clipping
        if self.config.output_clipping:
            sanitized_output = torch.clamp(sanitized_output, 0, 1)
        
        # Normalization
        if self.config.output_normalization:
            sanitized_output = F.softmax(sanitized_output, dim=-1)
        
        # Thresholding
        if self.config.output_threshold > 0:
            sanitized_output = torch.where(sanitized_output > self.config.output_threshold,
                                         sanitized_output, torch.zeros_like(sanitized_output))
        
        # Store sanitization history
        self.sanitization_history.append({
            'original_output': output,
            'sanitized_output': sanitized_output,
            'clipping_applied': self.config.output_clipping,
            'normalization_applied': self.config.output_normalization,
            'threshold_applied': self.config.output_threshold
        })
        
        return sanitized_output

class AccessController:
    """Access control system"""
    
    def __init__(self, config: ModelSecurityConfig):
        self.config = config
        self.access_history = []
        self.rate_limiter = {}
        logger.info("✅ Access Controller initialized")
    
    def authenticate_user(self, user_id: str, credentials: Dict[str, Any]) -> bool:
        """Authenticate user"""
        logger.info(f"🔍 Authenticating user: {user_id}")
        
        # Simplified authentication
        if self.config.authentication_required:
            # Check credentials (simplified)
            if 'password' in credentials and len(credentials['password']) >= 8:
                authenticated = True
            else:
                authenticated = False
        else:
            authenticated = True
        
        # Store authentication history
        self.access_history.append({
            'user_id': user_id,
            'authenticated': authenticated,
            'timestamp': time.time()
        })
        
        return authenticated
    
    def authorize_access(self, user_id: str, requested_level: str) -> bool:
        """Authorize user access"""
        logger.info(f"🔍 Authorizing access for user: {user_id}")
        
        # Check if requested level is in authorized levels
        authorized = requested_level in self.config.authorization_levels
        
        return authorized
    
    def check_rate_limit(self, user_id: str) -> bool:
        """Check rate limit for user"""
        logger.info(f"🔍 Checking rate limit for user: {user_id}")
        
        if not self.config.rate_limiting:
            return True
        
        current_time = time.time()
        
        # Initialize rate limiter for user
        if user_id not in self.rate_limiter:
            self.rate_limiter[user_id] = []
        
        # Remove old requests
        self.rate_limiter[user_id] = [
            req_time for req_time in self.rate_limiter[user_id]
            if current_time - req_time < 60  # Last minute
        ]
        
        # Check if under limit
        if len(self.rate_limiter[user_id]) < self.config.max_requests_per_minute:
            self.rate_limiter[user_id].append(current_time)
            return True
        else:
            return False

class ModelSecuritySystem:
    """Main model security system"""
    
    def __init__(self, config: ModelSecurityConfig):
        self.config = config
        
        # Components
        self.adversarial_defender = AdversarialDefender(config)
        self.privacy_protector = PrivacyProtector(config)
        self.model_watermarker = ModelWatermarker(config)
        self.input_validator = InputValidator(config)
        self.output_sanitizer = OutputSanitizer(config)
        self.access_controller = AccessController(config)
        
        # Security state
        self.security_history = []
        
        logger.info("✅ Model Security System initialized")
    
    def secure_model(self, model: nn.Module, input_data: torch.Tensor,
                    target: torch.Tensor = None) -> Dict[str, Any]:
        """Secure model"""
        logger.info(f"🔍 Securing model using {self.config.security_level.value} level")
        
        security_results = {
            'start_time': time.time(),
            'config': self.config,
            'security_results': {}
        }
        
        # Input validation
        if self.config.enable_input_validation:
            logger.info("🔍 Stage 1: Input validation")
            
            validation_results = self.input_validator.validate_input(input_data)
            security_results['security_results']['input_validation'] = validation_results
            
            if validation_results['is_valid']:
                input_data = self.input_validator.sanitize_input(input_data)
        
        # Adversarial defense
        if self.config.enable_adversarial_defense and target is not None:
            logger.info("🔍 Stage 2: Adversarial defense")
            
            defense_results = self.adversarial_defender.adversarial_training(model, input_data, target)
            security_results['security_results']['adversarial_defense'] = defense_results
        
        # Privacy protection
        if self.config.enable_privacy_protection:
            logger.info("🔍 Stage 3: Privacy protection")
            
            # Get gradients for privacy protection
            model.train()
            output = model(input_data)
            loss = F.cross_entropy(output, target) if target is not None else torch.tensor(0.0)
            loss.backward()
            
            gradients = [param.grad for param in model.parameters()]
            noisy_gradients = self.privacy_protector.add_differential_privacy_noise(gradients)
            
            privacy_budget = self.privacy_protector.calculate_privacy_budget(1, 1)
            
            security_results['security_results']['privacy_protection'] = {
                'noisy_gradients': noisy_gradients,
                'privacy_budget': privacy_budget
            }
        
        # Model watermarking
        if self.config.enable_model_watermarking:
            logger.info("🔍 Stage 4: Model watermarking")
            
            watermarked_model = self.model_watermarker.embed_watermark(model)
            watermark_detection = self.model_watermarker.detect_watermark(watermarked_model, model)
            
            security_results['security_results']['model_watermarking'] = {
                'watermarked_model': watermarked_model,
                'watermark_detection': watermark_detection
            }
        
        # Output sanitization
        if self.config.enable_output_sanitization:
            logger.info("🔍 Stage 5: Output sanitization")
            
            model.eval()
            with torch.no_grad():
                output = model(input_data)
            
            sanitized_output = self.output_sanitizer.sanitize_output(output)
            
            security_results['security_results']['output_sanitization'] = {
                'original_output': output,
                'sanitized_output': sanitized_output
            }
        
        # Access control
        if self.config.enable_access_control:
            logger.info("🔍 Stage 6: Access control")
            
            # Simulate user authentication and authorization
            user_id = "test_user"
            credentials = {"password": "secure_password123"}
            
            authenticated = self.access_controller.authenticate_user(user_id, credentials)
            authorized = self.access_controller.authorize_access(user_id, "read")
            rate_limited = self.access_controller.check_rate_limit(user_id)
            
            security_results['security_results']['access_control'] = {
                'authenticated': authenticated,
                'authorized': authorized,
                'rate_limited': rate_limited
            }
        
        # Final evaluation
        security_results['end_time'] = time.time()
        security_results['total_duration'] = security_results['end_time'] - security_results['start_time']
        
        # Store results
        self.security_history.append(security_results)
        
        logger.info("✅ Model security completed")
        return security_results
    
    def generate_security_report(self, security_results: Dict[str, Any]) -> str:
        """Generate security report"""
        logger.info("📋 Generating security report")
        
        report = []
        report.append("=" * 60)
        report.append("MODEL SECURITY ANALYSIS REPORT")
        report.append("=" * 60)
        
        # Configuration
        report.append("\nSECURITY CONFIGURATION:")
        report.append("-" * 22)
        report.append(f"Security Level: {self.config.security_level.value}")
        report.append(f"Security Type: {self.config.security_type.value}")
        report.append(f"Defense Strategy: {self.config.defense_strategy.value}")
        report.append(f"Adversarial Training Ratio: {self.config.adversarial_training_ratio}")
        report.append(f"Adversarial Epsilon: {self.config.adversarial_epsilon}")
        report.append(f"Adversarial Steps: {self.config.adversarial_steps}")
        report.append(f"Adversarial Step Size: {self.config.adversarial_step_size}")
        report.append(f"Differential Privacy Epsilon: {self.config.differential_privacy_epsilon}")
        report.append(f"Differential Privacy Delta: {self.config.differential_privacy_delta}")
        report.append(f"Noise Multiplier: {self.config.noise_multiplier}")
        report.append(f"Max Gradient Norm: {self.config.max_grad_norm}")
        report.append(f"Watermark Strength: {self.config.watermark_strength}")
        report.append(f"Watermark Pattern: {self.config.watermark_pattern}")
        report.append(f"Watermark Layers: {self.config.watermark_layers}")
        report.append(f"Input Range Min: {self.config.input_range_min}")
        report.append(f"Input Range Max: {self.config.input_range_max}")
        report.append(f"Input Shape Validation: {'Enabled' if self.config.input_shape_validation else 'Disabled'}")
        report.append(f"Input Type Validation: {'Enabled' if self.config.input_type_validation else 'Disabled'}")
        report.append(f"Output Clipping: {'Enabled' if self.config.output_clipping else 'Disabled'}")
        report.append(f"Output Normalization: {'Enabled' if self.config.output_normalization else 'Disabled'}")
        report.append(f"Output Threshold: {self.config.output_threshold}")
        report.append(f"Authentication Required: {'Enabled' if self.config.authentication_required else 'Disabled'}")
        report.append(f"Authorization Levels: {self.config.authorization_levels}")
        report.append(f"Rate Limiting: {'Enabled' if self.config.rate_limiting else 'Disabled'}")
        report.append(f"Max Requests Per Minute: {self.config.max_requests_per_minute}")
        report.append(f"Adversarial Defense: {'Enabled' if self.config.enable_adversarial_defense else 'Disabled'}")
        report.append(f"Privacy Protection: {'Enabled' if self.config.enable_privacy_protection else 'Disabled'}")
        report.append(f"Model Watermarking: {'Enabled' if self.config.enable_model_watermarking else 'Disabled'}")
        report.append(f"Input Validation: {'Enabled' if self.config.enable_input_validation else 'Disabled'}")
        report.append(f"Output Sanitization: {'Enabled' if self.config.enable_output_sanitization else 'Disabled'}")
        report.append(f"Access Control: {'Enabled' if self.config.enable_access_control else 'Disabled'}")
        
        # Security results
        report.append("\nSECURITY RESULTS:")
        report.append("-" * 17)
        
        for method, results in security_results.get('security_results', {}).items():
            report.append(f"\n{method.upper()}:")
            report.append("-" * len(method))
            
            if isinstance(results, dict):
                for key, value in results.items():
                    if isinstance(value, (list, tuple)) and len(value) > 5:
                        report.append(f"  {key}: {type(value).__name__} with {len(value)} items")
                    elif isinstance(value, dict) and len(value) > 5:
                        report.append(f"  {key}: Dict with {len(value)} items")
                    else:
                        report.append(f"  {key}: {value}")
            else:
                report.append(f"  Results: {results}")
        
        # Summary
        report.append("\nSUMMARY:")
        report.append("-" * 8)
        report.append(f"Total Duration: {security_results.get('total_duration', 0):.2f} seconds")
        report.append(f"Security History Length: {len(self.security_history)}")
        
        return "\n".join(report)

# Factory functions
def create_security_config(**kwargs) -> ModelSecurityConfig:
    """Create security configuration"""
    return ModelSecurityConfig(**kwargs)

def create_adversarial_defender(config: ModelSecurityConfig) -> AdversarialDefender:
    """Create adversarial defender"""
    return AdversarialDefender(config)

def create_privacy_protector(config: ModelSecurityConfig) -> PrivacyProtector:
    """Create privacy protector"""
    return PrivacyProtector(config)

def create_model_watermarker(config: ModelSecurityConfig) -> ModelWatermarker:
    """Create model watermarker"""
    return ModelWatermarker(config)

def create_input_validator(config: ModelSecurityConfig) -> InputValidator:
    """Create input validator"""
    return InputValidator(config)

def create_output_sanitizer(config: ModelSecurityConfig) -> OutputSanitizer:
    """Create output sanitizer"""
    return OutputSanitizer(config)

def create_access_controller(config: ModelSecurityConfig) -> AccessController:
    """Create access controller"""
    return AccessController(config)

def create_model_security_system(config: ModelSecurityConfig) -> ModelSecuritySystem:
    """Create model security system"""
    return ModelSecuritySystem(config)

# Example usage
def example_model_security():
    """Example of model security system"""
    # Create configuration
    config = create_security_config(
        security_level=SecurityLevel.INTERMEDIATE,
        security_type=SecurityType.ADVERSARIAL_DEFENSE,
        defense_strategy=DefenseStrategy.ADVERSARIAL_TRAINING,
        adversarial_training_ratio=0.3,
        adversarial_epsilon=0.03,
        adversarial_steps=10,
        adversarial_step_size=0.01,
        differential_privacy_epsilon=1.0,
        differential_privacy_delta=1e-5,
        noise_multiplier=1.1,
        max_grad_norm=1.0,
        watermark_strength=0.1,
        watermark_pattern="random",
        watermark_layers=[0, 1, 2],
        input_range_min=0.0,
        input_range_max=1.0,
        input_shape_validation=True,
        input_type_validation=True,
        output_clipping=True,
        output_normalization=True,
        output_threshold=0.5,
        authentication_required=True,
        authorization_levels=["read", "write", "admin"],
        rate_limiting=True,
        max_requests_per_minute=100,
        enable_adversarial_defense=True,
        enable_privacy_protection=True,
        enable_model_watermarking=True,
        enable_input_validation=True,
        enable_output_sanitization=True,
        enable_access_control=True
    )
    
    # Create model security system
    security_system = create_model_security_system(config)
    
    # Create dummy model
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(64, 10)
    )
    
    # Generate dummy data
    input_data = torch.randn(1, 3, 32, 32)
    target = torch.randint(0, 10, (1,))
    
    # Secure model
    security_results = security_system.secure_model(model, input_data, target)
    
    # Generate report
    security_report = security_system.generate_security_report(security_results)
    
    print(f"✅ Model Security Example Complete!")
    print(f"🚀 Model Security Statistics:")
    print(f"   Security Level: {config.security_level.value}")
    print(f"   Security Type: {config.security_type.value}")
    print(f"   Defense Strategy: {config.defense_strategy.value}")
    print(f"   Adversarial Training Ratio: {config.adversarial_training_ratio}")
    print(f"   Adversarial Epsilon: {config.adversarial_epsilon}")
    print(f"   Adversarial Steps: {config.adversarial_steps}")
    print(f"   Adversarial Step Size: {config.adversarial_step_size}")
    print(f"   Differential Privacy Epsilon: {config.differential_privacy_epsilon}")
    print(f"   Differential Privacy Delta: {config.differential_privacy_delta}")
    print(f"   Noise Multiplier: {config.noise_multiplier}")
    print(f"   Max Gradient Norm: {config.max_grad_norm}")
    print(f"   Watermark Strength: {config.watermark_strength}")
    print(f"   Watermark Pattern: {config.watermark_pattern}")
    print(f"   Watermark Layers: {config.watermark_layers}")
    print(f"   Input Range Min: {config.input_range_min}")
    print(f"   Input Range Max: {config.input_range_max}")
    print(f"   Input Shape Validation: {'Enabled' if config.input_shape_validation else 'Disabled'}")
    print(f"   Input Type Validation: {'Enabled' if config.input_type_validation else 'Disabled'}")
    print(f"   Output Clipping: {'Enabled' if config.output_clipping else 'Disabled'}")
    print(f"   Output Normalization: {'Enabled' if config.output_normalization else 'Disabled'}")
    print(f"   Output Threshold: {config.output_threshold}")
    print(f"   Authentication Required: {'Enabled' if config.authentication_required else 'Disabled'}")
    print(f"   Authorization Levels: {config.authorization_levels}")
    print(f"   Rate Limiting: {'Enabled' if config.rate_limiting else 'Disabled'}")
    print(f"   Max Requests Per Minute: {config.max_requests_per_minute}")
    print(f"   Adversarial Defense: {'Enabled' if config.enable_adversarial_defense else 'Disabled'}")
    print(f"   Privacy Protection: {'Enabled' if config.enable_privacy_protection else 'Disabled'}")
    print(f"   Model Watermarking: {'Enabled' if config.enable_model_watermarking else 'Disabled'}")
    print(f"   Input Validation: {'Enabled' if config.enable_input_validation else 'Disabled'}")
    print(f"   Output Sanitization: {'Enabled' if config.enable_output_sanitization else 'Disabled'}")
    print(f"   Access Control: {'Enabled' if config.enable_access_control else 'Disabled'}")
    
    print(f"\n📊 Model Security Results:")
    print(f"   Security History Length: {len(security_system.security_history)}")
    print(f"   Total Duration: {security_results.get('total_duration', 0):.2f} seconds")
    
    # Show security results summary
    if 'security_results' in security_results:
        print(f"   Number of Security Methods: {len(security_results['security_results'])}")
    
    print(f"\n📋 Model Security Report:")
    print(security_report)
    
    return security_system

# Export utilities
__all__ = [
    'SecurityLevel',
    'SecurityType',
    'DefenseStrategy',
    'ModelSecurityConfig',
    'AdversarialDefender',
    'PrivacyProtector',
    'ModelWatermarker',
    'InputValidator',
    'OutputSanitizer',
    'AccessController',
    'ModelSecuritySystem',
    'create_security_config',
    'create_adversarial_defender',
    'create_privacy_protector',
    'create_model_watermarker',
    'create_input_validator',
    'create_output_sanitizer',
    'create_access_controller',
    'create_model_security_system',
    'example_model_security'
]

if __name__ == "__main__":
    example_model_security()
    print("✅ Model security example completed successfully!")