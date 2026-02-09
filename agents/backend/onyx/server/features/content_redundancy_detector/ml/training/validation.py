"""
Training Validation
Validation utilities for training data and configurations
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class TrainingValidator:
    """
    Validates training setup and data
    """
    
    @staticmethod
    def validate_model(model: nn.Module) -> Dict[str, Any]:
        """
        Validate model structure
        
        Args:
            model: Model to validate
            
        Returns:
            Validation results
        """
        issues = []
        warnings = []
        
        # Check for empty modules
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:
                if not list(module.parameters()):
                    warnings.append(f"Module {name} has no parameters")
        
        # Check parameter initialization
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                issues.append(f"NaN in parameter {name}")
            if torch.isinf(param).any():
                issues.append(f"Inf in parameter {name}")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
        }
    
    @staticmethod
    def validate_data_loader(
        data_loader: torch.utils.data.DataLoader,
        num_samples: int = 5,
    ) -> Dict[str, Any]:
        """
        Validate data loader
        
        Args:
            data_loader: DataLoader to validate
            num_samples: Number of samples to check
            
        Returns:
            Validation results
        """
        issues = []
        warnings = []
        
        try:
            # Try to get a batch
            batch = next(iter(data_loader))
            inputs, targets = batch
            
            # Check shapes
            if inputs.dim() not in [3, 4]:
                issues.append(f"Invalid input dimensions: {inputs.dim()}")
            
            if targets.dim() not in [0, 1]:
                issues.append(f"Invalid target dimensions: {targets.dim()}")
            
            # Check for NaN/Inf
            if torch.isnan(inputs).any():
                issues.append("NaN values in inputs")
            
            if torch.isinf(inputs).any():
                issues.append("Inf values in inputs")
            
            # Check value ranges
            if inputs.min() < 0 or inputs.max() > 1.1:
                warnings.append("Input values outside [0, 1] range")
            
        except Exception as e:
            issues.append(f"Error loading data: {str(e)}")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
        }
    
    @staticmethod
    def validate_training_config(
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Validate training configuration
        
        Args:
            config: Training configuration
            
        Returns:
            Validation results
        """
        issues = []
        warnings = []
        
        # Check required fields
        required_fields = ['learning_rate', 'batch_size', 'num_epochs']
        for field in required_fields:
            if field not in config:
                issues.append(f"Missing required field: {field}")
        
        # Check value ranges
        if 'learning_rate' in config:
            lr = config['learning_rate']
            if lr <= 0 or lr > 1:
                issues.append(f"Invalid learning rate: {lr}")
        
        if 'batch_size' in config:
            bs = config['batch_size']
            if bs <= 0:
                issues.append(f"Invalid batch size: {bs}")
        
        if 'num_epochs' in config:
            epochs = config['num_epochs']
            if epochs <= 0:
                issues.append(f"Invalid number of epochs: {epochs}")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
        }



