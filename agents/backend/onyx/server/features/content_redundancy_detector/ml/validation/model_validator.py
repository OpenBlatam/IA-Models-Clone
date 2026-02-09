"""
Model Validator
Advanced model validation
"""

import torch
import torch.nn as nn
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class ModelValidator:
    """
    Advanced model validation
    """
    
    @staticmethod
    def validate_model_architecture(
        model: nn.Module,
        input_shape: tuple,
    ) -> Dict[str, Any]:
        """
        Validate model architecture
        
        Args:
            model: Model to validate
            input_shape: Input shape
            
        Returns:
            Dictionary with validation results
        """
        issues = []
        warnings = []
        
        # Test forward pass
        try:
            model.eval()
            dummy_input = torch.randn(input_shape)
            with torch.no_grad():
                output = model(dummy_input)
            
            if output.shape[0] != input_shape[0]:
                issues.append(f"Batch size mismatch: input {input_shape[0]}, output {output.shape[0]}")
        except Exception as e:
            issues.append(f"Forward pass failed: {str(e)}")
        
        # Check parameters
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                issues.append(f"NaN in parameter: {name}")
            
            if torch.isinf(param).any():
                issues.append(f"Inf in parameter: {name}")
            
            if param.requires_grad and param.grad is None:
                warnings.append(f"No gradient for parameter: {name}")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
        }
    
    @staticmethod
    def validate_model_output(
        model: nn.Module,
        inputs: torch.Tensor,
        expected_shape: tuple = None,
    ) -> Dict[str, Any]:
        """
        Validate model output
        
        Args:
            model: Model to validate
            inputs: Input tensor
            expected_shape: Expected output shape (optional)
            
        Returns:
            Dictionary with validation results
        """
        model.eval()
        with torch.no_grad():
            outputs = model(inputs)
        
        issues = []
        
        if torch.isnan(outputs).any():
            issues.append("Output contains NaN")
        
        if torch.isinf(outputs).any():
            issues.append("Output contains Inf")
        
        if expected_shape and outputs.shape != expected_shape:
            issues.append(f"Shape mismatch: expected {expected_shape}, got {outputs.shape}")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'output_shape': list(outputs.shape),
        }



