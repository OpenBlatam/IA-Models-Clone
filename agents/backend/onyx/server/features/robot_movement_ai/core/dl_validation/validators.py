"""
Model Validators - Modular Validation
=====================================

Validación modular para modelos y datos.
"""

import logging
from typing import Dict, Any, Optional, List, Callable
import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger(__name__)


class ModelValidator:
    """Validador de modelos."""
    
    def __init__(self, model: nn.Module):
        """
        Inicializar validador.
        
        Args:
            model: Modelo a validar
        """
        self.model = model
    
    def validate_architecture(self) -> Dict[str, Any]:
        """
        Validar arquitectura del modelo.
        
        Returns:
            Resultados de validación
        """
        issues = []
        warnings = []
        
        # Verificar que todos los módulos tienen forward
        for name, module in self.model.named_modules():
            if not hasattr(module, 'forward'):
                issues.append(f"Module {name} missing forward method")
        
        # Verificar parámetros
        num_params = sum(p.numel() for p in self.model.parameters())
        if num_params == 0:
            issues.append("Model has no parameters")
        
        # Verificar parámetros trainables
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        if trainable_params == 0:
            warnings.append("Model has no trainable parameters")
        
        # Verificar NaN/Inf en parámetros
        for name, param in self.model.named_parameters():
            if torch.isnan(param).any():
                issues.append(f"Parameter {name} contains NaN")
            if torch.isinf(param).any():
                issues.append(f"Parameter {name} contains Inf")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'num_parameters': num_params,
            'trainable_parameters': trainable_params
        }
    
    def validate_input_shape(
        self,
        input_shape: tuple,
        expected_output_shape: Optional[tuple] = None
    ) -> Dict[str, Any]:
        """
        Validar forma de entrada.
        
        Args:
            input_shape: Forma de entrada
            expected_output_shape: Forma de salida esperada (opcional)
            
        Returns:
            Resultados de validación
        """
        try:
            dummy_input = torch.randn(1, *input_shape)
            self.model.eval()
            
            with torch.no_grad():
                output = self.model(dummy_input)
            
            result = {
                'valid': True,
                'input_shape': input_shape,
                'output_shape': tuple(output.shape[1:])  # Sin batch dimension
            }
            
            if expected_output_shape:
                result['expected_output_shape'] = expected_output_shape
                result['shape_match'] = output.shape[1:] == expected_output_shape
            
            return result
        except Exception as e:
            return {
                'valid': False,
                'error': str(e)
            }
    
    def validate_gradients(
        self,
        input_shape: tuple,
        loss_fn: Optional[nn.Module] = None
    ) -> Dict[str, Any]:
        """
        Validar gradientes.
        
        Args:
            input_shape: Forma de entrada
            loss_fn: Función de pérdida
            
        Returns:
            Resultados de validación
        """
        try:
            if loss_fn is None:
                loss_fn = nn.MSELoss()
            
            self.model.train()
            dummy_input = torch.randn(1, *input_shape)
            dummy_target = torch.randn_like(self.model(dummy_input))
            
            output = self.model(dummy_input)
            loss = loss_fn(output, dummy_target)
            loss.backward()
            
            # Verificar gradientes
            issues = []
            for name, param in self.model.named_parameters():
                if param.grad is None:
                    issues.append(f"Parameter {name} has no gradient")
                elif torch.isnan(param.grad).any():
                    issues.append(f"Parameter {name} has NaN gradients")
                elif torch.isinf(param.grad).any():
                    issues.append(f"Parameter {name} has Inf gradients")
                elif param.grad.abs().sum() == 0:
                    issues.append(f"Parameter {name} has zero gradients")
            
            return {
                'valid': len(issues) == 0,
                'issues': issues,
                'loss': loss.item()
            }
        except Exception as e:
            return {
                'valid': False,
                'error': str(e)
            }


class DataValidator:
    """Validador de datos."""
    
    @staticmethod
    def validate_tensor(
        tensor: torch.Tensor,
        check_nan: bool = True,
        check_inf: bool = True,
        check_range: bool = False,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Validar tensor.
        
        Args:
            tensor: Tensor a validar
            check_nan: Verificar NaN
            check_inf: Verificar Inf
            check_range: Verificar rango
            min_val: Valor mínimo
            max_val: Valor máximo
            
        Returns:
            Resultados de validación
        """
        issues = []
        
        if check_nan and torch.isnan(tensor).any():
            issues.append("Tensor contains NaN values")
        
        if check_inf and torch.isinf(tensor).any():
            issues.append("Tensor contains Inf values")
        
        if check_range:
            if min_val is not None and (tensor < min_val).any():
                issues.append(f"Tensor contains values below {min_val}")
            if max_val is not None and (tensor > max_val).any():
                issues.append(f"Tensor contains values above {max_val}")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'shape': tuple(tensor.shape),
            'dtype': str(tensor.dtype),
            'min': tensor.min().item(),
            'max': tensor.max().item(),
            'mean': tensor.mean().item(),
            'std': tensor.std().item()
        }
    
    @staticmethod
    def validate_batch(
        batch: Dict[str, Any],
        required_keys: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Validar batch de datos.
        
        Args:
            batch: Batch a validar
            required_keys: Claves requeridas
            
        Returns:
            Resultados de validación
        """
        issues = []
        
        if required_keys:
            for key in required_keys:
                if key not in batch:
                    issues.append(f"Missing required key: {key}")
        
        # Validar que todos los tensores tienen el mismo batch size
        batch_sizes = []
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                if value.dim() > 0:
                    batch_sizes.append(value.shape[0])
        
        if batch_sizes and len(set(batch_sizes)) > 1:
            issues.append(f"Inconsistent batch sizes: {set(batch_sizes)}")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'batch_size': batch_sizes[0] if batch_sizes else None
        }








