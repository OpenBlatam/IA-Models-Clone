"""
Validation Utils - Utilidades de Validación de Datos
======================================================

Utilidades para validar datos de entrada y salida.
"""

import logging
import torch
import numpy as np
from typing import List, Dict, Optional, Callable, Any, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Resultado de validación."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    
    def __bool__(self):
        return self.is_valid


class DataValidator:
    """
    Validador de datos de entrada.
    """
    
    def __init__(self):
        """Inicializar validador."""
        self.validators: List[Callable] = []
    
    def add_validator(self, validator: Callable) -> 'DataValidator':
        """
        Agregar validador.
        
        Args:
            validator: Función validadora
            
        Returns:
            Self para chaining
        """
        self.validators.append(validator)
        return self
    
    def validate(self, data: Any) -> ValidationResult:
        """
        Validar datos.
        
        Args:
            data: Datos a validar
            
        Returns:
            Resultado de validación
        """
        errors = []
        warnings = []
        
        for validator in self.validators:
            try:
                result = validator(data)
                if isinstance(result, ValidationResult):
                    if not result.is_valid:
                        errors.extend(result.errors)
                        warnings.extend(result.warnings)
                elif not result:
                    errors.append(f"Validation failed: {validator.__name__}")
            except Exception as e:
                errors.append(f"Validator error: {str(e)}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )


class TensorValidator:
    """
    Validador específico para tensores.
    """
    
    @staticmethod
    def validate_shape(
        tensor: torch.Tensor,
        expected_shape: Tuple[int, ...],
        allow_broadcast: bool = False
    ) -> ValidationResult:
        """
        Validar forma de tensor.
        
        Args:
            tensor: Tensor a validar
            expected_shape: Forma esperada
            allow_broadcast: Permitir broadcasting
            
        Returns:
            Resultado de validación
        """
        errors = []
        warnings = []
        
        if not isinstance(tensor, torch.Tensor):
            errors.append(f"Expected tensor, got {type(tensor)}")
            return ValidationResult(False, errors, warnings)
        
        if not allow_broadcast and tensor.shape != expected_shape:
            errors.append(
                f"Shape mismatch: expected {expected_shape}, got {tensor.shape}"
            )
        elif allow_broadcast:
            # Verificar compatibilidad de broadcasting
            try:
                torch.broadcast_shapes(tensor.shape, expected_shape)
            except RuntimeError:
                errors.append(
                    f"Shape not broadcastable: {tensor.shape} vs {expected_shape}"
                )
        
        return ValidationResult(len(errors) == 0, errors, warnings)
    
    @staticmethod
    def validate_dtype(
        tensor: torch.Tensor,
        expected_dtype: torch.dtype
    ) -> ValidationResult:
        """
        Validar tipo de tensor.
        
        Args:
            tensor: Tensor a validar
            expected_dtype: Tipo esperado
            
        Returns:
            Resultado de validación
        """
        errors = []
        
        if tensor.dtype != expected_dtype:
            errors.append(
                f"Dtype mismatch: expected {expected_dtype}, got {tensor.dtype}"
            )
        
        return ValidationResult(len(errors) == 0, errors, [])
    
    @staticmethod
    def validate_range(
        tensor: torch.Tensor,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None
    ) -> ValidationResult:
        """
        Validar rango de valores.
        
        Args:
            tensor: Tensor a validar
            min_value: Valor mínimo (opcional)
            max_value: Valor máximo (opcional)
            
        Returns:
            Resultado de validación
        """
        errors = []
        warnings = []
        
        if min_value is not None and (tensor < min_value).any():
            errors.append(f"Values below minimum: {min_value}")
        
        if max_value is not None and (tensor > max_value).any():
            errors.append(f"Values above maximum: {max_value}")
        
        # Verificar NaN e Inf
        if torch.isnan(tensor).any():
            errors.append("Tensor contains NaN values")
        
        if torch.isinf(tensor).any():
            warnings.append("Tensor contains Inf values")
        
        return ValidationResult(len(errors) == 0, errors, warnings)
    
    @staticmethod
    def validate_not_empty(tensor: torch.Tensor) -> ValidationResult:
        """
        Validar que tensor no esté vacío.
        
        Args:
            tensor: Tensor a validar
            
        Returns:
            Resultado de validación
        """
        errors = []
        
        if tensor.numel() == 0:
            errors.append("Tensor is empty")
        
        return ValidationResult(len(errors) == 0, errors, [])


class ModelOutputValidator:
    """
    Validador de salidas de modelos.
    """
    
    @staticmethod
    def validate_logits(
        logits: torch.Tensor,
        num_classes: Optional[int] = None
    ) -> ValidationResult:
        """
        Validar logits.
        
        Args:
            logits: Logits del modelo
            num_classes: Número de clases esperado
            
        Returns:
            Resultado de validación
        """
        errors = []
        warnings = []
        
        # Validar forma
        if len(logits.shape) != 2:
            errors.append(f"Expected 2D tensor, got {len(logits.shape)}D")
        
        if num_classes is not None and logits.shape[1] != num_classes:
            errors.append(
                f"Expected {num_classes} classes, got {logits.shape[1]}"
            )
        
        # Validar valores
        if torch.isnan(logits).any():
            errors.append("Logits contain NaN values")
        
        if torch.isinf(logits).any():
            warnings.append("Logits contain Inf values")
        
        return ValidationResult(len(errors) == 0, errors, warnings)
    
    @staticmethod
    def validate_probabilities(
        probs: torch.Tensor,
        num_classes: Optional[int] = None,
        tolerance: float = 1e-6
    ) -> ValidationResult:
        """
        Validar probabilidades.
        
        Args:
            probs: Probabilidades
            num_classes: Número de clases esperado
            tolerance: Tolerancia para suma
            
        Returns:
            Resultado de validación
        """
        errors = []
        warnings = []
        
        # Validar forma
        if len(probs.shape) != 2:
            errors.append(f"Expected 2D tensor, got {len(probs.shape)}D")
        
        if num_classes is not None and probs.shape[1] != num_classes:
            errors.append(
                f"Expected {num_classes} classes, got {probs.shape[1]}"
            )
        
        # Validar rango [0, 1]
        if (probs < 0).any() or (probs > 1).any():
            errors.append("Probabilities outside [0, 1] range")
        
        # Validar que sumen a 1
        sums = probs.sum(dim=1)
        if not torch.allclose(sums, torch.ones_like(sums), atol=tolerance):
            errors.append("Probabilities do not sum to 1")
        
        # Validar NaN
        if torch.isnan(probs).any():
            errors.append("Probabilities contain NaN values")
        
        return ValidationResult(len(errors) == 0, errors, warnings)


class DatasetValidator:
    """
    Validador de datasets.
    """
    
    @staticmethod
    def validate_dataset(
        dataset: Any,
        expected_size: Optional[int] = None,
        sample_check: bool = True
    ) -> ValidationResult:
        """
        Validar dataset.
        
        Args:
            dataset: Dataset a validar
            expected_size: Tamaño esperado (opcional)
            sample_check: Verificar muestras (opcional)
            
        Returns:
            Resultado de validación
        """
        errors = []
        warnings = []
        
        # Verificar que tenga __len__
        if not hasattr(dataset, '__len__'):
            errors.append("Dataset does not have __len__ method")
            return ValidationResult(False, errors, warnings)
        
        # Verificar tamaño
        actual_size = len(dataset)
        if expected_size is not None and actual_size != expected_size:
            errors.append(
                f"Size mismatch: expected {expected_size}, got {actual_size}"
            )
        
        if actual_size == 0:
            errors.append("Dataset is empty")
        
        # Verificar muestras
        if sample_check and actual_size > 0:
            try:
                sample = dataset[0]
                if sample is None:
                    errors.append("Dataset returns None for sample")
            except Exception as e:
                errors.append(f"Error accessing dataset sample: {str(e)}")
        
        return ValidationResult(len(errors) == 0, errors, warnings)


def validate_input(
    data: Any,
    validators: List[Callable]
) -> ValidationResult:
    """
    Validar input con múltiples validadores.
    
    Args:
        data: Datos a validar
        validators: Lista de validadores
        
    Returns:
        Resultado de validación
    """
    validator = DataValidator()
    for v in validators:
        validator.add_validator(v)
    return validator.validate(data)




