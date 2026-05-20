"""
Data Validation - Validación de datos
======================================

Sistema para validar datos antes de entrenamiento.
Sigue mejores prácticas de validación de datos.
"""

import logging
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
import torch
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Resultado de validación"""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class DataValidationService:
    """Servicio de validación de datos"""
    
    @staticmethod
    def validate_tensor(
        tensor: torch.Tensor,
        check_nan: bool = True,
        check_inf: bool = True,
        check_shape: Optional[Tuple[int, ...]] = None,
        check_dtype: Optional[torch.dtype] = None,
        check_range: Optional[Tuple[float, float]] = None
    ) -> ValidationResult:
        """
        Validar tensor.
        
        Args:
            tensor: Tensor a validar
            check_nan: Verificar NaN
            check_inf: Verificar Inf
            check_shape: Verificar forma (opcional)
            check_dtype: Verificar dtype (opcional)
            check_range: Verificar rango (min, max) (opcional)
        
        Returns:
            ValidationResult
        """
        result = ValidationResult(is_valid=True)
        
        # Check NaN
        if check_nan and torch.isnan(tensor).any():
            result.is_valid = False
            result.errors.append("Tensor contains NaN values")
        
        # Check Inf
        if check_inf and torch.isinf(tensor).any():
            result.is_valid = False
            result.errors.append("Tensor contains Inf values")
        
        # Check shape
        if check_shape and tensor.shape != check_shape:
            result.is_valid = False
            result.errors.append(
                f"Tensor shape {tensor.shape} does not match expected {check_shape}"
            )
        
        # Check dtype
        if check_dtype and tensor.dtype != check_dtype:
            result.warnings.append(
                f"Tensor dtype {tensor.dtype} does not match expected {check_dtype}"
            )
        
        # Check range
        if check_range:
            min_val, max_val = check_range
            if tensor.min().item() < min_val or tensor.max().item() > max_val:
                result.warnings.append(
                    f"Tensor values outside expected range [{min_val}, {max_val}]"
                )
        
        return result
    
    @staticmethod
    def validate_dataset(
        dataset: Any,
        sample_size: int = 100,
        check_labels: bool = True
    ) -> ValidationResult:
        """
        Validar dataset.
        
        Args:
            dataset: Dataset a validar
            sample_size: Tamaño de muestra para validación
            check_labels: Verificar labels
        
        Returns:
            ValidationResult
        """
        result = ValidationResult(is_valid=True)
        
        try:
            dataset_size = len(dataset)
            if dataset_size == 0:
                result.is_valid = False
                result.errors.append("Dataset is empty")
                return result
            
            result.metadata["dataset_size"] = dataset_size
            
            # Sample and validate
            sample_indices = np.random.choice(
                min(sample_size, dataset_size),
                size=min(sample_size, dataset_size),
                replace=False
            )
            
            for idx in sample_indices:
                try:
                    item = dataset[int(idx)]
                    
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        data, label = item[0], item[1]
                        
                        # Validate data
                        if isinstance(data, torch.Tensor):
                            data_result = DataValidationService.validate_tensor(data)
                            if not data_result.is_valid:
                                result.is_valid = False
                                result.errors.extend(data_result.errors)
                            result.warnings.extend(data_result.warnings)
                        
                        # Validate label
                        if check_labels:
                            if isinstance(label, torch.Tensor):
                                label_result = DataValidationService.validate_tensor(label)
                                if not label_result.is_valid:
                                    result.is_valid = False
                                    result.errors.extend(label_result.errors)
                    
                    elif isinstance(item, torch.Tensor):
                        # Single tensor
                        tensor_result = DataValidationService.validate_tensor(item)
                        if not tensor_result.is_valid:
                            result.is_valid = False
                            result.errors.extend(tensor_result.errors)
                        result.warnings.extend(tensor_result.warnings)
                
                except Exception as e:
                    result.is_valid = False
                    result.errors.append(f"Error accessing dataset item {idx}: {str(e)}")
            
            result.metadata["samples_checked"] = len(sample_indices)
        
        except Exception as e:
            result.is_valid = False
            result.errors.append(f"Error validating dataset: {str(e)}")
        
        return result
    
    @staticmethod
    def validate_dataloader(
        dataloader: Any,
        num_batches: int = 10
    ) -> ValidationResult:
        """
        Validar DataLoader.
        
        Args:
            dataloader: DataLoader a validar
            num_batches: Número de batches a verificar
        
        Returns:
            ValidationResult
        """
        result = ValidationResult(is_valid=True)
        
        try:
            batch_count = 0
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= num_batches:
                    break
                
                try:
                    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                        inputs, targets = batch[0], batch[1]
                        
                        # Validate inputs
                        if isinstance(inputs, torch.Tensor):
                            input_result = DataValidationService.validate_tensor(inputs)
                            if not input_result.is_valid:
                                result.is_valid = False
                                result.errors.extend(input_result.errors)
                        
                        # Validate targets
                        if isinstance(targets, torch.Tensor):
                            target_result = DataValidationService.validate_tensor(targets)
                            if not target_result.is_valid:
                                result.is_valid = False
                                result.errors.extend(target_result.errors)
                    
                    batch_count += 1
                
                except Exception as e:
                    result.is_valid = False
                    result.errors.append(f"Error in batch {batch_idx}: {str(e)}")
            
            result.metadata["batches_checked"] = batch_count
        
        except Exception as e:
            result.is_valid = False
            result.errors.append(f"Error validating dataloader: {str(e)}")
        
        return result
    
    @staticmethod
    def validate_model_inputs(
        model: torch.nn.Module,
        input_shape: Tuple[int, ...],
        device: Optional[torch.device] = None
    ) -> ValidationResult:
        """
        Validar que el modelo puede procesar inputs de la forma dada.
        
        Args:
            model: Modelo a validar
            input_shape: Forma del input (sin batch dimension)
            device: Dispositivo (None = auto)
        
        Returns:
            ValidationResult
        """
        result = ValidationResult(is_valid=True)
        
        try:
            device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            model.eval()
            
            # Create dummy input
            dummy_input = torch.randn(1, *input_shape).to(device)
            
            # Validate input tensor
            input_result = DataValidationService.validate_tensor(dummy_input)
            if not input_result.is_valid:
                result.is_valid = False
                result.errors.extend(input_result.errors)
            
            # Try forward pass
            with torch.no_grad():
                try:
                    output = model(dummy_input)
                    
                    # Validate output
                    if isinstance(output, torch.Tensor):
                        output_result = DataValidationService.validate_tensor(output)
                        if not output_result.is_valid:
                            result.is_valid = False
                            result.errors.extend(output_result.errors)
                        result.metadata["output_shape"] = list(output.shape)
                    
                    result.metadata["forward_pass_successful"] = True
                
                except Exception as e:
                    result.is_valid = False
                    result.errors.append(f"Forward pass failed: {str(e)}")
                    result.metadata["forward_pass_successful"] = False
        
        except Exception as e:
            result.is_valid = False
            result.errors.append(f"Error validating model inputs: {str(e)}")
        
        return result




