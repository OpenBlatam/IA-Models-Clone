"""
Validation Generator - Generador de utilidades de validación
============================================================

Genera utilidades para validación robusta:
- Input validation
- Model validation
- Data validation
- Security validation
"""

import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ValidationGenerator:
    """Generador de utilidades de validación"""
    
    def __init__(self):
        """Inicializa el generador de validación"""
        pass
    
    def generate(
        self,
        utils_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """
        Genera utilidades de validación.
        
        Args:
            utils_dir: Directorio donde generar las utilidades
            keywords: Keywords extraídos
            project_info: Información del proyecto
        """
        utils_dir.mkdir(parents=True, exist_ok=True)
        
        validation_dir = utils_dir / "validation"
        validation_dir.mkdir(parents=True, exist_ok=True)
        
        self._generate_input_validator(validation_dir, keywords, project_info)
        self._generate_model_validator(validation_dir, keywords, project_info)
        self._generate_data_validator(validation_dir, keywords, project_info)
        self._generate_validation_init(validation_dir, keywords)
    
    def _generate_validation_init(
        self,
        validation_dir: Path,
        keywords: Dict[str, Any],
    ) -> None:
        """Genera __init__.py del módulo de validación"""
        
        init_content = '''"""
Validation Utilities Module
=============================

Utilidades para validación robusta de inputs, modelos y datos.
"""

from .input_validator import (
    InputValidator,
    validate_text_input,
    validate_image_input,
    validate_tensor_input,
)
from .model_validator import (
    ModelValidator,
    validate_model_architecture,
    validate_model_outputs,
    check_model_integrity,
)
from .data_validator import (
    DataValidator,
    validate_dataset,
    validate_batch,
    check_data_quality,
)

__all__ = [
    "InputValidator",
    "validate_text_input",
    "validate_image_input",
    "validate_tensor_input",
    "ModelValidator",
    "validate_model_architecture",
    "validate_model_outputs",
    "check_model_integrity",
    "DataValidator",
    "validate_dataset",
    "validate_batch",
    "check_data_quality",
]
'''
        
        (validation_dir / "__init__.py").write_text(init_content, encoding="utf-8")
    
    def _generate_input_validator(
        self,
        validation_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """Genera validador de inputs"""
        
        input_validator_content = '''"""
Input Validator - Validador de inputs
======================================

Utilidades para validar inputs de manera robusta y segura.
"""

import torch
import numpy as np
from typing import Any, Optional, Dict, List, Union
import logging
import re

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

logger = logging.getLogger(__name__)


class InputValidator:
    """
    Validador robusto de inputs.
    
    Valida diferentes tipos de inputs con seguridad.
    """
    
    def __init__(
        self,
        max_text_length: int = 10000,
        max_image_size: tuple = (4096, 4096),
        allowed_image_formats: List[str] = None,
    ):
        """
        Inicializa el validador.
        
        Args:
            max_text_length: Longitud máxima de texto
            max_image_size: Tamaño máximo de imagen (width, height)
            allowed_image_formats: Formatos de imagen permitidos
        """
        self.max_text_length = max_text_length
        self.max_image_size = max_image_size
        self.allowed_image_formats = allowed_image_formats or ["JPEG", "PNG", "WEBP"]
    
    def validate_text(
        self,
        text: Any,
        min_length: int = 1,
        max_length: Optional[int] = None,
        allowed_chars: Optional[str] = None,
        sanitize: bool = True,
    ) -> str:
        """
        Valida y sanitiza texto.
        
        Args:
            text: Texto a validar
            min_length: Longitud mínima
            max_length: Longitud máxima (usa self.max_text_length si None)
            allowed_chars: Caracteres permitidos (regex)
            sanitize: Si sanitizar el texto
        
        Returns:
            Texto validado y sanitizado
        
        Raises:
            ValueError: Si el texto no es válido
        """
        if not isinstance(text, str):
            raise ValueError(f"Texto debe ser string, recibido: {type(text)}")
        
        if sanitize:
            # Remover caracteres peligrosos
            text = re.sub(r'[<>"\']', '', text)
            text = text.strip()
        
        if len(text) < min_length:
            raise ValueError(f"Texto muy corto: mínimo {min_length} caracteres")
        
        max_len = max_length or self.max_text_length
        if len(text) > max_len:
            raise ValueError(f"Texto muy largo: máximo {max_len} caracteres")
        
        if allowed_chars:
            if not re.match(allowed_chars, text):
                raise ValueError(f"Texto contiene caracteres no permitidos")
        
        return text
    
    def validate_image(
        self,
        image: Any,
        check_size: bool = True,
        check_format: bool = True,
    ) -> Any:
        """
        Valida imagen.
        
        Args:
            image: Imagen a validar (PIL Image, numpy array, o tensor)
            check_size: Si verificar tamaño
            check_format: Si verificar formato
        
        Returns:
            Imagen validada
        
        Raises:
            ValueError: Si la imagen no es válida
        """
        if PIL_AVAILABLE and isinstance(image, Image.Image):
            # Validar formato
            if check_format and image.format not in self.allowed_image_formats:
                raise ValueError(f"Formato de imagen no permitido: {image.format}")
            
            # Validar tamaño
            if check_size:
                width, height = image.size
                max_width, max_height = self.max_image_size
                if width > max_width or height > max_height:
                    raise ValueError(
                        f"Imagen muy grande: máximo {max_width}x{max_height}, "
                        f"recibido {width}x{height}"
                    )
            
            return image
        
        elif isinstance(image, np.ndarray):
            if image.ndim not in [2, 3]:
                raise ValueError(f"Array de imagen debe ser 2D o 3D, recibido: {image.ndim}D")
            
            if check_size:
                if image.ndim == 3:
                    height, width = image.shape[:2]
                else:
                    height, width = image.shape
                
                max_width, max_height = self.max_image_size
                if width > max_width or height > max_height:
                    raise ValueError(
                        f"Imagen muy grande: máximo {max_width}x{max_height}, "
                        f"recibido {width}x{height}"
                    )
            
            return image
        
        elif isinstance(image, torch.Tensor):
            if image.ndim not in [2, 3, 4]:
                raise ValueError(f"Tensor de imagen debe ser 2D, 3D o 4D, recibido: {image.ndim}D")
            
            if check_size:
                if image.ndim == 4:
                    _, _, height, width = image.shape
                elif image.ndim == 3:
                    _, height, width = image.shape
                else:
                    height, width = image.shape
                
                max_width, max_height = self.max_image_size
                if width > max_width or height > max_height:
                    raise ValueError(
                        f"Imagen muy grande: máximo {max_width}x{max_height}, "
                        f"recibido {width}x{height}"
                    )
            
            return image
        
        else:
            raise ValueError(f"Tipo de imagen no soportado: {type(image)}")
    
    def validate_tensor(
        self,
        tensor: Any,
        expected_shape: Optional[tuple] = None,
        expected_dtype: Optional[torch.dtype] = None,
        check_finite: bool = True,
        check_range: Optional[tuple] = None,
    ) -> torch.Tensor:
        """
        Valida tensor.
        
        Args:
            tensor: Tensor a validar
            expected_shape: Shape esperado (opcional)
            expected_dtype: Dtype esperado (opcional)
            check_finite: Si verificar valores finitos
            check_range: Rango permitido (min, max) (opcional)
        
        Returns:
            Tensor validado
        
        Raises:
            ValueError: Si el tensor no es válido
        """
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Debe ser torch.Tensor, recibido: {type(tensor)}")
        
        if expected_shape and tensor.shape != expected_shape:
            raise ValueError(
                f"Shape incorrecto: esperado {expected_shape}, recibido {tensor.shape}"
            )
        
        if expected_dtype and tensor.dtype != expected_dtype:
            raise ValueError(
                f"Dtype incorrecto: esperado {expected_dtype}, recibido {tensor.dtype}"
            )
        
        if check_finite:
            if not torch.isfinite(tensor).all():
                raise ValueError("Tensor contiene valores NaN o Inf")
        
        if check_range:
            min_val, max_val = check_range
            if tensor.min() < min_val or tensor.max() > max_val:
                raise ValueError(
                    f"Valores fuera de rango: esperado [{min_val}, {max_val}], "
                    f"recibido [{tensor.min()}, {tensor.max()}]"
                )
        
        return tensor


def validate_text_input(
    text: Any,
    max_length: int = 10000,
    **kwargs,
) -> str:
    """
    Función helper para validar texto.
    
    Args:
        text: Texto a validar
        max_length: Longitud máxima
        **kwargs: Argumentos adicionales
    
    Returns:
        Texto validado
    """
    validator = InputValidator(max_text_length=max_length)
    return validator.validate_text(text, **kwargs)


def validate_image_input(
    image: Any,
    max_size: tuple = (4096, 4096),
    **kwargs,
) -> Any:
    """
    Función helper para validar imagen.
    
    Args:
        image: Imagen a validar
        max_size: Tamaño máximo
        **kwargs: Argumentos adicionales
    
    Returns:
        Imagen validada
    """
    validator = InputValidator(max_image_size=max_size)
    return validator.validate_image(image, **kwargs)


def validate_tensor_input(
    tensor: Any,
    **kwargs,
) -> torch.Tensor:
    """
    Función helper para validar tensor.
    
    Args:
        tensor: Tensor a validar
        **kwargs: Argumentos adicionales
    
    Returns:
        Tensor validado
    """
    validator = InputValidator()
    return validator.validate_tensor(tensor, **kwargs)
'''
        
        (validation_dir / "input_validator.py").write_text(input_validator_content, encoding="utf-8")
    
    def _generate_model_validator(
        self,
        validation_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """Genera validador de modelos"""
        
        model_validator_content = '''"""
Model Validator - Validador de modelos
======================================

Utilidades para validar arquitecturas y outputs de modelos.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


class ModelValidator:
    """
    Validador de modelos.
    
    Valida arquitecturas y outputs de modelos.
    """
    
    def __init__(self):
        """Inicializa el validador"""
        pass
    
    def validate_architecture(
        self,
        model: nn.Module,
        expected_input_shape: Tuple[int, ...],
        expected_output_shape: Optional[Tuple[int, ...]] = None,
    ) -> Dict[str, Any]:
        """
        Valida arquitectura de modelo.
        
        Args:
            model: Modelo a validar
            expected_input_shape: Shape esperado de input
            expected_output_shape: Shape esperado de output (opcional)
        
        Returns:
            Diccionario con resultados de validación
        """
        model.eval()
        
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
        }
        
        # Verificar que el modelo puede hacer forward
        try:
            dummy_input = torch.randn(1, *expected_input_shape)
            with torch.no_grad():
                output = model(dummy_input)
            
            validation_results["output_shape"] = tuple(output.shape)
            
            if expected_output_shape:
                if output.shape[1:] != expected_output_shape:
                    validation_results["valid"] = False
                    validation_results["errors"].append(
                        f"Output shape incorrecto: esperado {expected_output_shape}, "
                        f"recibido {output.shape[1:]}"
                    )
        except Exception as e:
            validation_results["valid"] = False
            validation_results["errors"].append(f"Error en forward pass: {str(e)}")
        
        # Verificar parámetros
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        validation_results["total_parameters"] = total_params
        validation_results["trainable_parameters"] = trainable_params
        
        if total_params == 0:
            validation_results["valid"] = False
            validation_results["errors"].append("Modelo no tiene parámetros")
        
        return validation_results
    
    def validate_outputs(
        self,
        outputs: torch.Tensor,
        expected_shape: Optional[Tuple[int, ...]] = None,
        check_finite: bool = True,
        check_range: Optional[Tuple[float, float]] = None,
    ) -> Dict[str, Any]:
        """
        Valida outputs de modelo.
        
        Args:
            outputs: Outputs a validar
            expected_shape: Shape esperado (opcional)
            check_finite: Si verificar valores finitos
            check_range: Rango permitido (opcional)
        
        Returns:
            Diccionario con resultados de validación
        """
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
        }
        
        if not isinstance(outputs, torch.Tensor):
            validation_results["valid"] = False
            validation_results["errors"].append(f"Outputs debe ser Tensor, recibido: {type(outputs)}")
            return validation_results
        
        if expected_shape and outputs.shape != expected_shape:
            validation_results["valid"] = False
            validation_results["errors"].append(
                f"Shape incorrecto: esperado {expected_shape}, recibido {outputs.shape}"
            )
        
        if check_finite:
            if not torch.isfinite(outputs).all():
                validation_results["valid"] = False
                validation_results["errors"].append("Outputs contiene NaN o Inf")
        
        if check_range:
            min_val, max_val = check_range
            if outputs.min() < min_val or outputs.max() > max_val:
                validation_results["warnings"].append(
                    f"Valores fuera de rango esperado: [{min_val}, {max_val}]"
                )
        
        validation_results["output_stats"] = {
            "min": float(outputs.min().item()),
            "max": float(outputs.max().item()),
            "mean": float(outputs.mean().item()),
            "std": float(outputs.std().item()),
        }
        
        return validation_results
    
    def check_integrity(
        self,
        model: nn.Module,
    ) -> Dict[str, Any]:
        """
        Verifica integridad del modelo.
        
        Args:
            model: Modelo a verificar
        
        Returns:
            Diccionario con resultados de verificación
        """
        integrity_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
        }
        
        # Verificar que todos los parámetros son finitos
        for name, param in model.named_parameters():
            if not torch.isfinite(param).all():
                integrity_results["valid"] = False
                integrity_results["errors"].append(f"Parámetro {name} contiene NaN o Inf")
        
        # Verificar que hay parámetros
        if len(list(model.parameters())) == 0:
            integrity_results["valid"] = False
            integrity_results["errors"].append("Modelo no tiene parámetros")
        
        # Verificar que hay buffers
        buffers = list(model.buffers())
        if buffers:
            for name, buffer in model.named_buffers():
                if not torch.isfinite(buffer).all():
                    integrity_results["warnings"].append(f"Buffer {name} contiene NaN o Inf")
        
        return integrity_results


def validate_model_architecture(
    model: nn.Module,
    expected_input_shape: Tuple[int, ...],
    **kwargs,
) -> Dict[str, Any]:
    """
    Función helper para validar arquitectura.
    
    Args:
        model: Modelo a validar
        expected_input_shape: Shape esperado
        **kwargs: Argumentos adicionales
    
    Returns:
        Resultados de validación
    """
    validator = ModelValidator()
    return validator.validate_architecture(model, expected_input_shape, **kwargs)


def validate_model_outputs(
    outputs: torch.Tensor,
    **kwargs,
) -> Dict[str, Any]:
    """
    Función helper para validar outputs.
    
    Args:
        outputs: Outputs a validar
        **kwargs: Argumentos adicionales
    
    Returns:
        Resultados de validación
    """
    validator = ModelValidator()
    return validator.validate_outputs(outputs, **kwargs)


def check_model_integrity(
    model: nn.Module,
) -> Dict[str, Any]:
    """
    Función helper para verificar integridad.
    
    Args:
        model: Modelo a verificar
    
    Returns:
        Resultados de verificación
    """
    validator = ModelValidator()
    return validator.check_integrity(model)
'''
        
        (validation_dir / "model_validator.py").write_text(model_validator_content, encoding="utf-8")
    
    def _generate_data_validator(
        self,
        validation_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """Genera validador de datos"""
        
        data_validator_content = '''"""
Data Validator - Validador de datos
====================================

Utilidades para validar datasets y batches.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Validador de datos.
    
    Valida calidad y consistencia de datasets y batches.
    """
    
    def __init__(self):
        """Inicializa el validador"""
        pass
    
    def validate_dataset(
        self,
        dataset,
        check_samples: int = 10,
    ) -> Dict[str, Any]:
        """
        Valida un dataset.
        
        Args:
            dataset: Dataset a validar
            check_samples: Número de muestras a verificar
        
        Returns:
            Diccionario con resultados de validación
        """
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "dataset_size": len(dataset),
        }
        
        if len(dataset) == 0:
            validation_results["valid"] = False
            validation_results["errors"].append("Dataset está vacío")
            return validation_results
        
        # Verificar muestras aleatorias
        indices = np.random.choice(len(dataset), min(check_samples, len(dataset)), replace=False)
        
        for idx in indices:
            try:
                sample = dataset[idx]
                
                if isinstance(sample, (tuple, list)):
                    data, label = sample[0], sample[1]
                else:
                    data, label = sample, None
                
                # Validar data
                if isinstance(data, torch.Tensor):
                    if not torch.isfinite(data).all():
                        validation_results["warnings"].append(
                            f"Muestra {idx}: data contiene NaN o Inf"
                        )
                elif isinstance(data, np.ndarray):
                    if not np.isfinite(data).all():
                        validation_results["warnings"].append(
                            f"Muestra {idx}: data contiene NaN o Inf"
                        )
                
                # Validar label si existe
                if label is not None:
                    if isinstance(label, torch.Tensor):
                        if not torch.isfinite(label).all():
                            validation_results["warnings"].append(
                                f"Muestra {idx}: label contiene NaN o Inf"
                            )
                
            except Exception as e:
                validation_results["valid"] = False
                validation_results["errors"].append(f"Error en muestra {idx}: {str(e)}")
        
        return validation_results
    
    def validate_batch(
        self,
        batch: Any,
        expected_batch_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Valida un batch.
        
        Args:
            batch: Batch a validar
            expected_batch_size: Tamaño de batch esperado (opcional)
        
        Returns:
            Diccionario con resultados de validación
        """
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
        }
        
        if isinstance(batch, (tuple, list)):
            data, labels = batch[0], batch[1] if len(batch) > 1 else None
        else:
            data, labels = batch, None
        
        # Validar data
        if isinstance(data, torch.Tensor):
            batch_size = data.shape[0]
            
            if expected_batch_size and batch_size != expected_batch_size:
                validation_results["warnings"].append(
                    f"Batch size incorrecto: esperado {expected_batch_size}, recibido {batch_size}"
                )
            
            if not torch.isfinite(data).all():
                validation_results["valid"] = False
                validation_results["errors"].append("Batch data contiene NaN o Inf")
        else:
            validation_results["errors"].append(f"Data debe ser Tensor, recibido: {type(data)}")
        
        # Validar labels si existen
        if labels is not None:
            if isinstance(labels, torch.Tensor):
                if not torch.isfinite(labels).all():
                    validation_results["valid"] = False
                    validation_results["errors"].append("Batch labels contiene NaN o Inf")
                
                if labels.shape[0] != batch_size:
                    validation_results["valid"] = False
                    validation_results["errors"].append(
                        f"Batch size inconsistente: data={batch_size}, labels={labels.shape[0]}"
                    )
        
        validation_results["batch_size"] = batch_size if isinstance(data, torch.Tensor) else None
        
        return validation_results
    
    def check_data_quality(
        self,
        data: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Verifica calidad de datos.
        
        Args:
            data: Datos a verificar
        
        Returns:
            Diccionario con métricas de calidad
        """
        quality_metrics = {
            "finite_ratio": float(torch.isfinite(data).sum() / data.numel()),
            "nan_count": int(torch.isnan(data).sum().item()),
            "inf_count": int(torch.isinf(data).sum().item()),
            "zero_count": int((data == 0).sum().item()),
            "mean": float(data.mean().item()) if torch.isfinite(data).any() else None,
            "std": float(data.std().item()) if torch.isfinite(data).any() else None,
            "min": float(data.min().item()) if torch.isfinite(data).any() else None,
            "max": float(data.max().item()) if torch.isfinite(data).any() else None,
        }
        
        return quality_metrics


def validate_dataset(
    dataset,
    **kwargs,
) -> Dict[str, Any]:
    """
    Función helper para validar dataset.
    
    Args:
        dataset: Dataset a validar
        **kwargs: Argumentos adicionales
    
    Returns:
        Resultados de validación
    """
    validator = DataValidator()
    return validator.validate_dataset(dataset, **kwargs)


def validate_batch(
    batch: Any,
    **kwargs,
) -> Dict[str, Any]:
    """
    Función helper para validar batch.
    
    Args:
        batch: Batch a validar
        **kwargs: Argumentos adicionales
    
    Returns:
        Resultados de validación
    """
    validator = DataValidator()
    return validator.validate_batch(batch, **kwargs)


def check_data_quality(
    data: torch.Tensor,
) -> Dict[str, Any]:
    """
    Función helper para verificar calidad.
    
    Args:
        data: Datos a verificar
    
    Returns:
        Métricas de calidad
    """
    validator = DataValidator()
    return validator.check_data_quality(data)
'''
        
        (validation_dir / "data_validator.py").write_text(data_validator_content, encoding="utf-8")

