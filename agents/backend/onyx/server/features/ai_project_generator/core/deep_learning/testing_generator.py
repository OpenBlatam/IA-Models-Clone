"""
Testing Generator - Generador de utilidades de testing
======================================================

Genera utilidades para testing de modelos:
- Unit tests para modelos
- Integration tests
- Performance tests
- Model validation tests
"""

import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


class TestingGenerator:
    """Generador de utilidades de testing"""
    
    def __init__(self):
        """Inicializa el generador de testing"""
        pass
    
    def generate(
        self,
        utils_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """
        Genera utilidades de testing.
        
        Args:
            utils_dir: Directorio donde generar las utilidades
            keywords: Keywords extraídos
            project_info: Información del proyecto
        """
        utils_dir.mkdir(parents=True, exist_ok=True)
        
        test_dir = utils_dir / "testing"
        test_dir.mkdir(parents=True, exist_ok=True)
        
        self._generate_model_tests(test_dir, keywords, project_info)
        self._generate_test_utils(test_dir, keywords, project_info)
        self._generate_testing_init(test_dir, keywords)
    
    def _generate_testing_init(
        self,
        test_dir: Path,
        keywords: Dict[str, Any],
    ) -> None:
        """Genera __init__.py del módulo de testing"""
        
        init_content = '''"""
Testing Utilities Module
=========================

Utilidades para testing de modelos y componentes.
"""

from .model_tests import (
    test_model_forward,
    test_model_output_shape,
    test_model_gradients,
    test_model_inference,
)
from .test_utils import (
    create_dummy_input,
    assert_model_outputs_valid,
    compare_model_outputs,
)

__all__ = [
    "test_model_forward",
    "test_model_output_shape",
    "test_model_gradients",
    "test_model_inference",
    "create_dummy_input",
    "assert_model_outputs_valid",
    "compare_model_outputs",
]
'''
        
        (test_dir / "__init__.py").write_text(init_content, encoding="utf-8")
    
    def _generate_model_tests(
        self,
        test_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """Genera tests para modelos"""
        
        tests_content = '''"""
Model Tests - Tests para modelos
=================================

Tests unitarios e integración para modelos de Deep Learning.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def test_model_forward(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: str = "cpu",
) -> bool:
    """
    Test básico de forward pass.
    
    Args:
        model: Modelo a testear
        input_shape: Shape del input
        device: Dispositivo a usar
    
    Returns:
        True si el test pasa
    """
    try:
        model.eval()
        dummy_input = torch.randn(1, *input_shape).to(device)
        model = model.to(device)
        
        with torch.no_grad():
            output = model(dummy_input)
        
        assert output is not None, "Output es None"
        assert isinstance(output, torch.Tensor), "Output no es un Tensor"
        
        logger.info("✓ Forward pass test pasado")
        return True
    
    except Exception as e:
        logger.error(f"✗ Forward pass test falló: {e}")
        return False


def test_model_output_shape(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    expected_output_shape: Tuple[int, ...],
    device: str = "cpu",
) -> bool:
    """
    Test de shape de output.
    
    Args:
        model: Modelo a testear
        input_shape: Shape del input
        expected_output_shape: Shape esperado del output
        device: Dispositivo a usar
    
    Returns:
        True si el test pasa
    """
    try:
        model.eval()
        dummy_input = torch.randn(1, *input_shape).to(device)
        model = model.to(device)
        
        with torch.no_grad():
            output = model(dummy_input)
        
        if isinstance(output, torch.Tensor):
            output_shape = output.shape[1:]  # Ignorar batch dimension
        elif isinstance(output, (list, tuple)):
            output_shape = output[0].shape[1:]
        else:
            output_shape = None
        
        assert output_shape == expected_output_shape, \
            f"Shape esperado {expected_output_shape}, obtenido {output_shape}"
        
        logger.info(f"✓ Output shape test pasado: {output_shape}")
        return True
    
    except Exception as e:
        logger.error(f"✗ Output shape test falló: {e}")
        return False


def test_model_gradients(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: str = "cpu",
) -> bool:
    """
    Test de gradientes.
    
    Args:
        model: Modelo a testear
        input_shape: Shape del input
        device: Dispositivo a usar
    
    Returns:
        True si el test pasa
    """
    try:
        model.train()
        dummy_input = torch.randn(1, *input_shape).to(device)
        dummy_target = torch.randn(1, *input_shape).to(device)
        model = model.to(device)
        
        # Forward pass
        output = model(dummy_input)
        
        # Loss simple
        if isinstance(output, torch.Tensor):
            loss = nn.MSELoss()(output, dummy_target)
        else:
            loss = nn.MSELoss()(output[0], dummy_target)
        
        # Backward pass
        loss.backward()
        
        # Verificar que hay gradientes
        has_gradients = False
        for param in model.parameters():
            if param.requires_grad and param.grad is not None:
                has_gradients = True
                # Verificar que no son NaN/Inf
                assert torch.isfinite(param.grad).all(), \
                    f"Gradientes no finitos en {param}"
                break
        
        assert has_gradients, "No se encontraron gradientes"
        
        logger.info("✓ Gradients test pasado")
        return True
    
    except Exception as e:
        logger.error(f"✗ Gradients test falló: {e}")
        return False


def test_model_inference(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: str = "cpu",
    num_runs: int = 5,
) -> Dict[str, Any]:
    """
    Test de inferencia con métricas de performance.
    
    Args:
        model: Modelo a testear
        input_shape: Shape del input
        device: Dispositivo a usar
        num_runs: Número de ejecuciones
    
    Returns:
        Diccionario con métricas
    """
    import time
    
    model.eval()
    dummy_input = torch.randn(1, *input_shape).to(device)
    model = model.to(device)
    
    times = []
    
    with torch.no_grad():
        # Warmup
        _ = model(dummy_input)
        if device == "cuda":
            torch.cuda.synchronize()
        
        # Medir tiempos
        for _ in range(num_runs):
            if device == "cuda":
                torch.cuda.synchronize()
            
            start = time.time()
            output = model(dummy_input)
            
            if device == "cuda":
                torch.cuda.synchronize()
            
            times.append(time.time() - start)
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    # Verificar output
    output_valid = True
    if isinstance(output, torch.Tensor):
        if torch.isnan(output).any():
            output_valid = False
        if torch.isinf(output).any():
            output_valid = False
    
    metrics = {
        "avg_inference_time_ms": avg_time * 1000,
        "min_inference_time_ms": min_time * 1000,
        "max_inference_time_ms": max_time * 1000,
        "throughput_samples_per_sec": 1.0 / avg_time if avg_time > 0 else 0,
        "output_valid": output_valid,
    }
    
    logger.info(f"✓ Inference test pasado: {avg_time*1000:.2f}ms promedio")
    return metrics
'''
        
        (test_dir / "model_tests.py").write_text(tests_content, encoding="utf-8")
    
    def _generate_test_utils(
        self,
        test_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """Genera utilidades de testing"""
        
        utils_content = '''"""
Test Utilities - Utilidades para testing
==========================================

Funciones helper para testing de modelos.
"""

import torch
from typing import Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)


def create_dummy_input(
    input_shape: Tuple[int, ...],
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Crea input dummy para testing.
    
    Args:
        input_shape: Shape del input
        dtype: Tipo de datos
        device: Dispositivo
    
    Returns:
        Tensor dummy
    """
    return torch.randn(1, *input_shape, dtype=dtype, device=device)


def assert_model_outputs_valid(
    outputs: Any,
    check_nan: bool = True,
    check_inf: bool = True,
    check_shape: bool = False,
    expected_shape: Optional[Tuple[int, ...]] = None,
) -> bool:
    """
    Verifica que los outputs del modelo son válidos.
    
    Args:
        outputs: Outputs del modelo
        check_nan: Si verificar NaN
        check_inf: Si verificar Inf
        check_shape: Si verificar shape
        expected_shape: Shape esperado (si check_shape=True)
    
    Returns:
        True si los outputs son válidos
    """
    if isinstance(outputs, torch.Tensor):
        if check_nan and torch.isnan(outputs).any():
            logger.error("Output contiene NaN")
            return False
        
        if check_inf and torch.isinf(outputs).any():
            logger.error("Output contiene Inf")
            return False
        
        if check_shape and expected_shape is not None:
            if outputs.shape[1:] != expected_shape:
                logger.error(
                    f"Shape incorrecto: esperado {expected_shape}, "
                    f"obtenido {outputs.shape[1:]}"
                )
                return False
    
    elif isinstance(outputs, (list, tuple)):
        for output in outputs:
            if not assert_model_outputs_valid(
                output, check_nan, check_inf, check_shape, expected_shape
            ):
                return False
    
    return True


def compare_model_outputs(
    output1: torch.Tensor,
    output2: torch.Tensor,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> bool:
    """
    Compara dos outputs de modelos.
    
    Args:
        output1: Primer output
        output2: Segundo output
        rtol: Tolerancia relativa
        atol: Tolerancia absoluta
    
    Returns:
        True si los outputs son similares
    """
    try:
        if output1.shape != output2.shape:
            logger.error(f"Shapes diferentes: {output1.shape} vs {output2.shape}")
            return False
        
        if not torch.allclose(output1, output2, rtol=rtol, atol=atol):
            max_diff = (output1 - output2).abs().max().item()
            logger.error(f"Outputs diferentes: max_diff={max_diff}")
            return False
        
        logger.info("Outputs son similares")
        return True
    
    except Exception as e:
        logger.error(f"Error comparando outputs: {e}")
        return False
'''
        
        (test_dir / "test_utils.py").write_text(utils_content, encoding="utf-8")

