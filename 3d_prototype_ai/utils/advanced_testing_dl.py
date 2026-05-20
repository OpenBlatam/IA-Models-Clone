"""
Advanced Testing for Deep Learning - Testing avanzado para DL
==============================================================
Testing framework especializado para modelos de deep learning
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
import unittest

logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    """Caso de prueba"""
    name: str
    test_fn: Callable
    expected_result: Any
    tolerance: float = 1e-5


class DLTestSuite:
    """Suite de tests para deep learning"""
    
    def __init__(self):
        self.test_cases: List[TestCase] = []
        self.test_results: List[Dict[str, Any]] = []
    
    def add_test(
        self,
        name: str,
        test_fn: Callable,
        expected_result: Any,
        tolerance: float = 1e-5
    ):
        """Agrega caso de prueba"""
        test_case = TestCase(name, test_fn, expected_result, tolerance)
        self.test_cases.append(test_case)
    
    def test_model_forward(self, model: nn.Module, input_shape: tuple) -> bool:
        """Test de forward pass"""
        try:
            model.eval()
            dummy_input = torch.randn(1, *input_shape)
            output = model(dummy_input)
            
            # Verificar que output es tensor válido
            assert isinstance(output, torch.Tensor), "Output must be a tensor"
            assert not torch.isnan(output).any(), "Output contains NaN"
            assert not torch.isinf(output).any(), "Output contains Inf"
            
            logger.info("Forward pass test passed")
            return True
        except Exception as e:
            logger.error(f"Forward pass test failed: {e}")
            return False
    
    def test_model_gradient(self, model: nn.Module, input_shape: tuple) -> bool:
        """Test de gradientes"""
        try:
            model.train()
            dummy_input = torch.randn(1, *input_shape)
            dummy_target = torch.randint(0, 10, (1,))
            
            output = model(dummy_input)
            if output.dim() > 1:
                loss = nn.CrossEntropyLoss()(output, dummy_target)
            else:
                loss = nn.MSELoss()(output, dummy_target.float())
            
            loss.backward()
            
            # Verificar que hay gradientes
            has_gradients = any(p.grad is not None for p in model.parameters() if p.requires_grad)
            assert has_gradients, "No gradients found"
            
            # Verificar que no hay NaN en gradientes
            for name, param in model.named_parameters():
                if param.grad is not None:
                    assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
                    assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"
            
            logger.info("Gradient test passed")
            return True
        except Exception as e:
            logger.error(f"Gradient test failed: {e}")
            return False
    
    def test_model_consistency(
        self,
        model: nn.Module,
        input_shape: tuple,
        num_runs: int = 5
    ) -> bool:
        """Test de consistencia"""
        try:
            model.eval()
            dummy_input = torch.randn(1, *input_shape)
            
            outputs = []
            with torch.no_grad():
                for _ in range(num_runs):
                    output = model(dummy_input)
                    outputs.append(output)
            
            # Verificar que outputs son consistentes
            first_output = outputs[0]
            for output in outputs[1:]:
                assert torch.allclose(first_output, output, atol=1e-5), "Inconsistent outputs"
            
            logger.info("Consistency test passed")
            return True
        except Exception as e:
            logger.error(f"Consistency test failed: {e}")
            return False
    
    def test_model_memory_leak(
        self,
        model: nn.Module,
        input_shape: tuple,
        num_iterations: int = 100
    ) -> bool:
        """Test de memory leak"""
        try:
            model.eval()
            initial_memory = self._get_memory_usage()
            
            with torch.no_grad():
                for _ in range(num_iterations):
                    dummy_input = torch.randn(1, *input_shape)
                    _ = model(dummy_input)
                    
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            final_memory = self._get_memory_usage()
            memory_increase = final_memory - initial_memory
            
            # Warning si aumenta mucho la memoria
            if memory_increase > 100:  # MB
                logger.warning(f"Potential memory leak: {memory_increase:.2f} MB increase")
                return False
            
            logger.info("Memory leak test passed")
            return True
        except Exception as e:
            logger.error(f"Memory leak test failed: {e}")
            return False
    
    def _get_memory_usage(self) -> float:
        """Obtiene uso de memoria"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)  # MB
        else:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)  # MB
    
    def run_all_tests(self, model: nn.Module, input_shape: tuple) -> Dict[str, Any]:
        """Ejecuta todos los tests"""
        results = {
            "forward": self.test_model_forward(model, input_shape),
            "gradient": self.test_model_gradient(model, input_shape),
            "consistency": self.test_model_consistency(model, input_shape),
            "memory_leak": self.test_model_memory_leak(model, input_shape)
        }
        
        passed = sum(1 for v in results.values() if v)
        total = len(results)
        
        self.test_results.append({
            "model": model.__class__.__name__,
            "results": results,
            "passed": passed,
            "total": total,
            "pass_rate": passed / total if total > 0 else 0.0
        })
        
        return {
            "results": results,
            "passed": passed,
            "total": total,
            "pass_rate": passed / total if total > 0 else 0.0
        }




