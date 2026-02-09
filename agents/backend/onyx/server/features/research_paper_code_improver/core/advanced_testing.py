"""
Advanced Model Testing - Testing avanzado de modelos
======================================================
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Resultado de test"""
    test_name: str
    passed: bool
    message: str
    metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class AdvancedModelTester:
    """Tester avanzado de modelos"""
    
    def __init__(self):
        self.test_results: List[TestResult] = []
    
    def test_model_inference(
        self,
        model: nn.Module,
        example_input: torch.Tensor,
        device: str = "cuda"
    ) -> TestResult:
        """Test de inferencia básica"""
        device = torch.device(device)
        model = model.to(device)
        model.eval()
        
        try:
            with torch.no_grad():
                output = model(example_input.to(device))
            
            result = TestResult(
                test_name="inference",
                passed=True,
                message="Inferencia exitosa",
                metrics={"output_shape": list(output.shape) if hasattr(output, 'shape') else []}
            )
        except Exception as e:
            result = TestResult(
                test_name="inference",
                passed=False,
                message=f"Error en inferencia: {e}"
            )
        
        self.test_results.append(result)
        return result
    
    def test_model_gradient_flow(
        self,
        model: nn.Module,
        example_input: torch.Tensor,
        device: str = "cuda"
    ) -> TestResult:
        """Test de flujo de gradientes"""
        device = torch.device(device)
        model = model.to(device)
        model.train()
        
        try:
            output = model(example_input.to(device))
            if hasattr(output, 'logits'):
                loss = output.logits.sum()
            else:
                loss = output.sum()
            
            loss.backward()
            
            # Verificar gradientes
            has_grad = any(p.grad is not None for p in model.parameters())
            grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
            
            result = TestResult(
                test_name="gradient_flow",
                passed=has_grad and grad_norm > 0,
                message="Gradientes calculados correctamente" if has_grad else "No hay gradientes",
                metrics={"grad_norm": grad_norm}
            )
        except Exception as e:
            result = TestResult(
                test_name="gradient_flow",
                passed=False,
                message=f"Error en flujo de gradientes: {e}"
            )
        
        self.test_results.append(result)
        return result
    
    def test_model_output_range(
        self,
        model: nn.Module,
        example_input: torch.Tensor,
        expected_range: Tuple[float, float] = (0.0, 1.0),
        device: str = "cuda"
    ) -> TestResult:
        """Test de rango de salida"""
        device = torch.device(device)
        model = model.to(device)
        model.eval()
        
        try:
            with torch.no_grad():
                output = model(example_input.to(device))
                
                if hasattr(output, 'logits'):
                    output = output.logits
                
                min_val = output.min().item()
                max_val = output.max().item()
                
                in_range = min_val >= expected_range[0] and max_val <= expected_range[1]
                
                result = TestResult(
                    test_name="output_range",
                    passed=in_range,
                    message=f"Rango: [{min_val:.4f}, {max_val:.4f}]",
                    metrics={"min": min_val, "max": max_val}
                )
        except Exception as e:
            result = TestResult(
                test_name="output_range",
                passed=False,
                message=f"Error: {e}"
            )
        
        self.test_results.append(result)
        return result
    
    def run_all_tests(
        self,
        model: nn.Module,
        example_input: torch.Tensor,
        device: str = "cuda"
    ) -> List[TestResult]:
        """Ejecuta todos los tests"""
        tests = [
            self.test_model_inference(model, example_input, device),
            self.test_model_gradient_flow(model, example_input, device),
            self.test_model_output_range(model, example_input, device=device)
        ]
        
        return tests
    
    def get_test_summary(self) -> Dict[str, Any]:
        """Obtiene resumen de tests"""
        if not self.test_results:
            return {}
        
        passed = sum(1 for r in self.test_results if r.passed)
        total = len(self.test_results)
        
        return {
            "total_tests": total,
            "passed_tests": passed,
            "failed_tests": total - passed,
            "pass_rate": passed / total if total > 0 else 0,
            "results": [
                {
                    "test_name": r.test_name,
                    "passed": r.passed,
                    "message": r.message
                }
                for r in self.test_results
            ]
        }




