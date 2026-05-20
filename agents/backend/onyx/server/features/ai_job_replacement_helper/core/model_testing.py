"""
Model Testing - Testing de modelos
==================================

Sistema para testing de modelos antes de producción.
Sigue mejores prácticas de testing en deep learning.
"""

import logging
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Resultado de test"""
    test_name: str
    passed: bool
    message: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestSuite:
    """Suite de tests"""
    name: str
    results: List[TestResult] = field(default_factory=list)
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    
    def add_result(self, result: TestResult) -> None:
        """Agregar resultado de test"""
        self.results.append(result)
        self.total_tests += 1
        if result.passed:
            self.passed_tests += 1
        else:
            self.failed_tests += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Obtener resumen de tests"""
        return {
            "name": self.name,
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "failed_tests": self.failed_tests,
            "pass_rate": self.passed_tests / self.total_tests if self.total_tests > 0 else 0.0,
        }


class ModelTestingService:
    """Servicio de testing de modelos"""
    
    def __init__(self):
        """Inicializar servicio"""
        self.test_suites: Dict[str, TestSuite] = {}
        logger.info("ModelTestingService initialized")
    
    def test_forward_pass(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        device: Optional[torch.device] = None
    ) -> TestResult:
        """
        Test de forward pass.
        
        Args:
            model: Modelo a testear
            input_shape: Forma del input (sin batch dimension)
            device: Dispositivo (None = auto)
        
        Returns:
            TestResult
        """
        test_name = "forward_pass"
        
        try:
            device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            model.eval()
            
            # Create dummy input
            dummy_input = torch.randn(1, *input_shape).to(device)
            
            # Forward pass
            with torch.no_grad():
                output = model(dummy_input)
            
            # Validate output
            if isinstance(output, torch.Tensor):
                if torch.isnan(output).any():
                    return TestResult(
                        test_name=test_name,
                        passed=False,
                        message="Output contains NaN values"
                    )
                
                if torch.isinf(output).any():
                    return TestResult(
                        test_name=test_name,
                        passed=False,
                        message="Output contains Inf values"
                    )
                
                return TestResult(
                    test_name=test_name,
                    passed=True,
                    message="Forward pass successful",
                    metadata={
                        "input_shape": list(dummy_input.shape),
                        "output_shape": list(output.shape),
                    }
                )
            else:
                return TestResult(
                    test_name=test_name,
                    passed=True,
                    message="Forward pass successful (non-tensor output)",
                )
        
        except Exception as e:
            return TestResult(
                test_name=test_name,
                passed=False,
                message=f"Forward pass failed: {str(e)}"
            )
    
    def test_backward_pass(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        device: Optional[torch.device] = None
    ) -> TestResult:
        """
        Test de backward pass.
        
        Args:
            model: Modelo a testear
            input_shape: Forma del input
            device: Dispositivo (None = auto)
        
        Returns:
            TestResult
        """
        test_name = "backward_pass"
        
        try:
            device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            model.train()
            
            # Create dummy input and target
            dummy_input = torch.randn(1, *input_shape).to(device)
            dummy_target = torch.randint(0, 10, (1,)).to(device)
            
            # Forward and backward
            output = model(dummy_input)
            
            if isinstance(output, torch.Tensor):
                if output.dim() > 1:
                    loss = nn.functional.cross_entropy(output, dummy_target)
                else:
                    loss = output.mean()
            else:
                loss = output.loss if hasattr(output, "loss") else output[0].mean()
            
            loss.backward()
            
            # Check gradients
            has_nan_grad = False
            has_inf_grad = False
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any():
                        has_nan_grad = True
                    if torch.isinf(param.grad).any():
                        has_inf_grad = True
            
            if has_nan_grad or has_inf_grad:
                return TestResult(
                    test_name=test_name,
                    passed=False,
                    message="Gradients contain NaN or Inf values"
                )
            
            # Clear gradients
            model.zero_grad()
            
            return TestResult(
                test_name=test_name,
                passed=True,
                message="Backward pass successful"
            )
        
        except Exception as e:
            return TestResult(
                test_name=test_name,
                passed=False,
                message=f"Backward pass failed: {str(e)}"
            )
    
    def test_model_consistency(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        num_runs: int = 5,
        device: Optional[torch.device] = None
    ) -> TestResult:
        """
        Test de consistencia del modelo (mismo input = mismo output).
        
        Args:
            model: Modelo a testear
            input_shape: Forma del input
            num_runs: Número de runs
            device: Dispositivo (None = auto)
        
        Returns:
            TestResult
        """
        test_name = "model_consistency"
        
        try:
            device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            model.eval()
            
            # Create fixed input
            torch.manual_seed(42)
            fixed_input = torch.randn(1, *input_shape).to(device)
            
            # Get reference output
            with torch.no_grad():
                reference_output = model(fixed_input)
                if isinstance(reference_output, torch.Tensor):
                    reference_output = reference_output.cpu()
            
            # Run multiple times
            outputs = []
            for _ in range(num_runs):
                with torch.no_grad():
                    output = model(fixed_input)
                    if isinstance(output, torch.Tensor):
                        outputs.append(output.cpu())
            
            # Check consistency
            if isinstance(reference_output, torch.Tensor):
                for output in outputs:
                    if not torch.allclose(reference_output, output, atol=1e-6):
                        return TestResult(
                            test_name=test_name,
                            passed=False,
                            message="Model outputs are not consistent"
                        )
            
            return TestResult(
                test_name=test_name,
                passed=True,
                message="Model is consistent",
                metadata={"num_runs": num_runs}
            )
        
        except Exception as e:
            return TestResult(
                test_name=test_name,
                passed=False,
                message=f"Consistency test failed: {str(e)}"
            )
    
    def run_test_suite(
        self,
        suite_name: str,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        device: Optional[torch.device] = None
    ) -> TestSuite:
        """
        Ejecutar suite completa de tests.
        
        Args:
            suite_name: Nombre de la suite
            model: Modelo a testear
            input_shape: Forma del input
            device: Dispositivo (None = auto)
        
        Returns:
            TestSuite con resultados
        """
        suite = TestSuite(name=suite_name)
        
        # Run tests
        suite.add_result(self.test_forward_pass(model, input_shape, device))
        suite.add_result(self.test_backward_pass(model, input_shape, device))
        suite.add_result(self.test_model_consistency(model, input_shape, device=device))
        
        self.test_suites[suite_name] = suite
        
        logger.info(
            f"Test suite '{suite_name}' completed: "
            f"{suite.passed_tests}/{suite.total_tests} tests passed"
        )
        
        return suite




