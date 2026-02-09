"""
Test Runner System
==================

Sistema de ejecución de tests avanzado.
"""

import subprocess
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Resultado de test."""
    test_name: str
    status: str  # "passed", "failed", "skipped", "error"
    duration: float
    message: Optional[str] = None
    traceback: Optional[str] = None


@dataclass
class TestSuiteResult:
    """Resultado de suite de tests."""
    suite_name: str
    total_tests: int
    passed: int
    failed: int
    skipped: int
    errors: int
    duration: float
    results: List[TestResult] = field(default_factory=list)
    coverage: Optional[float] = None


class TestRunner:
    """
    Ejecutor de tests.
    
    Ejecuta tests y genera reportes.
    """
    
    def __init__(self, test_directory: str = "tests"):
        """
        Inicializar ejecutor de tests.
        
        Args:
            test_directory: Directorio de tests
        """
        self.test_directory = Path(test_directory)
        self.test_history: List[TestSuiteResult] = []
    
    def run_tests(
        self,
        test_path: Optional[str] = None,
        verbose: bool = False,
        coverage: bool = False,
        parallel: bool = False
    ) -> TestSuiteResult:
        """
        Ejecutar tests.
        
        Args:
            test_path: Ruta de tests (None = todos)
            verbose: Modo verbose
            coverage: Incluir coverage
            parallel: Ejecutar en paralelo
            
        Returns:
            Resultado de la suite de tests
        """
        import time
        start_time = time.time()
        
        test_path = test_path or str(self.test_directory)
        logger.info(f"Running tests: {test_path}")
        
        # Construir comando pytest
        cmd = ["pytest", test_path]
        
        if verbose:
            cmd.append("-v")
        
        if coverage:
            cmd.extend(["--cov=robot_movement_ai", "--cov-report=term-missing"])
        
        if parallel:
            cmd.extend(["-n", "auto"])
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutos timeout
            )
            
            duration = time.time() - start_time
            
            # Parsear resultados
            suite_result = self._parse_pytest_output(
                result.stdout,
                result.stderr,
                duration,
                coverage
            )
            
            self.test_history.append(suite_result)
            logger.info(f"Tests completed: {suite_result.passed}/{suite_result.total_tests} passed")
            
            return suite_result
        
        except subprocess.TimeoutExpired:
            logger.error("Tests timed out")
            return TestSuiteResult(
                suite_name=test_path,
                total_tests=0,
                passed=0,
                failed=0,
                skipped=0,
                errors=1,
                duration=300.0
            )
        except Exception as e:
            logger.error(f"Error running tests: {e}")
            return TestSuiteResult(
                suite_name=test_path,
                total_tests=0,
                passed=0,
                failed=0,
                skipped=0,
                errors=1,
                duration=0.0
            )
    
    def _parse_pytest_output(
        self,
        stdout: str,
        stderr: str,
        duration: float,
        coverage: bool
    ) -> TestSuiteResult:
        """Parsear salida de pytest."""
        # Parsear básico (en producción, usar parser más robusto)
        lines = stdout.split('\n')
        
        total_tests = 0
        passed = 0
        failed = 0
        skipped = 0
        errors = 0
        coverage_value = None
        
        for line in lines:
            if "passed" in line.lower() and "failed" not in line.lower():
                # Intentar extraer números
                import re
                numbers = re.findall(r'\d+', line)
                if numbers:
                    total_tests = int(numbers[0])
                    if len(numbers) > 1:
                        passed = int(numbers[1])
            
            if "failed" in line.lower():
                numbers = re.findall(r'\d+', line)
                if numbers:
                    failed = int(numbers[0])
            
            if "skipped" in line.lower():
                numbers = re.findall(r'\d+', line)
                if numbers:
                    skipped = int(numbers[0])
            
            if "error" in line.lower() and "failed" not in line.lower():
                numbers = re.findall(r'\d+', line)
                if numbers:
                    errors = int(numbers[0])
            
            if coverage and "TOTAL" in line:
                # Extraer coverage
                numbers = re.findall(r'\d+', line)
                if numbers:
                    coverage_value = float(numbers[-1]) / 100.0
        
        return TestSuiteResult(
            suite_name="pytest",
            total_tests=total_tests or (passed + failed + skipped + errors),
            passed=passed,
            failed=failed,
            skipped=skipped,
            errors=errors,
            duration=duration,
            coverage=coverage_value
        )
    
    def get_test_history(self, limit: int = 100) -> List[TestSuiteResult]:
        """Obtener historial de tests."""
        return self.test_history[-limit:]
    
    def get_test_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas de tests."""
        if not self.test_history:
            return {
                "total_runs": 0,
                "average_pass_rate": 0.0,
                "average_coverage": 0.0
            }
        
        total_runs = len(self.test_history)
        total_tests = sum(r.total_tests for r in self.test_history)
        total_passed = sum(r.passed for r in self.test_history)
        
        pass_rate = total_passed / total_tests if total_tests > 0 else 0.0
        
        coverage_values = [r.coverage for r in self.test_history if r.coverage is not None]
        avg_coverage = sum(coverage_values) / len(coverage_values) if coverage_values else 0.0
        
        return {
            "total_runs": total_runs,
            "total_tests": total_tests,
            "total_passed": total_passed,
            "average_pass_rate": pass_rate,
            "average_coverage": avg_coverage
        }


# Instancia global
_test_runner: Optional[TestRunner] = None


def get_test_runner(test_directory: str = "tests") -> TestRunner:
    """Obtener instancia global del ejecutor de tests."""
    global _test_runner
    if _test_runner is None:
        _test_runner = TestRunner(test_directory=test_directory)
    return _test_runner






