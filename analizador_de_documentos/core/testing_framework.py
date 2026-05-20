"""
Sistema de Testing Automatizado
=================================

Framework para testing automatizado de modelos y análisis.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class TestStatus(Enum):
    """Estado de test"""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class TestCase:
    """Caso de prueba"""
    name: str
    description: str
    test_function: Callable
    expected_result: Any
    timeout: float = 30.0
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


@dataclass
class TestResult:
    """Resultado de test"""
    test_name: str
    status: TestStatus
    duration: float
    error: Optional[str] = None
    actual_result: Any = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class TestingFramework:
    """
    Framework de testing
    
    Proporciona:
    - Ejecución de tests automatizados
    - Assertions personalizadas
    - Reportes de tests
    - Tests de regresión
    - Tests de rendimiento
    """
    
    def __init__(self):
        """Inicializar framework"""
        self.test_cases: Dict[str, TestCase] = {}
        self.test_results: List[TestResult] = []
        logger.info("TestingFramework inicializado")
    
    def register_test(
        self,
        name: str,
        description: str,
        test_function: Callable,
        expected_result: Any = None,
        timeout: float = 30.0,
        tags: Optional[List[str]] = None
    ):
        """Registrar caso de prueba"""
        test_case = TestCase(
            name=name,
            description=description,
            test_function=test_function,
            expected_result=expected_result,
            timeout=timeout,
            tags=tags or []
        )
        
        self.test_cases[name] = test_case
        logger.info(f"Test registrado: {name}")
    
    async def run_test(self, test_name: str) -> TestResult:
        """Ejecutar test individual"""
        if test_name not in self.test_cases:
            return TestResult(
                test_name=test_name,
                status=TestStatus.ERROR,
                duration=0.0,
                error=f"Test no encontrado: {test_name}"
            )
        
        test_case = self.test_cases[test_name]
        start_time = datetime.now()
        
        try:
            # Ejecutar test con timeout
            if asyncio.iscoroutinefunction(test_case.test_function):
                actual_result = await asyncio.wait_for(
                    test_case.test_function(),
                    timeout=test_case.timeout
                )
            else:
                actual_result = test_case.test_function()
            
            duration = (datetime.now() - start_time).total_seconds()
            
            # Verificar resultado esperado si existe
            if test_case.expected_result is not None:
                if actual_result != test_case.expected_result:
                    return TestResult(
                        test_name=test_name,
                        status=TestStatus.FAILED,
                        duration=duration,
                        actual_result=actual_result,
                        error=f"Resultado esperado: {test_case.expected_result}, obtenido: {actual_result}"
                    )
            
            return TestResult(
                test_name=test_name,
                status=TestStatus.PASSED,
                duration=duration,
                actual_result=actual_result
            )
            
        except asyncio.TimeoutError:
            duration = (datetime.now() - start_time).total_seconds()
            return TestResult(
                test_name=test_name,
                status=TestStatus.ERROR,
                duration=duration,
                error=f"Test timeout después de {test_case.timeout}s"
            )
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            return TestResult(
                test_name=test_name,
                status=TestStatus.ERROR,
                duration=duration,
                error=str(e)
            )
    
    async def run_all_tests(
        self,
        filter_tags: Optional[List[str]] = None
    ) -> List[TestResult]:
        """Ejecutar todos los tests"""
        results = []
        
        tests_to_run = self.test_cases
        if filter_tags:
            tests_to_run = {
                name: test for name, test in self.test_cases.items()
                if any(tag in test.tags for tag in filter_tags)
            }
        
        for test_name in tests_to_run:
            result = await self.run_test(test_name)
            results.append(result)
            self.test_results.append(result)
        
        return results
    
    def get_test_summary(self) -> Dict[str, Any]:
        """Obtener resumen de tests"""
        if not self.test_results:
            return {"total": 0, "passed": 0, "failed": 0, "error": 0}
        
        total = len(self.test_results)
        passed = sum(1 for r in self.test_results if r.status == TestStatus.PASSED)
        failed = sum(1 for r in self.test_results if r.status == TestStatus.FAILED)
        error = sum(1 for r in self.test_results if r.status == TestStatus.ERROR)
        
        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "error": error,
            "pass_rate": passed / total if total > 0 else 0.0,
            "avg_duration": sum(r.duration for r in self.test_results) / total if total > 0 else 0.0
        }
    
    def generate_report(self) -> str:
        """Generar reporte de tests"""
        summary = self.get_test_summary()
        
        report = f"# Reporte de Tests\n\n"
        report += f"**Fecha**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        report += f"## Resumen\n\n"
        report += f"- **Total**: {summary['total']}\n"
        report += f"- **Pasados**: {summary['passed']}\n"
        report += f"- **Fallidos**: {summary['failed']}\n"
        report += f"- **Errores**: {summary['error']}\n"
        report += f"- **Tasa de éxito**: {summary['pass_rate']:.2%}\n"
        report += f"- **Duración promedio**: {summary['avg_duration']:.2f}s\n\n"
        
        report += "## Detalles\n\n"
        for result in self.test_results:
            status_icon = "✅" if result.status == TestStatus.PASSED else "❌"
            report += f"### {status_icon} {result.test_name}\n\n"
            report += f"- **Estado**: {result.status.value}\n"
            report += f"- **Duración**: {result.duration:.2f}s\n"
            if result.error:
                report += f"- **Error**: {result.error}\n"
            report += "\n"
        
        return report


# Instancia global
_testing_framework: Optional[TestingFramework] = None


def get_testing_framework() -> TestingFramework:
    """Obtener instancia global del framework"""
    global _testing_framework
    if _testing_framework is None:
        _testing_framework = TestingFramework()
    return _testing_framework
















