"""
Testing Framework para LLMs.

Permite crear y ejecutar tests automatizados para evaluar
la calidad y consistencia de respuestas de modelos LLM.
"""

import json
import asyncio
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
import os

from config.logging_config import get_logger

logger = get_logger(__name__)


class TestType(str, Enum):
    """Tipos de tests."""
    FUNCTIONAL = "functional"
    QUALITY = "quality"
    CONSISTENCY = "consistency"
    PERFORMANCE = "performance"
    SAFETY = "safety"
    CUSTOM = "custom"


class AssertionType(str, Enum):
    """Tipos de aserciones."""
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    EQUALS = "equals"
    REGEX = "regex"
    LENGTH = "length"
    CUSTOM = "custom"


@dataclass
class TestAssertion:
    """Aserción en un test."""
    name: str
    assertion_type: AssertionType
    expected: Any
    custom_function: Optional[str] = None  # Código Python como string para funciones custom
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "name": self.name,
            "assertion_type": self.assertion_type.value,
            "expected": self.expected,
            "custom_function": self.custom_function
        }


@dataclass
class TestCase:
    """Caso de prueba individual."""
    case_id: str
    name: str
    prompt: str
    system_prompt: Optional[str] = None
    assertions: List[TestAssertion] = field(default_factory=list)
    expected_output: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "case_id": self.case_id,
            "name": self.name,
            "prompt": self.prompt,
            "system_prompt": self.system_prompt,
            "assertions": [a.to_dict() for a in self.assertions],
            "expected_output": self.expected_output,
            "metadata": self.metadata
        }


@dataclass
class TestResult:
    """Resultado de ejecutar un test case."""
    case_id: str
    passed: bool
    response: str
    latency_ms: float
    tokens_used: int
    assertions_results: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "case_id": self.case_id,
            "passed": self.passed,
            "response": self.response,
            "latency_ms": self.latency_ms,
            "tokens_used": self.tokens_used,
            "assertions_results": self.assertions_results,
            "error": self.error,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class TestSuite:
    """Suite de tests."""
    suite_id: str
    name: str
    description: str
    test_type: TestType
    test_cases: List[TestCase]
    model: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "suite_id": self.suite_id,
            "name": self.name,
            "description": self.description,
            "test_type": self.test_type.value,
            "test_cases": [tc.to_dict() for tc in self.test_cases],
            "model": self.model,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class TestSuiteResult:
    """Resultado de ejecutar una suite de tests."""
    suite_id: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    results: List[TestResult]
    avg_latency_ms: float
    total_tokens: int
    execution_time: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "suite_id": self.suite_id,
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "failed_tests": self.failed_tests,
            "results": [r.to_dict() for r in self.results],
            "avg_latency_ms": self.avg_latency_ms,
            "total_tokens": self.total_tokens,
            "execution_time": self.execution_time,
            "timestamp": self.timestamp.isoformat()
        }


class LLMTestingFramework:
    """
    Framework para testing de LLMs.
    
    Características:
    - Creación de test suites
    - Ejecución de tests
    - Aserciones personalizables
    - Reportes detallados
    - Integración con CI/CD
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Inicializar framework de testing.
        
        Args:
            storage_path: Ruta para almacenar tests (opcional)
        """
        self.storage_path = storage_path or "data/llm_tests"
        self.test_suites: Dict[str, TestSuite] = {}
        self.test_results: Dict[str, List[TestSuiteResult]] = {}
        
        # Crear directorio si no existe
        os.makedirs(self.storage_path, exist_ok=True)
        
        # Cargar tests existentes
        self._load_test_suites()
    
    def create_test_suite(
        self,
        name: str,
        description: str,
        test_type: TestType,
        test_cases: List[TestCase],
        model: Optional[str] = None,
        suite_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Crear una nueva suite de tests.
        
        Args:
            name: Nombre de la suite
            description: Descripción
            test_type: Tipo de test
            test_cases: Casos de prueba
            model: Modelo a usar (opcional)
            suite_id: ID personalizado (opcional)
            metadata: Metadatos adicionales
            
        Returns:
            ID de la suite creada
        """
        import hashlib
        
        if suite_id is None:
            suite_id = hashlib.md5(
                f"{name}{description}{datetime.now().isoformat()}".encode()
            ).hexdigest()[:12]
        
        suite = TestSuite(
            suite_id=suite_id,
            name=name,
            description=description,
            test_type=test_type,
            test_cases=test_cases,
            model=model,
            metadata=metadata or {}
        )
        
        self.test_suites[suite_id] = suite
        self._save_test_suite(suite)
        
        logger.info(f"Test suite creada: {suite_id} - {name}")
        return suite_id
    
    def get_test_suite(self, suite_id: str) -> Optional[TestSuite]:
        """Obtener suite de tests por ID."""
        return self.test_suites.get(suite_id)
    
    def list_test_suites(self, test_type: Optional[TestType] = None) -> List[TestSuite]:
        """
        Listar todas las suites de tests.
        
        Args:
            test_type: Filtrar por tipo (opcional)
            
        Returns:
            Lista de suites
        """
        suites = list(self.test_suites.values())
        if test_type:
            suites = [s for s in suites if s.test_type == test_type]
        return suites
    
    async def run_test_suite(
        self,
        suite_id: str,
        llm_service: Any,  # LLMService
        model: Optional[str] = None
    ) -> Optional[TestSuiteResult]:
        """
        Ejecutar una suite de tests.
        
        Args:
            suite_id: ID de la suite
            llm_service: Instancia de LLMService
            model: Modelo a usar (sobrescribe el de la suite si se proporciona)
            
        Returns:
            Resultado de la ejecución o None
        """
        suite = self.test_suites.get(suite_id)
        if not suite:
            return None
        
        model_to_use = model or suite.model
        if not model_to_use:
            logger.error(f"No se especificó modelo para suite {suite_id}")
            return None
        
        start_time = datetime.now()
        results = []
        
        # Ejecutar tests en paralelo (con límite)
        semaphore = asyncio.Semaphore(5)  # Máximo 5 tests en paralelo
        
        async def run_test_case(test_case: TestCase) -> TestResult:
            async with semaphore:
                return await self._run_test_case(test_case, llm_service, model_to_use)
        
        test_results = await asyncio.gather(
            *[run_test_case(tc) for tc in suite.test_cases],
            return_exceptions=True
        )
        
        # Procesar resultados
        for i, result in enumerate(test_results):
            if isinstance(result, Exception):
                # Crear resultado de error
                test_case = suite.test_cases[i]
                results.append(TestResult(
                    case_id=test_case.case_id,
                    passed=False,
                    response="",
                    latency_ms=0,
                    tokens_used=0,
                    error=str(result)
                ))
            else:
                results.append(result)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        passed = sum(1 for r in results if r.passed)
        failed = len(results) - passed
        avg_latency = sum(r.latency_ms for r in results) / len(results) if results else 0
        total_tokens = sum(r.tokens_used for r in results)
        
        suite_result = TestSuiteResult(
            suite_id=suite_id,
            total_tests=len(results),
            passed_tests=passed,
            failed_tests=failed,
            results=results,
            avg_latency_ms=avg_latency,
            total_tokens=total_tokens,
            execution_time=execution_time
        )
        
        # Guardar resultado
        if suite_id not in self.test_results:
            self.test_results[suite_id] = []
        self.test_results[suite_id].append(suite_result)
        self._save_test_result(suite_id, suite_result)
        
        logger.info(
            f"Test suite {suite_id} ejecutada: {passed}/{len(results)} tests pasaron"
        )
        
        return suite_result
    
    async def _run_test_case(
        self,
        test_case: TestCase,
        llm_service: Any,
        model: str
    ) -> TestResult:
        """
        Ejecutar un caso de prueba individual.
        
        Args:
            test_case: Caso de prueba
            llm_service: Servicio LLM
            model: Modelo a usar
            
        Returns:
            Resultado del test
        """
        import time
        
        start_time = time.time()
        
        try:
            # Ejecutar generación
            response = await llm_service.generate(
                prompt=test_case.prompt,
                system_prompt=test_case.system_prompt,
                model=model
            )
            
            latency_ms = (time.time() - start_time) * 1000
            tokens_used = response.usage.get("total_tokens", 0) if response.usage else 0
            
            # Ejecutar aserciones
            assertions_results = []
            all_passed = True
            
            for assertion in test_case.assertions:
                assertion_result = self._evaluate_assertion(
                    assertion,
                    response.content
                )
                assertions_results.append(assertion_result)
                if not assertion_result.get("passed", False):
                    all_passed = False
            
            return TestResult(
                case_id=test_case.case_id,
                passed=all_passed,
                response=response.content,
                latency_ms=latency_ms,
                tokens_used=tokens_used,
                assertions_results=assertions_results
            )
        
        except Exception as e:
            logger.error(f"Error ejecutando test case {test_case.case_id}: {e}")
            return TestResult(
                case_id=test_case.case_id,
                passed=False,
                response="",
                latency_ms=(time.time() - start_time) * 1000,
                tokens_used=0,
                error=str(e)
            )
    
    def _evaluate_assertion(
        self,
        assertion: TestAssertion,
        response: str
    ) -> Dict[str, Any]:
        """
        Evaluar una aserción.
        
        Args:
            assertion: Aserción a evaluar
            response: Respuesta del modelo
            
        Returns:
            Resultado de la aserción
        """
        result = {
            "name": assertion.name,
            "assertion_type": assertion.assertion_type.value,
            "passed": False,
            "message": ""
        }
        
        try:
            if assertion.assertion_type == AssertionType.CONTAINS:
                result["passed"] = assertion.expected.lower() in response.lower()
                result["message"] = (
                    f"Expected '{assertion.expected}' in response"
                    if result["passed"]
                    else f"Expected '{assertion.expected}' not found in response"
                )
            
            elif assertion.assertion_type == AssertionType.NOT_CONTAINS:
                result["passed"] = assertion.expected.lower() not in response.lower()
                result["message"] = (
                    f"Expected '{assertion.expected}' not in response"
                    if result["passed"]
                    else f"Unexpected '{assertion.expected}' found in response"
                )
            
            elif assertion.assertion_type == AssertionType.EQUALS:
                result["passed"] = response.strip() == assertion.expected.strip()
                result["message"] = (
                    "Response matches expected"
                    if result["passed"]
                    else "Response does not match expected"
                )
            
            elif assertion.assertion_type == AssertionType.REGEX:
                import re
                pattern = assertion.expected
                result["passed"] = bool(re.search(pattern, response))
                result["message"] = (
                    f"Response matches regex '{pattern}'"
                    if result["passed"]
                    else f"Response does not match regex '{pattern}'"
                )
            
            elif assertion.assertion_type == AssertionType.LENGTH:
                expected_length = assertion.expected
                if isinstance(expected_length, dict):
                    min_len = expected_length.get("min", 0)
                    max_len = expected_length.get("max", float('inf'))
                    actual_len = len(response)
                    result["passed"] = min_len <= actual_len <= max_len
                    result["message"] = (
                        f"Response length {actual_len} is within range [{min_len}, {max_len}]"
                        if result["passed"]
                        else f"Response length {actual_len} is outside range [{min_len}, {max_len}]"
                    )
                else:
                    result["passed"] = len(response) == expected_length
                    result["message"] = (
                        f"Response length {len(response)} matches expected {expected_length}"
                        if result["passed"]
                        else f"Response length {len(response)} does not match expected {expected_length}"
                    )
            
            elif assertion.assertion_type == AssertionType.CUSTOM:
                # Ejecutar función custom (cuidado con seguridad)
                if assertion.custom_function:
                    try:
                        # En producción, usar un sandbox o validar el código
                        namespace = {"response": response, "assertion": assertion}
                        exec(assertion.custom_function, namespace)
                        result["passed"] = namespace.get("result", False)
                        result["message"] = namespace.get("message", "Custom assertion executed")
                    except Exception as e:
                        result["passed"] = False
                        result["message"] = f"Error in custom assertion: {e}"
        
        except Exception as e:
            result["passed"] = False
            result["message"] = f"Error evaluating assertion: {e}"
        
        return result
    
    def get_test_results(
        self,
        suite_id: str,
        limit: Optional[int] = None
    ) -> List[TestSuiteResult]:
        """
        Obtener resultados de una suite.
        
        Args:
            suite_id: ID de la suite
            limit: Límite de resultados (opcional)
            
        Returns:
            Lista de resultados
        """
        results = self.test_results.get(suite_id, [])
        if limit:
            results = results[-limit:]
        return results
    
    def _save_test_suite(self, suite: TestSuite) -> None:
        """Guardar suite en disco."""
        file_path = os.path.join(self.storage_path, f"suite_{suite.suite_id}.json")
        with open(file_path, 'w') as f:
            json.dump(suite.to_dict(), f, indent=2, default=str)
    
    def _save_test_result(self, suite_id: str, result: TestSuiteResult) -> None:
        """Guardar resultado en disco."""
        file_path = os.path.join(
            self.storage_path,
            f"result_{suite_id}_{result.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(file_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
    
    def _load_test_suites(self) -> None:
        """Cargar suites desde disco."""
        if not os.path.exists(self.storage_path):
            return
        
        for filename in os.listdir(self.storage_path):
            if filename.startswith("suite_") and filename.endswith(".json"):
                file_path = os.path.join(self.storage_path, filename)
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    test_cases = []
                    for tc_data in data.get("test_cases", []):
                        assertions = [
                            TestAssertion(
                                name=a["name"],
                                assertion_type=AssertionType(a["assertion_type"]),
                                expected=a["expected"],
                                custom_function=a.get("custom_function")
                            )
                            for a in tc_data.get("assertions", [])
                        ]
                        
                        test_cases.append(TestCase(
                            case_id=tc_data["case_id"],
                            name=tc_data["name"],
                            prompt=tc_data["prompt"],
                            system_prompt=tc_data.get("system_prompt"),
                            assertions=assertions,
                            expected_output=tc_data.get("expected_output"),
                            metadata=tc_data.get("metadata", {})
                        ))
                    
                    suite = TestSuite(
                        suite_id=data["suite_id"],
                        name=data["name"],
                        description=data["description"],
                        test_type=TestType(data["test_type"]),
                        test_cases=test_cases,
                        model=data.get("model"),
                        metadata=data.get("metadata", {})
                    )
                    
                    if isinstance(data.get("created_at"), str):
                        suite.created_at = datetime.fromisoformat(data["created_at"])
                    
                    self.test_suites[suite.suite_id] = suite
                except Exception as e:
                    logger.error(f"Error cargando suite desde {filename}: {e}")


def get_llm_testing_framework(storage_path: Optional[str] = None) -> LLMTestingFramework:
    """Factory function para obtener instancia singleton del framework."""
    if not hasattr(get_llm_testing_framework, "_instance"):
        get_llm_testing_framework._instance = LLMTestingFramework(storage_path)
    return get_llm_testing_framework._instance



