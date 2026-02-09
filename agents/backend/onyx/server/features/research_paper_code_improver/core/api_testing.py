"""
API Testing Framework - Framework de testing para APIs
======================================================
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class TestStatus(Enum):
    """Estados de test"""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TestCase:
    """Caso de prueba"""
    id: str
    name: str
    method: str
    url: str
    headers: Dict[str, str] = field(default_factory=dict)
    body: Optional[Any] = None
    expected_status: int = 200
    expected_response: Optional[Dict[str, Any]] = None
    assertions: List[Callable] = field(default_factory=list)
    setup: Optional[Callable] = None
    teardown: Optional[Callable] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario"""
        return {
            "id": self.id,
            "name": self.name,
            "method": self.method,
            "url": self.url,
            "expected_status": self.expected_status
        }


@dataclass
class TestResult:
    """Resultado de un test"""
    test_id: str
    test_name: str
    status: TestStatus
    duration: float
    response_status: Optional[int] = None
    response_body: Optional[Any] = None
    error: Optional[str] = None
    assertions_passed: int = 0
    assertions_failed: int = 0
    executed_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario"""
        return {
            "test_id": self.test_id,
            "test_name": self.test_name,
            "status": self.status.value,
            "duration": self.duration,
            "response_status": self.response_status,
            "error": self.error,
            "assertions_passed": self.assertions_passed,
            "assertions_failed": self.assertions_failed,
            "executed_at": self.executed_at.isoformat()
        }


class APITestingFramework:
    """Framework de testing para APIs"""
    
    def __init__(self, base_url: str = ""):
        self.base_url = base_url
        self.test_cases: Dict[str, TestCase] = {}
        self.test_results: List[TestResult] = []
        self.test_suites: Dict[str, List[str]] = {}  # suite_name -> test_ids
    
    def create_test_case(
        self,
        test_id: str,
        name: str,
        method: str,
        url: str,
        expected_status: int = 200,
        headers: Optional[Dict[str, str]] = None,
        body: Optional[Any] = None,
        expected_response: Optional[Dict[str, Any]] = None,
        assertions: Optional[List[Callable]] = None,
        setup: Optional[Callable] = None,
        teardown: Optional[Callable] = None
    ) -> TestCase:
        """Crea un caso de prueba"""
        test_case = TestCase(
            id=test_id,
            name=name,
            method=method.upper(),
            url=url,
            headers=headers or {},
            body=body,
            expected_status=expected_status,
            expected_response=expected_response,
            assertions=assertions or [],
            setup=setup,
            teardown=teardown
        )
        
        self.test_cases[test_id] = test_case
        logger.info(f"Test case {test_id} creado")
        return test_case
    
    async def run_test(self, test_id: str) -> TestResult:
        """Ejecuta un test"""
        if test_id not in self.test_cases:
            raise ValueError(f"Test {test_id} no encontrado")
        
        test_case = self.test_cases[test_id]
        start_time = datetime.now()
        
        result = TestResult(
            test_id=test_id,
            test_name=test_case.name,
            status=TestStatus.RUNNING
        )
        
        try:
            # Setup
            if test_case.setup:
                if asyncio.iscoroutinefunction(test_case.setup):
                    await test_case.setup()
                else:
                    test_case.setup()
            
            # Ejecutar request
            import httpx
            full_url = f"{self.base_url}{test_case.url}" if self.base_url else test_case.url
            
            async with httpx.AsyncClient() as client:
                response = await client.request(
                    method=test_case.method,
                    url=full_url,
                    headers=test_case.headers,
                    json=test_case.body if isinstance(test_case.body, dict) else None,
                    content=test_case.body if isinstance(test_case.body, bytes) else None
                )
                
                result.response_status = response.status_code
                try:
                    result.response_body = response.json()
                except:
                    result.response_body = response.text
            
            # Verificar status
            if result.response_status != test_case.expected_status:
                result.status = TestStatus.FAILED
                result.error = f"Expected status {test_case.expected_status}, got {result.response_status}"
            else:
                # Ejecutar assertions
                for assertion in test_case.assertions:
                    try:
                        if asyncio.iscoroutinefunction(assertion):
                            await assertion(result.response_body)
                        else:
                            assertion(result.response_body)
                        result.assertions_passed += 1
                    except Exception as e:
                        result.assertions_failed += 1
                        result.error = str(e)
                
                # Verificar expected response
                if test_case.expected_response:
                    if not self._match_response(result.response_body, test_case.expected_response):
                        result.status = TestStatus.FAILED
                        result.error = "Response doesn't match expected"
                    else:
                        result.status = TestStatus.PASSED
                elif result.assertions_failed == 0:
                    result.status = TestStatus.PASSED
                else:
                    result.status = TestStatus.FAILED
            
            # Teardown
            if test_case.teardown:
                if asyncio.iscoroutinefunction(test_case.teardown):
                    await test_case.teardown()
                else:
                    test_case.teardown()
        
        except Exception as e:
            result.status = TestStatus.FAILED
            result.error = str(e)
            logger.error(f"Error ejecutando test {test_id}: {e}")
        
        result.duration = (datetime.now() - start_time).total_seconds()
        self.test_results.append(result)
        
        return result
    
    def _match_response(self, actual: Any, expected: Dict[str, Any]) -> bool:
        """Verifica si la respuesta coincide con lo esperado"""
        if not isinstance(actual, dict):
            return False
        
        for key, expected_value in expected.items():
            if key not in actual:
                return False
            if actual[key] != expected_value:
                return False
        
        return True
    
    async def run_test_suite(self, suite_name: str) -> List[TestResult]:
        """Ejecuta una suite de tests"""
        if suite_name not in self.test_suites:
            raise ValueError(f"Test suite {suite_name} no encontrada")
        
        test_ids = self.test_suites[suite_name]
        results = []
        
        for test_id in test_ids:
            result = await self.run_test(test_id)
            results.append(result)
        
        return results
    
    def create_test_suite(self, suite_name: str, test_ids: List[str]):
        """Crea una suite de tests"""
        self.test_suites[suite_name] = test_ids
        logger.info(f"Test suite {suite_name} creada con {len(test_ids)} tests")
    
    def get_test_results(
        self,
        test_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Obtiene resultados de tests"""
        results = self.test_results
        
        if test_id:
            results = [r for r in results if r.test_id == test_id]
        
        results.sort(key=lambda r: r.executed_at, reverse=True)
        return [r.to_dict() for r in results[:limit]]
    
    def get_test_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas de tests"""
        total = len(self.test_results)
        passed = len([r for r in self.test_results if r.status == TestStatus.PASSED])
        failed = len([r for r in self.test_results if r.status == TestStatus.FAILED])
        
        return {
            "total_tests": total,
            "passed": passed,
            "failed": failed,
            "success_rate": passed / total if total > 0 else 0,
            "test_cases_count": len(self.test_cases),
            "test_suites_count": len(self.test_suites)
        }

