"""
Testing Utils - Utilidades para testing
"""

import logging
from typing import Dict, Any, Optional, List
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)


class TestRunner:
    """Ejecutor de tests"""

    def __init__(self):
        """Inicializar ejecutor de tests"""
        self.test_results: List[Dict[str, Any]] = []

    async def run_test(
        self,
        test_name: str,
        test_func: callable,
        *args,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Ejecutar un test.

        Args:
            test_name: Nombre del test
            test_func: Función de test
            *args: Argumentos posicionales
            **kwargs: Argumentos con nombre

        Returns:
            Resultado del test
        """
        start_time = datetime.utcnow()
        success = False
        error = None
        
        try:
            if asyncio.iscoroutinefunction(test_func):
                await test_func(*args, **kwargs)
            else:
                test_func(*args, **kwargs)
            success = True
        except Exception as e:
            error = str(e)
            logger.error(f"Test falló: {test_name} - {error}")
        
        duration = (datetime.utcnow() - start_time).total_seconds()
        
        result = {
            "test_name": test_name,
            "success": success,
            "duration": duration,
            "error": error,
            "timestamp": start_time.isoformat()
        }
        
        self.test_results.append(result)
        return result

    async def run_test_suite(
        self,
        tests: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Ejecutar suite de tests.

        Args:
            tests: Lista de tests

        Returns:
            Resumen de resultados
        """
        results = []
        
        for test in tests:
            result = await self.run_test(
                test["name"],
                test["func"],
                *test.get("args", []),
                **test.get("kwargs", {})
            )
            results.append(result)
        
        total = len(results)
        passed = sum(1 for r in results if r["success"])
        failed = total - passed
        
        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "success_rate": passed / total if total > 0 else 0.0,
            "results": results
        }

    def get_test_results(self) -> List[Dict[str, Any]]:
        """
        Obtener resultados de tests.

        Returns:
            Lista de resultados
        """
        return self.test_results


class MockContentEditor:
    """Editor de contenido mock para testing"""

    def __init__(self):
        """Inicializar editor mock"""
        self.operations = []

    async def add(self, content: str, addition: str, **kwargs) -> Dict[str, Any]:
        """Mock de add"""
        result = {
            "success": True,
            "content": content + "\n\n" + addition,
            "validation": {"valid": True}
        }
        self.operations.append(("add", result))
        return result

    async def remove(self, content: str, pattern: str, **kwargs) -> Dict[str, Any]:
        """Mock de remove"""
        result = {
            "success": True,
            "content": content.replace(pattern, ""),
            "validation": {"valid": True}
        }
        self.operations.append(("remove", result))
        return result


class PerformanceBenchmark:
    """Benchmark de rendimiento"""

    def __init__(self):
        """Inicializar benchmark"""
        pass

    async def benchmark_operation(
        self,
        operation: callable,
        iterations: int = 100,
        *args,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Hacer benchmark de una operación.

        Args:
            operation: Operación a medir
            iterations: Número de iteraciones
            *args: Argumentos posicionales
            **kwargs: Argumentos con nombre

        Returns:
            Resultados del benchmark
        """
        import time
        
        durations = []
        
        for i in range(iterations):
            start = time.time()
            if asyncio.iscoroutinefunction(operation):
                await operation(*args, **kwargs)
            else:
                operation(*args, **kwargs)
            duration = time.time() - start
            durations.append(duration)
        
        return {
            "iterations": iterations,
            "total_time": sum(durations),
            "avg_time": sum(durations) / len(durations),
            "min_time": min(durations),
            "max_time": max(durations),
            "median_time": sorted(durations)[len(durations) // 2],
            "ops_per_second": iterations / sum(durations) if sum(durations) > 0 else 0
        }






