"""
Performance Testing - Testing de performance y carga
====================================================
"""

import logging
import asyncio
import time
import statistics
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class LoadTestType(Enum):
    """Tipos de tests de carga"""
    STRESS = "stress"
    SPIKE = "spike"
    VOLUME = "volume"
    ENDURANCE = "endurance"


@dataclass
class LoadTestResult:
    """Resultado de un test de carga"""
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_duration: float
    requests_per_second: float
    avg_response_time: float
    min_response_time: float
    max_response_time: float
    p50_response_time: float
    p95_response_time: float
    p99_response_time: float
    response_times: List[float] = field(default_factory=list)
    status_codes: Dict[int, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario"""
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "total_duration": self.total_duration,
            "requests_per_second": self.requests_per_second,
            "avg_response_time": self.avg_response_time,
            "min_response_time": self.min_response_time,
            "max_response_time": self.max_response_time,
            "p50_response_time": self.p50_response_time,
            "p95_response_time": self.p95_response_time,
            "p99_response_time": self.p99_response_time,
            "status_codes": self.status_codes,
            "error_count": len(self.errors)
        }


class PerformanceTester:
    """Tester de performance"""
    
    def __init__(self):
        self.test_results: List[LoadTestResult] = []
    
    async def run_load_test(
        self,
        endpoint: str,
        method: str = "GET",
        total_requests: int = 100,
        concurrent_users: int = 10,
        headers: Optional[Dict[str, str]] = None,
        body: Optional[Any] = None,
        test_type: LoadTestType = LoadTestType.STRESS
    ) -> LoadTestResult:
        """Ejecuta un test de carga"""
        import httpx
        
        start_time = time.time()
        response_times = []
        status_codes = {}
        errors = []
        successful = 0
        failed = 0
        
        # Crear semáforo para limitar concurrencia
        semaphore = asyncio.Semaphore(concurrent_users)
        
        async def make_request(request_num: int):
            nonlocal successful, failed
            async with semaphore:
                try:
                    request_start = time.time()
                    async with httpx.AsyncClient(timeout=30.0) as client:
                        response = await client.request(
                            method=method,
                            url=endpoint,
                            headers=headers,
                            json=body if isinstance(body, dict) else None
                        )
                        request_duration = time.time() - request_start
                        
                        response_times.append(request_duration)
                        status_codes[response.status_code] = status_codes.get(response.status_code, 0) + 1
                        
                        if 200 <= response.status_code < 300:
                            successful += 1
                        else:
                            failed += 1
                            errors.append(f"Request {request_num}: Status {response.status_code}")
                except Exception as e:
                    failed += 1
                    errors.append(f"Request {request_num}: {str(e)}")
        
        # Ejecutar requests
        tasks = [make_request(i) for i in range(total_requests)]
        await asyncio.gather(*tasks)
        
        total_duration = time.time() - start_time
        
        # Calcular estadísticas
        if response_times:
            result = LoadTestResult(
                total_requests=total_requests,
                successful_requests=successful,
                failed_requests=failed,
                total_duration=total_duration,
                requests_per_second=total_requests / total_duration if total_duration > 0 else 0,
                avg_response_time=statistics.mean(response_times),
                min_response_time=min(response_times),
                max_response_time=max(response_times),
                p50_response_time=self._percentile(response_times, 50),
                p95_response_time=self._percentile(response_times, 95),
                p99_response_time=self._percentile(response_times, 99),
                response_times=response_times,
                status_codes=status_codes,
                errors=errors[:100]  # Limitar errores
            )
        else:
            result = LoadTestResult(
                total_requests=total_requests,
                successful_requests=0,
                failed_requests=failed,
                total_duration=total_duration,
                requests_per_second=0,
                avg_response_time=0,
                min_response_time=0,
                max_response_time=0,
                p50_response_time=0,
                p95_response_time=0,
                p99_response_time=0,
                status_codes=status_codes,
                errors=errors
            )
        
        self.test_results.append(result)
        return result
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calcula percentil"""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    async def run_stress_test(
        self,
        endpoint: str,
        method: str = "GET",
        max_concurrent: int = 100,
        duration_seconds: int = 60
    ) -> LoadTestResult:
        """Ejecuta un stress test"""
        start_time = time.time()
        request_count = 0
        
        async def continuous_requests():
            nonlocal request_count
            import httpx
            while time.time() - start_time < duration_seconds:
                try:
                    async with httpx.AsyncClient(timeout=10.0) as client:
                        await client.request(method=method, url=endpoint)
                        request_count += 1
                except:
                    pass
                await asyncio.sleep(0.1)
        
        tasks = [continuous_requests() for _ in range(max_concurrent)]
        await asyncio.gather(*tasks)
        
        # Retornar resultado simplificado
        return LoadTestResult(
            total_requests=request_count,
            successful_requests=request_count,
            failed_requests=0,
            total_duration=duration_seconds,
            requests_per_second=request_count / duration_seconds,
            avg_response_time=0,
            min_response_time=0,
            max_response_time=0,
            p50_response_time=0,
            p95_response_time=0,
            p99_response_time=0
        )
    
    def get_test_results(self) -> List[Dict[str, Any]]:
        """Obtiene resultados de tests"""
        return [r.to_dict() for r in self.test_results]




