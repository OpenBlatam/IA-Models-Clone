"""
API Testing - Testing de APIs Avanzado
======================================

Utilidades avanzadas para testing de APIs:
- API test client
- Test scenarios
- Load testing helpers
- Contract testing
- API mocking
"""

import logging
import asyncio
from typing import Optional, Dict, Any, List, Callable
from datetime import datetime
import httpx

logger = logging.getLogger(__name__)


class APITestClient:
    """
    Cliente de testing para APIs.
    """
    
    def __init__(
        self,
        base_url: str,
        timeout: float = 30.0,
        default_headers: Optional[Dict[str, str]] = None
    ) -> None:
        self.base_url = base_url
        self.timeout = timeout
        self.default_headers = default_headers or {}
        self.client = httpx.AsyncClient(
            base_url=base_url,
            timeout=timeout,
            headers=self.default_headers
        )
        self.request_history: List[Dict[str, Any]] = []
    
    async def get(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """GET request"""
        return await self._request("GET", path, params=params, headers=headers)
    
    async def post(
        self,
        path: str,
        json: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """POST request"""
        return await self._request("POST", path, json=json, data=data, headers=headers)
    
    async def put(
        self,
        path: str,
        json: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """PUT request"""
        return await self._request("PUT", path, json=json, headers=headers)
    
    async def delete(
        self,
        path: str,
        headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """DELETE request"""
        return await self._request("DELETE", path, headers=headers)
    
    async def _request(
        self,
        method: str,
        path: str,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Ejecuta request"""
        start_time = datetime.now()
        
        try:
            response = await self.client.request(method, path, **kwargs)
            duration = (datetime.now() - start_time).total_seconds()
            
            result = {
                "method": method,
                "path": path,
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "body": response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text,
                "duration": duration,
                "timestamp": start_time.isoformat()
            }
            
            self.request_history.append(result)
            return result
            
        except Exception as e:
            logger.error(f"API test request failed: {e}")
            return {
                "method": method,
                "path": path,
                "error": str(e),
                "timestamp": start_time.isoformat()
            }
    
    def assert_status_code(self, response: Dict[str, Any], expected: int) -> bool:
        """Assert status code"""
        actual = response.get("status_code")
        if actual != expected:
            raise AssertionError(f"Expected status {expected}, got {actual}")
        return True
    
    def assert_response_time(self, response: Dict[str, Any], max_time: float) -> bool:
        """Assert response time"""
        duration = response.get("duration", 0)
        if duration > max_time:
            raise AssertionError(f"Response time {duration}s exceeds {max_time}s")
        return True
    
    def get_request_history(self) -> List[Dict[str, Any]]:
        """Obtiene historial de requests"""
        return self.request_history
    
    async def close(self) -> None:
        """Cierra cliente"""
        await self.client.aclose()


class LoadTestRunner:
    """Runner para load testing"""
    
    def __init__(self, test_client: APITestClient) -> None:
        self.test_client = test_client
    
    async def run_load_test(
        self,
        endpoint: str,
        method: str = "GET",
        concurrent_users: int = 10,
        requests_per_user: int = 100,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Ejecuta load test"""
        results = []
        
        async def user_simulation(user_id: int) -> List[Dict[str, Any]]:
            user_results = []
            for i in range(requests_per_user):
                if method == "GET":
                    result = await self.test_client.get(endpoint, **kwargs)
                elif method == "POST":
                    result = await self.test_client.post(endpoint, **kwargs)
                else:
                    result = await self.test_client._request(method, endpoint, **kwargs)
                user_results.append(result)
            return user_results
        
        # Ejecutar usuarios concurrentes
        tasks = [user_simulation(i) for i in range(concurrent_users)]
        all_results = await asyncio.gather(*tasks)
        
        # Flatten results
        for user_results in all_results:
            results.extend(user_results)
        
        # Calcular estadísticas
        durations = [r.get("duration", 0) for r in results if "duration" in r]
        status_codes = [r.get("status_code") for r in results if "status_code" in r]
        
        return {
            "total_requests": len(results),
            "concurrent_users": concurrent_users,
            "requests_per_user": requests_per_user,
            "avg_response_time": sum(durations) / len(durations) if durations else 0,
            "min_response_time": min(durations) if durations else 0,
            "max_response_time": max(durations) if durations else 0,
            "status_codes": {code: status_codes.count(code) for code in set(status_codes)},
            "success_rate": (status_codes.count(200) / len(status_codes) * 100) if status_codes else 0
        }


def get_api_test_client(
    base_url: str,
    timeout: float = 30.0
) -> APITestClient:
    """Obtiene cliente de testing de API"""
    return APITestClient(base_url=base_url, timeout=timeout)


def get_load_test_runner(test_client: APITestClient) -> LoadTestRunner:
    """Obtiene runner de load testing"""
    return LoadTestRunner(test_client)















