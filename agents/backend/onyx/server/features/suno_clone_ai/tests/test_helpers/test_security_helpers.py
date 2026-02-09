"""
Helpers para tests de seguridad
"""

import pytest
import re
from typing import List, Dict, Any


class SecurityTestHelper:
    """Helper para tests de seguridad"""
    
    @staticmethod
    def generate_sql_injection_payloads() -> List[str]:
        """Genera payloads de SQL injection para testing"""
        return [
            "'; DROP TABLE songs; --",
            "' OR '1'='1",
            "'; SELECT * FROM users; --",
            "1' UNION SELECT NULL--",
            "admin'--",
            "' OR 1=1--"
        ]
    
    @staticmethod
    def generate_xss_payloads() -> List[str]:
        """Genera payloads de XSS para testing"""
        return [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "<svg onload=alert('XSS')>",
            "javascript:alert('XSS')",
            "<body onload=alert('XSS')>"
        ]
    
    @staticmethod
    def generate_path_traversal_payloads() -> List[str]:
        """Genera payloads de path traversal para testing"""
        return [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32",
            "/etc/passwd",
            "....//....//etc/passwd",
            "..%2F..%2F..%2Fetc%2Fpasswd"
        ]
    
    @staticmethod
    def generate_command_injection_payloads() -> List[str]:
        """Genera payloads de command injection para testing"""
        return [
            "; rm -rf /",
            "| cat /etc/passwd",
            "&& whoami",
            "`id`",
            "$(ls -la)"
        ]
    
    @staticmethod
    def is_safe_string(value: str) -> bool:
        """Verifica si un string es seguro (sin caracteres peligrosos)"""
        dangerous_patterns = [
            r'<script',
            r'javascript:',
            r'\.\./',
            r'\.\.\\',
            r';\s*(rm|cat|ls|whoami)',
            r'DROP\s+TABLE',
            r'UNION\s+SELECT'
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                return False
        
        return True
    
    @staticmethod
    def sanitize_input(value: str) -> str:
        """Sanitiza un input removiendo caracteres peligrosos"""
        # Remover tags HTML
        value = re.sub(r'<[^>]+>', '', value)
        # Remover caracteres de control
        value = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', value)
        # Escapar caracteres especiales SQL
        value = value.replace("'", "''")
        value = value.replace(";", "")
        return value


class RateLimitHelper:
    """Helper para tests de rate limiting"""
    
    @staticmethod
    async def make_rapid_requests(
        client,
        endpoint: str,
        payload: Dict[str, Any],
        count: int = 100,
        delay: float = 0.0
    ) -> List[int]:
        """
        Hace múltiples requests rápidos
        
        Args:
            client: Test client
            endpoint: Endpoint a testear
            payload: Payload para el request
            count: Número de requests
            delay: Delay entre requests (segundos)
        
        Returns:
            Lista de códigos de estado
        """
        import asyncio
        
        status_codes = []
        
        async def make_request():
            if endpoint.startswith("POST"):
                response = client.post(endpoint.split()[1], json=payload)
            elif endpoint.startswith("GET"):
                response = client.get(endpoint.split()[1], params=payload)
            else:
                response = client.get(endpoint, params=payload)
            
            status_codes.append(response.status_code)
            
            if delay > 0:
                await asyncio.sleep(delay)
        
        await asyncio.gather(*[make_request() for _ in range(count)])
        
        return status_codes
    
    @staticmethod
    def analyze_rate_limiting(status_codes: List[int]) -> Dict[str, Any]:
        """
        Analiza resultados de rate limiting
        
        Args:
            status_codes: Lista de códigos de estado
        
        Returns:
            Diccionario con análisis
        """
        from collections import Counter
        
        counter = Counter(status_codes)
        total = len(status_codes)
        
        return {
            "total_requests": total,
            "status_distribution": dict(counter),
            "rate_limited_count": counter.get(429, 0),
            "success_count": counter.get(200, 0) + counter.get(202, 0),
            "rate_limited_percentage": (counter.get(429, 0) / total) * 100 if total > 0 else 0
        }

