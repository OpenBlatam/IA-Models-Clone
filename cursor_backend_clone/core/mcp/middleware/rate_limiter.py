"""
MCP Rate Limiter - Rate Limiting para el servidor MCP
======================================================

Rate limiter simple para prevenir abuso en el servidor MCP.
"""

import time
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class RateLimiter:
    """Rate limiter simple para prevenir abuso"""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, List[float]] = {}
    
    def is_allowed(self, client_id: str = "default") -> bool:
        """Verificar si el cliente puede hacer una request"""
        now = time.time()
        
        if client_id not in self.requests:
            self.requests[client_id] = []
        
        requests = self.requests[client_id]
        requests[:] = [req_time for req_time in requests if now - req_time < self.window_seconds]
        
        if len(requests) >= self.max_requests:
            return False
        
        requests.append(now)
        return True




