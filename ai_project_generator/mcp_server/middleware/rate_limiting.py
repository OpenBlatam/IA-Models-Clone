"""
Rate Limiting Middleware
========================

Middleware para rate limiting de requests.
"""

import logging
import time
from typing import Dict, Optional
from collections import defaultdict
from fastapi import Request, HTTPException, status

logger = logging.getLogger(__name__)


class RateLimitMiddleware:
    """Middleware para rate limiting simple"""
    
    def __init__(
        self,
        app,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        per_user: bool = True
    ):
        self.app = app
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.per_user = per_user
        
        # Simple in-memory rate limiting
        self._minute_requests: Dict[str, list] = defaultdict(list)
        self._hour_requests: Dict[str, list] = defaultdict(list)
    
    def _get_user_id(self, request: Request) -> str:
        """Obtener ID de usuario desde request"""
        if self.per_user:
            # Intentar obtener de headers o state
            user_id = getattr(request.state, "user_id", None)
            if user_id:
                return str(user_id)
            
            # Fallback a IP
            client_host = request.client.host if request.client else "unknown"
            return f"ip:{client_host}"
        else:
            return "global"
    
    def _check_rate_limit(self, user_id: str) -> bool:
        """Verificar si el usuario excede el rate limit"""
        now = time.time()
        
        # Limpiar requests antiguos
        self._minute_requests[user_id] = [
            ts for ts in self._minute_requests[user_id]
            if now - ts < 60
        ]
        self._hour_requests[user_id] = [
            ts for ts in self._hour_requests[user_id]
            if now - ts < 3600
        ]
        
        # Verificar límites
        if len(self._minute_requests[user_id]) >= self.requests_per_minute:
            return False
        if len(self._hour_requests[user_id]) >= self.requests_per_hour:
            return False
        
        # Agregar request actual
        self._minute_requests[user_id].append(now)
        self._hour_requests[user_id].append(now)
        
        return True
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        # Crear request object para obtener user_id
        # Nota: Esto es una simplificación, en producción usar un sistema más robusto
        request = Request(scope, receive)
        user_id = self._get_user_id(request)
        
        if not self._check_rate_limit(user_id):
            logger.warning(f"Rate limit exceeded for user: {user_id}")
            response = {
                "type": "http.response.start",
                "status": status.HTTP_429_TOO_MANY_REQUESTS,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"retry-after", b"60"),
                ],
            }
            await send(response)
            body = b'{"success": false, "error": "Rate limit exceeded"}'
            await send({"type": "http.response.body", "body": body})
            return
        
        await self.app(scope, receive, send)

