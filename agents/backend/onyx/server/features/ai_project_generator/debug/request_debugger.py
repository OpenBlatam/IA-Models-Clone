"""
Request Debugger - Debugger de requests
=======================================

Herramientas para debugging de requests HTTP.
"""

import logging
import json
from typing import Dict, Any, Optional
from fastapi import Request
from datetime import datetime

logger = logging.getLogger(__name__)


class RequestDebugger:
    """Debugger para requests HTTP"""
    
    @staticmethod
    def debug_request(request: Request) -> Dict[str, Any]:
        """
        Extrae información de debug de un request.
        
        Args:
            request: Request de FastAPI
        
        Returns:
            Información de debug
        """
        return {
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "path_params": dict(request.path_params),
            "headers": dict(request.headers),
            "client": {
                "host": request.client.host if request.client else None,
                "port": request.client.port if request.client else None
            },
            "cookies": dict(request.cookies),
        }
    
    @staticmethod
    async def debug_request_body(request: Request) -> Optional[Dict[str, Any]]:
        """
        Extrae body del request para debugging.
        
        Args:
            request: Request de FastAPI
        
        Returns:
            Body del request o None
        """
        try:
            body = await request.body()
            if body:
                try:
                    return json.loads(body)
                except json.JSONDecodeError:
                    return {"raw": body.decode("utf-8", errors="ignore")[:1000]}
        except Exception as e:
            logger.error(f"Error reading request body: {e}")
            return None
    
    @staticmethod
    def debug_response(response, duration: float) -> Dict[str, Any]:
        """
        Extrae información de debug de un response.
        
        Args:
            response: Response
            duration: Duración de la request
        
        Returns:
            Información de debug
        """
        return {
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "duration": duration
        }















