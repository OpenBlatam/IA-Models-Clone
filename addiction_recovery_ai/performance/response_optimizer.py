"""
Response Optimizer
Ultra-fast response generation and optimization
"""

import logging
from typing import Any, Dict
from fastapi import Response
from fastapi.responses import JSONResponse

from performance.serialization_optimizer import get_serializer

logger = logging.getLogger(__name__)


class FastResponse:
    """Fast response generator"""
    
    def __init__(self):
        self.serializer = get_serializer()
    
    def json_response(self, data: Any, status_code: int = 200) -> Response:
        """Create fast JSON response"""
        # Use orjson for serialization
        content = self.serializer.serialize_json(data)
        
        return Response(
            content=content,
            status_code=status_code,
            media_type="application/json",
            headers={
                "Content-Length": str(len(content)),
                "X-Response-Optimized": "true"
            }
        )
    
    def cached_response(
        self,
        data: Any,
        etag: str,
        status_code: int = 200
    ) -> Response:
        """Create cached response with ETag"""
        response = self.json_response(data, status_code)
        response.headers["ETag"] = etag
        response.headers["Cache-Control"] = "public, max-age=300"
        return response
    
    def compressed_response(
        self,
        data: Any,
        status_code: int = 200,
        compress: bool = True
    ) -> Response:
        """Create compressed response"""
        content = self.serializer.serialize_json(data)
        
        if compress and len(content) > 1024:  # Only compress if > 1KB
            import gzip
            compressed = gzip.compress(content, compresslevel=6)
            return Response(
                content=compressed,
                status_code=status_code,
                media_type="application/json",
                headers={
                    "Content-Encoding": "gzip",
                    "Content-Length": str(len(compressed)),
                    "X-Compressed": "true"
                }
            )
        
        return Response(
            content=content,
            status_code=status_code,
            media_type="application/json"
        )


# Global response optimizer
_response_optimizer: FastResponse = None


def get_response_optimizer() -> FastResponse:
    """Get global response optimizer"""
    global _response_optimizer
    if _response_optimizer is None:
        _response_optimizer = FastResponse()
    return _response_optimizer


def fast_json_response(data: Any, status_code: int = 200) -> Response:
    """Fast JSON response helper"""
    return get_response_optimizer().json_response(data, status_code)















