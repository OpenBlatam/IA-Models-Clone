"""
Response optimization utilities
"""

import gzip
from typing import Any, Optional, Dict
from fastapi import Response
from fastapi.responses import JSONResponse

try:
    import orjson
    ORJSON_AVAILABLE = True
except ImportError:
    ORJSON_AVAILABLE = False


class ResponseOptimizer:
    """Optimize API responses for speed and size"""
    
    @staticmethod
    def compress_response(
        content: Any,
        threshold: int = 1024,
        compress_level: int = 6
    ) -> tuple[bytes, bool]:
        """Compress response content if large enough"""
        if ORJSON_AVAILABLE:
            try:
                body = orjson.dumps(content)
            except:
                import json
                body = json.dumps(content).encode('utf-8')
        else:
            import json
            body = json.dumps(content).encode('utf-8')
        
        if len(body) > threshold:
            compressed = gzip.compress(body, compresslevel=compress_level)
            if len(compressed) < len(body) * 0.9:
                return compressed, True
        
        return body, False
    
    @staticmethod
    def create_optimized_response(
        content: Any,
        status_code: int = 200,
        compress: bool = True,
        headers: Optional[Dict[str, str]] = None,
        threshold: int = 1024
    ) -> Response:
        """Create optimized response with optional compression"""
        if compress:
            body, is_compressed = ResponseOptimizer.compress_response(content, threshold=threshold)
        else:
            if ORJSON_AVAILABLE:
                try:
                    body = orjson.dumps(content)
                except:
                    import json
                    body = json.dumps(content).encode('utf-8')
            else:
                import json
                body = json.dumps(content).encode('utf-8')
            is_compressed = False
        
        response_headers = headers or {}
        response_headers["Content-Type"] = "application/json"
        
        if is_compressed:
            response_headers["Content-Encoding"] = "gzip"
            response_headers["Content-Length"] = str(len(body))
        
        return Response(
            content=body,
            status_code=status_code,
            headers=response_headers
        )
    
    @staticmethod
    def should_compress(content: Any, threshold: int = 1024) -> bool:
        """Check if content should be compressed"""
        if ORJSON_AVAILABLE:
            try:
                size = len(orjson.dumps(content))
            except:
                import json
                size = len(json.dumps(content).encode('utf-8'))
        else:
            import json
            size = len(json.dumps(content).encode('utf-8'))
        
        return size > threshold

