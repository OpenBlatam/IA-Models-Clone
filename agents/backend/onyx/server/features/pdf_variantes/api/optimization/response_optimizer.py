"""
Response Optimization
Optimize API responses for speed
"""

from typing import Any, Dict, Optional
from fastapi import Response
from fastapi.responses import JSONResponse
import orjson
import gzip


class ResponseOptimizer:
    """Optimize API responses"""
    
    @staticmethod
    def compress_response(
        content: Any,
        threshold: int = 1024,
        compress_level: int = 6
    ) -> tuple[bytes, bool]:
        """Compress response content if large enough"""
        # Serialize first
        try:
            body = orjson.dumps(content)
        except:
            import json
            body = json.dumps(content).encode('utf-8')
        
        # Compress if above threshold
        if len(body) > threshold:
            compressed = gzip.compress(body, compresslevel=compress_level)
            if len(compressed) < len(body):
                return compressed, True
        
        return body, False
    
    @staticmethod
    def create_optimized_response(
        content: Any,
        status_code: int = 200,
        compress: bool = True,
        headers: Optional[Dict[str, str]] = None
    ) -> Response:
        """Create optimized response"""
        # Compress if enabled
        body, is_compressed = ResponseOptimizer.compress_response(content) if compress else (content, False)
        
        # Set headers
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
    def minify_response(content: Dict[str, Any]) -> Dict[str, Any]:
        """Minify response by removing nulls and empty values"""
        if isinstance(content, dict):
            return {
                k: ResponseOptimizer.minify_response(v)
                for k, v in content.items()
                if v is not None and v != ""
            }
        elif isinstance(content, list):
            return [ResponseOptimizer.minify_response(item) for item in content]
        return content


class ChunkedResponse:
    """Stream large responses in chunks"""
    
    @staticmethod
    async def stream_response(
        generator,
        chunk_size: int = 8192
    ):
        """Stream response in chunks"""
        async for chunk in generator:
            yield chunk


def optimize_response(
    compress: bool = True,
    minify: bool = False,
    cache: bool = False,
    cache_ttl: int = 60
):
    """Decorator to optimize responses"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Execute function
            result = await func(*args, **kwargs)
            
            # Minify if requested
            if minify:
                result = ResponseOptimizer.minify_response(result)
            
            # Create optimized response
            response = ResponseOptimizer.create_optimized_response(
                result,
                compress=compress
            )
            
            # Add cache headers
            if cache:
                response.headers["Cache-Control"] = f"public, max-age={cache_ttl}"
            
            return response
        
        return wrapper
    return decorator






