"""
Azure Functions handler for Music Analyzer AI
Optimized for serverless deployment with cold start reduction
Uses Azure Functions v2 programming model with FastAPI
"""

import logging
import json
import os
from typing import Dict, Any, Optional
import azure.functions as func
from azure.functions import HttpRequest, HttpResponse

# Configure logging for Application Insights
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global app instance (reused across invocations)
_app: Optional[Any] = None
_handler: Optional[Any] = None


def get_app():
    """Lazy load FastAPI app to reduce cold start time"""
    global _app
    if _app is None:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        
        from main import app
        _app = app
        logger.info("FastAPI app initialized for Azure Functions")
    return _app


def get_handler():
    """Get or create Azure Functions handler"""
    global _handler
    if _handler is None:
        try:
            from azure.functions import AsgiMiddleware
            app = get_app()
            _handler = AsgiMiddleware(app)
            logger.info("AsgiMiddleware handler initialized")
        except ImportError:
            # Fallback to manual ASGI handling
            logger.warning("AsgiMiddleware not available, using manual ASGI handling")
            _handler = None
    return _handler


# Create Azure Function App
function_app = func.FunctionApp()


@function_app.route(route="{*path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def http_trigger(req: HttpRequest) -> HttpResponse:
    """
    HTTP trigger that routes all requests to FastAPI app
    """
    try:
        handler = get_handler()
        
        if handler:
            # Use AsgiMiddleware if available
            return await handler(req)
        else:
            # Manual ASGI handling
            app = get_app()
            
            # Convert Azure Function request to ASGI scope
            path = req.route_params.get('path', '') or '/'
            method = req.method
            headers = [[k.encode(), v.encode()] for k, v in req.headers.items()]
            query_string = "&".join([f"{k}={v}" for k, v in req.params.items()]).encode()
            
            # Get body
            body = b""
            if req.get_body():
                body = req.get_body() if isinstance(req.get_body(), bytes) else req.get_body().encode()
            
            # Create ASGI scope
            scope = {
                "type": "http",
                "method": method,
                "path": f"/{path}" if path != '/' else "/",
                "query_string": query_string,
                "headers": headers,
                "body": body,
            }
            
            # Process with FastAPI using ASGI
            from starlette.requests import Request
            from starlette.responses import Response
            
            async def receive():
                return {"type": "http.request", "body": body}
            
            response_parts = []
            
            async def send(message):
                response_parts.append(message)
            
            await app(scope, receive, send)
            
            # Extract response
            status_code = 200
            response_headers = {}
            response_body = b""
            
            for part in response_parts:
                if part["type"] == "http.response.start":
                    status_code = part["status"]
                    response_headers = {k.decode(): v.decode() if isinstance(v, bytes) else v 
                                       for k, v in part.get("headers", [])}
                elif part["type"] == "http.response.body":
                    response_body += part.get("body", b"")
            
            return HttpResponse(
                response_body.decode('utf-8') if isinstance(response_body, bytes) else response_body,
                status_code=status_code,
                headers=response_headers
            )
            
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        return HttpResponse(
            json.dumps({"error": "Internal server error", "message": str(e)}),
            status_code=500,
            headers={"Content-Type": "application/json"}
        )

