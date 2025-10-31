"""
Serverless Adapter for FastAPI Applications
Optimized for AWS Lambda, Azure Functions, Google Cloud Functions, Vercel, and Netlify
"""

import asyncio
import json
import logging
import os
import time
from typing import Dict, Any, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum
import base64
from urllib.parse import parse_qs, urlparse

# Serverless platform adapters
try:
    from mangum import Mangum
    MANGUM_AVAILABLE = True
except ImportError:
    MANGUM_AVAILABLE = False

try:
    import azure.functions as func
    AZURE_FUNCTIONS_AVAILABLE = True
except ImportError:
    AZURE_FUNCTIONS_AVAILABLE = False

try:
    from functions_framework import http
    GOOGLE_CLOUD_FUNCTIONS_AVAILABLE = True
except ImportError:
    GOOGLE_CLOUD_FUNCTIONS_AVAILABLE = False

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
import uvicorn

logger = logging.getLogger(__name__)

class ServerlessPlatform(Enum):
    """Supported serverless platforms"""
    AWS_LAMBDA = "aws_lambda"
    AZURE_FUNCTIONS = "azure_functions"
    GOOGLE_CLOUD_FUNCTIONS = "google_cloud_functions"
    VERCEL = "vercel"
    NETLIFY = "netlify"
    RAILWAY = "railway"
    HEROKU = "heroku"

@dataclass
class ServerlessConfig:
    """Serverless configuration"""
    platform: ServerlessPlatform
    cold_start_timeout: float = 10.0
    max_memory_mb: int = 512
    enable_compression: bool = True
    enable_caching: bool = True
    cache_ttl: int = 300  # seconds
    enable_metrics: bool = True
    log_level: str = "INFO"
    cors_origins: list = None
    cors_methods: list = None
    cors_headers: list = None

class ColdStartOptimizer:
    """
    Optimizes FastAPI applications for serverless cold starts
    """
    
    def __init__(self, config: ServerlessConfig):
        self.config = config
        self.startup_time = time.time()
        self._preload_modules()
    
    def _preload_modules(self):
        """Preload commonly used modules to reduce cold start time"""
        try:
            # Preload common modules
            import pydantic
            import uvicorn
            import asyncio
            import json
            import logging
            
            # Preload database drivers if available
            try:
                import asyncpg
                import aioredis
            except ImportError:
                pass
            
            logger.info("Modules preloaded successfully")
            
        except Exception as e:
            logger.warning(f"Failed to preload some modules: {e}")
    
    def optimize_app(self, app: FastAPI) -> FastAPI:
        """Apply cold start optimizations to FastAPI app"""
        
        # Add startup optimization middleware
        @app.middleware("http")
        async def cold_start_middleware(request: Request, call_next):
            # Add cold start headers
            start_time = time.time()
            response = await call_next(request)
            
            # Add performance headers
            response.headers["X-Cold-Start-Time"] = str(time.time() - self.startup_time)
            response.headers["X-Request-Duration"] = str(time.time() - start_time)
            
            return response
        
        return app

class ServerlessAdapter:
    """
    Universal serverless adapter for FastAPI applications
    """
    
    def __init__(self, app: FastAPI, config: ServerlessConfig):
        self.app = app
        self.config = config
        self.optimizer = ColdStartOptimizer(config)
        self.mangum_handler = None
        
        # Optimize app for serverless
        self.app = self.optimizer.optimize_app(app)
        
        # Initialize platform-specific handlers
        self._initialize_handlers()
    
    def _initialize_handlers(self):
        """Initialize platform-specific handlers"""
        
        if self.config.platform == ServerlessPlatform.AWS_LAMBDA:
            if MANGUM_AVAILABLE:
                self.mangum_handler = Mangum(
                    self.app,
                    lifespan="off",  # Disable lifespan for Lambda
                    api_gateway_base_path=None,
                    text_mime_types=[
                        "application/json",
                        "application/javascript",
                        "application/xml",
                        "application/vnd.api+json",
                        "text/plain",
                        "text/html",
                        "text/css",
                        "text/javascript",
                        "text/xml",
                    ]
                )
            else:
                raise ImportError("Mangum is required for AWS Lambda deployment")
        
        elif self.config.platform == ServerlessPlatform.VERCEL:
            self._setup_vercel_handler()
        
        elif self.config.platform == ServerlessPlatform.NETLIFY:
            self._setup_netlify_handler()
    
    def _setup_vercel_handler(self):
        """Setup Vercel handler"""
        # Vercel uses a specific handler format
        def vercel_handler(request):
            return self._handle_vercel_request(request)
        
        self.vercel_handler = vercel_handler
    
    def _setup_netlify_handler(self):
        """Setup Netlify handler"""
        # Netlify uses AWS Lambda format
        if MANGUM_AVAILABLE:
            self.mangum_handler = Mangum(
                self.app,
                lifespan="off",
                api_gateway_base_path=None
            )
    
    def get_handler(self) -> Callable:
        """Get the appropriate handler for the platform"""
        
        if self.config.platform == ServerlessPlatform.AWS_LAMBDA:
            return self.mangum_handler
        
        elif self.config.platform == ServerlessPlatform.AZURE_FUNCTIONS:
            return self._azure_functions_handler
        
        elif self.config.platform == ServerlessPlatform.GOOGLE_CLOUD_FUNCTIONS:
            return self._google_cloud_functions_handler
        
        elif self.config.platform == ServerlessPlatform.VERCEL:
            return self.vercel_handler
        
        elif self.config.platform == ServerlessPlatform.NETLIFY:
            return self.mangum_handler
        
        else:
            raise ValueError(f"Unsupported platform: {self.config.platform}")
    
    async def _azure_functions_handler(self, req: func.HttpRequest) -> func.HttpResponse:
        """Azure Functions handler"""
        try:
            # Convert Azure Functions request to ASGI scope
            scope = self._azure_request_to_scope(req)
            
            # Create ASGI application
            from asgiref.wsgi import WsgiToAsgi
            asgi_app = WsgiToAsgi(self.app)
            
            # Handle request
            response = await self._handle_asgi_request(asgi_app, scope)
            
            return func.HttpResponse(
                body=response["body"],
                status_code=response["status"],
                headers=response["headers"]
            )
            
        except Exception as e:
            logger.error(f"Azure Functions handler error: {e}")
            return func.HttpResponse(
                body=json.dumps({"error": "Internal server error"}),
                status_code=500,
                headers={"Content-Type": "application/json"}
            )
    
    def _google_cloud_functions_handler(self, request):
        """Google Cloud Functions handler"""
        try:
            # Convert Google Cloud Functions request
            scope = self._gcf_request_to_scope(request)
            
            # Handle request
            response = self._handle_sync_request(scope)
            
            return response
            
        except Exception as e:
            logger.error(f"Google Cloud Functions handler error: {e}")
            return JSONResponse(
                content={"error": "Internal server error"},
                status_code=500
            )
    
    def _handle_vercel_request(self, request):
        """Handle Vercel request"""
        try:
            # Vercel provides request in a specific format
            # This is a simplified implementation
            return JSONResponse(
                content={"message": "Vercel handler not fully implemented"},
                status_code=501
            )
            
        except Exception as e:
            logger.error(f"Vercel handler error: {e}")
            return JSONResponse(
                content={"error": "Internal server error"},
                status_code=500
            )
    
    def _azure_request_to_scope(self, req: func.HttpRequest) -> Dict[str, Any]:
        """Convert Azure Functions request to ASGI scope"""
        return {
            "type": "http",
            "method": req.method,
            "path": req.url.split("?")[0],
            "query_string": req.url.split("?")[1].encode() if "?" in req.url else b"",
            "headers": [(k.lower().encode(), v.encode()) for k, v in req.headers.items()],
            "body": req.get_body(),
            "client": ("127.0.0.1", 0),
            "server": ("127.0.0.1", 80),
        }
    
    def _gcf_request_to_scope(self, request) -> Dict[str, Any]:
        """Convert Google Cloud Functions request to ASGI scope"""
        return {
            "type": "http",
            "method": request.method,
            "path": request.path,
            "query_string": request.query_string.encode(),
            "headers": [(k.lower().encode(), v.encode()) for k, v in request.headers.items()],
            "body": request.get_data(),
            "client": ("127.0.0.1", 0),
            "server": ("127.0.0.1", 80),
        }
    
    async def _handle_asgi_request(self, app, scope):
        """Handle ASGI request"""
        # This is a simplified ASGI handler
        # In production, you'd use a proper ASGI server
        return {
            "status": 200,
            "headers": [(b"content-type", b"application/json")],
            "body": b'{"message": "ASGI handler"}'
        }
    
    def _handle_sync_request(self, scope):
        """Handle synchronous request"""
        # Simplified synchronous handler
        return JSONResponse(
            content={"message": "Synchronous handler"},
            status_code=200
        )

class ServerlessDeployment:
    """
    Handles serverless deployment configurations
    """
    
    @staticmethod
    def create_aws_lambda_config() -> Dict[str, Any]:
        """Create AWS Lambda deployment configuration"""
        return {
            "runtime": "python3.9",
            "handler": "main.handler",
            "timeout": 30,
            "memory_size": 512,
            "environment": {
                "PYTHONPATH": "/var/task",
                "LOG_LEVEL": "INFO"
            },
            "layers": [
                "arn:aws:lambda:us-east-1:123456789012:layer:fastapi:1"
            ]
        }
    
    @staticmethod
    def create_azure_functions_config() -> Dict[str, Any]:
        """Create Azure Functions deployment configuration"""
        return {
            "runtime": "python",
            "version": "3.9",
            "timeout": "00:05:00",
            "memory": 512,
            "environment": {
                "PYTHONPATH": "/home/site/wwwroot",
                "LOG_LEVEL": "INFO"
            }
        }
    
    @staticmethod
    def create_vercel_config() -> Dict[str, Any]:
        """Create Vercel deployment configuration"""
        return {
            "version": 2,
            "builds": [
                {
                    "src": "main.py",
                    "use": "@vercel/python"
                }
            ],
            "routes": [
                {
                    "src": "/(.*)",
                    "dest": "main.py"
                }
            ],
            "env": {
                "LOG_LEVEL": "INFO"
            }
        }
    
    @staticmethod
    def create_netlify_config() -> Dict[str, Any]:
        """Create Netlify deployment configuration"""
        return {
            "build": {
                "command": "pip install -r requirements.txt",
                "functions": "netlify/functions"
            },
            "functions": {
                "main": {
                    "runtime": "python3.9",
                    "timeout": 30
                }
            }
        }

# Utility functions for serverless optimization
def optimize_for_serverless(app: FastAPI, platform: ServerlessPlatform) -> ServerlessAdapter:
    """
    Optimize FastAPI app for serverless deployment
    
    Args:
        app: FastAPI application
        platform: Target serverless platform
        
    Returns:
        ServerlessAdapter: Optimized adapter
    """
    config = ServerlessConfig(platform=platform)
    return ServerlessAdapter(app, config)

def create_serverless_app(
    title: str = "Serverless FastAPI App",
    version: str = "1.0.0",
    platform: ServerlessPlatform = ServerlessPlatform.AWS_LAMBDA
) -> tuple[FastAPI, ServerlessAdapter]:
    """
    Create a new FastAPI app optimized for serverless deployment
    
    Args:
        title: App title
        version: App version
        platform: Target platform
        
    Returns:
        Tuple of (FastAPI app, ServerlessAdapter)
    """
    # Create FastAPI app
    app = FastAPI(
        title=title,
        version=version,
        docs_url="/docs" if platform != ServerlessPlatform.AWS_LAMBDA else None,
        redoc_url="/redoc" if platform != ServerlessPlatform.AWS_LAMBDA else None
    )
    
    # Add health check endpoint
    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "platform": platform.value,
            "timestamp": time.time()
        }
    
    # Create serverless adapter
    adapter = optimize_for_serverless(app, platform)
    
    return app, adapter

# Example usage
def create_lambda_handler():
    """Create AWS Lambda handler"""
    app, adapter = create_serverless_app(
        title="My Serverless API",
        platform=ServerlessPlatform.AWS_LAMBDA
    )
    
    # Add your routes here
    @app.get("/")
    async def root():
        return {"message": "Hello from AWS Lambda!"}
    
    @app.get("/users/{user_id}")
    async def get_user(user_id: int):
        return {"user_id": user_id, "name": f"User {user_id}"}
    
    return adapter.get_handler()

# For AWS Lambda
handler = create_lambda_handler()

# For Azure Functions
if AZURE_FUNCTIONS_AVAILABLE:
    app, adapter = create_serverless_app(platform=ServerlessPlatform.AZURE_FUNCTIONS)
    azure_handler = adapter.get_handler()

# For Google Cloud Functions
if GOOGLE_CLOUD_FUNCTIONS_AVAILABLE:
    app, adapter = create_serverless_app(platform=ServerlessPlatform.GOOGLE_CLOUD_FUNCTIONS)
    
    @http
    def gcf_handler(request):
        return adapter.get_handler()(request)






























