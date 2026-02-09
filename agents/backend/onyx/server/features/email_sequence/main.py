"""
FastAPI Email Sequence Application

This is the main FastAPI application for the email sequence system,
following best practices for async operations, error handling, and performance.
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, Response, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import uvicorn

from .core.config import get_settings
from .core.dependencies import lifespan
from .core.exceptions import EmailSequenceError
from .api.routes import email_sequence_router
from .api.advanced_routes import advanced_router
from .api.ml_routes import ml_router
from .api.websocket_routes import websocket_router
from .api.neural_network_routes import neural_network_router
from .api.edge_computing_routes import edge_computing_router
from .api.quantum_computing_routes import quantum_computing_router
from .api.advanced_ai_routes import advanced_ai_router
from .api.metaverse_routes import metaverse_router
from .api.space_computing_routes import space_computing_router
from .api.time_travel_routes import time_travel_router
from .api.schemas import ErrorResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Get settings
settings = get_settings()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""
    
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="AI-powered email sequence management system with LangChain integration",
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        openapi_url="/openapi.json" if settings.debug else None,
        lifespan=lifespan
    )
    
    # Add middleware
    setup_middleware(app)
    
    # Add exception handlers
    setup_exception_handlers(app)
    
    # Include routers
    app.include_router(
        email_sequence_router,
        prefix=settings.api_prefix
    )
    
    # Include advanced features router
    app.include_router(advanced_router)
    
    # Include ML features router
    app.include_router(ml_router)
    
    # Include WebSocket router
    app.include_router(websocket_router)
    
    # Include Neural Network router
    app.include_router(neural_network_router)
    
    # Include Edge Computing router
    app.include_router(edge_computing_router)
    
    # Include Quantum Computing router
    app.include_router(quantum_computing_router)
    
    # Include Advanced AI router
    app.include_router(advanced_ai_router)
    
    # Include Metaverse router
    app.include_router(metaverse_router)
    
    # Include Space Computing router
    app.include_router(space_computing_router)
    
    # Include Time Travel router
    app.include_router(time_travel_router)
    
    # Add health check endpoint
    @app.get("/health", tags=["Health"])
    async def health_check() -> Dict[str, Any]:
        """Health check endpoint"""
        return {
            "status": "healthy",
            "version": settings.app_version,
            "environment": settings.environment,
            "timestamp": time.time()
        }
    
    # Add root endpoint
    @app.get("/", tags=["Root"])
    async def root() -> Dict[str, str]:
        """Root endpoint"""
        return {
            "message": "Email Sequence AI API",
            "version": settings.app_version,
            "docs": "/docs" if settings.debug else "Documentation not available in production"
        }
    
    return app


def setup_middleware(app: FastAPI) -> None:
    """Setup middleware for the application"""
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=settings.cors_allow_methods,
        allow_headers=settings.cors_allow_headers,
    )
    
    # Trusted host middleware
    if settings.environment == "production":
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["*"]  # Configure with actual hosts in production
        )
    
    # GZip compression middleware
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """Log all requests"""
        start_time = time.time()
        
        # Log request
        logger.info(
            f"Request: {request.method} {request.url.path} "
            f"from {request.client.host if request.client else 'unknown'}"
        )
        
        # Process request
        response = await call_next(request)
        
        # Log response
        process_time = time.time() - start_time
        logger.info(
            f"Response: {response.status_code} "
            f"in {process_time:.4f}s"
        )
        
        # Add timing header
        response.headers["X-Process-Time"] = str(process_time)
        
        return response
    
    # Error handling middleware
    @app.middleware("http")
    async def error_handling_middleware(request: Request, call_next):
        """Handle errors and add error tracking"""
        try:
            response = await call_next(request)
            return response
        except Exception as e:
            logger.error(f"Unhandled error: {e}", exc_info=True)
            
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=ErrorResponse(
                    error="Internal server error",
                    error_code="INTERNAL_SERVER_ERROR",
                    details={"message": str(e)} if settings.debug else None
                ).dict()
            )


def setup_exception_handlers(app: FastAPI) -> None:
    """Setup exception handlers for the application"""
    
    @app.exception_handler(EmailSequenceError)
    async def email_sequence_error_handler(request: Request, exc: EmailSequenceError):
        """Handle custom email sequence errors"""
        logger.error(f"Email sequence error: {exc.message}")
        
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorResponse(
                error=exc.message,
                error_code=exc.error_code,
                details=exc.details
            ).dict()
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle request validation errors"""
        logger.warning(f"Validation error: {exc.errors()}")
        
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorResponse(
                error="Validation error",
                error_code="VALIDATION_ERROR",
                details={"errors": exc.errors()}
            ).dict()
        )
    
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        """Handle HTTP exceptions"""
        logger.warning(f"HTTP error: {exc.status_code} - {exc.detail}")
        
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error=exc.detail,
                error_code=f"HTTP_{exc.status_code}"
            ).dict()
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle general exceptions"""
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error="Internal server error",
                error_code="INTERNAL_SERVER_ERROR",
                details={"message": str(exc)} if settings.debug else None
            ).dict()
        )


# Create the application instance
app = create_app()


def run_server() -> None:
    """Run the development server"""
    uvicorn.run(
        "email_sequence.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
        access_log=True,
        use_colors=True
    )


if __name__ == "__main__":
    run_server()
