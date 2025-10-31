"""
FastAPI Application Factory
==========================

This module creates and configures the FastAPI application for the
AI History Comparison system with proper middleware, error handling,
and API documentation.
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from contextlib import asynccontextmanager
import logging
from typing import Optional
import uvicorn

from ..infrastructure.database import DatabaseManager, DatabaseConfig
from ..infrastructure.repositories import (
    HistoryRepository, ComparisonRepository, ReportRepository,
    AnalysisJobRepository, UserFeedbackRepository
)
from ..core.services import (
    ContentAnalysisService, ModelComparisonService,
    TrendAnalysisService, QualityAssessmentService
)
from ..application.use_cases import (
    AnalyzeContentUseCase, CompareModelsUseCase,
    GenerateReportUseCase, TrackTrendsUseCase, ManageAnalysisJobUseCase
)
from .endpoints import (
    AnalysisEndpoints, ComparisonEndpoints, ReportEndpoints,
    TrendEndpoints, SystemEndpoints
)
from .middleware import (
    ErrorHandlerMiddleware, LoggingMiddleware,
    AuthenticationMiddleware, RateLimitMiddleware
)

logger = logging.getLogger(__name__)

# Global app instance
_app: Optional[FastAPI] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("🚀 Starting AI History Comparison System...")
    
    try:
        # Initialize database
        db_config = DatabaseConfig(
            url="sqlite:///./ai_history.db",  # Default for development
            echo=False
        )
        db_manager = DatabaseManager(db_config)
        db_manager.initialize()
        db_manager.create_tables()
        
        # Store in app state
        app.state.db_manager = db_manager
        
        # Initialize repositories
        with db_manager.get_session() as session:
            app.state.history_repo = HistoryRepository(session)
            app.state.comparison_repo = ComparisonRepository(session)
            app.state.report_repo = ReportRepository(session)
            app.state.job_repo = AnalysisJobRepository(session)
            app.state.feedback_repo = UserFeedbackRepository(session)
        
        # Initialize services
        app.state.content_analysis_service = ContentAnalysisService()
        app.state.model_comparison_service = ModelComparisonService()
        app.state.trend_analysis_service = TrendAnalysisService()
        app.state.quality_assessment_service = QualityAssessmentService()
        
        # Initialize use cases
        app.state.analyze_content_use_case = AnalyzeContentUseCase(
            app.state.history_repo,
            app.state.comparison_repo,
            app.state.report_repo,
            app.state.job_repo,
            app.state.content_analysis_service
        )
        
        app.state.compare_models_use_case = CompareModelsUseCase(
            app.state.history_repo,
            app.state.comparison_repo,
            app.state.report_repo,
            app.state.job_repo,
            app.state.model_comparison_service
        )
        
        app.state.generate_report_use_case = GenerateReportUseCase(
            app.state.history_repo,
            app.state.comparison_repo,
            app.state.report_repo,
            app.state.job_repo,
            app.state.quality_assessment_service
        )
        
        app.state.track_trends_use_case = TrackTrendsUseCase(
            app.state.history_repo,
            app.state.comparison_repo,
            app.state.report_repo,
            app.state.job_repo,
            app.state.trend_analysis_service
        )
        
        app.state.manage_jobs_use_case = ManageAnalysisJobUseCase(
            app.state.history_repo,
            app.state.comparison_repo,
            app.state.report_repo,
            app.state.job_repo
        )
        
        logger.info("✅ AI History Comparison System initialized successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize system: {str(e)}")
        raise
    finally:
        # Shutdown
        logger.info("🛑 Shutting down AI History Comparison System...")
        if hasattr(app.state, 'db_manager'):
            app.state.db_manager.close()


def create_app(
    title: str = "AI History Comparison System",
    description: str = """
    A comprehensive system for analyzing, comparing, and tracking AI model outputs over time.
    
    ## Features
    
    * **Content Analysis** - Analyze content quality, readability, sentiment, and complexity
    * **Historical Comparison** - Compare content across different time periods and model versions
    * **Trend Analysis** - Track performance trends and identify patterns
    * **Quality Reporting** - Generate comprehensive quality reports with insights
    * **Model Comparison** - Compare different AI models and their performance
    * **Real-time Monitoring** - Monitor system health and performance metrics
    
    ## Use Cases
    
    * Monitor AI model performance over time
    * Track content quality improvements
    * Identify patterns in AI-generated content
    * Compare different model versions
    * Generate quality reports for stakeholders
    * Optimize content generation workflows
    """,
    version: str = "2.0.0",
    debug: bool = False,
    cors_origins: list = None,
    trusted_hosts: list = None
) -> FastAPI:
    """Create and configure FastAPI application"""
    
    global _app
    
    if _app is not None:
        return _app
    
    # Default CORS origins
    if cors_origins is None:
        cors_origins = ["*"] if debug else ["http://localhost:3000", "https://yourdomain.com"]
    
    # Default trusted hosts
    if trusted_hosts is None:
        trusted_hosts = ["*"] if debug else ["localhost", "127.0.0.1", "*.yourdomain.com"]
    
    # Create FastAPI app
    app = FastAPI(
        title=title,
        description=description,
        version=version,
        lifespan=lifespan,
        docs_url="/docs" if debug else None,
        redoc_url="/redoc" if debug else None,
        openapi_url="/openapi.json" if debug else None
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )
    
    # Add trusted host middleware for production
    if not debug:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=trusted_hosts
        )
    
    # Add custom middleware
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(ErrorHandlerMiddleware)
    
    # Add authentication middleware (optional)
    if not debug:
        app.add_middleware(AuthenticationMiddleware)
    
    # Add rate limiting middleware
    app.add_middleware(RateLimitMiddleware)
    
    # Include API routes
    app.include_router(AnalysisEndpoints.router, prefix="/api/v1/analysis", tags=["Analysis"])
    app.include_router(ComparisonEndpoints.router, prefix="/api/v1/comparison", tags=["Comparison"])
    app.include_router(ReportEndpoints.router, prefix="/api/v1/reports", tags=["Reports"])
    app.include_router(TrendEndpoints.router, prefix="/api/v1/trends", tags=["Trends"])
    app.include_router(SystemEndpoints.router, prefix="/api/v1/system", tags=["System"])
    
    # Root endpoint
    @app.get("/", tags=["Root"])
    async def root():
        """Root endpoint with system information"""
        return {
            "name": "AI History Comparison System",
            "version": version,
            "description": "Comprehensive AI content analysis and comparison system",
            "status": "operational",
            "docs": "/docs" if debug else "disabled in production",
            "endpoints": {
                "analysis": "/api/v1/analysis",
                "comparison": "/api/v1/comparison",
                "reports": "/api/v1/reports",
                "trends": "/api/v1/trends",
                "system": "/api/v1/system"
            }
        }
    
    # Health check endpoint
    @app.get("/health", tags=["System"])
    async def health_check():
        """Health check endpoint"""
        try:
            # Check database health
            db_manager = app.state.db_manager
            db_health = db_manager.health_check()
            
            health_status = {
                "status": "healthy" if db_health["healthy"] else "unhealthy",
                "version": version,
                "database": db_health,
                "components": {
                    "database": "healthy" if db_health["healthy"] else "unhealthy",
                    "api": "healthy",
                    "services": "healthy"
                }
            }
            
            if not db_health["healthy"]:
                return JSONResponse(
                    status_code=503,
                    content=health_status
                )
            
            return health_status
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "error": str(e)
                }
            )
    
    # Custom OpenAPI schema
    def custom_openapi():
        """Custom OpenAPI schema"""
        if app.openapi_schema:
            return app.openapi_schema
        
        openapi_schema = get_openapi(
            title=title,
            version=version,
            description=description,
            routes=app.routes,
        )
        
        # Add custom tags
        openapi_schema["tags"] = [
            {
                "name": "Analysis",
                "description": "Content analysis and quality assessment"
            },
            {
                "name": "Comparison",
                "description": "Model and content comparison operations"
            },
            {
                "name": "Reports",
                "description": "Quality reports and analytics"
            },
            {
                "name": "Trends",
                "description": "Trend analysis and forecasting"
            },
            {
                "name": "System",
                "description": "System management and monitoring"
            }
        ]
        
        app.openapi_schema = openapi_schema
        return app.openapi_schema
    
    app.openapi = custom_openapi
    
    # Custom docs endpoint
    @app.get("/docs", include_in_schema=False)
    async def custom_swagger_ui_html():
        """Custom Swagger UI with enhanced styling"""
        return get_swagger_ui_html(
            openapi_url="/openapi.json",
            title=f"{title} - API Documentation",
            swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
            swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
            swagger_favicon_url="https://fastapi.tiangolo.com/img/favicon.png"
        )
    
    _app = app
    return app


def get_app() -> FastAPI:
    """Get the global app instance"""
    global _app
    if _app is None:
        _app = create_app()
    return _app


def run_app(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
    log_level: str = "info"
):
    """Run the application"""
    app = get_app()
    
    uvicorn.run(
        "ai_history_comparison.refactored.presentation.api:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
        access_log=True
    )


if __name__ == "__main__":
    run_app()




