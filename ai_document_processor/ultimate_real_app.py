"""
Ultimate Real AI Document Processor Application
A comprehensive, real, working FastAPI application with all features
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import time

from real_working_processor import initialize_real_working_processor
from advanced_real_processor import initialize_advanced_real_processor
from document_upload_processor import document_upload_processor
from monitoring_system import monitoring_system
from improved_real_routes import router as basic_router
from advanced_real_routes import router as advanced_router
from upload_routes import router as upload_router
from monitoring_routes import router as monitoring_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting Ultimate Real AI Document Processor...")
    
    try:
        # Initialize basic AI processor
        await initialize_real_working_processor()
        logger.info("Basic AI processor initialized")
        
        # Initialize advanced AI processor
        await initialize_advanced_real_processor()
        logger.info("Advanced AI processor initialized")
        
        # Initialize monitoring system
        logger.info("Monitoring system initialized")
        
        logger.info("Ultimate Real AI Document Processor started successfully")
    except Exception as e:
        logger.error(f"Failed to start Ultimate Real AI Document Processor: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Ultimate Real AI Document Processor...")

# Create FastAPI app
app = FastAPI(
    title="Ultimate Real AI Document Processor",
    description="A comprehensive, real, working AI document processing system with all features",
    version="3.0.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add compression middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Add request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Include all routers
app.include_router(basic_router)
app.include_router(advanced_router)
app.include_router(upload_router)
app.include_router(monitoring_router)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Ultimate Real AI Document Processor",
        "version": "3.0.0",
        "status": "running",
        "description": "A comprehensive, real, working AI document processing system with all features",
        "features": [
            "Basic AI Processing",
            "Advanced AI Analysis",
            "Document Upload & Processing",
            "Real-time Monitoring",
            "Complexity Analysis",
            "Readability Analysis",
            "Language Pattern Analysis",
            "Quality Metrics",
            "Advanced Keyword Analysis",
            "Similarity Analysis",
            "Topic Analysis",
            "Batch Processing",
            "Caching",
            "Performance Monitoring",
            "System Monitoring",
            "Alert System",
            "Dashboard"
        ],
        "endpoints": {
            "basic": "/api/v1/real",
            "advanced": "/api/v1/advanced-real",
            "upload": "/api/v1/upload",
            "monitoring": "/api/v1/monitoring",
            "docs": "/docs",
            "health": "/health"
        }
    }

@app.get("/health")
async def health():
    """Basic health check"""
    return {
        "status": "healthy",
        "service": "Ultimate Real AI Document Processor",
        "version": "3.0.0",
        "timestamp": time.time()
    }

@app.get("/status")
async def status():
    """Detailed status endpoint"""
    try:
        from real_working_processor import real_working_processor
        from advanced_real_processor import advanced_real_processor
        from document_upload_processor import document_upload_processor
        
        basic_stats = real_working_processor.get_stats()
        advanced_stats = advanced_real_processor.get_stats()
        upload_stats = document_upload_processor.get_stats()
        basic_capabilities = real_working_processor.get_capabilities()
        advanced_capabilities = advanced_real_processor.get_capabilities()
        
        return {
            "status": "healthy",
            "version": "3.0.0",
            "uptime": time.time(),
            "processors": {
                "basic_processor": {
                    "stats": basic_stats["stats"],
                    "capabilities": basic_capabilities
                },
                "advanced_processor": {
                    "stats": advanced_stats["stats"],
                    "capabilities": advanced_capabilities
                },
                "upload_processor": {
                    "stats": upload_stats["stats"],
                    "supported_formats": upload_stats["supported_formats"]
                }
            },
            "features_available": {
                "basic_ai": True,
                "advanced_ai": True,
                "document_upload": True,
                "real_time_monitoring": True,
                "complexity_analysis": True,
                "readability_analysis": True,
                "language_patterns": True,
                "quality_metrics": True,
                "keyword_analysis": True,
                "similarity_analysis": advanced_capabilities["similarity_analysis"],
                "topic_analysis": advanced_capabilities["topic_analysis"],
                "caching": True,
                "batch_processing": True,
                "system_monitoring": True,
                "alert_system": True,
                "dashboard": True
            }
        }
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

@app.get("/metrics")
async def metrics():
    """Prometheus-style metrics endpoint"""
    try:
        from real_working_processor import real_working_processor
        from advanced_real_processor import advanced_real_processor
        from document_upload_processor import document_upload_processor
        
        basic_stats = real_working_processor.get_stats()
        advanced_stats = advanced_real_processor.get_stats()
        upload_stats = document_upload_processor.get_stats()
        basic_capabilities = real_working_processor.get_capabilities()
        advanced_capabilities = advanced_real_processor.get_capabilities()
        
        metrics_data = {
            "ai_requests_total": basic_stats["stats"]["total_requests"] + advanced_stats["stats"]["total_requests"],
            "ai_successful_requests_total": basic_stats["stats"]["successful_requests"] + advanced_stats["stats"]["successful_requests"],
            "ai_failed_requests_total": basic_stats["stats"]["failed_requests"] + advanced_stats["stats"]["failed_requests"],
            "ai_processing_time_seconds": (basic_stats["stats"]["average_processing_time"] + advanced_stats["stats"]["average_processing_time"]) / 2,
            "ai_uptime_seconds": basic_stats["uptime_seconds"],
            "ai_success_rate_percent": (basic_stats["success_rate"] + advanced_stats["success_rate"]) / 2,
            "ai_cache_hit_rate_percent": advanced_stats["cache_hit_rate"],
            "upload_total": upload_stats["stats"]["total_uploads"],
            "upload_successful_total": upload_stats["stats"]["successful_uploads"],
            "upload_failed_total": upload_stats["stats"]["failed_uploads"],
            "upload_success_rate_percent": upload_stats["success_rate"],
            "ai_spacy_available": 1 if basic_capabilities["nlp_analysis"] else 0,
            "ai_nltk_available": 1 if basic_capabilities["sentiment_analysis"] else 0,
            "ai_transformers_classifier_available": 1 if basic_capabilities["text_classification"] else 0,
            "ai_transformers_summarizer_available": 1 if basic_capabilities["text_summarization"] else 0,
            "ai_tfidf_vectorizer_available": 1 if advanced_capabilities["similarity_analysis"] else 0,
            "upload_pdf_available": 1 if upload_stats["supported_formats"]["pdf"] else 0,
            "upload_docx_available": 1 if upload_stats["supported_formats"]["docx"] else 0,
            "upload_excel_available": 1 if upload_stats["supported_formats"]["xlsx"] else 0,
            "upload_pptx_available": 1 if upload_stats["supported_formats"]["pptx"] else 0,
            "upload_ocr_available": 1 if upload_stats["supported_formats"]["image"] else 0
        }
        
        # Format as Prometheus metrics
        metrics_text = ""
        for key, value in metrics_data.items():
            metrics_text += f"# HELP {key} Ultimate AI Document Processor Metrics\n"
            metrics_text += f"# TYPE {key} gauge\n"
            metrics_text += f"{key} {value}\n"
        
        return metrics_text
        
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        return f"# Error getting metrics: {e}"

@app.get("/dashboard")
async def dashboard():
    """Get comprehensive dashboard data"""
    try:
        from real_working_processor import real_working_processor
        from advanced_real_processor import advanced_real_processor
        from document_upload_processor import document_upload_processor
        
        # Get monitoring dashboard
        dashboard_data = await monitoring_system.get_comprehensive_metrics(
            real_working_processor, advanced_real_processor, document_upload_processor
        )
        
        return JSONResponse(content=dashboard_data)
        
    except Exception as e:
        logger.error(f"Error getting dashboard: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/comparison")
async def comparison():
    """Compare all processors"""
    try:
        from real_working_processor import real_working_processor
        from advanced_real_processor import advanced_real_processor
        from document_upload_processor import document_upload_processor
        
        basic_stats = real_working_processor.get_stats()
        advanced_stats = advanced_real_processor.get_stats()
        upload_stats = document_upload_processor.get_stats()
        basic_capabilities = real_working_processor.get_capabilities()
        advanced_capabilities = advanced_real_processor.get_capabilities()
        
        return {
            "comparison": {
                "basic_processor": {
                    "capabilities": basic_capabilities,
                    "stats": basic_stats["stats"],
                    "success_rate": basic_stats["success_rate"]
                },
                "advanced_processor": {
                    "capabilities": advanced_capabilities,
                    "stats": advanced_stats["stats"],
                    "success_rate": advanced_stats["success_rate"],
                    "cache_hit_rate": advanced_stats["cache_hit_rate"]
                },
                "upload_processor": {
                    "capabilities": upload_stats["supported_formats"],
                    "stats": upload_stats["stats"],
                    "success_rate": upload_stats["success_rate"]
                },
                "monitoring_system": {
                    "features": [
                        "system_monitoring",
                        "ai_monitoring",
                        "upload_monitoring",
                        "performance_monitoring",
                        "alert_system",
                        "dashboard"
                    ]
                },
                "differences": {
                    "basic_features": [
                        "text_analysis",
                        "sentiment_analysis",
                        "text_classification",
                        "text_summarization",
                        "keyword_extraction",
                        "language_detection"
                    ],
                    "advanced_features": [
                        "complexity_analysis",
                        "readability_analysis",
                        "language_patterns",
                        "quality_metrics",
                        "similarity_analysis",
                        "topic_analysis",
                        "caching",
                        "batch_processing"
                    ],
                    "upload_features": [
                        "pdf_processing",
                        "docx_processing",
                        "excel_processing",
                        "powerpoint_processing",
                        "text_processing",
                        "ocr_processing"
                    ],
                    "monitoring_features": [
                        "system_monitoring",
                        "ai_monitoring",
                        "upload_monitoring",
                        "performance_monitoring",
                        "alert_system",
                        "dashboard"
                    ]
                }
            }
        }
    except Exception as e:
        logger.error(f"Error getting comparison: {e}")
        return {
            "error": str(e)
        }

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "path": str(request.url),
            "method": request.method
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "ultimate_real_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )













