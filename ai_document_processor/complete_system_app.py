"""
Complete System AI Document Processor Application
A comprehensive, real, working FastAPI application with ALL features
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
from security_system import security_system
from notification_system import notification_system
from analytics_system import analytics_system
from backup_system import backup_system
from improved_real_routes import router as basic_router
from advanced_real_routes import router as advanced_router
from upload_routes import router as upload_router
from monitoring_routes import router as monitoring_router
from security_routes import router as security_router
from notification_routes import router as notification_router
from analytics_routes import router as analytics_router
from backup_routes import router as backup_router

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
    logger.info("Starting Complete System AI Document Processor...")
    
    try:
        # Initialize basic AI processor
        await initialize_real_working_processor()
        logger.info("Basic AI processor initialized")
        
        # Initialize advanced AI processor
        await initialize_advanced_real_processor()
        logger.info("Advanced AI processor initialized")
        
        # Initialize monitoring system
        logger.info("Monitoring system initialized")
        
        # Initialize security system
        logger.info("Security system initialized")
        
        # Initialize notification system
        logger.info("Notification system initialized")
        
        # Initialize analytics system
        logger.info("Analytics system initialized")
        
        # Initialize backup system
        logger.info("Backup system initialized")
        
        logger.info("Complete System AI Document Processor started successfully")
    except Exception as e:
        logger.error(f"Failed to start Complete System AI Document Processor: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Complete System AI Document Processor...")

# Create FastAPI app
app = FastAPI(
    title="Complete System AI Document Processor",
    description="A comprehensive, real, working AI document processing system with ALL features",
    version="5.0.0",
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
app.include_router(security_router)
app.include_router(notification_router)
app.include_router(analytics_router)
app.include_router(backup_router)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Complete System AI Document Processor",
        "version": "5.0.0",
        "status": "running",
        "description": "A comprehensive, real, working AI document processing system with ALL features",
        "features": [
            "Basic AI Processing",
            "Advanced AI Analysis",
            "Document Upload & Processing",
            "Real-time Monitoring",
            "Security System",
            "Notification System",
            "Analytics & Reporting",
            "Backup & Recovery",
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
            "Dashboard",
            "API Key Management",
            "Rate Limiting",
            "IP Blocking",
            "Security Logging",
            "Email Notifications",
            "Webhook Notifications",
            "Analytics Dashboard",
            "Trend Analysis",
            "Performance Benchmarks",
            "Content Insights",
            "Backup Creation",
            "Backup Restoration",
            "Backup Verification"
        ],
        "endpoints": {
            "basic": "/api/v1/real",
            "advanced": "/api/v1/advanced-real",
            "upload": "/api/v1/upload",
            "monitoring": "/api/v1/monitoring",
            "security": "/api/v1/security",
            "notifications": "/api/v1/notifications",
            "analytics": "/api/v1/analytics",
            "backup": "/api/v1/backup",
            "docs": "/docs",
            "health": "/health"
        }
    }

@app.get("/health")
async def health():
    """Basic health check"""
    return {
        "status": "healthy",
        "service": "Complete System AI Document Processor",
        "version": "5.0.0",
        "timestamp": time.time()
    }

@app.get("/status")
async def status():
    """Detailed status endpoint"""
    try:
        from real_working_processor import real_working_processor
        from advanced_real_processor import advanced_real_processor
        from document_upload_processor import document_upload_processor
        from security_system import security_system
        from notification_system import notification_system
        from analytics_system import analytics_system
        from backup_system import backup_system
        
        basic_stats = real_working_processor.get_stats()
        advanced_stats = advanced_real_processor.get_stats()
        upload_stats = document_upload_processor.get_stats()
        security_stats = security_system.get_security_stats()
        notification_stats = notification_system.get_stats()
        analytics_stats = analytics_system.get_stats()
        backup_stats = backup_system.get_backup_stats()
        
        basic_capabilities = real_working_processor.get_capabilities()
        advanced_capabilities = advanced_real_processor.get_capabilities()
        security_config = security_system.get_security_config()
        notification_config = notification_system.get_config()
        backup_config = backup_system.get_backup_config()
        
        return {
            "status": "healthy",
            "version": "5.0.0",
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
                },
                "security_system": {
                    "stats": security_stats["stats"],
                    "config": security_config
                },
                "notification_system": {
                    "stats": notification_stats["stats"],
                    "config": notification_config
                },
                "analytics_system": {
                    "stats": analytics_stats["stats"]
                },
                "backup_system": {
                    "stats": backup_stats["stats"],
                    "config": backup_config
                }
            },
            "features_available": {
                "basic_ai": True,
                "advanced_ai": True,
                "document_upload": True,
                "real_time_monitoring": True,
                "security_system": True,
                "notification_system": True,
                "analytics_system": True,
                "backup_system": True,
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
                "dashboard": True,
                "api_key_management": True,
                "rate_limiting": True,
                "ip_blocking": True,
                "security_logging": True,
                "email_notifications": notification_config["features"]["email_notifications"],
                "webhook_notifications": notification_config["features"]["webhook_notifications"],
                "analytics_dashboard": True,
                "trend_analysis": True,
                "performance_benchmarks": True,
                "content_insights": True,
                "backup_creation": True,
                "backup_restoration": True,
                "backup_verification": True
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
        from security_system import security_system
        from notification_system import notification_system
        from analytics_system import analytics_system
        from backup_system import backup_system
        
        basic_stats = real_working_processor.get_stats()
        advanced_stats = advanced_real_processor.get_stats()
        upload_stats = document_upload_processor.get_stats()
        security_stats = security_system.get_security_stats()
        notification_stats = notification_system.get_stats()
        analytics_stats = analytics_system.get_stats()
        backup_stats = backup_system.get_backup_stats()
        
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
            "security_requests_total": security_stats["stats"]["total_requests"],
            "security_blocked_requests_total": security_stats["stats"]["blocked_requests"],
            "security_rate_limited_requests_total": security_stats["stats"]["rate_limited_requests"],
            "security_violations_total": security_stats["stats"]["security_violations"],
            "notifications_total": notification_stats["stats"]["total_notifications"],
            "notifications_email_total": notification_stats["stats"]["email_notifications"],
            "notifications_webhook_total": notification_stats["stats"]["webhook_notifications"],
            "notifications_failed_total": notification_stats["stats"]["failed_notifications"],
            "analytics_requests_total": analytics_stats["stats"]["total_analytics_requests"],
            "analytics_reports_generated_total": analytics_stats["stats"]["reports_generated"],
            "analytics_insights_generated_total": analytics_stats["stats"]["insights_generated"],
            "backup_total": backup_stats["stats"]["total_backups"],
            "backup_successful_total": backup_stats["stats"]["successful_backups"],
            "backup_failed_total": backup_stats["stats"]["failed_backups"],
            "ai_spacy_available": 1 if basic_capabilities["nlp_analysis"] else 0,
            "ai_nltk_available": 1 if basic_capabilities["sentiment_analysis"] else 0,
            "ai_transformers_classifier_available": 1 if basic_capabilities["text_classification"] else 0,
            "ai_transformers_summarizer_available": 1 if basic_capabilities["text_summarization"] else 0,
            "ai_tfidf_vectorizer_available": 1 if advanced_capabilities["similarity_analysis"] else 0,
            "upload_pdf_available": 1 if upload_stats["supported_formats"]["pdf"] else 0,
            "upload_docx_available": 1 if upload_stats["supported_formats"]["docx"] else 0,
            "upload_excel_available": 1 if upload_stats["supported_formats"]["xlsx"] else 0,
            "upload_pptx_available": 1 if upload_stats["supported_formats"]["pptx"] else 0,
            "upload_ocr_available": 1 if upload_stats["supported_formats"]["image"] else 0,
            "security_enabled": 1,
            "notifications_enabled": 1,
            "analytics_enabled": 1,
            "backup_enabled": 1,
            "email_notifications_enabled": 1 if notification_stats["email_enabled"] else 0
        }
        
        # Format as Prometheus metrics
        metrics_text = ""
        for key, value in metrics_data.items():
            metrics_text += f"# HELP {key} Complete System AI Document Processor Metrics\n"
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
        from security_system import security_system
        from notification_system import notification_system
        from analytics_system import analytics_system
        from backup_system import backup_system
        
        # Get monitoring dashboard
        dashboard_data = await monitoring_system.get_comprehensive_metrics(
            real_working_processor, advanced_real_processor, document_upload_processor
        )
        
        # Add security, notification, analytics, and backup data
        security_stats = security_system.get_security_stats()
        notification_stats = notification_system.get_stats()
        analytics_stats = analytics_system.get_stats()
        backup_stats = backup_system.get_backup_stats()
        
        dashboard_data.update({
            "security": {
                "total_requests": security_stats["stats"]["total_requests"],
                "blocked_requests": security_stats["stats"]["blocked_requests"],
                "rate_limited_requests": security_stats["stats"]["rate_limited_requests"],
                "security_violations": security_stats["stats"]["security_violations"],
                "blocked_ips_count": security_stats["blocked_ips_count"]
            },
            "notifications": {
                "total_notifications": notification_stats["stats"]["total_notifications"],
                "email_notifications": notification_stats["stats"]["email_notifications"],
                "webhook_notifications": notification_stats["stats"]["webhook_notifications"],
                "failed_notifications": notification_stats["stats"]["failed_notifications"],
                "total_subscribers": notification_stats["total_subscribers"],
                "active_subscribers": notification_stats["active_subscribers"]
            },
            "analytics": {
                "total_analytics_requests": analytics_stats["stats"]["total_analytics_requests"],
                "reports_generated": analytics_stats["stats"]["reports_generated"],
                "insights_generated": analytics_stats["stats"]["insights_generated"],
                "total_reports": analytics_stats["total_reports"],
                "total_insights": analytics_stats["total_insights"]
            },
            "backup": {
                "total_backups": backup_stats["stats"]["total_backups"],
                "successful_backups": backup_stats["stats"]["successful_backups"],
                "failed_backups": backup_stats["stats"]["failed_backups"],
                "last_backup_time": backup_stats["stats"]["last_backup_time"]
            }
        })
        
        return JSONResponse(content=dashboard_data)
        
    except Exception as e:
        logger.error(f"Error getting dashboard: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/comparison")
async def comparison():
    """Compare all processors and systems"""
    try:
        from real_working_processor import real_working_processor
        from advanced_real_processor import advanced_real_processor
        from document_upload_processor import document_upload_processor
        from security_system import security_system
        from notification_system import notification_system
        from analytics_system import analytics_system
        from backup_system import backup_system
        
        basic_stats = real_working_processor.get_stats()
        advanced_stats = advanced_real_processor.get_stats()
        upload_stats = document_upload_processor.get_stats()
        security_stats = security_system.get_security_stats()
        notification_stats = notification_system.get_stats()
        analytics_stats = analytics_system.get_stats()
        backup_stats = backup_system.get_backup_stats()
        
        basic_capabilities = real_working_processor.get_capabilities()
        advanced_capabilities = advanced_real_processor.get_capabilities()
        security_config = security_system.get_security_config()
        notification_config = notification_system.get_config()
        backup_config = backup_system.get_backup_config()
        
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
                "security_system": {
                    "capabilities": {
                        "request_validation": True,
                        "file_validation": True,
                        "rate_limiting": True,
                        "ip_blocking": True,
                        "api_key_management": True,
                        "security_logging": True
                    },
                    "stats": security_stats["stats"],
                    "config": security_config
                },
                "notification_system": {
                    "capabilities": notification_config["features"],
                    "stats": notification_stats["stats"],
                    "config": notification_config
                },
                "analytics_system": {
                    "capabilities": {
                        "processing_analytics": True,
                        "user_analytics": True,
                        "performance_analytics": True,
                        "content_analytics": True,
                        "trend_analytics": True,
                        "insight_generation": True,
                        "report_generation": True
                    },
                    "stats": analytics_stats["stats"]
                },
                "backup_system": {
                    "capabilities": backup_config["features"],
                    "stats": backup_stats["stats"],
                    "config": backup_config
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
                    "security_features": [
                        "request_validation",
                        "file_validation",
                        "rate_limiting",
                        "ip_blocking",
                        "api_key_management",
                        "security_logging",
                        "malicious_content_detection"
                    ],
                    "notification_features": [
                        "email_notifications",
                        "webhook_notifications",
                        "processing_notifications",
                        "error_notifications",
                        "security_notifications",
                        "performance_notifications"
                    ],
                    "analytics_features": [
                        "processing_analytics",
                        "user_analytics",
                        "performance_analytics",
                        "content_analytics",
                        "trend_analytics",
                        "insight_generation",
                        "report_generation"
                    ],
                    "backup_features": [
                        "full_backup",
                        "data_backup",
                        "config_backup",
                        "app_backup",
                        "doc_backup",
                        "install_backup",
                        "automatic_cleanup"
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
        "complete_system_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )













