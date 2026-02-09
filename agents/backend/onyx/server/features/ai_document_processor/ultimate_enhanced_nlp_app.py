"""
Ultimate Enhanced NLP Application for AI Document Processor
Real, working ultimate enhanced Natural Language Processing FastAPI application
"""

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import logging
import time
import asyncio
from datetime import datetime
import uvicorn
import os
import sys

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import all NLP systems
from nlp_system import nlp_system
from enhanced_nlp_system import enhanced_nlp_system
from advanced_nlp_features import advanced_nlp_system
from super_advanced_nlp import super_advanced_nlp_system
from hyper_advanced_nlp import hyper_advanced_nlp_system
from ultra_fast_nlp import ultra_fast_nlp_system
from ultimate_enhanced_nlp import ultimate_enhanced_nlp_system

# Import all route modules
from nlp_routes import router as nlp_router
from enhanced_nlp_routes import router as enhanced_nlp_router
from advanced_nlp_routes import router as advanced_nlp_router
from super_advanced_nlp_routes import router as super_advanced_nlp_router
from hyper_advanced_nlp_routes import router as hyper_advanced_nlp_router
from ultra_fast_nlp_routes import router as ultra_fast_nlp_router
from ultimate_enhanced_nlp_routes import router as ultimate_enhanced_nlp_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ultimate_enhanced_nlp_app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="Ultimate Enhanced NLP API",
    description="Real, working ultimate enhanced Natural Language Processing system for AI document processing",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Timing middleware
@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )

# Include all NLP routers
app.include_router(nlp_router, prefix="/api/v1")
app.include_router(enhanced_nlp_router, prefix="/api/v1")
app.include_router(advanced_nlp_router, prefix="/api/v1")
app.include_router(super_advanced_nlp_router, prefix="/api/v1")
app.include_router(hyper_advanced_nlp_router, prefix="/api/v1")
app.include_router(ultra_fast_nlp_router, prefix="/api/v1")
app.include_router(ultimate_enhanced_nlp_router, prefix="/api/v1")

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with system information"""
    return {
        "message": "Ultimate Enhanced NLP API",
        "version": "1.0.0",
        "description": "Real, working ultimate enhanced Natural Language Processing system",
        "features": [
            "Basic NLP",
            "Enhanced NLP", 
            "Advanced NLP",
            "Super Advanced NLP",
            "Hyper Advanced NLP",
            "Ultra Fast NLP",
            "Ultimate Enhanced NLP"
        ],
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "health": "/health",
            "stats": "/stats"
        },
        "timestamp": datetime.now().isoformat()
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Comprehensive health check for all NLP systems"""
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "systems": {}
        }
        
        # Check basic NLP system
        try:
            basic_nlp_stats = nlp_system.get_nlp_stats()
            health_status["systems"]["basic_nlp"] = {
                "status": "healthy",
                "uptime": basic_nlp_stats.get("uptime_seconds", 0),
                "requests": basic_nlp_stats.get("stats", {}).get("total_requests", 0)
            }
        except Exception as e:
            health_status["systems"]["basic_nlp"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # Check enhanced NLP system
        try:
            enhanced_nlp_stats = enhanced_nlp_system.get_enhanced_nlp_stats()
            health_status["systems"]["enhanced_nlp"] = {
                "status": "healthy",
                "uptime": enhanced_nlp_stats.get("uptime_seconds", 0),
                "requests": enhanced_nlp_stats.get("stats", {}).get("total_enhanced_requests", 0)
            }
        except Exception as e:
            health_status["systems"]["enhanced_nlp"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # Check advanced NLP system
        try:
            advanced_nlp_stats = advanced_nlp_system.get_advanced_nlp_stats()
            health_status["systems"]["advanced_nlp"] = {
                "status": "healthy",
                "uptime": advanced_nlp_stats.get("uptime_seconds", 0),
                "requests": advanced_nlp_stats.get("stats", {}).get("total_advanced_requests", 0)
            }
        except Exception as e:
            health_status["systems"]["advanced_nlp"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # Check super advanced NLP system
        try:
            super_advanced_nlp_stats = super_advanced_nlp_system.get_super_advanced_nlp_stats()
            health_status["systems"]["super_advanced_nlp"] = {
                "status": "healthy",
                "uptime": super_advanced_nlp_stats.get("uptime_seconds", 0),
                "requests": super_advanced_nlp_stats.get("stats", {}).get("total_super_advanced_requests", 0)
            }
        except Exception as e:
            health_status["systems"]["super_advanced_nlp"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # Check hyper advanced NLP system
        try:
            hyper_advanced_nlp_stats = hyper_advanced_nlp_system.get_hyper_advanced_nlp_stats()
            health_status["systems"]["hyper_advanced_nlp"] = {
                "status": "healthy",
                "uptime": hyper_advanced_nlp_stats.get("uptime_seconds", 0),
                "requests": hyper_advanced_nlp_stats.get("stats", {}).get("total_hyper_advanced_requests", 0)
            }
        except Exception as e:
            health_status["systems"]["hyper_advanced_nlp"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # Check ultra fast NLP system
        try:
            ultra_fast_nlp_stats = ultra_fast_nlp_system.get_ultra_fast_nlp_stats()
            health_status["systems"]["ultra_fast_nlp"] = {
                "status": "healthy",
                "uptime": ultra_fast_nlp_stats.get("uptime_seconds", 0),
                "requests": ultra_fast_nlp_stats.get("stats", {}).get("total_ultra_fast_requests", 0)
            }
        except Exception as e:
            health_status["systems"]["ultra_fast_nlp"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # Check ultimate enhanced NLP system
        try:
            ultimate_enhanced_nlp_stats = ultimate_enhanced_nlp_system.get_ultimate_enhanced_nlp_stats()
            health_status["systems"]["ultimate_enhanced_nlp"] = {
                "status": "healthy",
                "uptime": ultimate_enhanced_nlp_stats.get("uptime_seconds", 0),
                "requests": ultimate_enhanced_nlp_stats.get("stats", {}).get("total_ultimate_enhanced_requests", 0)
            }
        except Exception as e:
            health_status["systems"]["ultimate_enhanced_nlp"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # Determine overall health
        unhealthy_systems = [name for name, system in health_status["systems"].items() if system["status"] == "unhealthy"]
        if unhealthy_systems:
            health_status["status"] = "degraded"
            health_status["unhealthy_systems"] = unhealthy_systems
        else:
            health_status["status"] = "healthy"
        
        return health_status
        
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# Statistics endpoint
@app.get("/stats")
async def get_comprehensive_stats():
    """Get comprehensive statistics from all NLP systems"""
    try:
        stats = {
            "timestamp": datetime.now().isoformat(),
            "systems": {}
        }
        
        # Get basic NLP stats
        try:
            basic_nlp_stats = nlp_system.get_nlp_stats()
            stats["systems"]["basic_nlp"] = basic_nlp_stats
        except Exception as e:
            stats["systems"]["basic_nlp"] = {"error": str(e)}
        
        # Get enhanced NLP stats
        try:
            enhanced_nlp_stats = enhanced_nlp_system.get_enhanced_nlp_stats()
            stats["systems"]["enhanced_nlp"] = enhanced_nlp_stats
        except Exception as e:
            stats["systems"]["enhanced_nlp"] = {"error": str(e)}
        
        # Get advanced NLP stats
        try:
            advanced_nlp_stats = advanced_nlp_system.get_advanced_nlp_stats()
            stats["systems"]["advanced_nlp"] = advanced_nlp_stats
        except Exception as e:
            stats["systems"]["advanced_nlp"] = {"error": str(e)}
        
        # Get super advanced NLP stats
        try:
            super_advanced_nlp_stats = super_advanced_nlp_system.get_super_advanced_nlp_stats()
            stats["systems"]["super_advanced_nlp"] = super_advanced_nlp_stats
        except Exception as e:
            stats["systems"]["super_advanced_nlp"] = {"error": str(e)}
        
        # Get hyper advanced NLP stats
        try:
            hyper_advanced_nlp_stats = hyper_advanced_nlp_system.get_hyper_advanced_nlp_stats()
            stats["systems"]["hyper_advanced_nlp"] = hyper_advanced_nlp_stats
        except Exception as e:
            stats["systems"]["hyper_advanced_nlp"] = {"error": str(e)}
        
        # Get ultra fast NLP stats
        try:
            ultra_fast_nlp_stats = ultra_fast_nlp_system.get_ultra_fast_nlp_stats()
            stats["systems"]["ultra_fast_nlp"] = ultra_fast_nlp_stats
        except Exception as e:
            stats["systems"]["ultra_fast_nlp"] = {"error": str(e)}
        
        # Get ultimate enhanced NLP stats
        try:
            ultimate_enhanced_nlp_stats = ultimate_enhanced_nlp_system.get_ultimate_enhanced_nlp_stats()
            stats["systems"]["ultimate_enhanced_nlp"] = ultimate_enhanced_nlp_stats
        except Exception as e:
            stats["systems"]["ultimate_enhanced_nlp"] = {"error": str(e)}
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting comprehensive stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Comparison endpoint
@app.get("/compare")
async def compare_nlp_systems():
    """Compare performance across all NLP systems"""
    try:
        comparison = {
            "timestamp": datetime.now().isoformat(),
            "comparison": {}
        }
        
        # Get stats from all systems
        systems_stats = {}
        
        try:
            systems_stats["basic_nlp"] = nlp_system.get_nlp_stats()
        except:
            systems_stats["basic_nlp"] = None
        
        try:
            systems_stats["enhanced_nlp"] = enhanced_nlp_system.get_enhanced_nlp_stats()
        except:
            systems_stats["enhanced_nlp"] = None
        
        try:
            systems_stats["advanced_nlp"] = advanced_nlp_system.get_advanced_nlp_stats()
        except:
            systems_stats["advanced_nlp"] = None
        
        try:
            systems_stats["super_advanced_nlp"] = super_advanced_nlp_system.get_super_advanced_nlp_stats()
        except:
            systems_stats["super_advanced_nlp"] = None
        
        try:
            systems_stats["hyper_advanced_nlp"] = hyper_advanced_nlp_system.get_hyper_advanced_nlp_stats()
        except:
            systems_stats["hyper_advanced_nlp"] = None
        
        try:
            systems_stats["ultra_fast_nlp"] = ultra_fast_nlp_system.get_ultra_fast_nlp_stats()
        except:
            systems_stats["ultra_fast_nlp"] = None
        
        try:
            systems_stats["ultimate_enhanced_nlp"] = ultimate_enhanced_nlp_system.get_ultimate_enhanced_nlp_stats()
        except:
            systems_stats["ultimate_enhanced_nlp"] = None
        
        # Compare performance metrics
        performance_comparison = {}
        for system_name, stats in systems_stats.items():
            if stats:
                performance_comparison[system_name] = {
                    "average_processing_time": stats.get("average_processing_time", 0),
                    "throughput_per_second": stats.get("throughput_per_second", 0),
                    "success_rate": stats.get("success_rate", 0),
                    "total_requests": stats.get("stats", {}).get("total_requests", 0) if "total_requests" in stats.get("stats", {}) else stats.get("stats", {}).get(f"total_{system_name}_requests", 0)
                }
        
        comparison["performance_comparison"] = performance_comparison
        
        # Find best performing system
        if performance_comparison:
            best_throughput = max(performance_comparison.items(), key=lambda x: x[1]["throughput_per_second"])
            best_success_rate = max(performance_comparison.items(), key=lambda x: x[1]["success_rate"])
            fastest_processing = min(performance_comparison.items(), key=lambda x: x[1]["average_processing_time"])
            
            comparison["best_performance"] = {
                "highest_throughput": {
                    "system": best_throughput[0],
                    "throughput": best_throughput[1]["throughput_per_second"]
                },
                "highest_success_rate": {
                    "system": best_success_rate[0],
                    "success_rate": best_success_rate[1]["success_rate"]
                },
                "fastest_processing": {
                    "system": fastest_processing[0],
                    "processing_time": fastest_processing[1]["average_processing_time"]
                }
            }
        
        return comparison
        
    except Exception as e:
        logger.error(f"Error in comparison: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# System information endpoint
@app.get("/info")
async def get_system_info():
    """Get comprehensive system information"""
    return {
        "system": "Ultimate Enhanced NLP API",
        "version": "1.0.0",
        "description": "Real, working ultimate enhanced Natural Language Processing system",
        "features": [
            "Basic NLP - Tokenization, sentiment analysis, text classification",
            "Enhanced NLP - Advanced preprocessing, keyword extraction, similarity analysis",
            "Advanced NLP - Dependency parsing, coreference resolution, entity linking",
            "Super Advanced NLP - Transformer-based analysis, creative text generation",
            "Hyper Advanced NLP - Multimodal analysis, real-time processing, edge computing",
            "Ultra Fast NLP - Extreme performance optimizations, lightning processing",
            "Ultimate Enhanced NLP - Ultimate enhanced analysis with advanced optimizations"
        ],
        "capabilities": {
            "text_analysis": "Comprehensive text analysis with multiple levels of sophistication",
            "batch_processing": "High-performance batch processing capabilities",
            "real_time_processing": "Real-time text analysis with streaming support",
            "performance_optimization": "Extreme performance optimizations including caching, compression, quantization",
            "scalability": "Horizontal and vertical scaling capabilities",
            "monitoring": "Comprehensive monitoring and statistics",
            "health_checks": "System health monitoring and diagnostics"
        },
        "endpoints": {
            "basic_nlp": "/api/v1/nlp",
            "enhanced_nlp": "/api/v1/enhanced-nlp",
            "advanced_nlp": "/api/v1/advanced-nlp",
            "super_advanced_nlp": "/api/v1/super-advanced-nlp",
            "hyper_advanced_nlp": "/api/v1/hyper-advanced-nlp",
            "ultra_fast_nlp": "/api/v1/ultra-fast-nlp",
            "ultimate_enhanced_nlp": "/api/v1/ultimate-enhanced-nlp"
        },
        "documentation": {
            "swagger_ui": "/docs",
            "redoc": "/redoc",
            "openapi": "/openapi.json"
        }
    }

if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "ultimate_enhanced_nlp_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )












